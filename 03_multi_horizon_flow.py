"""
Module 3: Multi-Horizon ONI Forecast — Metaflow Flow
=====================================================
Productionized version of the multi-horizon experiment.

Models:
  - persistence: naive forecast (ONI_t+h = ONI_t), the hardest-to-beat control
  - linear: Ridge regression on baseline features
  - baseline: XGBoost on current-month features only
  - candidate: XGBoost on all lags + rolling stats (214 features)
  - adaptive: XGBoost with horizon-tuned features (fewer features at short
              horizons, more at long horizons) + feature selection

Architecture:
  start → train_horizon (foreach horizon × model_type) → join → end

Usage:
    python 03_multi_horizon_flow.py --environment=pypi run
    python 03_multi_horizon_flow.py card view end
"""

from metaflow import (
    step,
    FlowSpec,
    Flow,
    S3,
    Parameter,
    current,
    card,
    pypi_base,
    project,
)

HORIZONS = list(range(1, 13))  # 1 through 12 months
MODEL_TYPES = ["persistence", "linear", "baseline", "candidate", "adaptive"]


@project(name="oni_forecast")
@pypi_base(
    python="3.12",
    packages={
        "pandas": "2.2.3",
        "numpy": "1.26.4",
        "xgboost": "2.1.4",
        "scikit-learn": "1.6.1",
        "hvplot": "0.11.2",
        "holoviews": "1.20.0",
        "bokeh": "3.6.2",
    },
)
class ONIMultiHorizonFlow(FlowSpec):
    """
    Compare ONI forecasting models across multiple horizons.

    5 models × 5 horizons = 25 parallel branches, each running
    walk-forward cross-validation.
    """

    data_url = Parameter(
        "data_url",
        default="https://raw.githubusercontent.com/ahuang11/ninodata/master/nino_ml.csv",
        help="URL to the nino_ml.csv dataset.",
    )
    train_years = Parameter(
        "train_years", default=10, help="Years of training data per fold."
    )
    valid_years = Parameter(
        "valid_years", default=5, help="Years of validation data per fold."
    )

    @step
    def start(self):
        """Load data and fan out across (horizon, model_type) pairs."""
        import pandas as pd

        self.df = pd.read_csv(self.data_url, index_col=0, parse_dates=True)
        print(f"Loaded data: {self.df.shape}, "
              f"{self.df.index.min().date()} to {self.df.index.max().date()}")

        self.xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "eval_metric": ["mae", "rmse"],
            "seed": 42,
            "early_stopping_rounds": 10,
        }

        self.experiment_configs = [
            {"horizon": h, "model_type": m}
            for h in HORIZONS
            for m in MODEL_TYPES
        ]
        print(f"Launching {len(self.experiment_configs)} experiments: "
              f"{len(HORIZONS)} horizons × {len(MODEL_TYPES)} models")

        self.next(self.train_horizon, foreach="experiment_configs")

    @card(type="blank", id="training_progress")
    @step
    def train_horizon(self):
        """Train walk-forward CV for one (horizon, model_type) pair."""
        import numpy as np
        import pandas as pd
        import xgboost as xgb
        from sklearn.metrics import (
            mean_absolute_error,
            root_mean_squared_error,
        )
        from metaflow.cards import Markdown, Table

        config = self.input
        self.horizon = config["horizon"]
        self.model_type = config["model_type"]
        target = "nino3.4_anom"
        predictor_cols = [c for c in self.df.columns if c != target]
        anom_cols = [c for c in predictor_cols if "_anom" in c]

        current.card["training_progress"].append(
            Markdown(f"## {self.model_type.title()} — {self.horizon}-month horizon")
        )

        # --- Feature engineering (pluggable) ---
        if self.model_type == "persistence":
            X, y = self._persistence_features(self.df, target, self.horizon)
        elif self.model_type in ("linear", "baseline"):
            X, y = self._baseline_features(
                self.df, predictor_cols, target, self.horizon
            )
        elif self.model_type == "candidate":
            X, y = self._candidate_features(
                self.df, predictor_cols, target, self.horizon, anom_cols
            )
        else:  # adaptive
            X, y = self._adaptive_features(
                self.df, predictor_cols, target, self.horizon, anom_cols
            )

        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]

        current.card["training_progress"].append(
            Markdown(f"Features: {self.n_features}, Samples: {self.n_samples}")
        )

        # --- Walk-forward CV ---
        all_years = sorted(X.index.year.unique())
        fold_size = self.train_years + self.valid_years
        n_folds = max(1, len(all_years) - fold_size)

        fold_results = []
        all_actuals = []
        all_preds = []
        all_dates = []

        for i in range(n_folds):
            fold_years = all_years[i: i + fold_size]
            t_years = fold_years[: self.train_years]
            v_years = fold_years[self.train_years:]

            X_train = X[X.index.year.isin(t_years)]
            y_train = y[X_train.index]
            X_valid = X[X.index.year.isin(v_years)]
            y_valid = y[X_valid.index]

            # --- Model training (pluggable) ---
            if self.model_type == "persistence":
                # "Prediction" is just the current ONI value (already in X)
                y_pred = X_valid["current_target"].values
            elif self.model_type == "linear":
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)
            elif self.model_type == "adaptive":
                # Adaptive uses feature selection within each fold
                from sklearn.feature_selection import SelectKBest, f_regression
                k = min(40, X_train.shape[1])  # cap at 40 features
                selector = SelectKBest(f_regression, k=k)
                X_train_sel = selector.fit_transform(X_train, y_train)
                X_valid_sel = selector.transform(X_valid)
                model = xgb.XGBRegressor(**self.xgb_params)
                model.fit(
                    X_train_sel, y_train,
                    eval_set=[(X_train_sel, y_train), (X_valid_sel, y_valid)],
                    verbose=False,
                )
                y_pred = model.predict(X_valid_sel)
            else:
                # baseline and candidate both use XGBoost directly
                model = xgb.XGBRegressor(**self.xgb_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_valid, y_valid)],
                    verbose=False,
                )
                y_pred = model.predict(X_valid)

            mae = mean_absolute_error(y_valid, y_pred)
            rmse = root_mean_squared_error(y_valid, y_pred)

            fold_results.append({
                "fold": i,
                "train": f"{t_years[0]}–{t_years[-1]}",
                "valid": f"{v_years[0]}–{v_years[-1]}",
                "mae": mae,
                "rmse": rmse,
            })

            all_actuals.extend(y_valid.values.tolist())
            all_preds.extend(
                y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
            )
            all_dates.extend(y_valid.index.strftime("%Y-%m-%d").tolist())

        # Store results
        self.mae_mean = float(np.mean([r["mae"] for r in fold_results]))
        self.mae_std = float(np.std([r["mae"] for r in fold_results]))
        self.rmse_mean = float(np.mean([r["rmse"] for r in fold_results]))
        self.rmse_std = float(np.std([r["rmse"] for r in fold_results]))
        self.n_folds = n_folds

        self.prediction_data = {
            "dates": all_dates,
            "actuals": all_actuals,
            "preds": all_preds,
        }

        # --- Serialize the last fold's model for the model registry ---
        self.model_bytes = None
        self.model_format = None
        if self.model_type == "persistence":
            pass
        elif self.model_type == "linear":
            import pickle
            self.model_bytes = pickle.dumps(model)
            self.model_format = "pickle"
        elif self.model_type == "adaptive":
            import pickle
            self.model_bytes = pickle.dumps({"selector": selector, "model": model})
            self.model_format = "pickle"
        else:  # baseline, candidate — XGBoost models
            import io
            buf = io.BytesIO()
            model.save_model(buf)
            self.model_bytes = buf.getvalue()
            self.model_format = "xgboost"

        # Feature importances (where available)
        if self.model_type == "persistence":
            self.feature_importances = {"feature": [], "importance": []}
        elif self.model_type == "linear":
            coefs = np.abs(model.coef_)
            coefs = coefs / coefs.sum() if coefs.sum() > 0 else coefs
            self.feature_importances = {
                "feature": list(X_train.columns),
                "importance": coefs.tolist(),
            }
        elif self.model_type == "adaptive":
            # Map selected feature indices back to names
            mask = selector.get_support()
            sel_names = [X_train.columns[j] for j in range(len(mask)) if mask[j]]
            imp = model.feature_importances_.tolist()
            self.feature_importances = {
                "feature": sel_names,
                "importance": imp,
            }
        else:
            self.feature_importances = {
                "feature": model.feature_names_in_.tolist(),
                "importance": model.feature_importances_.tolist(),
            }

        current.card["training_progress"].append(
            Markdown(f"**MAE: {self.mae_mean:.3f} ± {self.mae_std:.3f}**")
        )
        current.card["training_progress"].append(
            Table(
                headers=["Fold", "Train", "Valid", "MAE", "RMSE"],
                data=[
                    [
                        Markdown(str(r["fold"])),
                        Markdown(r["train"]),
                        Markdown(r["valid"]),
                        Markdown(f"{r['mae']:.3f}"),
                        Markdown(f"{r['rmse']:.3f}"),
                    ]
                    for r in fold_results
                ],
            )
        )

        print(f"{self.model_type} @ {self.horizon}mo: "
              f"MAE={self.mae_mean:.3f}±{self.mae_std:.3f} "
              f"({n_folds} folds, {self.n_samples} samples, "
              f"{self.n_features} features)")

        self.next(self.join)

    # ------------------------------------------------------------------ #
    # PLUGGABLE FEATURE ENCODERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _persistence_features(df, target_col, horizon):
        """Persistence: predict ONI_t+h = ONI_t. No real features needed."""
        import pandas as pd
        X = pd.DataFrame({"current_target": df[target_col]}, index=df.index)
        y = df[target_col].shift(-horizon)
        valid = X.notna().all(axis=1) & y.notna()
        return X[valid], y[valid]

    @staticmethod
    def _baseline_features(df, predictor_cols, target_col, horizon):
        """Current values + month."""
        import pandas as pd
        X = df[predictor_cols].copy()
        X["month"] = df.index.month
        y = df[target_col].shift(-horizon)
        valid = X.notna().all(axis=1) & y.notna()
        return X[valid], y[valid]

    @staticmethod
    def _candidate_features(df, predictor_cols, target_col, horizon, anom_cols):
        """Current values + all lags + all rolling stats + diffs + month."""
        import pandas as pd

        lag_months = [1, 2, 3, 6, 9, 12]
        rolling_windows = [3, 6, 12]

        parts = [df[predictor_cols]]

        lag_frames = {}
        for col in anom_cols:
            for lag in lag_months:
                lag_frames[f"{col}_lag{lag}"] = df[col].shift(lag)
        parts.append(pd.DataFrame(lag_frames, index=df.index))

        roll_frames = {}
        for col in anom_cols:
            for w in rolling_windows:
                roll_frames[f"{col}_ma{w}"] = df[col].rolling(
                    window=w, min_periods=w).mean()
                roll_frames[f"{col}_std{w}"] = df[col].rolling(
                    window=w, min_periods=w).std()
        parts.append(pd.DataFrame(roll_frames, index=df.index))

        diff_frames = {}
        for col in anom_cols:
            diff_frames[f"{col}_diff1"] = df[col].diff(1)
            diff_frames[f"{col}_diff3"] = df[col].diff(3)
        parts.append(pd.DataFrame(diff_frames, index=df.index))

        parts.append(pd.DataFrame({"month": df.index.month}, index=df.index))

        X = pd.concat(parts, axis=1)
        y = df[target_col].shift(-horizon)
        valid = X.notna().all(axis=1) & y.notna()
        return X[valid], y[valid]

    @staticmethod
    def _adaptive_features(df, predictor_cols, target_col, horizon, anom_cols):
        """
        Horizon-adaptive features with seasonal awareness:

        Temporal features (horizon-tuned):
        - Short horizons (1-3mo): only short lags (1,2,3) + short rolling (3)
        - Medium horizons (4-6mo): medium lags (1,2,3,6) + medium rolling (3,6)
        - Long horizons (7-12mo): all lags + all rolling

        Seasonal features (new):
        - target_month_sin/cos: cyclical encoding of the month being predicted
        - init_month_sin/cos: cyclical encoding of the initialization month
        - crosses_spring: binary flag if forecast window crosses MAM (Mar-May)
        - months_through_spring: how many spring months (3,4,5) the forecast
          must predict through (0-3), capturing spring barrier severity

        Feature selection (SelectKBest) is applied during training, not here.
        """
        import pandas as pd
        import numpy as np

        if horizon <= 3:
            lag_months = [1, 2, 3]
            rolling_windows = [3]
        elif horizon <= 6:
            lag_months = [1, 2, 3, 6]
            rolling_windows = [3, 6]
        else:
            lag_months = [1, 2, 3, 6, 9, 12]
            rolling_windows = [3, 6, 12]

        parts = [df[predictor_cols]]

        # Lag features
        lag_frames = {}
        for col in anom_cols:
            for lag in lag_months:
                lag_frames[f"{col}_lag{lag}"] = df[col].shift(lag)
        parts.append(pd.DataFrame(lag_frames, index=df.index))

        # Rolling features
        roll_frames = {}
        for col in anom_cols:
            for w in rolling_windows:
                roll_frames[f"{col}_ma{w}"] = df[col].rolling(
                    window=w, min_periods=w).mean()
                roll_frames[f"{col}_std{w}"] = df[col].rolling(
                    window=w, min_periods=w).std()
        parts.append(pd.DataFrame(roll_frames, index=df.index))

        # Rate-of-change features
        diff_frames = {}
        for col in anom_cols:
            diff_frames[f"{col}_diff1"] = df[col].diff(1)
            if horizon > 3:
                diff_frames[f"{col}_diff3"] = df[col].diff(3)
        parts.append(pd.DataFrame(diff_frames, index=df.index))

        # --- Seasonal features ---
        init_month = df.index.month  # month when forecast is made
        target_month = (init_month - 1 + horizon) % 12 + 1  # month being predicted

        seasonal = pd.DataFrame(index=df.index)

        # Cyclical encoding of initialization month
        seasonal["init_month_sin"] = np.sin(2 * np.pi * init_month / 12)
        seasonal["init_month_cos"] = np.cos(2 * np.pi * init_month / 12)

        # Cyclical encoding of target month
        seasonal["target_month_sin"] = np.sin(2 * np.pi * target_month / 12)
        seasonal["target_month_cos"] = np.cos(2 * np.pi * target_month / 12)

        # Spring barrier features
        spring_months = {3, 4, 5}  # March, April, May
        crosses_spring = []
        months_through_spring = []
        for im in init_month:
            # Which months does this forecast cross through?
            forecast_months = set(
                (im - 1 + step) % 12 + 1 for step in range(1, horizon + 1)
            )
            spring_overlap = forecast_months & spring_months
            crosses_spring.append(1 if spring_overlap else 0)
            months_through_spring.append(len(spring_overlap))

        seasonal["crosses_spring"] = crosses_spring
        seasonal["months_through_spring"] = months_through_spring

        parts.append(seasonal)

        X = pd.concat(parts, axis=1)
        y = df[target_col].shift(-horizon)
        valid = X.notna().all(axis=1) & y.notna()
        return X[valid], y[valid]

    # ------------------------------------------------------------------ #
    # JOIN + END
    # ------------------------------------------------------------------ #

    @step
    def join(self, inputs):
        """Aggregate results, pick best model per horizon, save to S3."""
        import pandas as pd
        import tempfile
        import os

        rows = []
        self.all_predictions = {}
        self.all_importances = {}
        models_by_horizon = {}  # horizon -> list of (model_type, mae, bytes, format)

        for inp in inputs:
            rows.append({
                "horizon": inp.horizon,
                "model_type": inp.model_type,
                "mae_mean": inp.mae_mean,
                "mae_std": inp.mae_std,
                "rmse_mean": inp.rmse_mean,
                "rmse_std": inp.rmse_std,
                "n_folds": inp.n_folds,
                "n_samples": inp.n_samples,
                "n_features": inp.n_features,
            })
            key = f"{inp.model_type}_{inp.horizon}mo"
            self.all_predictions[key] = inp.prediction_data
            self.all_importances[key] = inp.feature_importances

            if inp.model_bytes is not None:
                models_by_horizon.setdefault(inp.horizon, []).append({
                    "model_type": inp.model_type,
                    "mae": inp.mae_mean,
                    "bytes": inp.model_bytes,
                    "format": inp.model_format,
                })

        self.summary_df = pd.DataFrame(rows).sort_values(
            ["horizon", "model_type"]
        )

        # --- Model registry: best model per horizon → S3 ---
        self.best_models = {}
        files_to_upload = []

        for h, candidates in models_by_horizon.items():
            best = min(candidates, key=lambda c: c["mae"])
            ext = "pkl" if best["format"] == "pickle" else "json"
            fname = f"oni_best_{best['model_type']}_{h}mo.{ext}"
            tmp_path = os.path.join(tempfile.gettempdir(), fname)
            with open(tmp_path, "wb") as f:
                f.write(best["bytes"])
            files_to_upload.append((fname, tmp_path))
            self.best_models[h] = {
                "model_type": best["model_type"],
                "mae": best["mae"],
                "format": best["format"],
                "filename": fname,
            }

        if files_to_upload:
            with S3(run=self) as s3:
                uploaded = s3.put_files(files_to_upload)
                for (fname, _), (s3_key, s3_url) in zip(files_to_upload, uploaded):
                    for h, info in self.best_models.items():
                        if info["filename"] == fname:
                            info["s3_key"] = s3_key
                            info["s3_url"] = s3_url
                            break

        # Tag the run as production
        run = Flow(current.flow_name)[current.run_id]
        run.add_tag("production")

        print("\n" + self.summary_df.to_string(index=False))
        print("\nBest models saved to S3:")
        for h in sorted(self.best_models):
            info = self.best_models[h]
            url = info.get("s3_url", "local")
            print(f"  {h:2d}mo: {info['model_type']:12s} "
                  f"MAE={info['mae']:.3f}  → {url}")

        self.next(self.end)

    @card(type="blank", id="summary")
    @card(type="blank", id="skill_curve")
    @card(type="blank", id="predictions_map")
    @card(type="blank", id="scatter_map")
    @card(type="blank", id="importance_map")
    @step
    def end(self):
        """Render summary cards with HoloMap visualizations."""
        import pandas as pd
        import numpy as np
        import holoviews as hv
        import hvplot.pandas  # noqa: F401
        from metaflow.cards import Markdown, Table

        hv.extension("bokeh")

        # --- Card 1: Summary table ---
        current.card["summary"].append(Markdown("# ONI Multi-Horizon Forecast Results"))
        current.card["summary"].append(
            Table.from_dataframe(
                self.summary_df[
                    ["horizon", "model_type", "mae_mean", "mae_std",
                     "rmse_mean", "n_folds", "n_samples", "n_features"]
                ].round(3)
            )
        )

        # --- Card 2: Skill degradation curve ---
        skill_plot = self.summary_df.hvplot.line(
            x="horizon",
            y="mae_mean",
            by="model_type",
            title="Forecast Skill vs Lead Time",
            xlabel="Forecast Horizon (months)",
            ylabel="Mean Absolute Error (°C)",
            width=750,
            height=420,
            line_width=2,
        ) * hv.HLine(0.5).opts(
            color="gray", line_dash="dotted", alpha=0.5
        )
        current.card["skill_curve"].append(Markdown("# Skill Degradation Curve"))
        current.card["skill_curve"].append(Markdown(
            "MAE vs forecast horizon. Gray line = 0.5°C (ONI threshold). "
            "Persistence = naive 'no change' forecast."
        ))
        self._embed_hv(current.card["skill_curve"], skill_plot)

        # --- Card 3: Predictions HoloMap (best model = adaptive) ---
        # Aggregate overlapping fold predictions into mean + envelope
        best_model = "adaptive"
        pred_plots = {}
        for h in HORIZONS:
            key = f"{best_model}_{h}mo"
            if key not in self.all_predictions:
                continue
            data = self.all_predictions[key]
            raw = pd.DataFrame({
                "date": pd.to_datetime(data["dates"]),
                "actual": data["actuals"],
                "predicted": data["preds"],
            })
            # Aggregate: mean, min, max across folds for each date
            agg = raw.groupby("date").agg(
                actual=("actual", "first"),
                pred_mean=("predicted", "mean"),
                pred_min=("predicted", "min"),
                pred_max=("predicted", "max"),
            ).sort_index()

            envelope = agg.hvplot.area(
                x="date", y="pred_min", y2="pred_max",
                alpha=0.25, color="#2ca02c", label="Fold spread",
            )
            mean_line = agg.hvplot.line(
                x="date", y="pred_mean", color="#2ca02c",
                label="Predicted (mean)", line_width=1.5,
            )
            actual_line = agg.hvplot.line(
                x="date", y="actual", color="black",
                label="Actual", line_width=1, alpha=0.7,
            )

            plot = (
                envelope * mean_line * actual_line
                * hv.HLine(0.5).opts(color="red", line_dash="dashed", alpha=0.4)
                * hv.HLine(-0.5).opts(color="blue", line_dash="dashed", alpha=0.4)
            ).opts(
                title=f"{h}-Month Ahead ONI (Adaptive)",
                ylabel="Niño 3.4 Anomaly (°C)",
                width=800, height=350,
            )
            pred_plots[h] = plot

        if pred_plots:
            hmap = hv.HoloMap(pred_plots, kdims=["Horizon (months)"])
            current.card["predictions_map"].append(
                Markdown("# Predictions by Horizon (Adaptive)")
            )
            current.card["predictions_map"].append(
                Markdown("Use the slider to switch between forecast horizons.")
            )
            self._embed_hv(current.card["predictions_map"], hmap)

        # --- Card 4: Scatter HoloMap ---
        scatter_plots = {}
        for h in HORIZONS:
            key = f"{best_model}_{h}mo"
            if key not in self.all_predictions:
                continue
            data = self.all_predictions[key]
            scatter_df = pd.DataFrame({
                "date": pd.to_datetime(data["dates"]),
                "actual": data["actuals"],
                "predicted": data["preds"],
            }).groupby("date").last()  # de-duplicate here too
            mae_row = self.summary_df[
                (self.summary_df.horizon == h) &
                (self.summary_df.model_type == best_model)
            ]
            mae = mae_row["mae_mean"].values[0] if len(mae_row) > 0 else 0

            plot = (
                scatter_df.hvplot.scatter(
                    x="actual", y="predicted",
                    alpha=0.4, size=15,
                    xlabel="Actual", ylabel="Predicted",
                    width=450, height=450,
                )
                * hv.Slope(slope=1, y_intercept=0).opts(
                    color="red", line_dash="dashed", alpha=0.5
                )
            ).opts(title=f"{h}-mo lead (MAE={mae:.3f})")
            scatter_plots[h] = plot

        if scatter_plots:
            smap = hv.HoloMap(scatter_plots, kdims=["Horizon (months)"])
            current.card["scatter_map"].append(
                Markdown("# Predicted vs Actual by Horizon (Adaptive)")
            )
            self._embed_hv(current.card["scatter_map"], smap)

        # --- Card 5: Feature importance HoloMap (adaptive, top 15) ---
        imp_plots = {}
        for h in HORIZONS:
            key = f"{best_model}_{h}mo"
            if key not in self.all_importances:
                continue
            imp_data = self.all_importances[key]
            if not imp_data["feature"]:
                continue
            imp_df = pd.DataFrame(imp_data).sort_values(
                "importance", ascending=True
            ).tail(15)

            plot = imp_df.hvplot.barh(
                x="feature", y="importance",
                title=f"Top Features — {h}-mo horizon (Adaptive)",
                xlabel="Importance (Gain)", ylabel="",
                width=700, height=450,
                color="#2ca02c",
            )
            imp_plots[h] = plot

        if imp_plots:
            imap = hv.HoloMap(imp_plots, kdims=["Horizon (months)"])
            current.card["importance_map"].append(
                Markdown("# Feature Importance by Horizon (Adaptive)")
            )
            current.card["importance_map"].append(
                Markdown("Watch how selected features and their importance shift at longer lead times.")
            )
            self._embed_hv(current.card["importance_map"], imap)

        print("Cards rendered.")

    @staticmethod
    def _embed_hv(card_section, hv_obj, width=950, height=550):
        """Render a HoloViews/hvplot object as HTML iframe in a Metaflow card."""
        import base64
        import tempfile
        import holoviews as hv
        from metaflow.cards import Markdown

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            hv.save(hv_obj, f.name, backend="bokeh")
            f.seek(0)
            html_str = open(f.name).read()

        b64 = base64.b64encode(html_str.encode()).decode()
        iframe = (
            f'<iframe src="data:text/html;base64,{b64}" '
            f'width="{width}" height="{height}" '
            f'frameborder="0" style="border:none;"></iframe>'
        )
        card_section.append(Markdown(iframe))


if __name__ == "__main__":
    ONIMultiHorizonFlow()

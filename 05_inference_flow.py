"""
ONI Inference Flow
==================
Loads the best models from the latest training run and makes
real forecasts from the most recent data.

Triggered by the sensor flow's 'new_oni_data' event, or run manually.

Usage:
    python 05_inference_flow.py --environment=pypi run
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
    trigger,
)

HORIZONS = list(range(1, 13))


@trigger(events=["new_oni_data"])
@project(name="oni_forecast")
@pypi_base(
    python="3.12",
    packages={
        "pandas": "2.2.3",
        "numpy": "1.26.4",
        "xgboost": "2.1.4",
        "scikit-learn": "1.6.1",
    },
)
class ONIInferenceFlow(FlowSpec):
    """
    Load best models per horizon from the latest training run,
    fetch current data, and produce forecasts.
    """

    data_url = Parameter(
        "data_url",
        default="https://raw.githubusercontent.com/ahuang11/ninodata/master/nino_ml.csv",
        help="URL to the latest nino_ml.csv.",
    )
    train_flow_name = Parameter(
        "train_flow_name",
        default="ONIMultiHorizonFlow",
        help="Name of the training flow to pull models from.",
    )

    @card(type="blank", id="forecast")
    @step
    def start(self):
        """Load data, retrieve best models, produce forecasts."""
        import pandas as pd
        import numpy as np
        import pickle
        import xgboost as xgb
        from metaflow.cards import Markdown, Table

        # --- Load latest data ---
        df = pd.read_csv(self.data_url, index_col=0, parse_dates=True)
        latest_date = df.index.max()
        print(f"Latest data: {latest_date.date()}")
        print(f"Data shape: {df.shape}")

        # --- Find latest production training run ---
        train_runs = list(Flow(self.train_flow_name).runs("production"))
        train_run = None
        for r in train_runs:
            if r.successful:
                train_run = r
                break
        if train_run is None:
            raise ValueError("No successful production training run found.")
        print(f"Using training run: {train_run.pathspec}")

        # --- Get best_models registry from the join step ---
        join_step = train_run["join"]
        join_task = list(join_step.tasks())[0]
        best_models = join_task.data.best_models
        xgb_params = train_run["start"].task.data.xgb_params

        print(f"Best models available for horizons: {sorted(best_models.keys())}")

        # --- Produce forecasts for each horizon ---
        target = "nino3.4_anom"
        predictor_cols = [c for c in df.columns if c != target]
        anom_cols = [c for c in predictor_cols if "_anom" in c]

        forecasts = []
        current.card["forecast"].append(Markdown("# ONI Forecast"))
        current.card["forecast"].append(
            Markdown(f"**Data through:** {latest_date.date()}")
        )
        current.card["forecast"].append(
            Markdown(f"**Training run:** `{train_run.pathspec}`")
        )

        for h in sorted(best_models.keys()):
            info = best_models[h]
            model_type = info["model_type"]
            s3_key = info.get("s3_key")

            if s3_key is None:
                print(f"  {h}mo: no S3 key, skipping")
                continue

            # Download model from S3
            with S3(run=train_run) as s3:
                obj = s3.get(s3_key)
                with open(obj.path, "rb") as f:
                    model_bytes = f.read()

            # Prepare features using the same encoder as training
            # (this is the training-serving parity principle from the course)
            if model_type in ("linear", "baseline"):
                X = df[predictor_cols].copy()
                X["month"] = df.index.month
                X = X.dropna()
            elif model_type == "candidate":
                X = self._candidate_features_only(df, predictor_cols, anom_cols)
            elif model_type == "adaptive":
                X = self._adaptive_features_only(df, predictor_cols, anom_cols, h)
            else:
                continue

            # Use the latest row for prediction
            if X.empty:
                print(f"  {h}mo: no valid features, skipping")
                continue
            X_latest = X.iloc[[-1]]

            # Load and predict
            if info["format"] == "pickle":
                obj = pickle.loads(model_bytes)
                if isinstance(obj, dict):  # adaptive: {selector, model}
                    X_sel = obj["selector"].transform(X_latest)
                    pred = obj["model"].predict(X_sel)[0]
                else:  # linear
                    pred = obj.predict(X_latest)[0]
            else:  # xgboost
                model = xgb.XGBRegressor(**xgb_params)
                import tempfile as _tf
                import os as _os
                tmp = _os.path.join(_tf.gettempdir(), "xgb_inf_model.json")
                with open(tmp, "wb") as f:
                    f.write(model_bytes)
                model.load_model(tmp)
                pred = model.predict(X_latest)[0]

            target_date = latest_date + pd.DateOffset(months=h)
            forecasts.append({
                "horizon": h,
                "model_type": model_type,
                "init_date": latest_date.date(),
                "target_date": target_date.date(),
                "prediction": round(float(pred), 3),
                "cv_mae": info["mae"],
            })
            print(f"  {h:2d}mo ({model_type:12s}): "
                  f"{latest_date.date()} → {target_date.date()} = {pred:.3f}°C")

        self.forecasts = forecasts
        self.forecast_df = pd.DataFrame(forecasts)

        # Render card
        if not self.forecast_df.empty:
            current.card["forecast"].append(Markdown("## Predictions"))
            current.card["forecast"].append(
                Table.from_dataframe(self.forecast_df.round(3))
            )

            # Classify
            def classify(val):
                if val >= 0.5:
                    return "🔴 El Niño"
                elif val <= -0.5:
                    return "🔵 La Niña"
                return "⚪ Neutral"

            current.card["forecast"].append(Markdown("## Classification"))
            for _, row in self.forecast_df.iterrows():
                label = classify(row["prediction"])
                current.card["forecast"].append(
                    Markdown(
                        f"- **{row['horizon']}mo** → {row['target_date']}: "
                        f"**{row['prediction']:.2f}°C** {label}"
                    )
                )

        self.next(self.end)

    @step
    def end(self):
        """Store forecast for monitoring."""
        print(f"Forecast complete: {len(self.forecasts)} horizons")
        if self.forecasts:
            print(self.forecast_df.to_string(index=False))

    # --- Feature encoders (must match training exactly) ---

    @staticmethod
    def _candidate_features_only(df, predictor_cols, anom_cols):
        """Same as training candidate encoder, returns X only."""
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
                roll_frames[f"{col}_ma{w}"] = df[col].rolling(window=w, min_periods=w).mean()
                roll_frames[f"{col}_std{w}"] = df[col].rolling(window=w, min_periods=w).std()
        parts.append(pd.DataFrame(roll_frames, index=df.index))
        diff_frames = {}
        for col in anom_cols:
            diff_frames[f"{col}_diff1"] = df[col].diff(1)
            diff_frames[f"{col}_diff3"] = df[col].diff(3)
        parts.append(pd.DataFrame(diff_frames, index=df.index))
        parts.append(pd.DataFrame({"month": df.index.month}, index=df.index))
        X = pd.concat(parts, axis=1)
        return X.dropna()

    @staticmethod
    def _adaptive_features_only(df, predictor_cols, anom_cols, horizon):
        """Same as training adaptive encoder, returns X only."""
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
        lag_frames = {}
        for col in anom_cols:
            for lag in lag_months:
                lag_frames[f"{col}_lag{lag}"] = df[col].shift(lag)
        parts.append(pd.DataFrame(lag_frames, index=df.index))
        roll_frames = {}
        for col in anom_cols:
            for w in rolling_windows:
                roll_frames[f"{col}_ma{w}"] = df[col].rolling(window=w, min_periods=w).mean()
                roll_frames[f"{col}_std{w}"] = df[col].rolling(window=w, min_periods=w).std()
        parts.append(pd.DataFrame(roll_frames, index=df.index))
        diff_frames = {}
        for col in anom_cols:
            diff_frames[f"{col}_diff1"] = df[col].diff(1)
            if horizon > 3:
                diff_frames[f"{col}_diff3"] = df[col].diff(3)
        parts.append(pd.DataFrame(diff_frames, index=df.index))
        init_month = df.index.month
        target_month = (init_month - 1 + horizon) % 12 + 1
        seasonal = pd.DataFrame(index=df.index)
        seasonal["init_month_sin"] = np.sin(2 * np.pi * init_month / 12)
        seasonal["init_month_cos"] = np.cos(2 * np.pi * init_month / 12)
        seasonal["target_month_sin"] = np.sin(2 * np.pi * target_month / 12)
        seasonal["target_month_cos"] = np.cos(2 * np.pi * target_month / 12)
        spring_months = {3, 4, 5}
        crosses = []
        through = []
        for im in init_month:
            fm = set((im - 1 + s) % 12 + 1 for s in range(1, horizon + 1))
            overlap = fm & spring_months
            crosses.append(1 if overlap else 0)
            through.append(len(overlap))
        seasonal["crosses_spring"] = crosses
        seasonal["months_through_spring"] = through
        parts.append(seasonal)
        X = pd.concat(parts, axis=1)
        return X.dropna()


if __name__ == "__main__":
    ONIInferenceFlow()

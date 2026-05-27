"""
Module 2: ONI Candidate Model — Lag & Rolling Features
=======================================================
Goal: Beat the baseline by giving XGBoost temporal context.

The baseline only sees the current month's values. This candidate adds:
  1. Lag features — values from the past N months
  2. Rolling statistics — moving averages and std over recent windows
  3. Rate-of-change features — momentum signals

Same walk-forward CV, same hyperparameters. The only change is features.

NaN handling: No ffill/bfill — we only use rows where all selected columns
have real observed values. Columns that start later simply reduce the usable
date range. This is honest: no fabricated data.

Usage:
    pip install pandas numpy xgboost scikit-learn hvplot
    python 02_candidate.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import hvplot.pandas  # noqa: F401
import holoviews as hv
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, max_error
from pprint import pprint

hv.extension("bokeh")

# --------------------------------------------------------------------------- #
# 1. LOAD DATA
# --------------------------------------------------------------------------- #

DATA_URL = (
    "https://raw.githubusercontent.com/ahuang11/ninodata/master/nino_ml.csv"
)

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df_raw = pd.read_csv(DATA_URL, index_col=0, parse_dates=True)

print(f"Raw shape: {df_raw.shape}")
print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

# Show non-null counts to understand data availability
print(f"\nNon-null counts per column:")
for col in df_raw.columns:
    n = df_raw[col].notna().sum()
    print(f"  {col:25s} {n:4d} / {len(df_raw)}")

TARGET = "nino3.4_anom"
FORECAST_HORIZON = 3  # months ahead

# Use all columns except target as predictors
PREDICTOR_COLS = [c for c in df_raw.columns if c != TARGET]

# NO ffill/bfill — just use the raw data as-is.
# Rows with NaN in any base predictor column will be dropped after
# feature engineering (the valid_mask step handles this cleanly).
df = df_raw.copy()

# Report how many rows have all base predictors present
base_complete = df[PREDICTOR_COLS].notna().all(axis=1).sum()
print(f"\nRows with all base predictors present: {base_complete} / {len(df)}")

# --------------------------------------------------------------------------- #
# 2. FEATURE ENGINEERING — Candidate features
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 2: Feature Engineering (Candidate)")
print("=" * 60)

# --- Configuration ---
LAG_MONTHS = [1, 2, 3, 6, 9, 12]  # how many months back to look
ROLLING_WINDOWS = [3, 6, 12]       # months for rolling statistics

# Columns to compute lags/rolling on (anomaly columns)
LAG_COLS = [c for c in PREDICTOR_COLS if "_anom" in c]
ROLLING_COLS = [c for c in PREDICTOR_COLS if "_anom" in c]


def create_candidate_features(df, predictor_cols, target_col, horizon,
                              lag_cols, lag_months,
                              rolling_cols, rolling_windows):
    """
    Create candidate features using pd.concat to avoid DataFrame fragmentation.

    NaN policy: features are computed from raw data (NaN propagates naturally
    through shift/rolling). The valid_mask at the end drops any row where
    any feature or the target is NaN — no fabrication.
    """
    feature_parts = []

    # Layer 1: All current-month values
    feature_parts.append(df[predictor_cols])

    # Layer 2: Lag features
    lag_frames = {}
    for col in lag_cols:
        for lag in lag_months:
            lag_frames[f"{col}_lag{lag}"] = df[col].shift(lag)
    feature_parts.append(pd.DataFrame(lag_frames, index=df.index))

    # Layer 3: Rolling statistics (min_periods=window ensures no partial windows)
    rolling_frames = {}
    for col in rolling_cols:
        for window in rolling_windows:
            rolling_frames[f"{col}_ma{window}"] = df[col].rolling(
                window=window, min_periods=window
            ).mean()
            rolling_frames[f"{col}_std{window}"] = df[col].rolling(
                window=window, min_periods=window
            ).std()
    feature_parts.append(pd.DataFrame(rolling_frames, index=df.index))

    # Layer 4: Rate of change
    diff_frames = {}
    for col in lag_cols:
        diff_frames[f"{col}_diff1"] = df[col].diff(1)
        diff_frames[f"{col}_diff3"] = df[col].diff(3)
    feature_parts.append(pd.DataFrame(diff_frames, index=df.index))

    # Layer 5: Month
    feature_parts.append(pd.DataFrame({"month": df.index.month}, index=df.index))

    # Concat all at once
    X = pd.concat(feature_parts, axis=1)

    # Shift target forward
    y = df[target_col].shift(-horizon)

    # Drop rows where ANY feature or target is NaN — no cheating
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y


X, y = create_candidate_features(
    df, PREDICTOR_COLS, TARGET, FORECAST_HORIZON,
    LAG_COLS, LAG_MONTHS,
    ROLLING_COLS, ROLLING_WINDOWS,
)

# Count features by type
n_base = len(PREDICTOR_COLS)
n_lag = len(LAG_COLS) * len(LAG_MONTHS)
n_rolling = len(ROLLING_COLS) * len(ROLLING_WINDOWS) * 2
n_roc = len(LAG_COLS) * 2
n_total = X.shape[1]

print(f"\nFeature breakdown:")
print(f"  Base predictors:   {n_base}")
print(f"  Lag features:      {n_lag} ({len(LAG_COLS)} cols × {len(LAG_MONTHS)} lags)")
print(f"  Rolling features:  {n_rolling} ({len(ROLLING_COLS)} cols × {len(ROLLING_WINDOWS)} windows × 2 stats)")
print(f"  Rate-of-change:    {n_roc} ({len(LAG_COLS)} cols × 2 diffs)")
print(f"  Month:             1")
print(f"  TOTAL:             {n_total}")
print(f"\nFeature matrix: {X.shape}")
print(f"Target vector:  {y.shape}")
print(f"Date range used: {X.index.min().date()} to {X.index.max().date()}")

# --------------------------------------------------------------------------- #
# 3. WALK-FORWARD CROSS-VALIDATION
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 3: Walk-Forward Cross-Validation")
print("=" * 60)

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
    "eval_metric": ["mae", "rmse"],
    "seed": 42,
    "early_stopping_rounds": 10,
}
print("\nXGBoost hyperparameters (same as baseline):")
pprint(XGB_PARAMS)

TRAIN_YEARS = 10
VALID_YEARS = 5

all_years = sorted(X.index.year.unique())
fold_size = TRAIN_YEARS + VALID_YEARS
n_folds = len(all_years) - fold_size
n_folds = max(1, n_folds)

print(f"\nYears available: {all_years[0]}–{all_years[-1]} ({len(all_years)} years)")
print(f"Train window: {TRAIN_YEARS} years, Valid window: {VALID_YEARS} years")
print(f"Number of CV folds: {n_folds}")

results = []

for i in range(n_folds):
    fold_years = all_years[i: i + fold_size]
    train_years = fold_years[:TRAIN_YEARS]
    valid_years = fold_years[TRAIN_YEARS:]

    X_train = X[X.index.year.isin(train_years)]
    y_train = y[X_train.index]
    X_valid = X[X.index.year.isin(valid_years)]
    y_valid = y[X_valid.index]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=False,
    )

    y_pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pred)
    rmse = root_mean_squared_error(y_valid, y_pred)
    me = max_error(y_valid, y_pred)

    results.append({
        "fold": i,
        "train": f"{train_years[0]}–{train_years[-1]}",
        "valid": f"{valid_years[0]}–{valid_years[-1]}",
        "mae": mae,
        "rmse": rmse,
        "max_error": me,
        "model": model,
        "y_valid": y_valid,
        "y_pred": y_pred,
    })

    print(f"  Fold {i}: train {train_years[0]}–{train_years[-1]}, "
          f"valid {valid_years[0]}–{valid_years[-1]}  |  "
          f"MAE={mae:.3f}  RMSE={rmse:.3f}  MaxErr={me:.3f}")

mae_vals = [r["mae"] for r in results]
rmse_vals = [r["rmse"] for r in results]
print(f"\n--- CV Summary ({FORECAST_HORIZON}-month ahead forecast) ---")
print(f"  MAE:  {np.mean(mae_vals):.3f} ± {np.std(mae_vals):.3f}")
print(f"  RMSE: {np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f}")

# --------------------------------------------------------------------------- #
# 4. FEATURE IMPORTANCE — Top 20
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 4: Feature Importance (Top 20)")
print("=" * 60)

last_model = results[-1]["model"]
importances = pd.DataFrame({
    "feature": last_model.feature_names_in_,
    "importance": last_model.feature_importances_,
}).sort_values("importance", ascending=True)

top20 = importances.tail(20)

print("\nTop 20 feature importances (gain, last fold):")
for _, row in top20.iterrows():
    bar = "█" * int(row["importance"] * 200)
    print(f"  {row['feature']:35s} {row['importance']:.4f} {bar}")

importance_plot = top20.hvplot.barh(
    x="feature",
    y="importance",
    title=f"Top 20 Feature Importance — {FORECAST_HORIZON}-Month Ahead ONI Forecast (Candidate)",
    xlabel="Importance (Gain)",
    ylabel="",
    width=800,
    height=600,
    color="#2ca02c",
)
hvplot.save(importance_plot, "02_feature_importance.html")
print("\nSaved: 02_feature_importance.html")

# --------------------------------------------------------------------------- #
# 5. PREDICTIONS — Interactive overlay
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 5: Predictions")
print("=" * 60)

elnino_line = hv.HLine(0.5).opts(color="red", line_dash="dashed", alpha=0.5)
lanina_line = hv.HLine(-0.5).opts(color="blue", line_dash="dashed", alpha=0.5)

actual_plot = y.hvplot.line(
    label="Actual",
    color="black",
    line_width=1,
    alpha=0.7,
)

fold_plots = actual_plot
for r in results:
    pred_series = pd.Series(r["y_pred"], index=r["y_valid"].index, name="predicted")
    fold_plots = fold_plots * pred_series.hvplot.line(
        label=f"Fold {r['fold']} ({r['valid']})",
        line_width=1.2,
        alpha=0.6,
    )

predictions_plot = (
    fold_plots * elnino_line * lanina_line
).opts(
    title=f"Walk-Forward CV: {FORECAST_HORIZON}-Month Ahead ONI (Candidate)",
    ylabel="Niño 3.4 Anomaly (°C)",
    width=1000,
    height=450,
    legend_position="top_left",
)
hvplot.save(predictions_plot, "02_predictions.html")
print("Saved: 02_predictions.html")

# Scatter
scatter_df = pd.DataFrame({
    "actual": np.concatenate([r["y_valid"].values for r in results]),
    "predicted": np.concatenate([r["y_pred"] for r in results]),
})
perfect_line = hv.Slope(slope=1, y_intercept=0).opts(
    color="red", line_dash="dashed", alpha=0.5
)
scatter_plot = scatter_df.hvplot.scatter(
    x="actual",
    y="predicted",
    title=f"Predicted vs Actual ({FORECAST_HORIZON}-month lead, Candidate)",
    xlabel="Actual Niño 3.4 Anomaly",
    ylabel="Predicted Niño 3.4 Anomaly",
    width=500,
    height=500,
    alpha=0.4,
    size=15,
) * perfect_line
hvplot.save(scatter_plot, "02_scatter.html")
print("Saved: 02_scatter.html")

# --------------------------------------------------------------------------- #
# 6. COMPARISON SUMMARY
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 6: Baseline vs Candidate Comparison")
print("=" * 60)

print(f"""
                    Baseline (Module 1)     Candidate (Module 2)
  Features:         32                      {n_total}
  Lag months:       none                    {LAG_MONTHS}
  Rolling windows:  none                    {ROLLING_WINDOWS}
  Rate of change:   none                    1-month, 3-month diffs
  MAE:              (run module 1)          {np.mean(mae_vals):.3f} ± {np.std(mae_vals):.3f}
  RMSE:             (run module 1)          {np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f}

Next steps (Module 3):
  • Turn this into a reusable feature encoder function (pluggable pattern)
  • Experiment with different lag/window configurations
  • Try longer forecast horizons (6, 9, 12 months)
""")

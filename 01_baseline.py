"""
Module 1: ONI Baseline Model
=============================
Goal: Build a baseline XGBoost model to forecast Niño 3.4 anomalies.

This mirrors Module 00 from the ml-end-to-end course:
  - Load data
  - EDA
  - Feature engineering (timestamp + lag features)
  - Walk-forward cross-validation
  - Evaluation

The target is `nino3.4_anom` — the sea surface temperature anomaly in the
Niño 3.4 region, which is the basis for the Oceanic Niño Index (ONI).

Usage:
    pip install pandas numpy xgboost scikit-learn hvplot
    python 01_baseline.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import hvplot.pandas  # noqa: F401 — activates .hvplot accessor
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

df = pd.read_csv(DATA_URL, index_col=0, parse_dates=True)
df = df.dropna()  # Drop rows with missing values (pre-1982 data)

print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}")
print(f"\nFirst 5 rows:")
print(df.head())

# --------------------------------------------------------------------------- #
# 2. EDA — Understand the target
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

TARGET = "nino3.4_anom"

print(f"\nTarget: {TARGET}")
print(f"  Mean:   {df[TARGET].mean():.3f}")
print(f"  Std:    {df[TARGET].std():.3f}")
print(f"  Min:    {df[TARGET].min():.3f}")
print(f"  Max:    {df[TARGET].max():.3f}")

# Interactive plot: target over time
elnino_line = hv.HLine(0.5).opts(color="red", line_dash="dashed", alpha=0.5)
lanina_line = hv.HLine(-0.5).opts(color="blue", line_dash="dashed", alpha=0.5)

target_plot = df.hvplot.line(
    y=TARGET,
    title="Niño 3.4 Anomaly Over Time",
    ylabel="SST Anomaly (°C)",
    width=900,
    height=400,
    line_width=1,
) * elnino_line * lanina_line

# Distribution
dist_plot = df.hvplot.hist(
    y=TARGET,
    bins=40,
    title="Distribution of Niño 3.4 Anomaly",
    xlabel="SST Anomaly (°C)",
    ylabel="Count",
    width=500,
    height=400,
)

eda_layout = (target_plot + dist_plot).cols(1)
hvplot.save(eda_layout, "01_eda_target.html")
print("\nSaved: 01_eda_target.html")

# Correlation of all predictors vs target
predictor_cols = [c for c in df.columns if c != TARGET]
correlations = df[predictor_cols].corrwith(df[TARGET]).sort_values(ascending=False)
print(f"\nTop 10 features correlated with {TARGET}:")
for col, corr in correlations.head(10).items():
    print(f"  {col:25s} {corr:+.3f}")
print(f"\nBottom 5 features correlated with {TARGET}:")
for col, corr in correlations.tail(5).items():
    print(f"  {col:25s} {corr:+.3f}")

# --------------------------------------------------------------------------- #
# 3. FEATURE ENGINEERING — Baseline features (ALL columns)
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering (Baseline)")
print("=" * 60)

# Forecasting horizon: predict 3 months ahead.
FORECAST_HORIZON = 3  # months

# Use ALL columns except the target
PREDICTOR_COLS = [c for c in df.columns if c != TARGET]
print(f"\nPredictor columns ({len(PREDICTOR_COLS)}):")
for c in PREDICTOR_COLS:
    print(f"  {c}")


def create_baseline_features(df, predictor_cols, target_col, horizon):
    """
    Create baseline features for ONI forecasting.

    For each row at time t, we want to predict target at time t+horizon.
    Features are the predictor values at time t (the "current" state of the ocean).

    We also add a month feature since ONI has seasonal structure.

    Returns X (features), y (target shifted by horizon).
    """
    # Shift target forward: y[t] = target[t + horizon]
    y = df[target_col].shift(-horizon)

    # Features: all predictor values + month
    X = df[predictor_cols].copy()
    X["month"] = df.index.month

    # Drop rows where y is NaN (last `horizon` rows)
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y


X, y = create_baseline_features(df, PREDICTOR_COLS, TARGET, FORECAST_HORIZON)
print(f"\nFeature matrix: {X.shape}")
print(f"Target vector:  {y.shape}")
print(f"Predicting {FORECAST_HORIZON} months ahead")

# --------------------------------------------------------------------------- #
# 4. WALK-FORWARD CROSS-VALIDATION
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 4: Walk-Forward Cross-Validation")
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
print("\nXGBoost hyperparameters:")
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
# 5. FEATURE IMPORTANCE
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 5: Feature Importance")
print("=" * 60)

last_model = results[-1]["model"]
importances = pd.DataFrame({
    "feature": last_model.feature_names_in_,
    "importance": last_model.feature_importances_,
}).sort_values("importance", ascending=True)

print("\nFeature importances (gain, last fold):")
for _, row in importances.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['feature']:25s} {row['importance']:.3f} {bar}")

importance_plot = importances.hvplot.barh(
    x="feature",
    y="importance",
    title=f"Feature Importance — {FORECAST_HORIZON}-Month Ahead ONI Forecast",
    xlabel="Importance (Gain)",
    ylabel="",
    width=700,
    height=600,
    color="#1f77b4",
)
hvplot.save(importance_plot, "01_feature_importance.html")
print("\nSaved: 01_feature_importance.html")

# --------------------------------------------------------------------------- #
# 6. ERROR ANALYSIS — Interactive predictions vs actuals
# --------------------------------------------------------------------------- #

print("\n" + "=" * 60)
print("STEP 6: Error Analysis")
print("=" * 60)

# Build a DataFrame of all predictions for interactive overlay
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
    fold_plots
    * elnino_line
    * lanina_line
).opts(
    title=f"Walk-Forward CV: {FORECAST_HORIZON}-Month Ahead ONI Predictions",
    ylabel="Niño 3.4 Anomaly (°C)",
    width=1000,
    height=450,
    legend_position="top_left",
)
hvplot.save(predictions_plot, "01_predictions.html")
print("Saved: 01_predictions.html")

# Scatter: predicted vs actual (all folds)
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
    title=f"Predicted vs Actual ({FORECAST_HORIZON}-month lead)",
    xlabel="Actual Niño 3.4 Anomaly",
    ylabel="Predicted Niño 3.4 Anomaly",
    width=500,
    height=500,
    alpha=0.4,
    size=15,
) * perfect_line
hvplot.save(scatter_plot, "01_scatter.html")
print("Saved: 01_scatter.html")

print("\n" + "=" * 60)
print("DONE — Baseline complete.")
print("=" * 60)
print(f"""
Key takeaways:
  • Target: {TARGET} ({FORECAST_HORIZON}-month ahead)
  • Predictors: {len(PREDICTOR_COLS)} columns (all features) + month = {X.shape[1]} total
  • CV: Walk-forward, {TRAIN_YEARS}yr train / {VALID_YEARS}yr valid, {n_folds} folds
  • MAE: {np.mean(mae_vals):.3f} ± {np.std(mae_vals):.3f}
  • RMSE: {np.mean(rmse_vals):.3f} ± {np.std(rmse_vals):.3f}

Next steps (Module 2):
  • Add lag features (past N months of each predictor)
  • Add rolling statistics (moving averages, trends)
  • Compare candidate vs baseline metrics
""")

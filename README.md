# ONI Forecast 🌊

An end-to-end ML system for forecasting the [Oceanic Niño Index (ONI)](https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-nino-index), built with [XGBoost](https://xgboost.readthedocs.io/) and [Metaflow](https://metaflow.org/) on [Outerbounds](https://outerbounds.com/).

The system compares 5 forecasting models across 12 monthly horizons using walk-forward cross-validation, selects the best model per horizon, saves them to S3, and produces real forecasts triggered by new data — all orchestrated as Metaflow flows with interactive [HoloViews](https://holoviews.org/) cards.

![Skill Curve](https://github.com/user-attachments/assets/placeholder-skill-curve.png)

## Why ONI?

ENSO (El Niño–Southern Oscillation) is the dominant mode of year-to-year climate variability. Predicting it months in advance matters for agriculture, fisheries, disaster preparedness, and energy markets. The ONI — a 3-month running mean of SST anomalies in the Niño 3.4 region — is NOAA's official metric for classifying El Niño (> +0.5°C) and La Niña (< −0.5°C) events.

The well-known **spring predictability barrier** makes ENSO forecasting especially interesting: forecast skill drops sharply for predictions that must cross through boreal spring (March–May). This project encodes that barrier directly as a feature.

## Quick Start

### Prerequisites

- Python 3.12+
- A [Metaflow](https://docs.metaflow.org/getting-started/install) / [Outerbounds](https://docs.outerbounds.com/) environment configured with S3

### Run the training flow

```bash
python 03_multi_horizon_flow.py --environment=pypi run
```

This trains 5 models × 12 horizons = 60 branches in parallel, selects the best model per horizon, and saves them to S3. View the results:

```bash
python 03_multi_horizon_flow.py card view end
```

### Deploy the full pipeline

```bash
# Deploy the data sensor (checks GitHub daily for new data)
python 04_sensor_flow.py --environment=pypi argo-workflows create

# The inference flow triggers automatically via ArgoEvent,
# or run it manually:
python 05_inference_flow.py --environment=pypi run
```

## Project Structure

```
oni-forecast/
├── 01_baseline.py              # EDA + baseline XGBoost (standalone script)
├── 02_candidate.py             # Candidate model with lag/rolling features
├── 03_multi_horizon_flow.py    # Training flow (Metaflow)
├── 04_sensor_flow.py           # Data sensor flow (Metaflow)
└── 05_inference_flow.py        # Inference flow (Metaflow)
```

### Module Progression

The project follows a **prototype → productionize** arc inspired by the [Outerbounds ML course](https://learn.outerbounds.com):

| Module | Type | Purpose |
|--------|------|---------|
| `01_baseline.py` | Script | Explore the data, establish a baseline XGBoost model with walk-forward CV |
| `02_candidate.py` | Script | Add lag, rolling, and rate-of-change features. Compare against baseline |
| `03_multi_horizon_flow.py` | Metaflow flow | Productionized experiment: 5 models × 12 horizons, model registry, Metaflow cards |
| `04_sensor_flow.py` | Metaflow flow | Monitors GitHub for new data, publishes `ArgoEvent("new_oni_data")` on change |
| `05_inference_flow.py` | Metaflow flow | Loads best models from S3, produces forecasts with El Niño/La Niña classification |

### Event-Driven Architecture

```
@schedule(daily)            @trigger(new_oni_data)          @trigger(new_oni_data)
┌──────────────┐   event   ┌────────────────────────┐  S3  ┌──────────────────────┐
│ ONIDataSensor│──────────→│ ONIMultiHorizonFlow    │─────→│ ONIInferenceFlow     │
│ (check GitHub)│          │ (retrain, save best)    │      │ (forecast, classify) │
└──────────────┘           └────────────────────────┘      └──────────────────────┘
```

## Models

| Model | Features | Algorithm | Strengths |
|-------|----------|-----------|-----------|
| **Persistence** | 1 | None (ONI_t+h = ONI_t) | Hard-to-beat naive control |
| **Linear** | 32 | Ridge regression | Best at 1-month lead (~0.19 MAE) |
| **Baseline** | 32 | XGBoost | Current-month snapshot only |
| **Candidate** | 214 | XGBoost | Full temporal context (all lags + rolling) |
| **Adaptive** | 70–214 + seasonal | XGBoost + SelectKBest(k=40) | Horizon-tuned features, best at 6+ months |

The adaptive model adjusts its feature set by horizon (fewer features at short leads, more at long leads) and adds seasonal features that encode the spring predictability barrier:

- **target_month_sin/cos** — cyclical encoding of the month being predicted
- **crosses_spring** — does the forecast window cross March–May?
- **months_through_spring** — how many spring months the forecast must cross (0–3)

## Data

Monthly Pacific Ocean observations from [ninodata](https://github.com/ahuang11/ninodata), sourced from NOAA/PMEL and CPC:

| Variable | Description |
|----------|-------------|
| t300 | Subsurface ocean temperature at 300m depth (east/west/central Pacific) |
| wwv | Warm water volume (east/west/central) |
| olr | Outgoing longwave radiation (proxy for deep convection) |
| u850 | 850hPa zonal wind (trade winds, east/west/central) |
| nino | SST indices (Niño 1+2, 3, 3.4, 4 regions) |

**Target:** `nino3.4_anom` — the Niño 3.4 SST anomaly.

## Key Findings

1. **Linear regression beats XGBoost at 1-month lead.** The short-term relationship is nearly linear; tree-based models add complexity without adding skill.

2. **Temporal features hurt at short horizons but help at long ones.** The candidate model (214 features) loses to the 32-feature baseline at 1–3 months but wins at 6+ months. Feature selection fixes this.

3. **Persistence is the hardest baseline to beat.** "Predict no change" achieves ~0.21 MAE at 1-month lead thanks to ENSO's strong autocorrelation.

4. **All models cross the 0.5°C error threshold around 4–6 months.** Beyond that, average error exceeds the threshold used to classify El Niño/La Niña events.

5. **Feature importance shifts with horizon:** current SST dominates at 1 month → subsurface heat content at 6 months → lagged historical state at 12 months.

## Metaflow Cards

The training flow produces interactive cards viewable in the Outerbounds UI:

- **Summary table** — MAE/RMSE for all model × horizon combinations
- **Skill degradation curve** — MAE vs lead time for all 5 models
- **Predictions HoloMap** — actual vs predicted with fold-spread envelope (slider by horizon)
- **Scatter HoloMap** — predicted vs actual scatter plots (slider by horizon)
- **Feature importance HoloMap** — top features at each horizon (slider)

## Acknowledgments

- Data: [ahuang11/ninodata](https://github.com/ahuang11/ninodata), sourced from [NOAA/PMEL](https://www.pmel.noaa.gov/tao/wwv/) and [CPC](https://www.cpc.ncep.noaa.gov/)
- Course: [Outerbounds ML End-to-End](https://learn.outerbounds.com)
- Tools: [Metaflow](https://metaflow.org/), [XGBoost](https://xgboost.readthedocs.io/), [HoloViews](https://holoviews.org/), [hvplot](https://hvplot.holoviz.org/)

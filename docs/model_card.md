# Model card

## Task and intended use

Exploratory regression of tract-mean satellite land surface temperature (LST, °C) in Long Beach,
California. Intended for demonstrating an environmental-data ML workflow and comparing observed
and estimated surface temperatures. Not validated for health, exposure, causal cooling effects,
future weather, equity rankings, or final infrastructure siting.

## Data and features

338 observations across 114 city-clipped 2023 Census tracts, from three Landsat 8/9 Collection 2
Level 2 scenes dated 2023-07-12, 2023-08-05, and 2023-08-29. See the [data sheet](data_sheet.md).
Target: mean clear-land USGS LST retrieval. Predictors: mean NDVI, NDBI, and NDWI calculated from
same-date surface reflectance. No thermal band, target, tract ID, date, or coverage field enters ML.
This uses simultaneous satellite information; it is not forecasting or validated thermal downscaling.

## Models and selection

Mean baseline, standardized linear regression, and Random Forest (300 trees, minimum leaf size 3,
seed 42, other scikit-learn defaults). Select lowest mean fold MAE. Random Forest was selected,
then refit on all observations. The saved model bundle contains the estimator, predictor order,
and configuration; actual package versions are in `results/run_metadata.json`.

## Evaluation

Preserved five-fold tract-grouped spatial CV. Every date of a tract is entirely in train or test
within a fold. Adjacent tracts can be in different folds. No spatial buffer, temporal holdout,
nested CV, field comparison, or independent external test was performed.

| Model | Mean fold MAE (°C) | Mean fold RMSE (°C) | Mean fold R² |
|---|---:|---:|---:|
| Mean baseline | 2.100 | 2.748 | −0.010 |
| Linear | 1.989 | 2.652 | 0.050 |
| Random Forest | 1.984 | 2.594 | 0.079 |

Pooled Random Forest held-out MAE: **1.987°C**, RMSE: **2.621°C**, R²: **0.095**.
Fold averages and pooled scores answer slightly different aggregation questions; do not mix them.
The Random Forest/linear difference is very small. Low R² and fold variation limit confidence.
Reusing folds for model selection can make selected-model performance optimistic.

## Failure analysis

Selected-model held-out MAE by date: July 12 **2.547°C**, August 5 **1.497°C**, August 29 **1.937°C**.
Mean residual (observed minus predicted) is +2.052°C in July, −0.800°C on August 5, and −1.231°C
on August 29: the model underpredicts the hotter July observations and overpredicts the August
observations on average. This is a diagnostic across spatially held-out rows, not a temporal test.

The largest tract-average MAEs are GEOIDs `06037577501` (8.845°C, two dates), `06037577504`
(8.228°C, two dates), and `06037577604` (7.633°C, two dates). The model overpredicts their observed
means. No cause has been established; retrieval quality, shoreline/mixed surfaces, and missing-date
selection require investigation. See `results/error_by_unit_id.csv` for all tracts. These are errors
against satellite retrievals, not against independent physical truth.

## Prediction labels and remaining limits

The app displays observations or spatially held-out estimates. `results/predictions.parquet` also
contains final-model `fitted_in_sample` predictions; these are never used as accuracy evidence.
No uncertainty intervals, demographic error analysis, or fairness claims are available. Generalization
beyond these selected dates and Long Beach is untested. Read the [limitations](limitations.md).

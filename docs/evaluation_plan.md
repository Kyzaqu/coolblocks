# Evaluation plan and current status

## Implemented

Compare a mean predictor (`DummyRegressor`), standardized linear regression, and Random Forest
(300 trees, minimum leaf size 3, seed 42). Use only mean NDVI, NDBI, and NDWI as predictors.
An allowlist prevents raw thermal bands, the target, coverage counts, and metadata from entering ML.
All rows must have finite target/predictor values and unique tract/scene keys.

Preserve five-fold `GroupKFold`, grouped by tract GEOID. All dates from a tract stay in the same
fold. Group overlap is checked and memberships are written to `run_metadata.json`. Linear scaling
is fit inside each training fold through a scikit-learn pipeline. Random pixel splitting is never used.
Adjacent tracts may be on opposite sides of a fold boundary; there is no buffer or contiguous-region
holdout. Performance therefore tests held-out tract IDs, not distant-region transfer.

Report MAE and RMSE in Celsius and dimensionless R² per fold, with mean and sample standard
deviation. Save each model's out-of-fold predictions. Also report pooled metrics over those rows;
pooled R² is not the mean fold R². Select lowest mean fold MAE, then refit on all observations.
The saved final model's in-sample predictions are explicitly labeled and excluded from evaluation.

`make eval` recomputes pooled metrics from the saved held-out predictions, checks them against
run metadata, checks fold membership against the spatial audit, and writes selected-model error
summaries by date and tract. Those date summaries use the same spatial CV predictions; they are
**not temporal holdout scores**. Read with the [model card](model_card.md).

## Not performed

- Temporal generalization: three summer 2023 dates do not support the proposed held-out-year test.
- Independent model-selection evaluation: selection and reporting reuse folds; no nested CV or
  untouched external test set. Small model differences should not be interpreted as proven gains.
- Stakeholder interviews, usability study, demographic error analysis, field validation, and
  spatial-buffer sensitivity checks. No results are claimed for these.

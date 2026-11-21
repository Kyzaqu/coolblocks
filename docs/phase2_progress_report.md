# Phase 2 progress report

**Project:** CoolBlocks — Long Beach neighborhood-scale surface temperature proof of concept  
**Date:** November 20, 2026  
**Implementation branch:** `implementation/real-data-ml-mvp`

## Objective and repository assessment

Deliver the simplest working public-data → ML → evaluation → predictions → interactive-map workflow.
The starting repository contained reusable Landsat QA/scaling helpers, NDVI/NDBI/NDWI functions,
mean/linear/Random Forest model constructors, regression metrics, and whole-unit spatial validation.
Download and feature generation were unimplemented. Training produced CV metric tables only;
evaluation was a placeholder. The app expected unavailable SVI/canopy/temperature columns and
had no neighborhood comparison. Documentation described planned capabilities rather than a run.

The implementation retains those useful scientific and ML building blocks. It chooses Census
tracts clipped to the city as explicit neighborhood proxies and uses the three spectral predictors
already supported by the code. It defers additional land-cover, demographic, prioritization, and
model dependencies rather than inserting fabricated data or unsupported UI controls.

## Increment 1: real data, training, evaluation, predictions

`data/download.py` caches Census 2023 California place and tract boundaries and seven Landsat
band windows per scene. Three fixed scene IDs make selection repeatable. The source manifest
records scene dates, public asset URLs, and local SHA-256 checksums.

`features/build.py` selects Long Beach place GEOID `0643000`, clips Los Angeles County tracts,
and drops empty and line-only intersections. This last check is necessary: a tract sharing only a
city boundary line is not a city study area, even if rasterization can assign pixels to the line.
A regression test covers that boundary case.

Cloud, shadow, snow, water, fill, saturation, invalid reflectance, missing thermal retrievals, and
undefined indices are excluded. Targets and predictors share the same valid pixels. At least 30
pixels and 50% coverage are required. There are **338 accepted observations in 114 clipped tracts**
from 342 candidate tract/dates: 110 on July 12 and 114 on each August date.

`models/train.py` compares mean, linear, and Random Forest models with the original five-fold
whole-tract `GroupKFold`. Every tract's dates stay together. An explicit predictor allowlist excludes
thermal data, target values, and metadata. The code writes fold metrics, every held-out prediction,
fold membership audit, a final refitted model, prediction table, and a latest-date geographic export.

For a junior developer: the feature table is the interface between the geospatial and ML code.
Each row describes one place at one time. Cross-validation trains five separate copies of each
model, each predicting places it did not see. The final saved model is a sixth fit using all rows;
its fitted predictions are useful output but cannot be used to measure generalization.

## Increment 2: map and tract comparison

The Streamlit app reads precomputed files without downloading or training. Users select a date,
switch between observed satellite LST and held-out model estimates, inspect tract tooltips, compare
two tracts, and download the selected date's table. The two map layers share a fixed temperature
scale across all dates. Coverage and spectral means accompany the comparison.

Observed means are explicitly labeled as satellite retrievals. Model estimates are explicitly
labeled as spatially held out and use same-date predictors, not future weather. Final-model
in-sample predictions are stored but not offered as an app layer. The previous SVI/canopy weights
are removed from the app because those data were not collected.

For a junior developer: joining the prediction table to polygons by `unit_id` attaches each row's
numbers to a map shape. Keeping this join in a read-only app makes page interactions fast and
avoids retraining whenever a user changes the selected date or layer.

## Increment 3: verification and reproducibility

`make eval` independently recomputes pooled metrics from saved held-out predictions, checks fold
membership and reported scores, and produces date/tract error summaries. Tests cover QA/scaling,
whole-tract holdouts, leakage and invalid inputs, duplicate observations, geographic clipping,
saved-model prediction consistency, map export, and detection of corrupted fold assignments.
Existing prioritization tests are retained, although prioritization is not part of the current app.

The README, data sheet, scope, evaluation plan, model card, and limitations now describe the
implemented workflow. CI installs declared project dependencies so the geographic pipeline tests
can run. Unused LightGBM, SHAP, stackstac, and matplotlib dependencies were removed; joblib,
already required by scikit-learn, is explicitly declared for saving the model. No new service or
infrastructure is introduced.

For a junior developer: a checksum is a fingerprint of a file. Recording source and feature hashes
helps detect accidental changes between preparation and training. Recording settings and package
versions helps another developer reconstruct the run, though cross-version numerical equality
is not guaranteed. Cached raw data permits feature building, training, and evaluation offline.

## Actual results

Mean ± sample standard deviation across five spatial folds:

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| Mean baseline | 2.100 ± 0.188 | 2.748 ± 0.291 | −0.010 ± 0.016 |
| Linear | 1.989 ± 0.154 | 2.652 ± 0.280 | 0.050 ± 0.150 |
| Random Forest | 1.984 ± 0.269 | 2.594 ± 0.395 | 0.079 ± 0.264 |

Random Forest is selected by lowest mean fold MAE. Its pooled held-out MAE is 1.987°C, RMSE
2.621°C, and R² 0.095. Gains over linear regression are tiny; no statistically supported superiority
is claimed. Error differs by date: MAE 2.547°C in July versus 1.497°C and 1.937°C in August.
Large errors in several tracts remain; the [model card](model_card.md) lists examples without
claiming a cause. These values are computed exclusively from actual downloaded satellite data.

## Verification record

- Downloaded actual Census archives and three Landsat scenes' cropped assets.
- Ran `make features train eval` on those inputs; checked row counts, geographic bounds,
  positive-area polygons, finite values, coverage thresholds, and zero train/test tract overlap.
- Ran `make test lint`: 22 tests pass and lint passes.
- Used Streamlit `AppTest` with real outputs to check initial rendering, both layers, all three dates,
  and comparison selections including none. This checks app execution, not a browser screenshot
  or external map-tile availability.
- Repeated cached feature building and model training to check deterministic held-out predictions
  within numerical precision in the recorded environment.

## Limits and remaining work

LST measures surfaces at a morning overpass, not air temperature or personal exposure. Thermal
resolution is roughly 100 m despite the 30 m product grid. Clear-sky sampling, retrieval error,
shorelines, and generalized boundaries affect means. Three dates are not a climatology. Whole-tract
CV allows adjacent tracts in different folds; no buffer or unseen-year test was run. Model selection
reuses evaluation folds. No field validation, uncertainty estimates, stakeholder study, demographic
fairness evaluation, causal intervention modeling, or production deployment has been completed.

The current MVP is a working local demonstration with modest predictive performance. A next
scientific step would be a broader date sample and independent temporal/spatial testing before
adding predictors or making planning claims. No results are claimed for those future steps.

# CoolBlocks: Long Beach land surface temperature

A small, reproducible proof of concept: public Landsat data → tract-level features →
spatial model validation → predictions → an interactive Streamlit map.

The current dataset contains **338 tract/date observations across 114 city-clipped Census tracts**
on July 12, August 5, and August 29, 2023. Census tracts are neighborhood proxies, not named
neighborhoods. Surface temperature is **not air temperature, heat index, or personal exposure**.

## Run locally

Use Python 3.11 or newer, from the repository root:

```bash
make setup
make data       # Internet: public Census boundaries and clipped Landsat rasters
make features   # Quality masking and tract aggregation
make train      # Spatial CV, final model, predictions, and map-ready output
make eval       # Independently check held-out metrics; summarize errors
make app        # Open the URL printed by Streamlit
```

The Makefile uses `.venv/bin/python`; activation is optional. Existing Conda users can run
`make setup-conda`, activate `coolblocks`, and use `PYTHON=python` with Make commands.
Windows users can create a venv with `py -3 -m venv .venv`, activate it, install with
`python -m pip install -e .`, and run the modules listed in the Makefile directly.

The first download needs network access to Census, Planetary Computer, and Azure Blob Storage.
Planetary Computer signing is automatic; no account or secret is configured. Raw rasters are
cached locally. After downloading, `make features train eval` works offline. The interactive
basemap uses OpenStreetMap tiles and needs browser network access. To replace a cached source,
remove its local cached file and rerun `make data`; the manifest records checksums.

```bash
make test lint
```

Tests use explicitly artificial fixtures to check software behavior; reported scientific results
come exclusively from the public environmental data. CI runs tests and lint without downloading scenes.

## How it works

1. [`download.py`](src/coolblocks/data/download.py) resolves three scene IDs pinned in
   [`config.yaml`](configs/config.yaml), downloads Census 2023 generalized place/tract boundaries,
   and caches only the Landsat raster windows around Long Beach. A manifest records source URLs,
   scene timestamps, and SHA-256 checksums (file fingerprints).
2. [`build.py`](src/coolblocks/features/build.py) clips Los Angeles County tracts to Long Beach,
   removes empty and line-only intersections, applies Landsat scaling and quality masks, and
   averages LST, NDVI, NDBI, and NDWI over the **same pixels** in each tract/date. Observations
   need at least 30 valid pixels and 50% valid coverage. Coverage includes water in its denominator;
   water pixels cannot contribute to the means.
3. [`train.py`](src/coolblocks/models/train.py) compares mean, linear, and Random Forest models.
   The existing five-fold `GroupKFold` approach holds out entire tracts, including all their dates.
   Only NDVI, NDBI, and NDWI enter the model. It saves every held-out prediction, selects the model
   with lowest mean fold MAE, and refits that model on all observations.
4. [`run_eval.py`](src/coolblocks/evaluation/run_eval.py) checks the spatial audit and recomputes
   pooled held-out metrics, then summarizes the selected model's errors by tract and date.
5. [`streamlit_app.py`](app/streamlit_app.py) reads the outputs without training. Choose a date,
   switch between observed LST and spatially held-out estimates, and compare two tracts.

This keeps the original masking, spectral indices, model constructors, metrics, and spatial splitting.
NLCD, demographic/SVI ingestion, prioritization, LightGBM, and intervention scenarios are outside
this minimal workflow. The existing prioritization module is retained but unused by the app.

## Verified results

Five-fold spatial validation; mean ± sample standard deviation across folds:

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| Mean baseline | 2.100 ± 0.188 | 2.748 ± 0.291 | −0.010 ± 0.016 |
| Linear regression | 1.989 ± 0.154 | 2.652 ± 0.280 | 0.050 ± 0.150 |
| Random Forest | 1.984 ± 0.269 | 2.594 ± 0.395 | 0.079 ± 0.264 |

Random Forest is selected by mean fold MAE. Its improvement over linear regression is tiny;
these results do not establish a meaningful superiority. Low R² indicates limited predictive
explanation. The same folds are used for selection and reporting, so this is exploratory evaluation.

See [fold metrics](results/cv_metrics_by_fold.csv), [pooled metrics](results/pooled_oof_metrics.csv),
[run provenance and fold memberships](results/run_metadata.json), and the
[Phase 2 progress report](docs/phase2_progress_report.md). Pooled metrics combine all held-out rows;
they differ from averaging fold scores. No future-date accuracy has been measured.

## Outputs and temperature labels

| File | Meaning |
|---|---|
| `data/raw/MANIFEST.json` | Sources, scene timestamps, cached-file checksums |
| `data/processed/features.parquet` | Observed LST and predictor means per tract/date |
| `data/processed/coverage.csv` | All candidate tract/dates and acceptance status |
| `data/processed/units.geojson` | City-clipped tract polygons |
| `results/oof_predictions.csv` | All models' **spatially held-out** predictions used for evaluation |
| `results/predictions.parquet` | Observations, selected model's fitted estimates, and held-out estimates |
| `results/model.joblib` | Final fitted model, ordered predictor names, and settings |
| `results/error_by_*.csv` | Selected model's held-out errors by date and tract |
| `data/processed/neighborhoods.geojson` | Latest date's joined observations/estimates, for export |

`observed_lst_celsius` is the mean USGS satellite retrieval, not a thermometer reading.
`oof_predicted_lst_celsius` is predicted without training on that tract.
`predicted_lst_celsius` in `predictions.parquet` comes from the final model fitted on all rows:
its `prediction_type` is **fitted_in_sample** and it is not used to report accuracy or shown on the map.
All estimates use same-date satellite predictors; they are not forecasts.

Raw data, processed datasets, and binary model/prediction files are ignored by Git. Small CSV
results and run metadata are retained for review. No server, database, or extra service is needed.
Dependencies are declared in `pyproject.toml` and `environment.yml`; the actual training package
versions are recorded in the run metadata. Exact cross-version numerical equality is not guaranteed.

## Scientific boundaries

This prototype has three clear-sky summer dates, approximate Census boundaries, coarse thermal
resolution, and no temporal holdout, field validation, spatial buffer, uncertainty model, or equity
analysis. It does not estimate causal benefits from trees or justify investment rankings.
See the [data sheet](docs/data_sheet.md), [model card](docs/model_card.md),
[evaluation plan](docs/evaluation_plan.md), and [limitations](docs/limitations.md).

## Team

Kevin Luu, Fristine Sok, Eric Lee, Yasasvi Kotagiri

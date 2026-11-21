# Data and included demonstration

The application works immediately after installation using these existing pipeline outputs:

- **Included:** `processed/units.geojson` — 114 public Census tract polygons clipped to Long Beach.
- **Included:** `../results/predictions.parquet` — 338 real tract/date observations with spectral
  features, final-model fitted estimates, and selected-model spatially held-out estimates.
- **Included:** `../results/run_metadata.json` — source URLs/checksums, settings, package versions,
  evaluation metrics, and spatial fold memberships. Published evaluation CSVs are also included.

The app displays only the **spatially held-out** estimates. In-sample estimates are explicitly
labeled in the prediction table and are not reported as validation accuracy. There are three dates:
2023-07-12, 2023-08-05, and 2023-08-29. This is the actual small pilot dataset, not synthetic data.

The included boundaries and predictions were produced by the normal workflow, in order:

```bash
make data
make features
make train
make eval
```

The first command downloads real Census archives and cropped Landsat assets. Reproduction
replaces the included outputs at the same paths; no demo-specific processing or duplicate dataset
is maintained. See [the README](../README.md) for installation and separate demo/reproduction paths.

Other generated files remain ignored by Git:

- `raw/place.zip`, `raw/tract.zip`: Census 2023 California cartographic boundaries.
- `raw/city.geojson`: selected Long Beach place polygon.
- `raw/<scene_id>/`: unsigned STAC metadata and seven cropped, unscaled Landsat band TIFFs.
- `raw/MANIFEST.json`: source URLs, scene times, and SHA-256 checksums.
- `processed/features.parquet`: one row per accepted tract/date with observed LST and spectral means.
- `processed/coverage.csv`: accepted/rejected tract-dates, pixel counts, and valid fractions.
- `processed/feature_metadata.json`: settings and source/feature fingerprints.
- `processed/neighborhoods.geojson`: latest date's joined observations and model outputs.
- `../results/model.joblib`: final fitted model, predictors, and settings.

The [data sheet](../docs/data_sheet.md) describes source versions, quality rules, and measurement
limits. Keep GEOIDs as strings when reading CSVs. App error is predicted minus observed;
evaluation residuals use observed minus predicted, with each convention labeled explicitly.

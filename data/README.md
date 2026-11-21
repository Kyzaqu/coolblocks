# Local data

From the repository root, run `make data features train eval`. Raw and processed data are ignored by
Git. No fabricated observations or fallback sample dataset are used by the app.

- `raw/place.zip`, `raw/tract.zip`: Census 2023 California cartographic boundaries.
- `raw/city.geojson`: selected Long Beach place polygon.
- `raw/<scene_id>/`: unsigned STAC metadata and seven cropped, unscaled Landsat band TIFFs.
- `raw/MANIFEST.json`: source URLs, scene times, and SHA-256 checksums.
- `processed/features.parquet`: one row per accepted tract/date with observed LST and spectral means.
- `processed/coverage.csv`: accepted and rejected tract/dates, pixel counts, valid fractions.
- `processed/feature_metadata.json`: settings and source/feature fingerprints.
- `processed/units.geojson`: city-clipped tracts, with stable tract GEOIDs as `unit_id`.
- `processed/neighborhoods.geojson`: latest date's joined observations and model outputs.

See [the data sheet](../docs/data_sheet.md) for sources and quality rules and
[the README](../README.md) for output labels and commands. Keep GEOIDs as strings when reading CSVs.

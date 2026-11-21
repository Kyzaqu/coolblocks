# Data

Nothing in `raw/`, `interim/` or `processed/` is committed (see `.gitignore`). Recreate it with `make data features`.

| Source | Resolution | Date range | Download method | Notes |
|---|---|---|---|---|
| Landsat 8/9 C2 L2 (ST_B10, SR bands, QA_PIXEL) | 30 m (thermal resampled from 100 m) | TODO | Planetary Computer STAC | summer, low cloud |
| NLCD impervious / tree canopy | 30 m | TODO | MRLC | check product year |
| Census TIGER boundaries + ACS | block group / tract | TODO | data.census.gov | |
| CDC/ATSDR SVI | tract | TODO | ATSDR | |

Record every download in `raw/MANIFEST.json` (URL, retrieval date, version).

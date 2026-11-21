# CoolBlocks: AI-Powered Neighborhood Heat Analysis

Machine-learning estimates of neighborhood-level land surface temperature (LST), with an
interactive map and a configurable prioritization tool. Pilot city: **Long Beach, California**.

> Status: Phase 2 in progress. Data pipeline under development. See [docs/scope.md](docs/scope.md).

## Problem
Heat is unevenly distributed inside cities, but planners usually work from city-wide or census
tract averages and from complaints. Cooling projects can end up where requests are loudest rather
than where heat is worst, and grant applications often lack block-level evidence. Tree canopy and
impervious surface are two main drivers of surface heat, and both can be changed through planning.

## Solution
A web platform that:
1. estimates and maps neighborhood LST using supervised regression,
2. compares neighborhoods and the environmental factors associated with their temperatures, and
3. lets users adjust weights for heat and community vulnerability to rank areas for cooling
   investment, then export the ranked list as a CSV.

**Target users:** city sustainability departments, urban planners, researchers, and environmental
justice organizations.

## Scope
| In scope (MVP) | Out of scope |
|---|---|
| Selected Long Beach neighborhoods | Real-time forecasting |
| LST regression (baseline, Random Forest, LightGBM) | Cooling intervention simulations |
| Interactive map and neighborhood comparison | Automated report generation |
| Configurable prioritization with CSV export | Air temperature or personal exposure estimates |
| Stakeholder interviews | Cities other than Long Beach |

Full details: [docs/scope.md](docs/scope.md).

## Architecture
```
Landsat 8/9 + NLCD + Census + CDC SVI
        -> src/coolblocks/data, features      data pipeline
        -> src/coolblocks/models, evaluation  ML and spatial validation
        -> src/coolblocks/prioritization      weighted ranking
        -> app/streamlit_app.py               interactive map
```

## Quickstart
```bash
conda env create -f environment.yml
conda activate coolblocks

make data       # download and cloud-mask Landsat, NLCD, Census, SVI   (in development)
make features   # build data/processed/features.parquet                (in development)
make train      # spatial-CV model comparison, writes to results/
make app        # launch the Streamlit map
make test       # run unit tests
```
Settings (study area, dates, seeds, weights) live in [configs/config.yaml](configs/config.yaml).
If `make` is unavailable (for example on Windows), run the underlying commands from the Makefile directly.

## Data
| Source | Use | License / terms |
|---|---|---|
| USGS Landsat 8/9 Collection 2 Level 2 | LST target and spectral bands | Public domain (USGS) |
| NLCD (impervious surface, tree canopy) | Land cover predictors | Public domain (MRLC) |
| U.S. Census Bureau (TIGER, ACS) | Boundaries and demographics | Public domain |
| CDC/ATSDR Social Vulnerability Index | Vulnerability indicators | Public domain |

Details and limitations: [docs/data_sheet.md](docs/data_sheet.md). Raw data is never committed;
download it with the scripts.

## Evaluation
Four evaluations: accuracy vs baselines, spatial generalization, temporal generalization, and
stakeholder/usability feedback. Metrics are MAE, RMSE, and R², using spatial holdout by
neighborhood (never random pixel splits). See [docs/evaluation_plan.md](docs/evaluation_plan.md).

## Results
_To be added after the Phase 3 evaluations._

## Limitations
- LST is satellite-measured surface temperature, **not** pedestrian-level air temperature or human heat exposure.
- Relationships learned by the model are associations, not causal effects.
- Predictions may not generalize to unseen neighborhoods or dates; see [docs/model_card.md](docs/model_card.md).
- Existing free tools (Tree Equity Score, California Heat Assessment Tool) may already meet some
  users' needs; the added value is being validated through stakeholder interviews.

## Repository layout
```
configs/    study area, dates, seeds, weights
data/       data folders (not committed) and sources
src/        data pipeline, features, models, evaluation, prioritization
app/        Streamlit web map
tests/      unit tests
docs/       scope, evaluation plan, data sheet, model card, limitations, interviews
results/    metrics and figures
```

## Team
Kevin Luu
Fristine Sok
Eric Lee
Yasasvi Kotagiri

## References
See [docs/references.md](docs/references.md).

## License
MIT. See [LICENSE](LICENSE).
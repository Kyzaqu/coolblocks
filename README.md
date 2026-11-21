# CoolBlocks: AI-Powered Neighborhood Heat Analysis

Machine-learning estimates of neighborhood-level land surface temperature (LST), with an interactive
map and a configurable prioritization tool. Pilot city: **Long Beach, California**.

> Status: Phase 2 (in progress). See [docs/scope.md](docs/scope.md) for what is and isn't in scope.

## Problem
Heat is unevenly distributed inside cities, but planners usually work from city-wide or tract averages
and from complaints. Cooling projects can end up where requests are loudest rather than where heat is
worst, and grant applications often lack block-level evidence. Tree canopy and impervious surface are
two main drivers of surface heat, and both can be changed through planning.

## Solution
A web platform that (1) estimates and maps neighborhood LST using supervised regression,
(2) compares neighborhoods and the environmental factors associated with their temperatures, and
(3) lets users re-weight heat and community vulnerability to rank areas for cooling investment.

**Target users:** city sustainability departments, urban planners, researchers, environmental justice organizations.

## Scope
| In scope (MVP) | Out of scope |
|---|---|
| Selected Long Beach neighborhoods | Real-time forecasting |
| LST regression (baseline, Random Forest, LightGBM) | Pedestrian-level air temperature |
| Interactive map + neighborhood comparison | TODO: confirm scenario tool / brief generator (see docs/scope.md) |
| Configurable prioritization | |

## Architecture
```
Landsat 8/9 + NLCD + Census + CDC SVI
        -> src/coolblocks/data, features      (data pipeline)
        -> src/coolblocks/models, evaluation  (ML + spatial validation)
        -> src/coolblocks/prioritization      (weighted ranking)
        -> app/streamlit_app.py               (interactive map)
```

## Quickstart
```bash
conda env create -f environment.yml
conda activate coolblocks

make data       # download + cloud-mask Landsat, NLCD, Census, SVI   (TODO: implement)
make features   # build data/processed/features.parquet               (TODO: implement)
make train      # spatial-CV model comparison -> results/
make app        # launch the Streamlit map
make test       # pytest
```
Settings (study area, dates, seeds, weights) live in [configs/config.yaml](configs/config.yaml).

## Data
| Source | Use | License / terms |
|---|---|---|
| USGS Landsat 8/9 Collection 2 Level 2 | LST target + spectral bands | Public domain (USGS) |
| NLCD (impervious, tree canopy) | Land cover predictors | Public domain (MRLC) |
| U.S. Census Bureau (TIGER, ACS) | Boundaries, demographics | Public domain |
| CDC/ATSDR Social Vulnerability Index | Vulnerability indicators | Public domain |

Details and limitations: [docs/data_sheet.md](docs/data_sheet.md). Raw data is never committed.

## Evaluation
MAE, RMSE and R² against mean and linear baselines, using **spatial holdout by unit** (never random
pixel splits), plus temporal holdout if data allows. See [docs/evaluation_plan.md](docs/evaluation_plan.md).

## Results
_To be filled in after Phase 3 evaluations._

## Limitations
- LST is satellite-measured surface temperature, **not** pedestrian-level air temperature or human exposure.
- Relationships learned by the model are associations, not causal effects.
- Predictions may not generalize to unseen neighborhoods or dates; see the model card.

## Team and timeline
TODO: names and roles. Milestones: PoC Oct 30, MVP Nov 13, Phase 2 Nov 20, Phase 3 Dec 11.

## References
See [docs/references.md](docs/references.md).

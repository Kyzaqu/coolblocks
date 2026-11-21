# Model card

- **Task:** supervised regression of land surface temperature (°C).
- **Intended use:** preliminary, neighborhood-level screening to support planning discussions.
- **Not for:** individual health decisions, air temperature or heat-index claims, causal conclusions, or final infrastructure siting without field validation.
- **Models compared:** mean baseline, linear regression, Random Forest, LightGBM.
- **Training data:** TODO (scenes, dates, units).
- **Validation:** spatial group CV (+ temporal holdout if available).
- **Results:** TODO table of MAE / RMSE / R² per model.
- **Failure modes:** TODO (e.g. waterfront, industrial roofs, scenes with thin cloud, units with few clear observations).
- **Fairness / equity considerations:** TODO: does error differ across high- and low-SVI neighborhoods?

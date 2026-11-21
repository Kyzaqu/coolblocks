# MVP scope

## Implemented

- Long Beach city-clipped Census tracts as explicit neighborhood-scale proxies.
- Three pinned public Landsat summer 2023 scenes and public Census boundaries.
- Quality masking, coverage filtering, mean LST and three spectral predictors.
- Mean/linear/Random Forest comparison using existing whole-tract spatial CV.
- Saved held-out predictions, final fitted model, source/checksum and fold audit metadata.
- Streamlit prediction/observation/error maps, click inspection, date selection, and two-tract comparison.
- Bundled real-data boundaries and predictions for a runnable demonstration without imagery downloads.
- Reproducible local commands, automated workflow checks, and documented scientific limitations.

## Deferred

NLCD, ACS/SVI ingestion and vulnerability prioritization; LightGBM/SHAP; named-neighborhood
boundaries; temporal holdout; field and stakeholder evaluation; hosted deployment.
The original prioritization module remains available but is not connected to this MVP.

## Out of scope

Real-time or weather forecasting, air-temperature/exposure estimates, causal intervention simulation,
automated planning briefs, or unsupported health/equity/investment recommendations.

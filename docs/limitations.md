# Scientific and product limitations

1. LST is a satellite surface retrieval at overpass time, not air temperature, heat index, human
   exposure, or a ground thermometer reading. No personal health inference is supported.
2. Three summer 2023 scenes are a small, selected sample, not a summer or annual average. Same-date
   reflectance predictors make this a reconstruction experiment, not future-temperature forecasting.
3. Thermal observations have approximately 100 m native resolution, resampled to 30 m. Pixels are
   spatially dependent; tract averages do not establish independent block-level detail.
4. Clear-sky masking, water exclusion, missing emissivity retrievals, and coverage thresholds select
   which pixels and tracts appear. Valid fraction includes water in its denominator. Missing areas
   must not be interpreted as cool. Residual cloud and retrieval errors may remain.
5. Census tracts clipped to a generalized city polygon are neighborhood proxies. Small clipped
   fragments, shoreline mixing, and pixel-center allocation can affect means. They are not official
   named neighborhoods, and cartographic boundaries are not legal boundaries.
6. Whole-tract spatial CV is preserved, but adjacent tracts may share a fold boundary and all dates
   occur in training folds. No spatial buffer, temporal holdout, or outside-city test was performed.
7. Model selection and reporting reuse the same folds. R² is low and differences between linear
   regression and Random Forest are tiny. No uncertainty intervals or superiority claims are supported.
8. Spectral associations do not show causal cooling benefits. NDVI is not canopy percentage, NDBI
   is not impervious percentage, and NDWI is not a land-cover classification.
9. No SVI, demographic, equity, stakeholder, or field evaluation was completed. The MVP cannot
   justify vulnerability rankings, intervention simulations, or final investment decisions.
10. Pinned source IDs, cached checksums, configuration, and package versions improve reproducibility.
    Remote availability and upstream product revisions can still change. This is a local prototype,
    without hosted deployment, production monitoring, or guaranteed bitwise cross-platform results.

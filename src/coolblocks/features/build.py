"""Build the model-ready feature table (TODO).

Inputs:
  data/interim/lst_by_unit.parquet   (from `make data`: LST + spectral indices per unit and scene)
  NLCD impervious / canopy, ACS, CDC SVI aggregated to block groups

Expected output: data/processed/features.parquet with one row per (block group, scene date):
  unit_id          12-digit block group GEOID (string)
  tract_id         first 11 digits of unit_id; SVI is joined on this (string)
  scene_id, date
  lst_celsius      prediction target
  ndvi, ndbi, ndwi
  impervious_pct, tree_canopy_pct
  svi_overall, population   (optional; svi_overall is also used by the prioritization tool)

Use coolblocks.features.ids.add_tract_id to create tract_id. Set modeling.group_column in
configs/config.yaml to `tract_id` to hold out whole tracts (stricter spatial test) or `unit_id`
to hold out block groups.
"""
import argparse

from coolblocks.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    load_config(parser.parse_args().config)
    raise NotImplementedError("Join LST, NLCD, ACS and SVI per unit and write features.parquet")


if __name__ == "__main__":
    main()

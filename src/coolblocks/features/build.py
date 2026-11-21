"""Build the model-ready feature table (TODO).

Expected output: data/processed/features.parquet with columns
  unit_id, scene_id, date, lst_celsius, ndvi, ndbi, ndwi,
  impervious_pct, tree_canopy_pct, (optional) lat, lon, svi_overall, population
One row per (spatial unit, scene date).
"""
import argparse

from coolblocks.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    load_config(parser.parse_args().config)
    raise NotImplementedError("Aggregate rasters to units and write features.parquet")


if __name__ == "__main__":
    main()

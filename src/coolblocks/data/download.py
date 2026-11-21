"""Download pipeline (TODO).

Suggested sources:
  - Landsat 8/9 C2 L2: Microsoft Planetary Computer STAC (pystac-client + planetary-computer)
  - NLCD impervious + tree canopy: MRLC (mrlc.gov)
  - Boundaries + ACS: Census TIGER and data.census.gov
  - CDC/ATSDR Social Vulnerability Index (California, tract level)

Every download should write a small provenance entry (source, date, URL) to data/raw/MANIFEST.json
and be documented in docs/data_sheet.md.
"""
import argparse

from coolblocks.config import load_config


def search_landsat(cfg: dict):
    """TODO: query the STAC API with cfg['study_area']['bbox'], months, years, cloud cover."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(parser.parse_args().config)
    search_landsat(cfg)


if __name__ == "__main__":
    main()

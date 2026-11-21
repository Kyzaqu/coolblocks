"""Data download pipeline.

  python -m coolblocks.data.download --step boundaries   # Census block groups for Long Beach
  python -m coolblocks.data.download --step landsat      # Landsat LST + indices per block group
  python -m coolblocks.data.download                     # both

TODO (other team members): NLCD impervious/canopy, ACS demographics, CDC SVI. Each should write a
MANIFEST.json entry via coolblocks.data.manifest.record and be described in docs/data_sheet.md.
"""
import argparse

from coolblocks.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--step", choices=["boundaries", "landsat", "all"], default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.step in ("boundaries", "all"):
        from coolblocks.data.boundaries import build_block_groups

        build_block_groups(cfg)
    if args.step in ("landsat", "all"):
        from coolblocks.data import landsat

        landsat.run(cfg)


if __name__ == "__main__":
    main()

"""Build the model-ready feature table.

    python -m coolblocks.features.build

Input (required):  data/interim/lst_by_unit.parquet     from `python -m coolblocks.data.download`
Inputs (optional, joined automatically when the file exists):
  data/interim/static_by_unit.parquet   unit_id, impervious_pct, tree_canopy_pct      (NLCD)
  data/interim/svi_by_tract.parquet     tract_id, svi_overall (+ any other columns)   (CDC SVI)

Output: data/processed/features.parquet, one row per (block group, scene), with
  unit_id, tract_id, scene_id, date, platform, n_clear_pixels, pct_clear   identifiers/bookkeeping
  lst_celsius      raw land surface temperature (deg C)
  scene_mean_lst   mean LST across all units in that scene (captures the weather of the day)
  lst_anomaly      lst_celsius - scene_mean_lst: how much hotter/cooler than the city that day
  ndvi, ndbi, ndwi (+ impervious_pct, tree_canopy_pct, canopy_deficit, svi_* when available)

Choose the prediction target with `modeling.target` in configs/config.yaml
(`lst_anomaly` recommended: removes day-to-day weather swings, so the model learns place effects).
Caveat: scene_mean_lst is computed from all units in the scene, including ones later held out in
cross-validation. With ~300 units the effect is tiny, but mention it in the model card.
"""
import argparse
from pathlib import Path

import pandas as pd

from coolblocks.config import load_config
from coolblocks.features.ids import add_tract_id

REQUIRED = ["unit_id", "scene_id", "date", "lst_celsius", "ndvi", "ndbi", "ndwi"]


def add_scene_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Add scene_mean_lst and lst_anomaly (LST minus that scene's mean over units)."""
    out = df.copy()
    out["scene_mean_lst"] = out.groupby("scene_id")["lst_celsius"].transform("mean")
    out["lst_anomaly"] = out["lst_celsius"] - out["scene_mean_lst"]
    return out


def _join_optional(df: pd.DataFrame, path: Path, key: str) -> pd.DataFrame:
    if not path.exists():
        print(f"  (skipping {path.name}: not found)")
        return df
    extra = pd.read_parquet(path)
    extra[key] = extra[key].astype(str)
    out = df.merge(extra, on=key, how="left", validate="many_to_one")
    new_cols = [c for c in extra.columns if c != key]
    missing = out[new_cols].isna().any(axis=1).mean()
    print(f"  joined {path.name} on {key}: {len(new_cols)} columns, {missing:.1%} rows missing values")
    return out


def build_features(lst: pd.DataFrame, interim_dir: str | Path) -> pd.DataFrame:
    interim_dir = Path(interim_dir)
    missing_cols = [c for c in REQUIRED if c not in lst.columns]
    if missing_cols:
        raise KeyError(f"lst_by_unit is missing columns: {missing_cols}")

    df = add_tract_id(lst)
    df = df.dropna(subset=["lst_celsius", "ndvi", "ndbi", "ndwi"])
    if df.duplicated(["unit_id", "scene_id"]).any():
        raise ValueError("Duplicate (unit_id, scene_id) rows found")

    df = add_scene_anomaly(df)
    df = _join_optional(df, interim_dir / "static_by_unit.parquet", "unit_id")
    df = _join_optional(df, interim_dir / "svi_by_tract.parquet", "tract_id")
    if "tree_canopy_pct" in df.columns:
        df["canopy_deficit"] = 100 - df["tree_canopy_pct"]
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(parser.parse_args().config)
    interim, processed = Path(cfg["paths"]["interim"]), Path(cfg["paths"]["processed"])

    lst = pd.read_parquet(interim / "lst_by_unit.parquet")
    df = build_features(lst, interim)

    processed.mkdir(parents=True, exist_ok=True)
    out = processed / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {out}: {len(df)} rows, {df['unit_id'].nunique()} units, {df['scene_id'].nunique()} scenes")
    print(f"  std of lst_celsius: {df['lst_celsius'].std():.2f} C | std of lst_anomaly: {df['lst_anomaly'].std():.2f} C")


if __name__ == "__main__":
    main()

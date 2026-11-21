"""Train and compare models with spatial group cross-validation.

Reads data/processed/features.parquet, writes results/cv_metrics_*.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coolblocks.config import load_config
from coolblocks.evaluation.leakage import assert_no_group_overlap, forbidden_features
from coolblocks.evaluation.metrics import regression_metrics
from coolblocks.evaluation.splits import spatial_group_splits
from coolblocks.models.registry import build_model

NON_FEATURES = {"unit_id", "scene_id", "date"}


def run_cv(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = cfg["modeling"]
    target, group = m["target"], m["group_column"]
    features = [c for c in df.columns if c not in NON_FEATURES | {target}]
    bad = forbidden_features(features, cfg["features"]["exclude_from_model"])
    if bad:
        raise ValueError(f"Leaky features present: {bad}")

    rows = []
    for name in m["models"]:
        for fold, (tr, te) in enumerate(spatial_group_splits(df, group, m["n_splits"])):
            assert_no_group_overlap(df, tr, te, group)
            model = build_model(name, cfg["seed"])
            model.fit(df.iloc[tr][features], df.iloc[tr][target])
            pred = model.predict(df.iloc[te][features])
            scores = regression_metrics(df.iloc[te][target], pred)
            rows.append({"model": name, "fold": fold, **scores})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(parser.parse_args().config)
    np.random.seed(cfg["seed"])
    df = pd.read_parquet(Path(cfg["paths"]["processed"]) / "features.parquet")
    folds = run_cv(df, cfg)
    out = Path(cfg["paths"]["results"])
    out.mkdir(exist_ok=True)
    folds.to_csv(out / "cv_metrics_by_fold.csv", index=False)
    summary = folds.groupby("model")[["mae", "rmse", "r2"]].agg(["mean", "std"])
    summary.to_csv(out / "cv_metrics_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()

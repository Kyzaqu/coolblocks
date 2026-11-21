"""Train and compare models with spatial group cross-validation.

    python -m coolblocks.models.train

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

# identifiers and bookkeeping columns that must never be used as model inputs
NON_FEATURES = {"unit_id", "tract_id", "scene_id", "date", "platform", "n_clear_pixels", "pct_clear"}
# every possible target form: whichever is not the target would leak it, so none are features
TARGET_COLUMNS = {"lst_celsius", "lst_anomaly"}
# date-level weather proxy: excluded unless modeling.use_scene_mean_feature is true
SCENE_MEAN = "scene_mean_lst"


def model_features(df, cfg):
    """Allowlist predictors so coverage counts, dates and targets cannot silently enter ML."""
    features = cfg["features"]["predictors"]
    target, group = cfg["modeling"]["target"], cfg["modeling"]["group_column"]
    bad = forbidden_features(features, cfg["features"]["exclude_from_model"] + [target, group])
    if not features or len(set(features)) != len(features) or bad:
        raise ValueError(f"Invalid or leaky predictor list: {features}")
    required = features + [target, "unit_id", "scene_id", "date"]
    if group != "unit_id":
        required.append(group)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[required].isna().any().any() or not np.isfinite(df[features + [target]]).all().all():
        raise ValueError("Training columns contain missing or non-finite values")
    # Distinct units can share a holdout group (e.g. block groups in one tract).
    if df.duplicated(["unit_id", "scene_id"]).any():
        raise ValueError("Duplicate spatial unit / scene observations")
    return features


def run_cv(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = cfg["modeling"]
    target, group = m["target"], m["group_column"]
    features = feature_columns(df, cfg)
    print(f"Target: {target} | group: {group} | features: {features}")

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
    print(summary.round(3))


if __name__ == "__main__":
    main()

"""Spatial cross-validation, explicit held-out predictions, and a final fitted model."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from coolblocks.config import load_config
from coolblocks.data.download import sha256
from coolblocks.evaluation.leakage import assert_no_group_overlap, forbidden_features
from coolblocks.evaluation.metrics import regression_metrics
from coolblocks.evaluation.splits import spatial_group_splits
from coolblocks.models.registry import build_model


def model_features(df, cfg):
    """Allowlist predictors so coverage counts, dates and targets cannot silently enter ML."""
    features = cfg["features"]["predictors"]
    target, group = cfg["modeling"]["target"], cfg["modeling"]["group_column"]
    bad = forbidden_features(features, cfg["features"]["exclude_from_model"] + [target, group])
    if not features or len(set(features)) != len(features) or bad:
        raise ValueError(f"Invalid or leaky predictor list: {features}")
    required = features + [target, group, "scene_id", "date"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[required].isna().any().any() or not np.isfinite(df[features + [target]]).all().all():
        raise ValueError("Training columns contain missing or non-finite values")
    if df.duplicated([group, "scene_id"]).any():
        raise ValueError("Duplicate spatial unit / scene observations")
    return features


def cross_validate(df, cfg):
    m = cfg["modeling"]
    target, group = m["target"], m["group_column"]
    features = model_features(df, cfg)
    rows, predictions, audit = [], [], []
    splits = list(spatial_group_splits(df, group, m["n_splits"]))
    for fold, (tr, te) in enumerate(splits):
        assert_no_group_overlap(df, tr, te, group)
        audit.append(
            {
                "fold": fold,
                "train_units": sorted(df.iloc[tr][group].unique().tolist()),
                "test_units": sorted(df.iloc[te][group].unique().tolist()),
                "train_rows": len(tr),
                "test_rows": len(te),
                "overlap_count": 0,
            }
        )
    for name in m["models"]:
        for fold, (tr, te) in enumerate(splits):
            model = build_model(name, cfg["seed"])
            model.fit(df.iloc[tr][features], df.iloc[tr][target])
            pred = model.predict(df.iloc[te][features])
            scores = regression_metrics(df.iloc[te][target], pred)
            rows.append({"model": name, "fold": fold, "test_rows": len(te), **scores})
            held = df.iloc[te][[group, "scene_id", "date"]].copy()
            held["observed_lst_celsius"] = df.iloc[te][target].to_numpy()
            held["predicted_lst_celsius"] = pred
            held["residual_celsius"] = held.observed_lst_celsius - pred
            held["model"], held["fold"] = name, fold
            held["prediction_type"] = "spatial_out_of_fold"
            predictions.append(held)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), audit


def run_cv(df, cfg):
    """Retain the original metric-table interface."""
    return cross_validate(df, cfg)[0]


def train(cfg):
    processed, out = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(processed / "features.parquet")
    feature_metadata = json.loads((processed / "feature_metadata.json").read_text())
    if feature_metadata["features_sha256"] != sha256(processed / "features.parquet"):
        raise ValueError("Feature checksum mismatch; rerun make features")
    if feature_metadata["settings"] != cfg:
        raise ValueError("Config differs from feature build; rerun make features")
    folds, oof, audit = cross_validate(df, cfg)
    summary = folds.groupby("model")[["mae", "rmse", "r2"]].agg(["mean", "std"])
    # Fixed tie-break by model name. This comparison is exploratory, not nested CV.
    selected = summary[("mae", "mean")].sort_index().sort_values(kind="stable").index[0]
    features = model_features(df, cfg)
    model = build_model(selected, cfg["seed"])
    model.fit(df[features], df[cfg["modeling"]["target"]])
    predictions = df.rename(columns={cfg["modeling"]["target"]: "observed_lst_celsius"}).copy()
    predictions["predicted_lst_celsius"] = model.predict(df[features])
    predictions["prediction_type"] = "fitted_in_sample"
    predictions["model"] = selected
    held = oof.loc[oof.model == selected, ["unit_id", "scene_id", "predicted_lst_celsius"]]
    predictions = predictions.merge(
        held.rename(columns={"predicted_lst_celsius": "oof_predicted_lst_celsius"}),
        on=["unit_id", "scene_id"],
        validate="one_to_one",
    )
    folds.to_csv(out / "cv_metrics_by_fold.csv", index=False)
    summary.to_csv(out / "cv_metrics_summary.csv")
    oof.to_csv(out / "oof_predictions.csv", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    joblib.dump({"model": model, "predictors": features, "config": cfg}, out / "model.joblib")
    # A single latest observed date for the upcoming map, with no cross-date averaging.
    latest = predictions.loc[predictions.date == predictions.date.max()]
    units = gpd.read_file(processed / "units.geojson")
    units.merge(latest, on="unit_id", validate="one_to_one").to_file(
        processed / "neighborhoods.geojson", driver="GeoJSON"
    )
    pooled = {
        name: regression_metrics(g.observed_lst_celsius, g.predicted_lst_celsius)
        for name, g in oof.groupby("model")
    }
    metadata = {
        "selected_model": selected,
        "selection_rule": "lowest mean spatial-fold MAE",
        "pooled_oof_metrics": pooled,
        "rows": len(df),
        "units": df.unit_id.nunique(),
        "dates": sorted(df.date.unique().tolist()),
        "predictors": features,
        "feature_provenance": feature_metadata,
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "geopandas", "rasterio", "pyarrow"]
        },
        "spatial_folds": audit,
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string())
    print(f"Selected {selected}; saved held-out predictions and final fitted model to {out}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    train(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()

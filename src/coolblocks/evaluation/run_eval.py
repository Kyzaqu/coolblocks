"""Recompute reported held-out metrics and summarize errors without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coolblocks.config import load_config
from coolblocks.evaluation.metrics import regression_metrics


def evaluate(cfg):
    out = Path(cfg["paths"]["results"])
    metadata = json.loads((out / "run_metadata.json").read_text())
    predictions = pd.read_csv(out / "oof_predictions.csv", dtype={"unit_id": str})
    if set(predictions.model) != set(metadata["pooled_oof_metrics"]):
        raise ValueError("Prediction models differ from the reported run")
    if set(predictions.fold) != {fold["fold"] for fold in metadata["spatial_folds"]}:
        raise ValueError("Prediction folds differ from the spatial audit")
    for fold in metadata["spatial_folds"]:
        if set(fold["train_units"]) & set(fold["test_units"]):
            raise ValueError("Spatial fold audit contains overlapping units")
        rows = predictions.loc[predictions.fold == fold["fold"]]
        if set(rows.unit_id) != set(fold["test_units"]):
            raise ValueError("Predictions do not match held-out spatial units")
    scores = {}
    for name, group in predictions.groupby("model"):
        if len(group) != metadata["rows"] or group.duplicated(["unit_id", "scene_id"]).any():
            raise ValueError("Held-out predictions are missing or duplicated")
        if not group.prediction_type.eq("spatial_out_of_fold").all():
            raise ValueError("Evaluation requires held-out predictions")
        scores[name] = regression_metrics(group.observed_lst_celsius, group.predicted_lst_celsius)
        for metric, value in scores[name].items():
            if not np.isclose(value, metadata["pooled_oof_metrics"][name][metric]):
                raise ValueError("Recomputed metrics differ from the reported run")
    selected = predictions.loc[predictions.model == metadata["selected_model"]].copy()
    selected["absolute_error_celsius"] = (
        selected.observed_lst_celsius - selected.predicted_lst_celsius
    ).abs()
    selected["residual_celsius"] = selected.observed_lst_celsius - selected.predicted_lst_celsius
    for column in ("date", "unit_id"):
        errors = selected.groupby(column).agg(
            observations=("absolute_error_celsius", "size"),
            mae_celsius=("absolute_error_celsius", "mean"),
            mean_residual_celsius=("residual_celsius", "mean"),
        )
        errors.sort_values("mae_celsius", ascending=False).to_csv(out / f"error_by_{column}.csv")
    frame = pd.DataFrame(scores).T
    frame.index.name = "model"
    frame.to_csv(out / "pooled_oof_metrics.csv")
    print(frame.to_string())
    print("Spatial fold audit and reported held-out metrics verified.")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    evaluate(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()

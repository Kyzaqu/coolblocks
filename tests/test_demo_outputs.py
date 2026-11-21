"""Check that the bundled demonstration and published evaluation describe the same run."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from coolblocks.config import load_config
from coolblocks.evaluation.metrics import regression_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_demo_matches_published_held_out_predictions_and_scores():
    cfg = load_config(ROOT / "configs/config.yaml")
    metadata = json.loads((ROOT / "results/run_metadata.json").read_text())
    demo = pd.read_parquet(ROOT / "results/predictions.parquet")
    units = gpd.read_file(ROOT / "data/processed/units.geojson")
    oof = pd.read_csv(ROOT / "results/oof_predictions.csv", dtype={"unit_id": str})
    folds = pd.read_csv(ROOT / "results/cv_metrics_by_fold.csv")
    keys = ["unit_id", "scene_id", "date"]

    assert metadata["predictors"] == cfg["features"]["predictors"] == ["ndvi", "ndbi", "ndwi"]
    assert len(demo) == metadata["rows"]
    assert not demo.duplicated(keys).any()
    assert set(demo.unit_id) == set(units.unit_id)
    assert len(units) == metadata["units"] and units.unit_id.is_unique
    assert units.geometry.is_valid.all() and not units.geometry.is_empty.any()
    assert units.to_crs(3310).area.gt(0).all()
    assert sorted(demo.date.unique()) == metadata["dates"]
    assert set(demo.scene_id) == set(cfg["landsat"]["scene_ids"])
    numeric = ["observed_lst_celsius", "oof_predicted_lst_celsius", *metadata["predictors"]]
    assert np.isfinite(demo[numeric]).all().all()
    assert demo.valid_pixels.ge(cfg["features"]["min_valid_pixels"]).all()
    assert demo.valid_fraction.between(cfg["features"]["min_valid_fraction"], 1).all()
    assert demo.prediction_type.eq("fitted_in_sample").all()
    assert demo.model.eq(metadata["selected_model"]).all()

    selected = oof.loc[oof.model == metadata["selected_model"]]
    joined = demo.merge(selected, on=keys, suffixes=("_demo", "_oof"), validate="one_to_one")
    assert len(joined) == len(demo)
    np.testing.assert_allclose(
        joined.oof_predicted_lst_celsius, joined.predicted_lst_celsius_oof, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        joined.observed_lst_celsius_demo, joined.observed_lst_celsius_oof, rtol=1e-12, atol=1e-12
    )

    for model, predictions in oof.groupby("model"):
        assert len(predictions) == len(demo)
        assert not predictions.duplicated(keys).any()
        assert predictions.prediction_type.eq("spatial_out_of_fold").all()
        assert predictions.groupby("unit_id").fold.nunique().eq(1).all()
        scores = regression_metrics(
            predictions.observed_lst_celsius, predictions.predicted_lst_celsius
        )
        for metric, score in scores.items():
            assert np.isclose(score, metadata["pooled_oof_metrics"][model][metric])
        for audit in metadata["spatial_folds"]:
            assert not set(audit["train_units"]) & set(audit["test_units"])
            held = predictions.loc[predictions.fold == audit["fold"]]
            assert set(held.unit_id) == set(audit["test_units"])
            assert len(held) == audit["test_rows"]
            reported = folds.loc[(folds.model == model) & (folds.fold == audit["fold"])].iloc[0]
            actual = regression_metrics(held.observed_lst_celsius, held.predicted_lst_celsius)
            for metric, score in actual.items():
                assert np.isclose(score, reported[metric])

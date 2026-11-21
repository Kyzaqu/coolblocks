"""Small artificial fixtures test software behavior; they are never reported as results."""

import copy
import json

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from coolblocks.config import load_config
from coolblocks.data.download import BANDS, sha256
from coolblocks.features.build import pixel_features
from coolblocks.models.train import cross_validate, model_features, train


def test_pixel_mask_and_physical_scaling():
    bands = {name: np.full(10, 20000, dtype="uint16") for name in BANDS}
    bands["lwir11"][:] = 45000
    bands["qa_pixel"] = np.array([0, 1, 2, 4, 8, 16, 32, 128, 0, 0], dtype="uint16")
    bands["qa_radsat"][:] = 0
    bands["qa_radsat"][8] = 1
    bands["red"][9] = 0  # missing reflectance is never treated as an observation
    values, valid = pixel_features(bands)
    assert valid.tolist() == [True] + [False] * 9
    assert values["lst_celsius"][0] == pytest.approx(45000 * 0.00341802 + 149 - 273.15)
    assert values["ndvi"][0] == 0
    bands["lwir11"][0] = 0
    assert not pixel_features(bands)[1].any()


@pytest.fixture
def training_data(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["processed"] = str(tmp_path / "processed")
    cfg["paths"]["results"] = str(tmp_path / "results")
    cfg["modeling"]["models"] = ["mean_baseline", "linear"]
    rows = []
    for i in range(10):
        for day in (1, 2):
            rows.append(
                {
                    "unit_id": f"unit-{i}",
                    "scene_id": f"scene-{day}",
                    "date": f"2023-08-0{day}",
                    "ndvi": i / 20,
                    "ndbi": -i / 20,
                    "ndwi": day / 10,
                    "lst_celsius": 40 - i + day,
                    "valid_pixels": 100 + i,
                }
            )
    return pd.DataFrame(rows), cfg


def test_oof_holds_entire_units_and_predicts_each_row_once(training_data):
    df, cfg = training_data
    folds, predictions, audit = cross_validate(df, cfg)
    assert len(folds) == 10
    for item in audit:
        assert not set(item["train_units"]) & set(item["test_units"])
    for _, group in predictions.groupby("model"):
        assert len(group) == len(df)
        assert not group.duplicated(["unit_id", "scene_id"]).any()
        assert group.groupby("unit_id").fold.nunique().eq(1).all()
        assert group.prediction_type.eq("spatial_out_of_fold").all()
    # The coverage count is deliberately present but cannot enter the model.
    assert model_features(df, cfg) == ["ndvi", "ndbi", "ndwi"]


@pytest.mark.parametrize("problem", ["leak", "missing", "infinite", "duplicate"])
def test_reject_invalid_training_inputs(training_data, problem):
    df, cfg = training_data
    if problem == "leak":
        cfg["features"]["predictors"].append("lst_celsius")
    elif problem == "missing":
        df.loc[0, "unit_id"] = None
    elif problem == "infinite":
        df.loc[0, "ndvi"] = np.inf
    else:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        cross_validate(df, cfg)


def test_saved_model_predictions_and_map_outputs_agree(training_data):
    df, cfg = training_data
    from pathlib import Path

    processed, results = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])
    processed.mkdir()
    df.to_parquet(processed / "features.parquet", index=False)
    (processed / "feature_metadata.json").write_text(
        json.dumps({"settings": cfg, "features_sha256": sha256(processed / "features.parquet")})
    )
    gpd.GeoDataFrame(
        {"unit_id": df.unit_id.unique(), "name": df.unit_id.unique()},
        geometry=[box(i, 0, i + 0.5, 0.5) for i in range(10)],
        crs=4326,
    ).to_file(processed / "units.geojson", driver="GeoJSON")
    metadata = train(cfg)
    assert metadata["selected_model"] == "linear"
    from coolblocks.evaluation.run_eval import evaluate

    scores = evaluate(cfg)
    assert scores.loc["linear", "mae"] < 1e-10
    bundle = joblib.load(results / "model.joblib")
    predictions = pd.read_parquet(results / "predictions.parquet")
    np.testing.assert_allclose(
        bundle["model"].predict(predictions[bundle["predictors"]]),
        predictions.predicted_lst_celsius,
    )
    assert predictions.prediction_type.eq("fitted_in_sample").all()
    mapped = gpd.read_file(processed / "neighborhoods.geojson")
    assert len(mapped) == 10
    assert mapped.date.eq("2023-08-02").all()
    assert mapped.oof_predicted_lst_celsius.notna().all()
    # A saved prediction moved into the wrong fold must fail the independent audit.
    oof = pd.read_csv(results / "oof_predictions.csv")
    oof.loc[0, "fold"] = (oof.loc[0, "fold"] + 1) % 5
    oof.to_csv(results / "oof_predictions.csv", index=False)
    with pytest.raises(ValueError, match="held-out spatial units"):
        evaluate(cfg)


def test_city_clip_excludes_neighbors_that_only_share_an_edge():
    from coolblocks.features.build import clip_tracts

    city = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=4326)
    tracts = gpd.GeoDataFrame(
        {"GEOID": ["inside", "edge", "outside"], "NAME": ["1", "2", "3"], "COUNTYFP": ["037"] * 3},
        geometry=[box(0, 0, 0.5, 1), box(1, 0, 2, 1), box(3, 0, 4, 1)],
        crs=4326,
    )
    clipped = clip_tracts(tracts, city, "037")
    assert clipped.unit_id.tolist() == ["inside"]
    assert clipped.geom_type.eq("Polygon").all()

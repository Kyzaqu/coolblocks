import numpy as np
import pandas as pd
import pytest

from coolblocks.models.train import cross_validate, model_features, run_cv


def _df():
    rng = np.random.default_rng(0)
    n_units = 20
    # 12-digit GEOIDs: 11-digit tract (10 tracts) + 1-digit block group (2 per tract)
    unit = np.repeat([f"06037{i // 2:06d}{i % 2}" for i in range(n_units)], 4)
    df = pd.DataFrame({"unit_id": unit, "ndvi": rng.normal(size=len(unit))})
    df["tract_id"] = df["unit_id"].str[:11]
    # One observation per unit/scene, with four dates from each spatial unit.
    observation = np.tile(np.arange(4), n_units)
    df["scene_id"] = [f"s{i}" for i in observation]
    df["date"] = [f"2023-07-{i + 1:02d}" for i in observation]
    df["lst_celsius"] = 35 - 5 * df["ndvi"] + rng.normal(scale=0.5, size=len(df))
    return df


def _cfg(group):
    return {
        "seed": 0,
        "features": {"predictors": ["ndvi"], "exclude_from_model": ["lwir11"]},
        "modeling": {
            "target": "lst_celsius",
            "group_column": group,
            "n_splits": 4,
            "models": ["mean_baseline", "linear"],
        },
    }


def test_identifier_columns_not_used_as_features():
    assert model_features(_df(), _cfg("tract_id")) == ["ndvi"]


def test_cv_runs_with_tract_grouping_and_linear_beats_mean():
    folds, predictions, audit = cross_validate(_df(), _cfg("tract_id"))
    res = folds.groupby("model")["rmse"].mean()
    assert res["linear"] < res["mean_baseline"]
    for fold in audit:
        assert not set(fold["train_units"]) & set(fold["test_units"])
    for _, model in predictions.groupby("model"):
        assert model.groupby("tract_id").fold.nunique().eq(1).all()


def test_cv_runs_with_unit_grouping():
    assert len(run_cv(_df(), _cfg("unit_id"))) == 8  # 2 models x 4 folds


@pytest.mark.parametrize("group", ["unit_id", "tract_id"])
def test_cv_still_rejects_duplicate_unit_scene_observations(group):
    df = _df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate spatial unit / scene observations"):
        run_cv(df, _cfg(group))

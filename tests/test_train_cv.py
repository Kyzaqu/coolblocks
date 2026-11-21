import numpy as np
import pandas as pd

from coolblocks.models.train import NON_FEATURES, run_cv


def _df():
    rng = np.random.default_rng(0)
    n_units = 20
    # 12-digit GEOIDs: 11-digit tract (10 tracts) + 1-digit block group (2 per tract)
    unit = np.repeat([f"06037{i // 2:06d}{i % 2}" for i in range(n_units)], 4)
    df = pd.DataFrame({"unit_id": unit, "ndvi": rng.normal(size=len(unit))})
    df["tract_id"] = df["unit_id"].str[:11]
    df["scene_id"], df["date"] = "s", "2023-07-01"
    df["lst_celsius"] = 35 - 5 * df["ndvi"] + rng.normal(scale=0.5, size=len(df))
    return df


def _cfg(group):
    return {
        "seed": 0,
        "features": {"exclude_from_model": ["lwir11"]},
        "modeling": {"target": "lst_celsius", "group_column": group, "n_splits": 4,
                     "models": ["mean_baseline", "linear"]},
    }


def test_identifier_columns_not_used_as_features():
    assert {"unit_id", "tract_id", "scene_id", "date"} <= NON_FEATURES


def test_cv_runs_with_tract_grouping_and_linear_beats_mean():
    res = run_cv(_df(), _cfg("tract_id")).groupby("model")["rmse"].mean()
    assert res["linear"] < res["mean_baseline"]


def test_cv_runs_with_unit_grouping():
    assert len(run_cv(_df(), _cfg("unit_id"))) == 8  # 2 models x 4 folds

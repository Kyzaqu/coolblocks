import numpy as np
import pandas as pd
import pytest

from coolblocks.features.build import add_scene_anomaly, build_features
from coolblocks.models.train import feature_columns


def _lst(n_units=6, n_scenes=3):
    rng = np.random.default_rng(1)
    rows = []
    for s in range(n_scenes):
        day_effect = 5.0 * s  # hot days shift every unit
        for u in range(n_units):
            rows.append({
                "unit_id": f"06037{u // 2:06d}{u % 2}", "scene_id": f"S{s}", "date": f"2023-07-{s + 1:02d}",
                "platform": "landsat-8", "n_clear_pixels": 100, "pct_clear": 1.0,
                "lst_celsius": 30 + day_effect + u + rng.normal(scale=0.1),
                "ndvi": 0.3, "ndbi": 0.0, "ndwi": -0.3,
            })
    return pd.DataFrame(rows)


def test_anomaly_removes_day_effect():
    out = add_scene_anomaly(_lst())
    assert out.groupby("scene_id")["lst_anomaly"].mean().abs().max() < 1e-9
    # raw LST differs a lot between scenes, anomaly does not
    assert out.groupby("scene_id")["lst_celsius"].mean().std() > 4
    assert out.groupby("scene_id")["lst_anomaly"].mean().std() < 1e-9


def test_build_adds_tract_id_and_skips_missing_optional_files(tmp_path):
    out = build_features(_lst(), tmp_path)
    assert out["tract_id"].str.len().eq(11).all()
    assert {"lst_anomaly", "scene_mean_lst"} <= set(out.columns)
    assert "svi_overall" not in out.columns


def test_build_joins_optional_tables(tmp_path):
    lst = _lst()
    units = lst["unit_id"].unique()
    pd.DataFrame({"unit_id": units, "impervious_pct": 60.0, "tree_canopy_pct": 15.0}).to_parquet(
        tmp_path / "static_by_unit.parquet")
    tracts = pd.Series(units).str[:11].unique()
    pd.DataFrame({"tract_id": tracts, "svi_overall": 0.7}).to_parquet(tmp_path / "svi_by_tract.parquet")
    out = build_features(lst, tmp_path)
    assert len(out) == len(lst)                      # joins must not duplicate rows
    assert (out["canopy_deficit"] == 85.0).all()
    assert (out["svi_overall"] == 0.7).all()


def test_build_rejects_duplicates_and_missing_columns(tmp_path):
    lst = _lst()
    with pytest.raises(ValueError):
        build_features(pd.concat([lst, lst.iloc[:1]]), tmp_path)
    with pytest.raises(KeyError):
        build_features(lst.drop(columns=["ndvi"]), tmp_path)


def _cfg(target, use_scene_mean=False):
    return {"features": {"exclude_from_model": ["lwir11"]},
            "modeling": {"target": target, "use_scene_mean_feature": use_scene_mean}}


def test_anomaly_target_never_sees_raw_lst_or_scene_mean(tmp_path):
    df = build_features(_lst(), tmp_path)
    feats = feature_columns(df, _cfg("lst_anomaly"))
    assert feats == ["ndvi", "ndbi", "ndwi"]


def test_raw_target_excludes_anomaly_and_optionally_allows_scene_mean(tmp_path):
    df = build_features(_lst(), tmp_path)
    assert "lst_anomaly" not in feature_columns(df, _cfg("lst_celsius"))
    assert "scene_mean_lst" not in feature_columns(df, _cfg("lst_celsius"))
    assert "scene_mean_lst" in feature_columns(df, _cfg("lst_celsius", use_scene_mean=True))

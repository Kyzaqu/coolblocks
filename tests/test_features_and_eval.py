import numpy as np
import pandas as pd
import pytest

from coolblocks.data.cloud_mask import clear_mask
from coolblocks.evaluation.leakage import assert_no_group_overlap, forbidden_features
from coolblocks.evaluation.metrics import regression_metrics
from coolblocks.evaluation.splits import spatial_group_splits
from coolblocks.features.indices import ndvi


def test_ndvi_known_value():
    assert ndvi(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(0.4 / 0.6)


def test_clear_mask_flags_cloud_bit():
    qa = np.array([0, 1 << 3, 1 << 4, 1 << 5])  # clear, cloud, shadow, snow
    assert clear_mask(qa).tolist() == [True, False, False, False]


def test_spatial_splits_do_not_share_groups():
    df = pd.DataFrame({"unit_id": np.repeat(np.arange(10), 5), "x": range(50)})
    for tr, te in spatial_group_splits(df, "unit_id", 5):
        assert_no_group_overlap(df, tr, te, "unit_id")


def test_forbidden_features():
    assert forbidden_features(["ndvi", "LWIR11"], ["lwir11"]) == ["LWIR11"]


def test_metrics_perfect_prediction():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from coolblocks.data.landsat import filter_items, process_bands, zonal_means


def _item(month, platform="landsat-8", cat="T1"):
    return SimpleNamespace(
        datetime=datetime(2023, month, 15),
        properties={"platform": platform, "landsat:collection_category": cat},
    )


def test_filter_items():
    items = [_item(7), _item(1), _item(8, "landsat-7"), _item(8, cat="T2"), _item(9, "landsat-9")]
    kept = filter_items(items, months=[6, 7, 8, 9], platforms=["landsat-8", "landsat-9"])
    assert [i.datetime.month for i in kept] == [7, 9]


def test_zonal_means_basic():
    labels = np.array([[1, 1, 2], [1, 2, 0]])
    vals = np.array([[10.0, 20.0, 5.0], [30.0, 15.0, 99.0]])
    valid = np.ones_like(labels, dtype=bool)
    df = zonal_means(labels, {"v": vals}, valid, n_units=2)
    assert df["v"].tolist() == [20.0, 10.0]
    assert df["n_total"].tolist() == [3, 2]


def test_zonal_means_respects_valid_mask_and_nan():
    labels = np.array([[1, 1, 1, 1]])
    vals = np.array([[10.0, 50.0, np.nan, 30.0]])
    valid = np.array([[True, False, True, True]])
    df = zonal_means(labels, {"v": vals}, valid, n_units=1)
    assert df["v"].iloc[0] == 20.0          # 10 and 30 only
    assert df["n_clear"].iloc[0] == 3       # valid mask only; NaN handled per layer


def test_zonal_means_unit_with_no_valid_pixels_is_nan():
    labels = np.array([[1, 2]])
    df = zonal_means(labels, {"v": np.array([[1.0, 2.0]])}, np.array([[True, False]]), 2)
    assert df["v"].iloc[0] == 1.0 and np.isnan(df["v"].iloc[1])


def _dn_for_reflectance(r):
    return (r + 0.2) / 0.0000275


def test_process_bands_end_to_end():
    shape = (2, 2)
    labels = np.array([[1, 1], [2, 2]])
    # unit 1: clear land; unit 2: cloudy
    qa = np.array([[0, 0], [1 << 3, 1 << 3]], dtype=float)
    st_dn = np.full(shape, (310.0 - 149.0) / 0.00341802)   # 310 K = 36.85 C
    land = {"green": 0.10, "red": 0.10, "nir08": 0.30, "swir16": 0.20}
    bands = {b: np.full(shape, _dn_for_reflectance(v)) for b, v in land.items()}
    bands.update({"lwir11": st_dn, "qa_pixel": qa})
    df = process_bands(bands, labels, n_units=2)
    assert df["lst_celsius"].iloc[0] == pytest.approx(36.85, abs=0.05)
    assert df["ndvi"].iloc[0] == pytest.approx(0.5, abs=0.01)
    assert df["pct_clear"].iloc[0] == 1.0
    assert df["pct_clear"].iloc[1] == 0.0 and np.isnan(df["lst_celsius"].iloc[1])


def test_water_pixels_are_masked():
    labels = np.array([[1, 1]])
    qa = np.zeros((1, 2))
    st_dn = np.full((1, 2), (300.0 - 149.0) / 0.00341802)
    # pixel 0 land (nir > green), pixel 1 water (green > nir)
    g = np.array([[0.05, 0.10]]); n = np.array([[0.30, 0.02]])
    bands = {
        "green": _dn_for_reflectance(g), "red": _dn_for_reflectance(np.full((1, 2), 0.08)),
        "nir08": _dn_for_reflectance(n), "swir16": _dn_for_reflectance(np.full((1, 2), 0.1)),
        "lwir11": st_dn, "qa_pixel": qa,
    }
    assert process_bands(bands, labels, 1, mask_water=True)["n_clear"].iloc[0] == 1
    assert process_bands(bands, labels, 1, mask_water=False)["n_clear"].iloc[0] == 2

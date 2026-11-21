import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box

from coolblocks.data.boundaries import select_block_groups, tiger_url


def test_tiger_url():
    assert tiger_url("BG", 2023, "06").endswith("/TIGER2023/BG/tl_2023_06_bg.zip")
    assert tiger_url("PLACE", 2023, "06").endswith("/PLACE/tl_2023_06_place.zip")


def _bg():
    rows = [
        # inside place, land
        {"GEOID": "060375701001", "COUNTYFP": "037", "TRACTCE": "570100", "BLKGRPCE": "1",
             "ALAND": 900, "AWATER": 100, "geometry": box(0, 0, 1, 1)},
        # inside place but mostly water
        {"GEOID": "060375701002", "COUNTYFP": "037", "TRACTCE": "570100", "BLKGRPCE": "2",
             "ALAND": 100, "AWATER": 900, "geometry": box(1, 0, 2, 1)},
        # outside place
        {"GEOID": "060375999001", "COUNTYFP": "037", "TRACTCE": "599900", "BLKGRPCE": "1",
             "ALAND": 900, "AWATER": 0, "geometry": box(10, 10, 11, 11)},
        # wrong county
        {"GEOID": "060595701001", "COUNTYFP": "059", "TRACTCE": "570100", "BLKGRPCE": "1",
             "ALAND": 900, "AWATER": 0, "geometry": box(0, 0, 1, 1)},
    ]
    return gpd.GeoDataFrame(rows, crs=4269)


def test_select_block_groups():
    place = gpd.GeoDataFrame({"NAME": ["Long Beach"]}, geometry=[box(-1, -1, 3, 2)], crs=4269)
    out = select_block_groups(_bg(), place, "037", min_land_fraction=0.5)
    assert out["unit_id"].tolist() == ["060375701001"]
    assert out["tract_id"].tolist() == ["06037570100"]
    assert out["name"].iloc[0] == "Tract 5701.00, BG 1"

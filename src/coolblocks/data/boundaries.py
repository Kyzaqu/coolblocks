"""Census TIGER block groups for the study area."""
import urllib.request
from pathlib import Path

from coolblocks.data import manifest
from coolblocks.features.ids import add_tract_id

TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER{year}"


def tiger_url(layer: str, year: int, state_fips: str) -> str:
    """layer is 'BG' (block groups) or 'PLACE' (cities)."""
    return f"{TIGER_BASE.format(year=year)}/{layer}/tl_{year}_{state_fips}_{layer.lower()}.zip"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def select_block_groups(bg, place, county_fips: str, min_land_fraction: float):
    """Pick block groups whose representative point lies inside the place polygon.

    bg and place are GeoDataFrames in the same CRS. Returns a GeoDataFrame with
    unit_id (12-digit GEOID), tract_id, name, land_fraction and geometry.
    """
    bg = bg[bg["COUNTYFP"] == county_fips].copy()
    boundary = place.geometry.union_all() if hasattr(place.geometry, "union_all") else place.unary_union
    inside = bg.geometry.representative_point().within(boundary)
    bg = bg[inside].copy()

    total = bg["ALAND"] + bg["AWATER"]
    bg["land_fraction"] = (bg["ALAND"] / total.where(total > 0)).fillna(0.0)
    bg = bg[bg["land_fraction"] >= min_land_fraction].copy()

    bg["unit_id"] = bg["GEOID"].astype(str)
    bg["name"] = (
        "Tract " + bg["TRACTCE"].str[:4] + "." + bg["TRACTCE"].str[4:] + ", BG " + bg["BLKGRPCE"]
    )
    bg = add_tract_id(bg)
    return bg[["unit_id", "tract_id", "name", "land_fraction", "ALAND", "geometry"]].reset_index(
        drop=True
    )


def build_block_groups(cfg: dict):
    """Download TIGER files, select Long Beach block groups, save to data/interim/block_groups.gpkg."""
    import geopandas as gpd

    sa = cfg["study_area"]
    raw, interim = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"])
    bg_url = tiger_url("BG", sa["tiger_year"], sa["state_fips"])
    place_url = tiger_url("PLACE", sa["tiger_year"], sa["state_fips"])

    bg = gpd.read_file(_download(bg_url, raw / Path(bg_url).name))
    places = gpd.read_file(_download(place_url, raw / Path(place_url).name))
    place = places[places["NAME"] == sa["place_name"]]
    if place.empty:
        raise ValueError(f"Place {sa['place_name']!r} not found in TIGER place file")

    units = select_block_groups(bg, place, sa["county_fips"], sa["min_land_fraction"])
    units = units.to_crs(4326)
    interim.mkdir(parents=True, exist_ok=True)
    out = interim / "block_groups.gpkg"
    units.to_file(out, driver="GPKG")

    manifest.record(raw, "Census TIGER block groups + places", urls=[bg_url, place_url],
                    year=sa["tiger_year"], n_block_groups=len(units))
    print(f"Saved {len(units)} block groups to {out}")
    return units

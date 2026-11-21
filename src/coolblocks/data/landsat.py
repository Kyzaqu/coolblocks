"""Landsat 8/9 Collection 2 Level 2 -> per-block-group LST and spectral indices.

Flow: STAC search (Planetary Computer) -> lazy stack of scenes -> per scene: cloud mask, scale,
indices -> average pixels inside each block group -> one row per (unit, scene).

Note: Landsat passes over Long Beach at roughly 10:30 am local time, so LST reflects late-morning
conditions, not the afternoon peak.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from coolblocks.data import manifest
from coolblocks.data.cloud_mask import clear_mask, sr_to_reflectance, st_to_celsius
from coolblocks.features.ids import add_tract_id
from coolblocks.features.indices import ndbi, ndvi, ndwi

# Planetary Computer asset names for Landsat C2 L2
ASSETS = ["green", "red", "nir08", "swir16", "lwir11", "qa_pixel"]
REFLECTANCE_BANDS = ["green", "red", "nir08", "swir16"]


# ---------- pure functions (unit tested) ----------
def filter_items(items, months, platforms):
    """Keep items acquired in the given months, from the given platforms, in Tier 1."""
    keep = []
    for it in items:
        props = it.properties
        if it.datetime.month not in months:
            continue
        if props.get("platform") not in platforms:
            continue
        if props.get("landsat:collection_category", "T1") != "T1":
            continue
        keep.append(it)
    return keep


def zonal_means(labels: np.ndarray, layers: dict, valid: np.ndarray, n_units: int) -> pd.DataFrame:
    """Average each layer over valid pixels inside each unit.

    labels: int array, 0 = outside any unit, k = unit k (1-based).
    Returns a DataFrame with one row per unit (in label order) with columns
    n_total (pixels in unit), n_clear (valid pixels) and one mean column per layer.
    """
    flat = labels.ravel()
    inside = flat > 0
    size = n_units + 1
    n_total = np.bincount(flat[inside], minlength=size)[1:]
    ok = valid.ravel() & inside
    out = {"n_total": n_total, "n_clear": np.bincount(flat[ok], minlength=size)[1:]}
    for name, arr in layers.items():
        v = arr.ravel()
        good = ok & np.isfinite(v)
        cnt = np.bincount(flat[good], minlength=size)[1:]
        tot = np.bincount(flat[good], weights=v[good], minlength=size)[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = np.where(cnt > 0, tot / cnt, np.nan)
    return pd.DataFrame(out)


def process_bands(bands: dict, labels: np.ndarray, n_units: int, mask_water: bool = True):
    """Turn raw Landsat digital numbers for one scene into per-unit means."""
    qa = np.nan_to_num(bands["qa_pixel"], nan=1).astype(np.uint16)  # NaN (no data) -> fill bit
    clear = clear_mask(qa)

    st = np.where(bands["lwir11"] > 0, bands["lwir11"], np.nan)
    lst = st_to_celsius(st)

    refl = {}
    for b in REFLECTANCE_BANDS:
        r = sr_to_reflectance(bands[b])
        refl[b] = np.where((r >= 0) & (r <= 1), r, np.nan)

    layers = {
        "lst_celsius": lst,
        "ndvi": ndvi(refl["nir08"], refl["red"]),
        "ndbi": ndbi(refl["swir16"], refl["nir08"]),
        "ndwi": ndwi(refl["green"], refl["nir08"]),
    }

    valid = clear.copy()
    for arr in layers.values():
        valid &= np.isfinite(arr)
    if mask_water:
        valid &= ~(layers["ndwi"] > 0)

    df = zonal_means(labels, layers, valid, n_units)
    df["pct_clear"] = np.where(df["n_total"] > 0, df["n_clear"] / df["n_total"].clip(lower=1), np.nan)
    return df


# ---------- network / geospatial steps (need Planetary Computer access) ----------
def search_scenes(cfg: dict, bbox):
    import planetary_computer
    from pystac_client import Client

    lc = cfg["landsat"]
    catalog = Client.open(lc["stac_url"], modifier=planetary_computer.sign_inplace)
    items = []
    for year in lc["years"]:
        search = catalog.search(
            collections=[lc["collection"]],
            bbox=list(bbox),
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": lc["max_cloud_cover"]}},
        )
        items.extend(search.items())
    items = filter_items(items, lc["months"], lc["platforms"])
    print(f"Found {len(items)} candidate scenes")
    return items


def run(cfg: dict) -> pd.DataFrame:
    import geopandas as gpd
    import stackstac
    from rasterio import features

    lc = cfg["landsat"]
    interim = Path(cfg["paths"]["interim"])
    units_path = interim / "block_groups.gpkg"
    if not units_path.exists():
        raise FileNotFoundError(f"{units_path} not found. Run the boundaries step first.")
    units = gpd.read_file(units_path)

    minx, miny, maxx, maxy = units.total_bounds
    bbox = (minx - 0.01, miny - 0.01, maxx + 0.01, maxy + 0.01)
    items = search_scenes(cfg, bbox)
    if not items:
        raise RuntimeError("No Landsat scenes matched the search. Check years, months and cloud cover.")
    by_id = {it.id: it for it in items}

    stack = stackstac.stack(
        items, assets=ASSETS, epsg=lc["epsg"], resolution=lc["resolution"],
        bounds_latlon=bbox, rescale=False, fill_value=np.nan, dtype="float64",
    )
    height, width = stack.sizes["y"], stack.sizes["x"]
    transform = stack.attrs["transform"]

    units_proj = units.to_crs(lc["epsg"])
    shapes = ((geom, i + 1) for i, geom in enumerate(units_proj.geometry))
    labels = features.rasterize(
        shapes, out_shape=(height, width), transform=transform, fill=0, dtype="int32"
    )

    frames = []
    for i in range(stack.sizes["time"]):
        scene_id = str(stack.id.values[i])
        scene = stack.isel(time=i).compute()
        bands = {b: scene.sel(band=b).values for b in ASSETS}
        df = process_bands(bands, labels, len(units), lc["mask_water"])
        item = by_id[scene_id]
        df.insert(0, "unit_id", units["unit_id"].to_numpy())
        df.insert(1, "scene_id", scene_id)
        df.insert(2, "date", item.datetime.date().isoformat())
        df.insert(3, "platform", item.properties.get("platform"))
        frames.append(df)
        print(f"  [{i + 1}/{stack.sizes['time']}] {scene_id}")

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"n_clear": "n_clear_pixels"}).drop(columns=["n_total"])
    before = len(out)
    out = out[out["pct_clear"] >= lc["min_clear_fraction"]].dropna(subset=["lst_celsius"])
    out = add_tract_id(out).reset_index(drop=True)
    print(f"Kept {len(out)} of {before} (unit, scene) rows after clear-pixel filter")

    interim.mkdir(parents=True, exist_ok=True)
    path = interim / "lst_by_unit.parquet"
    out.to_parquet(path, index=False)
    manifest.record(
        cfg["paths"]["raw"], "Landsat 8/9 C2 L2 via Planetary Computer",
        stac_url=lc["stac_url"], n_scenes=len(items), years=lc["years"], months=lc["months"],
        max_cloud_cover=lc["max_cloud_cover"], rows=len(out),
    )
    print(f"Saved {path}")
    return out

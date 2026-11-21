"""Aggregate co-located clear land pixels to city-clipped Census tract polygons."""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from coolblocks.config import load_config
from coolblocks.data.cloud_mask import clear_mask, sr_to_reflectance, st_to_celsius
from coolblocks.data.download import BANDS, sha256
from coolblocks.features.indices import ndbi, ndvi, ndwi


def pixel_features(bands):
    """One shared validity mask ensures target and predictors describe the same pixels."""
    qa = bands["qa_pixel"]
    valid = clear_mask(qa) & ((qa & (1 << 7)) == 0) & (bands["qa_radsat"] == 0)
    valid &= bands["lwir11"] >= 293
    for key in ("green", "red", "nir08", "swir16"):
        valid &= (bands[key] >= 7273) & (bands[key] <= 43636)
    green, red, nir, swir = [
        sr_to_reflectance(bands[k]) for k in ("green", "red", "nir08", "swir16")
    ]
    values = {
        "lst_celsius": st_to_celsius(bands["lwir11"]),
        "ndvi": ndvi(nir, red),
        "ndbi": ndbi(swir, nir),
        "ndwi": ndwi(green, nir),
    }
    for value in values.values():
        valid &= np.isfinite(value)
    return values, valid


def clip_tracts(tracts, city, county_fips):
    """Keep polygon area inside the city; shared edges alone are not study units."""
    tracts = tracts.loc[tracts.COUNTYFP == county_fips].copy()
    tracts["geometry"] = tracts.geometry.intersection(city.geometry.iloc[0])
    # Touching tracts can intersect only along a line. Rasterizing that line
    # would incorrectly turn an outside tract into a measured neighborhood.
    tracts = tracts.explode(index_parts=False)
    tracts = tracts.loc[
        ~tracts.is_empty & tracts.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    tracts = tracts.dissolve(by="GEOID", as_index=False)
    tracts = tracts.rename(columns={"GEOID": "unit_id"})
    tracts["name"] = "Census tract " + tracts.NAME
    tracts = tracts[["unit_id", "name", "geometry"]].sort_values("unit_id")
    return tracts


def build_features(cfg):
    raw, out = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw / "MANIFEST.json").read_text())
    if set(cfg["landsat"]["scene_ids"]) != {s["id"] for s in manifest["scenes"]}:
        raise ValueError("Configured scenes differ from cache; rerun make data")
    if manifest["study_area"] != cfg["study_area"]:
        raise ValueError("Study area differs from cache; rerun make data with a new raw directory")
    for source in manifest["boundaries"]:
        if sha256(source["file"]) != source["sha256"]:
            raise ValueError(f"Boundary checksum mismatch: {source['file']}")
    if sha256(raw / "city.geojson") != manifest["city_sha256"]:
        raise ValueError("Cached city geometry checksum mismatch; rerun make data")
    city = gpd.read_file(raw / "city.geojson").to_crs(4326)
    tracts = gpd.read_file(raw / "tract.zip").to_crs(4326)
    tracts = clip_tracts(tracts, city, cfg["study_area"]["county_fips"])
    rows, coverage = [], []
    for scene in manifest["scenes"]:
        bands = {}
        reference = None
        for band in BANDS:
            path = raw / scene["id"] / f"{band}.tif"
            expected = next(a["sha256"] for a in scene["assets"] if a["band"] == band)
            if sha256(path) != expected:
                raise ValueError(f"Cached raster checksum mismatch: {path}")
            with rasterio.open(path) as src:
                grid = (src.crs, src.transform, src.shape)
                if reference is not None and grid != reference:
                    raise ValueError("Landsat bands must have identical grids")
                reference = grid
                bands[band] = src.read(1)
        crs, transform, shape = reference
        values, valid = pixel_features(bands)
        for unit in tracts.to_crs(crs).itertuples():
            inside = geometry_mask([unit.geometry], shape, transform, invert=True)
            total = int(inside.sum())
            keep = inside & valid
            count = int(keep.sum())
            fraction = count / total if total else 0.0
            accepted = (
                count >= cfg["features"]["min_valid_pixels"]
                and fraction >= cfg["features"]["min_valid_fraction"]
            )
            metadata = {
                "unit_id": unit.unit_id,
                "scene_id": scene["id"],
                "date": scene["datetime"][:10],
                "valid_pixels": count,
                "total_pixels": total,
                "valid_fraction": fraction,
            }
            coverage.append({**metadata, "accepted": accepted})
            if accepted:
                rows.append({**metadata, **{k: float(v[keep].mean()) for k, v in values.items()}})
    if not rows:
        raise ValueError("No tract-scene observations pass coverage checks")
    frame = pd.DataFrame(rows).sort_values(["unit_id", "date"]).reset_index(drop=True)
    frame.to_parquet(out / "features.parquet", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    tracts.to_file(out / "units.geojson", driver="GeoJSON")
    (out / "feature_metadata.json").write_text(
        json.dumps(
            {
                "manifest_sha256": sha256(raw / "MANIFEST.json"),
                "source_manifest": manifest,
                "features_sha256": sha256(out / "features.parquet"),
                "settings": cfg,
                "rows": len(frame),
                "units": frame.unit_id.nunique(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Built {len(frame)} observations across {frame.unit_id.nunique()} city-clipped tracts")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    build_features(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()

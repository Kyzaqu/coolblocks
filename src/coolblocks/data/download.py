"""Cache public Census boundaries and small, unscaled Landsat raster windows."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from coolblocks.config import load_config

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ("green", "red", "nir08", "swir16", "lwir11", "qa_pixel", "qa_radsat")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def search_landsat(cfg):
    """Resolve explicitly pinned scene IDs, rather than changing search results."""
    catalog = Client.open(STAC_URL)
    items = list(
        catalog.search(
            collections=[cfg["landsat"]["collection"]], ids=cfg["landsat"]["scene_ids"]
        ).items()
    )
    if {i.id for i in items} != set(cfg["landsat"]["scene_ids"]):
        raise ValueError("Not all configured Landsat scenes were found")
    return sorted(items, key=lambda i: i.id)


def download(cfg):
    raw = Path(cfg["paths"]["raw"])
    raw.mkdir(parents=True, exist_ok=True)
    previous = raw / "MANIFEST.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if old.get("study_area", cfg["study_area"]) != cfg["study_area"]:
            raise ValueError("Study area changed; choose a new raw cache directory")
    sources = []
    for kind in ("place", "tract"):
        url = f"https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_06_{kind}_500k.zip"
        path = raw / f"{kind}.zip"
        if not path.exists():
            with urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        sources.append({"url": url, "file": str(path), "sha256": sha256(path)})
    places = gpd.read_file(raw / "place.zip")
    city = places.loc[places.GEOID == cfg["study_area"]["place_geoid"]].to_crs(4326)
    if len(city) != 1:
        raise ValueError("Expected exactly one Census place")
    city.to_file(raw / "city.geojson", driver="GeoJSON")
    bounds = city.total_bounds
    scenes = []
    for item in search_landsat(cfg):
        print(f"Caching {item.id}", flush=True)
        folder = raw / item.id
        folder.mkdir(exist_ok=True)
        # Keep unsigned source metadata; SAS access tokens expire and are not provenance.
        (folder / "item.json").write_text(json.dumps(item.to_dict(), indent=2))
        assets = []
        signed = planetary_computer.sign(item)
        for band in BANDS:
            path = folder / f"{band}.tif"
            if not path.exists():
                with rasterio.open(signed.assets[band].href) as src:
                    box = transform_bounds("EPSG:4326", src.crs, *bounds, densify_pts=21)
                    window = from_bounds(*box, src.transform).round_offsets().round_lengths()
                    array = src.read(1, window=window)
                    profile = src.profile.copy()
                    profile.update(
                        height=array.shape[0],
                        width=array.shape[1],
                        transform=src.window_transform(window),
                        compress="deflate",
                    )
                    temporary = path.with_suffix(".partial.tif")
                    with rasterio.open(temporary, "w", **profile) as dst:
                        dst.write(array, 1)
                    temporary.replace(path)
            assets.append(
                {
                    "band": band,
                    "source_url": item.assets[band].href,
                    "file": str(path),
                    "sha256": sha256(path),
                }
            )
        scenes.append({"id": item.id, "datetime": item.datetime.isoformat(), "assets": assets})
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "stac_url": STAC_URL,
        "study_area": cfg["study_area"],
        "city_sha256": sha256(raw / "city.geojson"),
        "boundaries": sources,
        "scenes": scenes,
    }
    (raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {raw / 'MANIFEST.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    download(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()

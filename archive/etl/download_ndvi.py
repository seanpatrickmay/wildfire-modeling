#!/usr/bin/env python3
"""
Download pre-fire MODIS NDVI for each fire, resampled to GOES grid.

Uses MODIS/061/MOD13A1 (500m, 16-day Terra Vegetation Indices) via
Google Earth Engine.  For each fire we:

  1. Read the GOES JSON to get grid shape, CRS, and geo_transform.
  2. Compute a 30-day pre-fire NDVI median composite (30 days before
     the first GOES timestep) — capturing vegetation state *before*
     the fire started.
  3. Reproject the NDVI composite to the fire's GOES grid
     (EPSG:3857, ~2 km pixels).
  4. Save a JSON file: {fire_dir}/{fire}_ndvi.json  with keys:
       - "data": 2-D array [H, W] of NDVI values (scaled 0–1)
       - "metadata": grid_shape, geo_transform, crs, composite window

Usage:
  python scripts/download_ndvi.py            # all fires
  python scripts/download_ndvi.py Kincade    # single fire
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np

ee.Initialize()


def find_repo_root(start: Path) -> Path:
    for path in [start] + list(start.parents):
        if (path / "data").exists() and (path / "scripts").exists():
            return path
    raise FileNotFoundError("Could not find repo root.")


def goes_bounds_from_meta(meta: dict) -> tuple[list[float], list[int], list[float]]:
    """Return (bounds [xmin,ymin,xmax,ymax], shape [H,W], geo_transform)."""
    gt = meta["geo_transform"]
    shape = meta["grid_shape"]
    x_min = gt[2]
    y_max = gt[5]
    x_max = x_min + gt[0] * shape[1]
    y_min = y_max + gt[4] * shape[0]
    return [x_min, y_min, x_max, y_max], shape, gt


def parse_fire_start(goes_meta: dict) -> datetime:
    """Get the first timestep as a datetime."""
    ts = goes_meta.get("time_steps", [])
    if not ts:
        raise ValueError("No time_steps in GOES metadata.")
    first = str(ts[0])
    if first.endswith("Z"):
        first = first[:-1] + "+00:00"
    return datetime.fromisoformat(first)


def download_ndvi_for_fire(fire_dir: Path, fire_name: str) -> Path | None:
    """Download pre-fire NDVI composite for a single fire."""
    goes_files = sorted(fire_dir.glob("*GOES*json"))
    if not goes_files:
        print(f"  SKIP {fire_name}: no GOES JSON found")
        return None

    out_path = fire_dir / f"{fire_name}_ndvi.json"
    if out_path.exists():
        print(f"  SKIP {fire_name}: {out_path.name} already exists")
        return out_path

    with goes_files[0].open() as f:
        goes_json = json.load(f)
    goes_meta = goes_json["metadata"]

    bounds, shape, gt = goes_bounds_from_meta(goes_meta)
    crs = goes_meta.get("crs", "EPSG:3857")
    fire_start = parse_fire_start(goes_meta)

    # 30-day window before fire start for pre-fire NDVI
    window_end = fire_start
    window_start = fire_start - timedelta(days=30)

    print(f"  {fire_name}: shape={shape}, window={window_start.date()} to {window_end.date()}")

    # MODIS Terra 16-day NDVI at 500m
    modis = (
        ee.ImageCollection("MODIS/061/MOD13A1")
        .filterDate(window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"))
        .select("NDVI")
    )

    count = modis.size().getInfo()
    if count == 0:
        # Widen window to 60 days if no composites in 30-day window
        window_start = fire_start - timedelta(days=60)
        modis = (
            ee.ImageCollection("MODIS/061/MOD13A1")
            .filterDate(window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"))
            .select("NDVI")
        )
        count = modis.size().getInfo()
        if count == 0:
            print(f"  SKIP {fire_name}: no MODIS composites found")
            return None
        print(f"    widened window to 60 days, found {count} composites")

    # Median composite, scale from raw int to 0-1 (MODIS NDVI scale = 0.0001)
    composite = modis.median().multiply(0.0001).clamp(-1.0, 1.0)

    # Define the GOES grid as the target
    goes_proj = ee.Projection(crs)
    goes_region = ee.Geometry.Rectangle(bounds, proj=goes_proj, geodesic=False)

    # Sample at GOES resolution
    pixel_size = abs(gt[0])
    ndvi_reprojected = composite.reproject(crs=goes_proj, scale=pixel_size)

    # Get as numpy array via getRegion or computePixels
    try:
        result = ndvi_reprojected.sampleRectangle(
            region=goes_region,
            defaultValue=0.0,
        )
        ndvi_array = np.array(result.get("NDVI").getInfo(), dtype=np.float32)
    except ee.EEException as exc:
        # sampleRectangle fails for large grids; fall back to getDownloadURL
        print(f"    sampleRectangle failed ({exc}), using getDownloadURL...")
        ndvi_array = _download_via_url(ndvi_reprojected, goes_region, crs, pixel_size, shape)

    # Ensure shape matches GOES grid (may differ by ±1 pixel due to rounding)
    if ndvi_array.shape != tuple(shape):
        target_h, target_w = shape
        h, w = ndvi_array.shape
        # Crop or pad to match
        arr = np.zeros((target_h, target_w), dtype=np.float32)
        copy_h = min(h, target_h)
        copy_w = min(w, target_w)
        arr[:copy_h, :copy_w] = ndvi_array[:copy_h, :copy_w]
        ndvi_array = arr
        print(f"    resized from ({h},{w}) to ({target_h},{target_w})")

    # Replace NaN with 0
    ndvi_array = np.nan_to_num(ndvi_array, nan=0.0).astype(np.float32)

    # Save
    out_data = {
        "data": ndvi_array.tolist(),
        "metadata": {
            "grid_shape": shape,
            "geo_transform": gt,
            "crs": crs,
            "source": "MODIS/061/MOD13A1",
            "band": "NDVI",
            "composite_method": "median",
            "window_start": window_start.strftime("%Y-%m-%d"),
            "window_end": window_end.strftime("%Y-%m-%d"),
            "modis_composites_used": count,
        },
    }
    with out_path.open("w") as f:
        json.dump(out_data, f)

    stats = ndvi_array[ndvi_array != 0]
    if stats.size > 0:
        print(f"    saved: mean NDVI={stats.mean():.3f}, min={stats.min():.3f}, max={stats.max():.3f}")
    else:
        print(f"    saved (all zero)")

    return out_path


def _download_via_url(image, region, crs, scale, shape):
    """Fallback: download NDVI via GeoTIFF URL and read with rasterio."""
    import rasterio
    import tempfile
    import urllib.request

    url = image.getDownloadURL({
        "region": region,
        "crs": crs,
        "scale": scale,
        "format": "GEO_TIFF",
    })

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    urllib.request.urlretrieve(url, tmp_path)

    with rasterio.open(tmp_path) as ds:
        arr = ds.read(1).astype(np.float32)

    Path(tmp_path).unlink(missing_ok=True)
    return arr


def main():
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    base = repo_root / "data" / "multi_fire"

    # Optional: filter to specific fires from command line
    requested = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    fire_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    success = 0
    skipped = 0

    for fire_dir in fire_dirs:
        fire_name = fire_dir.name
        if requested and fire_name not in requested:
            continue
        result = download_ndvi_for_fire(fire_dir, fire_name)
        if result:
            success += 1
        else:
            skipped += 1

    print(f"\nDone: {success} downloaded, {skipped} skipped")


if __name__ == "__main__":
    main()

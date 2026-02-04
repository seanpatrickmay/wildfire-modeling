#!/usr/bin/env python3
"""
Download RTMA hourly grids from Earth Engine, aligned to a GOES GeoTIFF.

Example:
  python3 scripts/ee_download_rtma.py \
    --goes-tif data/ee_exports/August_Complex_2020_GOESEast_MaxConf_part01.tif \
    --start 2020-08-16T21:00:00Z --end 2020-10-18T21:00:00Z \
    --output-dir data/rtma/august --chunk-hours 168
"""

import argparse
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta
import math

import rasterio

WEB_MERCATOR_R = 6378137.0


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = (x / WEB_MERCATOR_R) * 180.0 / math.pi
    lat = (2 * (math.atan(math.exp(y / WEB_MERCATOR_R))) - (math.pi / 2)) * 180.0 / math.pi
    return lon, lat


def parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def chunks_time(start_dt: datetime, end_dt: datetime, chunk_hours: int):
    cur = start_dt
    while cur <= end_dt:
        cur_end = min(end_dt, cur + timedelta(hours=chunk_hours - 1))
        yield cur, cur_end
        cur = cur_end + timedelta(hours=1)


def download_zip(url: str, zip_path: str) -> None:
    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as f:
        shutil.copyfileobj(response, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RTMA to GeoTIFF chunks")
    parser.add_argument("--goes-tif", required=True, help="Template GOES GeoTIFF")
    parser.add_argument("--start", required=True, help="ISO start time (UTC)")
    parser.add_argument("--end", required=True, help="ISO end time (UTC)")
    parser.add_argument(
        "--variables",
        default="TMP,WIND,WDIR,SPFH,ACPC01",
        help="Comma-separated RTMA bands",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-hours", type=int, default=168)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    try:
        import ee  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing earthengine-api. Install with: pip install earthengine-api") from exc

    if args.project:
        ee.Initialize(project=args.project)
    else:
        ee.Initialize()

    with rasterio.open(args.goes_tif) as ds:
        transform = ds.transform
        crs = ds.crs
        bounds = ds.bounds

    if not crs:
        raise SystemExit("GOES GeoTIFF missing CRS.")

    variables = [v.strip() for v in args.variables.split(",") if v.strip()]
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    start_dt = parse_time(args.start)
    end_dt = parse_time(args.end)

    manifest = {
        "dataset": "NOAA/NWS/RTMA",
        "variables": variables,
        "start": args.start,
        "end": args.end,
        "chunk_hours": args.chunk_hours,
        "files": {v: [] for v in variables},
        "time_steps": [],
        "grid": {
            "crs": crs.to_string(),
            "geo_transform": [
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            ],
            "bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        },
    }

    rtma = ee.ImageCollection("NOAA/NWS/RTMA")
    # Build WGS84 region polygon from GOES bounds (EPSG:3857 -> EPSG:4326)
    minlon, minlat = mercator_to_lonlat(bounds.left, bounds.bottom)
    maxlon, maxlat = mercator_to_lonlat(bounds.right, bounds.top)
    region_wgs84 = [
        [
            [minlon, minlat],
            [maxlon, minlat],
            [maxlon, maxlat],
            [minlon, maxlat],
            [minlon, minlat],
        ]
    ]
    ee_region = ee.Geometry.Polygon(region_wgs84, proj="EPSG:4326", geodesic=False)

    for chunk_idx, (chunk_start, chunk_end) in enumerate(
        chunks_time(start_dt, end_dt, args.chunk_hours), start=1
    ):
        # EE filterDate end is exclusive; add 1 hour to include last hour
        ee_start = chunk_start.isoformat()
        ee_end = (chunk_end + timedelta(hours=1)).isoformat()
        col = rtma.filterDate(ee_start, ee_end)

        # Build time steps list for this chunk
        cur = chunk_start
        chunk_steps = []
        while cur <= chunk_end:
            chunk_steps.append(cur.strftime("%Y-%m-%dT%H:00:00Z"))
            cur += timedelta(hours=1)
        if chunk_idx == 1:
            manifest["time_steps"] = []
        manifest["time_steps"].extend(chunk_steps)

        params = {
            "crs": "EPSG:3857",
            "scale": 2500,
            "region": region_wgs84,
            "filePerBand": False,
        }

        for var in variables:
            img = col.select(var).toBands().clip(ee_region)
            filename = f"rtma_{var}_part{chunk_idx:02d}.tif"
            out_path = os.path.join(output_dir, filename)
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "download.zip")
                try:
                    url = img.getDownloadURL(params)
                    download_zip(url, zip_path)
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    raise SystemExit(f"EE download error: {body}")

                with zipfile.ZipFile(zip_path, "r") as zf:
                    tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
                    if not tif_names:
                        raise SystemExit("No .tif in download")
                    zf.extract(tif_names[0], tmpdir)
                    shutil.move(os.path.join(tmpdir, tif_names[0]), out_path)

            manifest["files"][var].append(out_path)
            print(f"Saved: {out_path}")

    manifest_path = os.path.join(output_dir, "rtma_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

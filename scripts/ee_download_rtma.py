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
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta
import math

import rasterio

WEB_MERCATOR_R = 6378137.0
RETRIABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    def __init__(self, message: str, *, retriable: bool) -> None:
        super().__init__(message)
        self.retriable = retriable


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


def is_retriable_error(code: int | None, body: str) -> bool:
    if code in RETRIABLE_HTTP_CODES:
        return True
    text = body.lower()
    return (
        "memory capacity exceeded" in text
        or '"status": "unavailable"' in text
        or '"status": "internal"' in text
        or "internal error has occurred" in text
    )


def download_image_zip_with_retries(
    img,
    params: dict,
    zip_path: str,
    *,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> None:
    backoff = retry_backoff_seconds
    for attempt in range(1, max_attempts + 1):
        try:
            url = img.getDownloadURL(params)
            download_zip(url, zip_path)
            return
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retriable = is_retriable_error(exc.code, body)
            if retriable and attempt < max_attempts:
                print(
                    f"EE download retry {attempt}/{max_attempts - 1} "
                    f"(HTTP {exc.code}); waiting {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2.0
                continue
            raise DownloadError(
                f"EE download error (HTTP {exc.code}): {body}",
                retriable=retriable,
            ) from exc
        except urllib.error.URLError as exc:
            retriable = attempt < max_attempts
            if retriable:
                print(
                    f"EE download retry {attempt}/{max_attempts - 1} "
                    f"(URL error: {exc.reason}); waiting {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2.0
                continue
            raise DownloadError(f"EE URL error: {exc}", retriable=True) from exc


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
    parser.add_argument(
        "--max-download-attempts",
        type=int,
        default=5,
        help="Retries per EE download request before giving up (default: 5).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Initial retry backoff for EE request retries (default: 2.0).",
    )
    parser.add_argument(
        "--min-window-hours",
        type=int,
        default=1,
        help="Smallest adaptive RTMA window size in hours (default: 1).",
    )
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

    windows = list(chunks_time(start_dt, end_dt, args.chunk_hours))
    part_idx = 1
    manifest["time_steps"] = []

    while windows:
        chunk_start, chunk_end = windows.pop(0)
        chunk_hours = int((chunk_end - chunk_start).total_seconds() // 3600) + 1

        ee_start = chunk_start.isoformat()
        ee_end = (chunk_end + timedelta(hours=1)).isoformat()
        col = rtma.filterDate(ee_start, ee_end)

        chunk_steps = []
        cur = chunk_start
        while cur <= chunk_end:
            chunk_steps.append(cur.strftime("%Y-%m-%dT%H:00:00Z"))
            cur += timedelta(hours=1)

        params = {
            "crs": "EPSG:3857",
            "scale": 2500,
            "region": region_wgs84,
            "filePerBand": False,
        }

        print(
            f"Downloading RTMA window part{part_idx:03d}: "
            f"{chunk_steps[0]} .. {chunk_steps[-1]} ({chunk_hours}h)"
        )
        try:
            tmp_outputs: dict[str, str] = {}
            with tempfile.TemporaryDirectory() as tmpdir:
                for var in variables:
                    img = col.select(var).toBands().clip(ee_region)
                    zip_path = os.path.join(tmpdir, f"{var}.zip")
                    download_image_zip_with_retries(
                        img,
                        params,
                        zip_path,
                        max_attempts=args.max_download_attempts,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                    )

                    with zipfile.ZipFile(zip_path, "r") as zf:
                        tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
                        if not tif_names:
                            raise DownloadError("No .tif in RTMA download archive.", retriable=False)
                        zf.extract(tif_names[0], tmpdir)
                        extracted_path = os.path.join(tmpdir, tif_names[0])
                        unique_var_path = os.path.join(tmpdir, f"{var}.tif")
                        shutil.move(extracted_path, unique_var_path)
                        tmp_outputs[var] = unique_var_path

                for var in variables:
                    filename = f"rtma_{var}_part{part_idx:03d}.tif"
                    out_path = os.path.join(output_dir, filename)
                    shutil.move(tmp_outputs[var], out_path)
                    manifest["files"][var].append(out_path)
                    print(f"Saved: {out_path}")

            manifest["time_steps"].extend(chunk_steps)
            part_idx += 1
        except DownloadError as exc:
            can_split = chunk_hours > max(1, args.min_window_hours)
            if exc.retriable and can_split:
                left_hours = chunk_hours // 2
                right_hours = chunk_hours - left_hours
                left_end = chunk_start + timedelta(hours=left_hours - 1)
                right_start = left_end + timedelta(hours=1)
                right_end = right_start + timedelta(hours=right_hours - 1)
                print(
                    "Window failed with retriable error; splitting into "
                    f"{left_hours}h and {right_hours}h windows."
                )
                windows = [(chunk_start, left_end), (right_start, right_end)] + windows
                continue
            raise SystemExit(str(exc))

    manifest_path = os.path.join(output_dir, "rtma_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

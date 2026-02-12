#!/usr/bin/env python3
"""
Download FIRMS VIIRS data for a bounding box and date range, then grid into JSON.

Example (Kincade AOI, GOES date range, VIIRS S-NPP SP):
  FIRMS_MAP_KEY=YOUR_KEY python3 scripts/firms_viirs_to_json.py \
    --fire-name "Kincade" --year 2019 \
    --bbox -122.96 38.50 -122.59 38.87 \
    --start-date 2019-10-24 --day-range 7 \
    --source VIIRS_SNPP_SP \
    --output data/viirs/Kincade_2019_VIIRS_SNPP_SP.json
"""

import argparse
import csv
import json
import math
import os
import sys
import urllib.request

WEB_MERCATOR_R = 6378137.0
ENV_KEYS = ("FIRMS_MAP_KEY", "MAP_KEY", "map-key", "map_key")


def load_dotenv(path: str = ".env") -> dict:
    if not os.path.exists(path):
        return {}
    env = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")  # remove simple quotes
            env[key] = value
    return env


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    x = WEB_MERCATOR_R * math.radians(lon)
    # clamp latitude to avoid infinity
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = WEB_MERCATOR_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def parse_acq_datetime(acq_date: str, acq_time: str) -> str:
    # acq_time can be HHMM or HMM; zero-pad to 4
    acq_time = acq_time.zfill(4)
    hour = acq_time[:2]
    minute = acq_time[2:]
    return f"{acq_date}T{hour}:{minute}:00Z"


def hour_key(acq_date: str, acq_time: str) -> str:
    acq_time = acq_time.zfill(4)
    hour = acq_time[:2]
    return f"{acq_date}T{hour}:00:00Z"


def build_time_steps(start_date: str, day_range: int) -> list[str]:
    # start_date: YYYY-MM-DD
    y, m, d = map(int, start_date.split("-"))
    # naive UTC days
    from datetime import datetime, timedelta

    start = datetime(y, m, d)
    steps = []
    for day in range(day_range):
        for hour in range(24):
            steps.append((start + timedelta(days=day, hours=hour)).isoformat() + "Z")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FIRMS VIIRS data and grid into GOFER-style JSON"
    )
    parser.add_argument("--fire-name", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        type=float,
        required=True,
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--day-range", type=int, required=True, help="1-10")
    parser.add_argument(
        "--source",
        default="VIIRS_SNPP_SP",
        help="FIRMS source (e.g., VIIRS_SNPP_SP, VIIRS_NOAA20_SP)",
    )
    parser.add_argument(
        "--value",
        default="frp",
        choices=["frp", "bright_ti4", "bright_ti5", "confidence", "count"],
        help="Value to grid",
    )
    parser.add_argument(
        "--agg",
        default="max",
        choices=["max", "sum", "mean"],
        help="Aggregation per cell per hour",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=375.0,
        help="Grid resolution in meters (default 375m)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save the raw FIRMS CSV response.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print basic diagnostics (row count, header).",
    )
    parser.add_argument(
        "--map-key",
        default=None,
        help="FIRMS MAP_KEY. If omitted, reads FIRMS_MAP_KEY env var.",
    )
    args = parser.parse_args()

    map_key = args.map_key
    if not map_key:
        for key in ENV_KEYS:
            map_key = os.environ.get(key)
            if map_key:
                break
    if not map_key:
        dotenv = load_dotenv()
        for key in ENV_KEYS:
            map_key = dotenv.get(key)
            if map_key:
                break
    if not map_key:
        print(
            "Missing FIRMS MAP_KEY. Set FIRMS_MAP_KEY (or MAP_KEY/map-key) or pass --map-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    minlon, minlat, maxlon, maxlat = args.bbox
    bbox_str = f"{minlon},{minlat},{maxlon},{maxlat}"

    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{map_key}/{args.source}/{bbox_str}/{args.day_range}/{args.start_date}"
    )

    try:
        with urllib.request.urlopen(url) as response:
            csv_text = response.read().decode("utf-8")
    except Exception as exc:
        print(f"Failed to download FIRMS data: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        with open(args.save_csv, "w", encoding="utf-8") as f:
            f.write(csv_text)

    reader = csv.DictReader(csv_text.splitlines())
    if args.debug:
        print(f"CSV header: {reader.fieldnames}", file=sys.stderr)
    rows = list(reader)
    if args.debug:
        print(f"Rows: {len(rows)}", file=sys.stderr)

    if not reader.fieldnames or "latitude" not in reader.fieldnames:
        preview = csv_text[:300].replace("\n", "\\n")
        print(
            "Unexpected response from FIRMS (missing latitude field). "
            f"Preview: {preview}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build grid in EPSG:3857
    xmin, ymin = lonlat_to_mercator(minlon, minlat)
    xmax, ymax = lonlat_to_mercator(maxlon, maxlat)

    resolution = args.resolution_m
    width = max(1, math.ceil((xmax - xmin) / resolution))
    height = max(1, math.ceil((ymax - ymin) / resolution))

    # Initialize data structure: hour -> 2D list
    time_steps = build_time_steps(args.start_date, args.day_range)
    time_index = {t: i for i, t in enumerate(time_steps)}
    data = [
        [[0.0 for _ in range(width)] for _ in range(height)]
        for _ in range(len(time_steps))
    ]
    counts = None
    if args.agg == "mean":
        counts = [
            [[0 for _ in range(width)] for _ in range(height)]
            for _ in range(len(time_steps))
        ]

    for row in rows:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except Exception:
            continue

        tkey = hour_key(row["acq_date"], row["acq_time"])
        if tkey not in time_index:
            continue
        t_idx = time_index[tkey]

        x, y = lonlat_to_mercator(lon, lat)
        col = int((x - xmin) // resolution)
        row_idx = int((ymax - y) // resolution)
        if col < 0 or col >= width or row_idx < 0 or row_idx >= height:
            continue

        if args.value == "count":
            val = 1.0
        else:
            try:
                val = float(row[args.value])
            except Exception:
                continue

        if args.agg == "max":
            if val > data[t_idx][row_idx][col]:
                data[t_idx][row_idx][col] = val
        elif args.agg == "sum":
            data[t_idx][row_idx][col] += val
        elif args.agg == "mean":
            data[t_idx][row_idx][col] += val
            counts[t_idx][row_idx][col] += 1

    if args.agg == "mean":
        for t_idx in range(len(time_steps)):
            for r in range(height):
                for c in range(width):
                    count = counts[t_idx][r][c]
                    if count > 0:
                        data[t_idx][r][c] /= count

    metadata = {
        "fire_name": args.fire_name,
        "year": args.year,
        "source": args.source,
        "crs": "EPSG:3857",
        "geo_transform": [
            resolution,
            0.0,
            xmin,
            0.0,
            -resolution,
            ymax,
        ],
        "bbox": [minlon, minlat, maxlon, maxlat],
        "grid_shape": [height, width],
        "grid_origin": {"x": xmin, "y": ymax, "origin": "upper_left_corner"},
        "pixel_size_m": [resolution, resolution],
        "temporal_resolution": "1h",
        "time_steps": time_steps,
        "band_names": [f"h_{i:05d}" for i in range(1, len(time_steps) + 1)],
        "units": args.value,
        "nodata_value": 0.0,
        "nan_strategy": "none",
        "value_aggregation": args.agg,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": metadata, "data": data},
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

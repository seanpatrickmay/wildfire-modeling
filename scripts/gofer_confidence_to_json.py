#!/usr/bin/env python3
"""
Convert a GOFER confidence GeoTIFF stack (multi-band, hourly) into JSON.

Example:
  python scripts/gofer_confidence_to_json.py \
    --input /path/to/Kincade_2019_GOESEast_MaxConf.tif \
    --output /path/to/Kincade_2019_GOESEast_MaxConf.json \
    --fire-name "Kincade" --year 2019 --source "GOES-East" \
    --start-time "2019-10-24T04:00:00Z"
"""

import argparse
import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import rasterio


def parse_iso8601(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def build_time_steps(count: int, start_time: Optional[datetime], step_hours: float) -> list:
    if start_time is None:
        # 1-based hour offsets to match GOFER timeStep convention
        return list(range(1, count + 1))
    return [
        (start_time + timedelta(hours=step_hours * i)).isoformat()
        for i in range(count)
    ]


def pixel_size_meters(crs, transform):
    if crs is None:
        return None

    # rasterio CRS exposes linear_units and linear_units_factor for projected CRSs
    if crs.is_projected:
        factor = getattr(crs, "linear_units_factor", None)
        if factor is None:
            factor = 1.0
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            factor = 1.0
        return [abs(transform.a) * factor, abs(transform.e) * factor]

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GOFER confidence stack to JSON with metadata and matrices"
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Path(s) to multi-band GeoTIFF(s). Provide in chronological order.",
    )
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--fire-name", required=True, help="Fire name")
    parser.add_argument("--year", required=True, type=int, help="Fire year")
    parser.add_argument(
        "--source",
        required=True,
        help='Source name (e.g., "GOES-East", "GOES-West", "Combined")',
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help=(
            "ISO8601 start time (e.g., 2019-10-24T04:00:00Z). "
            "If omitted, uses hour offsets."
        ),
    )
    parser.add_argument(
        "--time-step-hours",
        type=float,
        default=1.0,
        help="Temporal resolution in hours (default: 1.0)",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="Override nodata value if missing in the raster",
    )
    parser.add_argument(
        "--nan-strategy",
        choices=["null", "allow", "error"],
        default="null",
        help=(
            "How to handle NaN values: 'null' replaces with JSON null (default), "
            "'allow' writes NaN (non-standard JSON), 'error' raises."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation (easier to view in vim).",
    )
    args = parser.parse_args()

    start_time = parse_iso8601(args.start_time) if args.start_time else None

    data = []
    band_names = []
    nodata_value = args.nodata
    height = width = None
    transform = None
    crs = None
    bounds = None
    nan_found = False

    for input_path in args.input:
        with rasterio.open(input_path) as ds:
            if height is None:
                height = ds.height
                width = ds.width
                transform = ds.transform
                crs = ds.crs
                bounds = ds.bounds
                if ds.nodata is not None:
                    nodata_value = ds.nodata
            else:
                if ds.height != height or ds.width != width:
                    raise ValueError("Input rasters must have identical dimensions.")
                if ds.transform != transform:
                    raise ValueError("Input rasters must have identical transforms.")
                if (ds.crs and crs) and ds.crs != crs:
                    raise ValueError("Input rasters must have identical CRS.")

            if ds.descriptions and any(ds.descriptions):
                band_names.extend(
                    [
                        (name if name not in (None, "") else f"band_{i + 1}")
                        for i, name in enumerate(ds.descriptions)
                    ]
                )
            else:
                band_names.extend([f"band_{i + 1}" for i in range(ds.count)])

            for band_index in range(1, ds.count + 1):
                arr = ds.read(band_index).astype("float32")
                mask = np.isfinite(arr)
                if not mask.all():
                    nan_found = True
                    if args.nan_strategy == "error":
                        raise ValueError(
                            "NaN/Inf values found in data. Use --nan-strategy null or allow."
                        )
                    if args.nan_strategy == "null":
                        arr_obj = arr.astype(object)
                        arr_obj[~mask] = None
                        arr = arr_obj
                data.append(arr.tolist())

    total_bands = len(data)
    time_steps = build_time_steps(total_bands, start_time, args.time_step_hours)

    metadata = {
        "fire_name": args.fire_name,
        "year": args.year,
        "source": args.source,
        "crs": crs.to_string() if crs else None,
        "geo_transform": [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        "bbox": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "grid_shape": [height, width],
        "grid_origin": {
            "x": transform.c,
            "y": transform.f,
            "origin": "upper_left_corner",
        },
        "pixel_size_m": pixel_size_meters(crs, transform),
        "temporal_resolution": f"{args.time_step_hours}h",
        "time_steps": time_steps,
        "band_names": band_names,
        "units": "confidence",
        "nodata_value": nodata_value,
        "nan_strategy": args.nan_strategy,
    }

    metadata["nan_found"] = nan_found

    output = {"metadata": metadata, "data": data}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=True,
            allow_nan=(args.nan_strategy == "allow"),
            indent=2 if args.pretty else None,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert RTMA GeoTIFF chunks to a single JSON with normalization.

Example:
  python3 scripts/rtma_to_json.py \
    --manifest data/rtma/august/rtma_manifest.json \
    --output data/rtma/august/rtma_normalized.json \
    --normalize zscore
"""

import argparse
import json
import math
import os

import numpy as np
import rasterio


def compute_stats(values: np.ndarray):
    flat = values.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    mean = float(flat.mean())
    std = float(flat.std())
    min_v = float(flat.min())
    max_v = float(flat.max())
    return {"mean": mean, "std": std, "min": min_v, "max": max_v, "count": int(flat.size)}


def normalize(values: np.ndarray, stats: dict, method: str) -> np.ndarray:
    if stats is None:
        return values
    if method == "zscore":
        std = stats["std"] or 0.0
        if std == 0.0:
            return np.zeros_like(values)
        return (values - stats["mean"]) / std
    if method == "minmax":
        span = stats["max"] - stats["min"]
        if span == 0.0:
            return np.zeros_like(values)
        return (values - stats["min"]) / span
    if method == "none":
        return values
    raise ValueError(f"Unknown method: {method}")


def read_stack(paths: list[str]):
    data = []
    height = width = None
    transform = None
    crs = None
    bounds = None
    for path in paths:
        with rasterio.open(path) as ds:
            if height is None:
                height, width = ds.height, ds.width
                transform = ds.transform
                crs = ds.crs
                bounds = ds.bounds
            for b in range(1, ds.count + 1):
                arr = ds.read(b).astype("float32")
                data.append(arr)
    return np.stack(data), height, width, transform, crs, bounds


def main() -> None:
    parser = argparse.ArgumentParser(description="RTMA GeoTIFF to JSON")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--normalize",
        default="zscore",
        choices=["zscore", "minmax", "none"],
    )
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    variables = manifest.get("variables", [])
    files = manifest.get("files", {})
    time_steps = manifest.get("time_steps", [])

    if not variables or not files:
        raise SystemExit("Manifest missing variables/files.")

    data_out = {}
    stats_out = {}
    grid_meta = None

    for var in variables:
        paths = files.get(var, [])
        if not paths:
            continue
        stack, height, width, transform, crs, bounds = read_stack(paths)
        if grid_meta is None:
            grid_meta = {
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
                "grid_origin": {"x": transform.c, "y": transform.f, "origin": "upper_left_corner"},
            }
        stats = compute_stats(stack)
        stats_out[var] = stats
        stack = normalize(stack, stats, args.normalize)
        # Convert to list of matrices
        data_out[var] = [frame.tolist() for frame in stack]

    output = {
        "metadata": {
            **(grid_meta or {}),
            "dataset": manifest.get("dataset"),
            "variables": variables,
            "time_steps": time_steps,
            "temporal_resolution": "1h",
            "normalization": args.normalize,
        },
        "stats": stats_out,
        "data": data_out,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

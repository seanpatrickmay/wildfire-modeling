#!/usr/bin/env python3
"""
Normalize RAWS hourly time series per station.

Default: z-score per station per variable over the full time window.

Example:
  python3 scripts/raws_normalize.py \
    --input data/raws/august/raws_timeseries_hourly.json \
    --output data/raws/august/raws_timeseries_hourly_normalized.json \
    --method zscore
"""

import argparse
import json
import math
import os
import sys


def compute_stats(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    std = math.sqrt(var)
    return {
        "count": len(clean),
        "mean": mean,
        "std": std,
        "min": min(clean),
        "max": max(clean),
    }


def normalize(values, stats, method):
    if stats is None:
        return [None for _ in values]
    if method == "zscore":
        std = stats["std"] or 0.0
        if std == 0.0:
            return [0.0 if v is not None else None for v in values]
        return [None if v is None else (v - stats["mean"]) / std for v in values]
    if method == "minmax":
        span = stats["max"] - stats["min"]
        if span == 0.0:
            return [0.0 if v is not None else None for v in values]
        return [None if v is None else (v - stats["min"]) / span for v in values]
    if method == "robust":
        # median and MAD
        clean = [v for v in values if v is not None]
        if not clean:
            return [None for _ in values]
        clean_sorted = sorted(clean)
        mid = len(clean_sorted) // 2
        if len(clean_sorted) % 2 == 0:
            median = (clean_sorted[mid - 1] + clean_sorted[mid]) / 2
        else:
            median = clean_sorted[mid]
        mad = sorted([abs(v - median) for v in clean_sorted])[mid]
        if mad == 0.0:
            return [0.0 if v is not None else None for v in values]
        return [None if v is None else (v - median) / (1.4826 * mad) for v in values]
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize RAWS hourly time series")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        default="zscore",
        choices=["zscore", "minmax", "robust"],
        help="Normalization method per station per variable.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    stations = data.get("data", {})
    if not stations:
        raise SystemExit("No station data found in input.")

    out = {
        "metadata": data.get("metadata", {}),
        "normalization": {"method": args.method},
        "data": {},
        "stats": {},
    }

    for stid, entry in stations.items():
        obs = entry.get("observations", {})
        out_obs = {}
        out_stats = {}
        for var, values in obs.items():
            stats = compute_stats(values)
            out_stats[var] = stats
            out_obs[var] = normalize(values, stats, args.method)
        out["data"][stid] = {
            "latitude": entry.get("latitude"),
            "longitude": entry.get("longitude"),
            "elevation": entry.get("elevation"),
            "observations": out_obs,
        }
        out["stats"][stid] = out_stats

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

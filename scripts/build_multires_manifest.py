#!/usr/bin/env python3
"""
Build a versioned multi-resolution covariate manifest (v2) from a config JSON.

Example:
  python3 scripts/build_multires_manifest.py \
    --config docs/examples/multires_config.example.json \
    --output data/multi_fire/North_Complex/multires_manifest_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any


TOP_LEVEL_KEYS = ("study_area", "time_window", "target_grid", "sources")
REQUIRED_SOURCE_KEYS = (
    "id",
    "provider",
    "dataset",
    "source_type",
    "native_resolution_m",
    "native_temporal_resolution",
    "variables",
)
ALLOWED_SOURCE_TYPES = {"dynamic", "static"}


def now_utc_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(cfg: dict[str, Any]) -> None:
    for key in TOP_LEVEL_KEYS:
        _require(key in cfg, f"Missing top-level key: {key}")

    grid = cfg["target_grid"]
    _require(isinstance(grid, dict), "target_grid must be an object")
    _require(float(grid.get("resolution_m", 0)) > 0, "target_grid.resolution_m must be > 0")

    sources = cfg["sources"]
    _require(isinstance(sources, list) and sources, "sources must be a non-empty list")

    seen_ids: set[str] = set()
    for idx, src in enumerate(sources):
        _require(isinstance(src, dict), f"sources[{idx}] must be an object")
        for key in REQUIRED_SOURCE_KEYS:
            _require(key in src, f"sources[{idx}] missing key: {key}")

        sid = str(src["id"])
        _require(sid not in seen_ids, f"Duplicate source id: {sid}")
        seen_ids.add(sid)

        stype = str(src["source_type"]).lower()
        _require(
            stype in ALLOWED_SOURCE_TYPES,
            f"sources[{idx}].source_type must be one of {sorted(ALLOWED_SOURCE_TYPES)}",
        )

        _require(
            float(src["native_resolution_m"]) > 0,
            f"sources[{idx}].native_resolution_m must be > 0",
        )

        variables = src["variables"]
        _require(
            isinstance(variables, list) and variables,
            f"sources[{idx}].variables must be a non-empty list",
        )

        for vidx, var in enumerate(variables):
            _require(isinstance(var, dict), f"sources[{idx}].variables[{vidx}] must be an object")
            _require("name" in var, f"sources[{idx}].variables[{vidx}] missing key: name")


def default_resampling_for_source(source: dict[str, Any], target_grid: dict[str, Any]) -> str:
    # Prefer explicit source setting.
    if source.get("resampling_method"):
        return str(source["resampling_method"])

    defaults = target_grid.get("resampling_defaults", {})
    variables = source.get("variables", [])
    kinds = {str(v.get("kind", "continuous")) for v in variables if isinstance(v, dict)}

    if kinds and kinds.issubset({"categorical"}):
        return str(defaults.get("categorical", "nearest"))

    return str(defaults.get("continuous", "bilinear"))


def normalize_source(source: dict[str, Any], target_grid: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "id": str(source["id"]),
        "provider": str(source["provider"]),
        "dataset": str(source["dataset"]),
        "source_type": str(source["source_type"]).lower(),
        "native_resolution_m": float(source["native_resolution_m"]),
        "native_temporal_resolution": str(source["native_temporal_resolution"]),
        "resampling_method": default_resampling_for_source(source, target_grid),
        "source_priority": int(source.get("source_priority", 100)),
        "variables": [],
    }

    optional_passthrough = ("latency", "coverage", "license", "access")
    for key in optional_passthrough:
        if key in source:
            normalized[key] = source[key]

    for var in source.get("variables", []):
        variable = {
            "name": str(var["name"]),
            "dtype": str(var.get("dtype", "float32")),
            "units": str(var.get("units", "unknown")),
            "kind": str(var.get("kind", "continuous")),
        }
        for key in ("null_value", "valid_range"):
            if key in var:
                variable[key] = var[key]
        normalized["variables"].append(variable)

    return normalized


def flatten_variables(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        sid = source["id"]
        for var in source["variables"]:
            rows.append(
                {
                    "source_id": sid,
                    "name": var["name"],
                    "dtype": var["dtype"],
                    "units": var["units"],
                    "kind": var["kind"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-resolution manifest v2")
    parser.add_argument("--config", required=True, help="Input config JSON")
    parser.add_argument("--output", required=True, help="Output manifest JSON")
    parser.add_argument(
        "--notes",
        default="",
        help="Optional provenance notes saved in manifest.provenance.notes",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    validate_config(cfg)

    normalized_sources = [
        normalize_source(source, cfg["target_grid"])
        for source in cfg["sources"]
    ]

    manifest = {
        "manifest_version": "2.0",
        "generated_at": now_utc_z(),
        "study_area": cfg["study_area"],
        "time_window": cfg["time_window"],
        "target_grid": cfg["target_grid"],
        "sources": normalized_sources,
        "variables": flatten_variables(normalized_sources),
        "provenance": {
            "config_path": os.path.abspath(args.config),
            "notes": args.notes,
        },
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {out_path}")
    print(f"Sources: {len(manifest['sources'])}, variables: {len(manifest['variables'])}")


if __name__ == "__main__":
    main()

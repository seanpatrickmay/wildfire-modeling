#!/usr/bin/env python3
"""
Validate a multi-resolution manifest (v2) for structural correctness.

Example:
  python3 scripts/validate_multires_manifest.py \
    --manifest data/multi_fire/North_Complex/multires_manifest_v2.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any


REQUIRED_TOP_LEVEL = (
    "manifest_version",
    "generated_at",
    "study_area",
    "time_window",
    "target_grid",
    "sources",
    "variables",
    "provenance",
)
REQUIRED_SOURCE = (
    "id",
    "provider",
    "dataset",
    "source_type",
    "native_resolution_m",
    "native_temporal_resolution",
    "resampling_method",
    "source_priority",
    "variables",
)
REQUIRED_VARIABLE = ("source_id", "name", "dtype", "units", "kind")
ALLOWED_SOURCE_TYPES = {"dynamic", "static"}
ALLOWED_KINDS = {"continuous", "categorical"}


class ValidationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_source(src: dict[str, Any], idx: int) -> None:
    for key in REQUIRED_SOURCE:
        require(key in src, f"sources[{idx}] missing key: {key}")

    require(
        str(src["source_type"]) in ALLOWED_SOURCE_TYPES,
        f"sources[{idx}].source_type must be one of {sorted(ALLOWED_SOURCE_TYPES)}",
    )
    require(float(src["native_resolution_m"]) > 0, f"sources[{idx}].native_resolution_m must be > 0")
    require(isinstance(src["variables"], list) and src["variables"], f"sources[{idx}].variables must be non-empty")



def validate_variable(var: dict[str, Any], idx: int) -> None:
    for key in REQUIRED_VARIABLE:
        require(key in var, f"variables[{idx}] missing key: {key}")
    require(str(var["kind"]) in ALLOWED_KINDS, f"variables[{idx}].kind must be one of {sorted(ALLOWED_KINDS)}")



def validate_manifest(manifest: dict[str, Any]) -> None:
    for key in REQUIRED_TOP_LEVEL:
        require(key in manifest, f"Missing top-level key: {key}")

    require(str(manifest["manifest_version"]) == "2.0", "manifest_version must be '2.0'")

    grid = manifest["target_grid"]
    require(isinstance(grid, dict), "target_grid must be an object")
    require(float(grid.get("resolution_m", 0)) > 0, "target_grid.resolution_m must be > 0")

    sources = manifest["sources"]
    variables = manifest["variables"]
    require(isinstance(sources, list) and sources, "sources must be a non-empty list")
    require(isinstance(variables, list), "variables must be a list")

    seen_source_ids: set[str] = set()
    for idx, src in enumerate(sources):
        require(isinstance(src, dict), f"sources[{idx}] must be an object")
        validate_source(src, idx)
        sid = str(src["id"])
        require(sid not in seen_source_ids, f"Duplicate source id: {sid}")
        seen_source_ids.add(sid)

    for idx, var in enumerate(variables):
        require(isinstance(var, dict), f"variables[{idx}] must be an object")
        validate_variable(var, idx)
        require(
            str(var["source_id"]) in seen_source_ids,
            f"variables[{idx}].source_id does not exist in sources",
        )



def summarize(manifest: dict[str, Any]) -> str:
    sources = manifest["sources"]
    variables = manifest["variables"]
    dynamic = sum(1 for s in sources if s.get("source_type") == "dynamic")
    static = sum(1 for s in sources if s.get("source_type") == "static")
    return (
        f"manifest_version={manifest['manifest_version']}\n"
        f"sources={len(sources)} (dynamic={dynamic}, static={static})\n"
        f"variables={len(variables)}\n"
        f"target_resolution_m={manifest['target_grid'].get('resolution_m')}"
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multi-resolution manifest v2")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    try:
        validate_manifest(manifest)
    except ValidationError as exc:
        raise SystemExit(f"Validation failed: {exc}")

    print("Validation passed")
    print(summarize(manifest))


if __name__ == "__main__":
    main()

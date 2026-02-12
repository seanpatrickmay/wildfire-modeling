#!/usr/bin/env python3
"""
Download and build the full multi-fire dataset for the 28 GOFER fires.

Pipeline per fire:
1) Download GOES confidence stack(s) from Earth Engine.
2) Convert GOES stack(s) to GOFER-style JSON.
3) Download aligned RTMA chunks + manifest.
4) Build normalized RTMA JSON.

Default behavior is resumable:
- Reuses existing valid outputs.
- Rebuilds missing/invalid downstream artifacts from local files when possible.
- Downloads only when required.

Examples:
  python3 scripts/download_all_28_fires.py
  python3 scripts/download_all_28_fires.py --fires "Creek,Dixie"
  python3 scripts/download_all_28_fires.py --dry-run
  python3 scripts/download_all_28_fires.py --overwrite --chunk-hours 24
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FireSpec:
    name: str
    year: int
    start_utc_hour: str  # "YYYY-MM-DD HH"


FIRE_CATALOG: list[FireSpec] = [
    FireSpec("Kincade", 2019, "2019-10-24 04"),
    FireSpec("Walker", 2019, "2019-09-04 07"),
    FireSpec("August Complex", 2020, "2020-08-16 21"),
    FireSpec("Bobcat", 2020, "2020-09-06 19"),
    FireSpec("CZU Lightning Complex", 2020, "2020-08-16 15"),
    FireSpec("Creek", 2020, "2020-09-05 01"),
    FireSpec("Dolan", 2020, "2020-08-18 18"),
    FireSpec("Glass", 2020, "2020-09-27 10"),
    FireSpec("July Complex", 2020, "2020-07-22 17"),
    FireSpec("LNU Lightning Complex", 2020, "2020-08-17 13"),
    FireSpec("North Complex", 2020, "2020-08-17 16"),
    FireSpec("Red Salmon Complex", 2020, "2020-07-27 18"),
    FireSpec("SCU Lightning Complex", 2020, "2020-08-16 11"),
    FireSpec("SQF Complex", 2020, "2020-08-19 14"),
    FireSpec("Slater and Devil", 2020, "2020-09-08 13"),
    FireSpec("W-5 Cold Springs", 2020, "2020-08-18 18"),
    FireSpec("Zogg", 2020, "2020-09-27 21"),
    FireSpec("Antelope", 2021, "2021-08-01 17"),
    FireSpec("Beckwourth Complex", 2021, "2021-06-30 23"),
    FireSpec("Caldor", 2021, "2021-08-15 01"),
    FireSpec("Dixie", 2021, "2021-07-14 00"),
    FireSpec("KNP Complex", 2021, "2021-09-10 14"),
    FireSpec("McCash", 2021, "2021-08-01 02"),
    FireSpec("McFarland", 2021, "2021-07-30 01"),
    FireSpec("Monument", 2021, "2021-07-31 01"),
    FireSpec("River Complex", 2021, "2021-07-30 21"),
    FireSpec("Tamarack", 2021, "2021-07-04 18"),
    FireSpec("Windy", 2021, "2021-09-10 00"),
]


def slugify_fire_name(name: str) -> str:
    return name.replace(" ", "_")


def parse_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_utc_hour_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def parse_catalog_start_to_iso(start_utc_hour: str) -> str:
    dt = datetime.strptime(start_utc_hour, "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
    return to_utc_hour_z(dt)


def source_name_for_file(source: str) -> str:
    return "GOESEast" if source == "east" else "GOESWest"


def source_name_for_json(source: str) -> str:
    return "GOES-East" if source == "east" else "GOES-West"


def normalize_selector(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def select_fires(fires_arg: str | None) -> list[FireSpec]:
    if not fires_arg:
        return FIRE_CATALOG

    by_name: dict[str, FireSpec] = {}
    for fire in FIRE_CATALOG:
        by_name[normalize_selector(fire.name)] = fire
        by_name[normalize_selector(slugify_fire_name(fire.name))] = fire

    selected: list[FireSpec] = []
    unknown: list[str] = []
    for raw in fires_arg.split(","):
        key = normalize_selector(raw)
        if not key:
            continue
        fire = by_name.get(key)
        if fire is None:
            unknown.append(raw.strip())
            continue
        if fire not in selected:
            selected.append(fire)

    if unknown:
        available = ", ".join(sorted(f.name for f in FIRE_CATALOG))
        raise SystemExit(
            "Unknown fire(s): "
            + ", ".join(unknown)
            + "\nAvailable fires: "
            + available
        )

    if not selected:
        raise SystemExit("No fires selected after parsing --fires.")
    return selected


def sorted_goes_tifs(fire_dir: Path, fire_slug: str, year: int, source: str) -> list[Path]:
    source_label = source_name_for_file(source)
    base = fire_dir / f"{fire_slug}_{year}_{source_label}_MaxConf.tif"
    part_glob = f"{fire_slug}_{year}_{source_label}_MaxConf_part*.tif"
    parts = sorted(fire_dir.glob(part_glob))
    if parts:
        return parts
    if base.exists():
        return [base]
    return []


def is_valid_goes_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        time_steps = metadata.get("time_steps", [])
        data = payload.get("data", [])
        return isinstance(time_steps, list) and len(time_steps) > 0 and isinstance(data, list) and len(data) > 0
    except Exception:
        return False


def resolve_manifest_file_path(path_str: str, repo_root: Path, manifest_dir: Path) -> Path | None:
    p = Path(path_str).expanduser()
    if p.exists():
        return p

    parts = p.parts
    if "data" in parts:
        idx = parts.index("data")
        candidate = repo_root.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate

    candidate = (manifest_dir / path_str).resolve()
    if candidate.exists():
        return candidate

    return None


def is_valid_rtma_manifest(manifest_path: Path, repo_root: Path) -> bool:
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        variables = payload.get("variables", [])
        files = payload.get("files", {})
        time_steps = payload.get("time_steps", [])
        if not isinstance(variables, list) or not variables:
            return False
        if not isinstance(files, dict) or not files:
            return False
        if not isinstance(time_steps, list) or not time_steps:
            return False
        manifest_dir = manifest_path.parent
        for var in variables:
            paths = files.get(var)
            if not isinstance(paths, list) or not paths:
                return False
            for item in paths:
                if not isinstance(item, str):
                    return False
                resolved = resolve_manifest_file_path(item, repo_root, manifest_dir)
                if resolved is None:
                    return False
        return True
    except Exception:
        return False


def is_valid_rtma_normalized(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata")
        data = payload.get("data")
        return isinstance(metadata, dict) and isinstance(data, dict) and len(data) > 0
    except Exception:
        return False


def parse_goes_time_bounds(goes_json: Path) -> tuple[str, str]:
    payload = json.loads(goes_json.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    raw_steps = metadata.get("time_steps", [])
    start_time = metadata.get("start_time")
    data = payload.get("data", [])

    if not raw_steps:
        raise ValueError(f"GOES JSON has no time_steps: {goes_json}")

    steps: list[datetime] = []
    if isinstance(raw_steps[0], (int, float)):
        if not start_time:
            raise ValueError("GOES JSON has numeric time_steps but metadata.start_time is missing.")
        start_dt = parse_iso(start_time)
        for item in raw_steps:
            steps.append(start_dt + timedelta(hours=int(item) - 1))
    else:
        for item in raw_steps:
            steps.append(parse_iso(str(item)))

    if len(steps) != len(data):
        raise ValueError(
            f"GOES JSON time length mismatch for {goes_json.name}: "
            f"{len(steps)} time_steps vs {len(data)} data bands"
        )

    start = to_utc_hour_z(steps[0])
    end_plus_one = to_utc_hour_z(steps[-1] + timedelta(hours=1))
    return start, end_plus_one


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    printable = " ".join(subprocess.list2cmdline([part]) if " " in part else part for part in cmd)
    print(f"  $ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def process_fire(
    fire: FireSpec,
    args: argparse.Namespace,
    repo_root: Path,
) -> dict:
    source_label = source_name_for_file(args.source)
    source_json_name = source_name_for_json(args.source)
    fire_slug = slugify_fire_name(fire.name)
    fire_dir = args.output_root / fire_slug
    rtma_dir = fire_dir / "rtma"

    goes_tif_base = fire_dir / f"{fire_slug}_{fire.year}_{source_label}_MaxConf.tif"
    goes_json = fire_dir / f"{fire_slug}_{fire.year}_{source_label}_MaxConf.json"
    rtma_manifest = rtma_dir / "rtma_manifest.json"
    rtma_normalized = rtma_dir / "rtma_normalized.json"

    status = {
        "fire": fire.name,
        "slug": fire_slug,
        "reused": [],
        "rebuilt_local": [],
        "downloaded": [],
    }

    ensure_dir(fire_dir, args.dry_run)
    ensure_dir(rtma_dir, args.dry_run)

    # Step 1: GOES TIF download (only if missing unless overwrite).
    goes_tifs = sorted_goes_tifs(fire_dir, fire_slug, fire.year, args.source)
    if args.overwrite or not goes_tifs:
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "ee_download_confidence_stack.py"),
            "--fire-name",
            fire.name,
            "--year",
            str(fire.year),
            "--source",
            args.source,
            "--output",
            str(goes_tif_base),
        ]
        if args.project:
            cmd.extend(["--project", args.project])
        run_cmd(cmd, args.dry_run)
        status["downloaded"].append("goes_tif")
    else:
        status["reused"].append("goes_tif")

    goes_tifs = sorted_goes_tifs(fire_dir, fire_slug, fire.year, args.source)
    if args.dry_run and not goes_tifs:
        goes_tifs = [goes_tif_base]
    if not goes_tifs and not args.dry_run:
        raise RuntimeError(f"No GOES TIFF found after step for {fire.name}.")

    # Step 2: GOES JSON conversion (rebuild locally when needed).
    goes_json_valid = is_valid_goes_json(goes_json)
    if args.overwrite or not goes_json_valid:
        if not goes_tifs:
            raise RuntimeError(f"Cannot build GOES JSON for {fire.name}: no local GOES TIFFs.")
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "gofer_confidence_to_json.py"),
            "--input",
        ]
        cmd.extend(str(p) for p in goes_tifs)
        cmd.extend(
            [
                "--output",
                str(goes_json),
                "--fire-name",
                fire.name,
                "--year",
                str(fire.year),
                "--source",
                source_json_name,
                "--start-time",
                parse_catalog_start_to_iso(fire.start_utc_hour),
            ]
        )
        run_cmd(cmd, args.dry_run)
        if args.dry_run:
            status["downloaded"].append("goes_json")
        elif goes_json.exists() and not args.overwrite:
            status["rebuilt_local"].append("goes_json")
        else:
            status["downloaded"].append("goes_json")
    else:
        status["reused"].append("goes_json")

    # Step 3: RTMA download (requires valid GOES JSON + template tif).
    rtma_manifest_valid = is_valid_rtma_manifest(rtma_manifest, repo_root)
    if args.overwrite or not rtma_manifest_valid:
        if not is_valid_goes_json(goes_json) and not args.dry_run:
            raise RuntimeError(f"Cannot build RTMA for {fire.name}: GOES JSON invalid.")
        if not goes_tifs:
            raise RuntimeError(f"Cannot build RTMA for {fire.name}: no GOES TIFF template.")
        if is_valid_goes_json(goes_json):
            start_iso, end_iso = parse_goes_time_bounds(goes_json)
        else:
            start_iso = parse_catalog_start_to_iso(fire.start_utc_hour)
            end_iso = to_utc_hour_z(parse_iso(start_iso) + timedelta(hours=1))
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "ee_download_rtma.py"),
            "--goes-tif",
            str(goes_tifs[0]),
            "--start",
            start_iso,
            "--end",
            end_iso,
            "--output-dir",
            str(rtma_dir),
            "--chunk-hours",
            str(args.chunk_hours),
        ]
        if args.project:
            cmd.extend(["--project", args.project])
        run_cmd(cmd, args.dry_run)
        status["downloaded"].append("rtma_manifest_and_chunks")
    else:
        status["reused"].append("rtma_manifest_and_chunks")

    # Step 4: RTMA normalized JSON (rebuild locally when needed).
    rtma_norm_valid = is_valid_rtma_normalized(rtma_normalized)
    if args.overwrite or not rtma_norm_valid:
        if not is_valid_rtma_manifest(rtma_manifest, repo_root) and not args.dry_run:
            raise RuntimeError(f"Cannot build rtma_normalized.json for {fire.name}: manifest invalid.")
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "rtma_to_json.py"),
            "--manifest",
            str(rtma_manifest),
            "--output",
            str(rtma_normalized),
            "--normalize",
            args.normalize,
        ]
        run_cmd(cmd, args.dry_run)
        if rtma_normalized.exists() and not args.overwrite:
            status["rebuilt_local"].append("rtma_normalized")
        else:
            status["downloaded"].append("rtma_normalized")
    else:
        status["reused"].append("rtma_normalized")

    return status


def print_summary(successes: list[dict], failures: list[tuple[str, str]]) -> None:
    print("\n=== Batch Summary ===")
    print(f"Successes: {len(successes)}")
    print(f"Failures:  {len(failures)}")
    for item in successes:
        print(
            f"  [OK] {item['fire']}: "
            f"reused={','.join(item['reused']) or '-'} "
            f"rebuilt_local={','.join(item['rebuilt_local']) or '-'} "
            f"downloaded={','.join(item['downloaded']) or '-'}"
        )
    for fire_name, err in failures:
        print(f"  [FAIL] {fire_name}: {err}")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/build all 28 GOFER fires into data/multi_fire.")
    parser.add_argument(
        "--output-root",
        default="data/multi_fire",
        help="Root output directory (default: data/multi_fire).",
    )
    parser.add_argument(
        "--source",
        default="east",
        choices=["east", "west"],
        help="GOES source to use for all fires (default: east).",
    )
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=24,
        help="RTMA chunk size in hours (default: 24).",
    )
    parser.add_argument(
        "--normalize",
        default="zscore",
        choices=["zscore", "minmax", "none"],
        help="Normalization mode for rtma_to_json (default: zscore).",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional Earth Engine project ID.",
    )
    parser.add_argument(
        "--fires",
        default=None,
        help='Optional comma-separated subset, e.g. --fires "Creek,Dixie".',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run all steps even if outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing downloads/conversions.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first fire failure.",
    )
    parser.add_argument(
        "--fire-retries",
        type=int,
        default=1,
        help="Number of retry attempts per fire after a failed run (default: 1).",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    args.output_root = Path(args.output_root).resolve()

    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "scripts").exists() or not (repo_root / "data").exists():
        raise SystemExit(f"Could not resolve repo root from {__file__}.")

    fires = select_fires(args.fires)

    print(f"Repo root: {repo_root}")
    print(f"Output root: {args.output_root}")
    print(f"Selected fires: {len(fires)}")
    print(f"Source: {args.source}")
    print(f"Dry run: {args.dry_run}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Retries per fire: {args.fire_retries}")

    successes: list[dict] = []
    failures: list[tuple[str, str]] = []

    for idx, fire in enumerate(fires, start=1):
        print(f"\n[{idx}/{len(fires)}] {fire.name} ({fire.year})")
        max_attempts = max(1, int(args.fire_retries) + 1)
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                print(f"  Retry attempt {attempt - 1}/{args.fire_retries} for {fire.name}...")
            try:
                status = process_fire(fire, args, repo_root)
                successes.append(status)
                break
            except subprocess.CalledProcessError as exc:
                msg = f"Command failed with exit {exc.returncode}"
                print(f"  ERROR: {msg}")
                if attempt == max_attempts:
                    failures.append((fire.name, msg))
                    if args.fail_fast:
                        break
            except Exception as exc:
                msg = str(exc)
                print(f"  ERROR: {msg}")
                if attempt == max_attempts:
                    failures.append((fire.name, msg))
                    if args.fail_fast:
                        break
        if args.fail_fast and failures:
            break

    print_summary(successes, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Download GOES + RTMA for multiple fires and run aggregated regression.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_fire_metadata():
    text = (ROOT / "gee" / "largeFires_metadata.js").read_text()
    # Extract fire params: name -> (year, start, end)
    # Start/end lines: start: ee.Date.parse('Y-MM-dd HH','YYYY-MM-DD HH')
    fire_data = {}
    year = None
    current = None
    for line in text.splitlines():
        m_year = re.search(r"'([0-9]{4})':\s*\{", line)
        if m_year:
            year = m_year.group(1)
        m_fire = re.search(r"'([^']+)':\s*\{", line)
        if m_fire:
            name = m_fire.group(1)
            if not name.isdigit():
                current = name
                fire_data.setdefault(current, {"year": year})
        m_start = re.search(r"start:\s*ee.Date.parse\('Y-MM-dd HH','([^']+)'\)", line)
        if m_start and current:
            fire_data[current]["start"] = m_start.group(1) + ":00Z"
        m_end = re.search(r"end:\s*ee.Date.parse\('Y-MM-dd HH','([^']+)'\)", line)
        if m_end and current:
            fire_data[current]["end"] = m_end.group(1) + ":00Z"
    return fire_data


def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/multi_fire")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-hours", type=int, default=24)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--max-samples-per-fire", type=int, default=150000)
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing outputs.")
    args = parser.parse_args()

    fires = ["Dixie", "SCU Lightning Complex", "Creek", "LNU Lightning Complex", "North Complex"]
    meta = parse_fire_metadata()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {"fires": []}

    for fire in fires:
        if fire not in meta:
            print(f"Missing metadata for {fire}")
            continue
        year = meta[fire]["year"]
        start = meta[fire].get("start")
        end = meta[fire].get("end")
        if not start or not end:
            print(f"Missing start/end for {fire}")
            continue

        fire_slug = fire.replace(" ", "_")
        fire_dir = output_dir / fire_slug
        fire_dir.mkdir(parents=True, exist_ok=True)

        goes_tif = fire_dir / f"{fire_slug}_{year}_GOESEast_MaxConf.tif"

        goes_json = fire_dir / f"{fire_slug}_{year}_GOESEast_MaxConf.json"
        if not goes_json.exists() or args.force:
            # Collect GOES part files if they exist
            parts = sorted(fire_dir.glob(f"{fire_slug}_{year}_GOESEast_MaxConf_part*.tif"))
            if not parts or args.force:
                # Download GOES confidence
                run([
                    "python3",
                    str(ROOT / "scripts" / "ee_download_confidence_stack.py"),
                    "--fire-name", fire,
                    "--year", str(year),
                    "--source", "east",
                    "--chunk-size", str(args.chunk_size),
                    "--output", str(goes_tif),
                ])
                parts = sorted(fire_dir.glob(f"{fire_slug}_{year}_GOESEast_MaxConf_part*.tif"))
            if not parts:
                if goes_tif.exists():
                    parts = [goes_tif]
            if not parts:
                print(f"No GOES files for {fire}")
                continue

            # Convert GOES to JSON
            cmd = [
                "python3",
                str(ROOT / "scripts" / "gofer_confidence_to_json.py"),
                "--input",
            ] + [str(p) for p in parts] + [
                "--output", str(goes_json),
                "--fire-name", fire,
                "--year", str(year),
                "--source", "GOES-East",
                "--start-time", start.replace(" ", "T")
            ]
            run(cmd)

        # Download RTMA (24h chunks)
        rtma_dir = fire_dir / "rtma"
        rtma_manifest = rtma_dir / "rtma_manifest.json"
        if not rtma_manifest.exists() or args.force:
            # Ensure we have at least one GOES part for geometry
            parts = sorted(fire_dir.glob(f"{fire_slug}_{year}_GOESEast_MaxConf_part*.tif"))
            if not parts and goes_tif.exists():
                parts = [goes_tif]
            if not parts:
                print(f"Missing GOES part for RTMA geometry in {fire}")
                continue
            run([
                "python3",
                str(ROOT / "scripts" / "ee_download_rtma.py"),
                "--goes-tif", str(parts[0]),
                "--start", start.replace(" ", "T"),
                "--end", end.replace(" ", "T"),
                "--output-dir", str(rtma_dir),
                "--chunk-hours", str(args.chunk_hours),
            ])

        # Normalize RTMA (optional) - create normalized JSON for reference
        rtma_norm_json = rtma_dir / "rtma_normalized.json"
        if not rtma_norm_json.exists() or args.force:
            run([
                "python3",
                str(ROOT / "scripts" / "rtma_to_json.py"),
                "--manifest", str(rtma_manifest),
                "--output", str(rtma_norm_json),
                "--normalize", "zscore",
            ])

        config["fires"].append(
            {
                "name": fire,
                "goes_json": str(goes_json),
                "rtma_manifest": str(rtma_manifest),
                "goes_start": start.replace(" ", "T"),
            }
        )

    config_path = output_dir / "aggregate_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # Run aggregated regression
    output_report = output_dir / "aggregate_regression_report.json"
    run([
        "python3",
        str(ROOT / "scripts" / "run_locational_regressions_aggregate.py"),
        "--config", str(config_path),
        "--threshold", "0.1",
        "--neg-ratio", str(args.neg_ratio),
        "--max-samples-per-fire", str(args.max_samples_per_fire),
        "--output", str(output_report),
    ])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download GOFER product files from Zenodo.

Default record is GOFER v0.2 (as referenced in README). You can override via
--record-id or --doi.

Example:
  python scripts/download_gofer_zenodo.py --target data/zenodo
  python scripts/download_gofer_zenodo.py --record-id 14642378 --target data/zenodo
  python scripts/download_gofer_zenodo.py --doi 10.5281/zenodo.8327264 --target data/zenodo
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from typing import Optional


def extract_record_id_from_doi(doi: str) -> Optional[str]:
    match = re.search(r"zenodo\.(\d+)", doi)
    return match.group(1) if match else None


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def download_file(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as f:
        f.write(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GOFER data from Zenodo")
    parser.add_argument(
        "--record-id",
        default=None,
        help="Zenodo record id (e.g., 14642378)",
    )
    parser.add_argument(
        "--doi",
        default=None,
        help="Zenodo DOI (e.g., 10.5281/zenodo.14642378)",
    )
    parser.add_argument(
        "--target",
        default="data/zenodo",
        help="Target directory for downloads",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    record_id = args.record_id
    if args.doi:
        record_id = extract_record_id_from_doi(args.doi)

    if not record_id:
        # Default to GOFER v0.2 record (as in README)
        record_id = "14642378"

    record_url = f"https://zenodo.org/api/records/{record_id}"

    try:
        record = fetch_json(record_url)
    except Exception as exc:
        print(f"Failed to fetch record {record_id}: {exc}", file=sys.stderr)
        raise

    os.makedirs(args.target, exist_ok=True)
    manifest_path = os.path.join(args.target, "record.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=True, indent=2)

    files = record.get("files", [])
    if not files:
        print("No files found in record.", file=sys.stderr)
        return

    for entry in files:
        name = entry.get("key") or entry.get("filename")
        link = entry.get("links", {}).get("self")
        if not name or not link:
            continue

        dest_path = os.path.join(args.target, name)
        if os.path.exists(dest_path) and not args.overwrite:
            print(f"Skip existing: {dest_path}")
            continue

        print(f"Downloading: {name}")
        download_file(link, dest_path)

    print(f"Done. Files saved in: {os.path.abspath(args.target)}")


if __name__ == "__main__":
    main()

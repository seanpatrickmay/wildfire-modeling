#!/usr/bin/env python3
"""
Download a GOFER confidence stack (multi-band GeoTIFF) from Earth Engine.

This script pulls the public GOFER confidence asset (GOES-East/West MaxConf)
and downloads it locally as a multi-band GeoTIFF.

Example:
  python scripts/ee_download_confidence_stack.py \
    --fire-name "Kincade" --year 2019 --source east \
    --output data/ee_exports/Kincade_2019_GOESEast_MaxConf.tif

If you already know the asset id:
  python scripts/ee_download_confidence_stack.py \
    --asset-id projects/GlobalFires/GOFER/GOESEast_MaxConf/Kincade_2019 \
    --output data/ee_exports/Kincade_2019_GOESEast_MaxConf.tif

Requirements:
  pip install earthengine-api
  earthengine authenticate
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import urllib.error
import zipfile

RETRIABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    def __init__(self, message: str, *, retriable: bool) -> None:
        super().__init__(message)
        self.retriable = retriable


def build_asset_id(fire_name: str, year: int, source: str) -> str:
    fire_name_yr = f"{fire_name.replace(' ', '_')}_{year}"
    if source.lower() in ("east", "goes-east", "goeseast"):
        folder = "GOESEast_MaxConf"
    elif source.lower() in ("west", "goes-west", "goeswest"):
        folder = "GOESWest_MaxConf"
    else:
        raise ValueError("source must be 'east' or 'west'")
    return f"projects/GlobalFires/GOFER/{folder}/{fire_name_yr}"


def download_zip(url: str, zip_path: str) -> None:
    with urllib.request.urlopen(url) as response, open(zip_path, "wb") as f:
        shutil.copyfileobj(response, f)


def should_retry(code: int | None, body: str) -> bool:
    if code in RETRIABLE_HTTP_CODES:
        return True
    text = body.lower()
    return (
        "internal error has occurred" in text
        or "memory capacity exceeded" in text
        or '"status": "internal"' in text
        or '"status": "unavailable"' in text
    )


def is_projection_error(code: int | None, body: str) -> bool:
    return code == 400 and "unable to write geotiffs in projection" in body.lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GOFER confidence stack from Earth Engine"
    )
    parser.add_argument("--fire-name", help="Fire name (e.g., Kincade)")
    parser.add_argument("--year", type=int, help="Fire year (e.g., 2019)")
    parser.add_argument(
        "--source",
        help="GOES source: east or west",
    )
    parser.add_argument("--asset-id", help="Explicit Earth Engine asset id")
    parser.add_argument("--output", required=True, help="Output GeoTIFF path")
    parser.add_argument(
        "--scale",
        type=float,
        default=2004.017315487541,
        help="Pixel size in meters (default: 2004.017315487541)",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="Override CRS (e.g., EPSG:4326). Defaults to asset CRS.",
    )
    parser.add_argument(
        "--region-bounds",
        nargs=4,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        type=float,
        help="Override region bounds (lon/lat). If omitted, uses asset bounds.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Google Cloud project ID for Earth Engine quota/billing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Max number of bands per file. If omitted, auto-split when >1024 bands.",
    )
    parser.add_argument(
        "--band-start",
        type=int,
        default=None,
        help="1-based start band index (inclusive) for manual band range.",
    )
    parser.add_argument(
        "--band-end",
        type=int,
        default=None,
        help="1-based end band index (inclusive) for manual band range.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file next to the output.",
    )
    parser.add_argument(
        "--max-download-attempts",
        type=int,
        default=5,
        help="Retries per EE download request before failing (default: 5).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Initial retry backoff seconds for EE retries (default: 2.0).",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=32,
        help="Smallest adaptive band chunk size when splitting failing requests (default: 32).",
    )
    args = parser.parse_args()

    if not args.asset_id:
        if not (args.fire_name and args.year and args.source):
            parser.error("Provide --asset-id or --fire-name, --year, and --source")
        asset_id = build_asset_id(args.fire_name, args.year, args.source)
    else:
        asset_id = args.asset_id

    try:
        import ee  # type: ignore
    except Exception as exc:  # pragma: no cover
        print("Missing dependency: earthengine-api", file=sys.stderr)
        print("Install with: pip install earthengine-api", file=sys.stderr)
        raise exc

    try:
        if args.project:
            ee.Initialize(project=args.project)
        else:
            ee.Initialize()
    except Exception:
        print(
            "Earth Engine init failed. If you see a project warning, set a project with "
            "`earthengine set_project YOUR_PROJECT_ID` or pass --project YOUR_PROJECT_ID.",
            file=sys.stderr,
        )
        raise

    image = ee.Image(asset_id)
    band_names = image.bandNames().getInfo()
    band_count = len(band_names)

    if args.band_start or args.band_end:
        start = args.band_start or 1
        end = args.band_end or band_count
        if start < 1 or end > band_count or start > end:
            raise ValueError("Invalid band range.")
        band_names = band_names[start - 1 : end]
        band_count = len(band_names)
        image = image.select(band_names)
    if args.region_bounds:
        minlon, minlat, maxlon, maxlat = args.region_bounds
        region = ee.Geometry.Rectangle([minlon, minlat, maxlon, maxlat])
    else:
        region = image.geometry().bounds()

    params = {
        "scale": args.scale,
        "region": region.getInfo()["coordinates"],
        "filePerBand": False,
    }
    if args.crs:
        params["crs"] = args.crs
    if args.chunk_size:
        chunk_size = args.chunk_size
    else:
        # Default to a smaller chunk size to avoid request-size limits.
        chunk_size = 256 if band_count > 256 else None

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    def download_image(img, out_path):
        download_params = dict(params)
        backoff = args.retry_backoff_seconds
        projection_retry_done = bool(args.crs)

        for attempt in range(1, args.max_download_attempts + 1):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "download.zip")
                try:
                    url = img.getDownloadURL(download_params)
                    download_zip(url, zip_path)
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    print(f"HTTP error body:\\n{body}", file=sys.stderr)

                    if not projection_retry_done and is_projection_error(exc.code, body):
                        fallback_crs = "EPSG:3857"
                        print(f"Retrying with CRS override: {fallback_crs}", file=sys.stderr)
                        download_params["crs"] = fallback_crs
                        projection_retry_done = True
                        continue

                    retriable = should_retry(exc.code, body)
                    if retriable and attempt < args.max_download_attempts:
                        print(
                            f"Retrying download attempt {attempt}/{args.max_download_attempts - 1} "
                            f"after HTTP {exc.code}; waiting {backoff:.1f}s...",
                            file=sys.stderr,
                        )
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue

                    raise DownloadError(
                        f"HTTP {exc.code}: {body}",
                        retriable=retriable,
                    ) from exc
                except urllib.error.URLError as exc:
                    if attempt < args.max_download_attempts:
                        print(
                            f"Retrying URL error attempt {attempt}/{args.max_download_attempts - 1} "
                            f"({exc.reason}); waiting {backoff:.1f}s...",
                            file=sys.stderr,
                        )
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    raise DownloadError(f"URL error: {exc}", retriable=True) from exc

                with zipfile.ZipFile(zip_path, "r") as zf:
                    tif_names = [name for name in zf.namelist() if name.lower().endswith(".tif")]
                    if not tif_names:
                        raise DownloadError("No .tif found in downloaded archive", retriable=False)
                    tif_name = tif_names[0]
                    zf.extract(tif_name, tmpdir)
                    extracted_path = os.path.join(tmpdir, tif_name)
                    shutil.move(extracted_path, out_path)

                if args.keep_zip:
                    zip_keep_path = os.path.splitext(out_path)[0] + ".zip"
                    shutil.copyfile(zip_path, zip_keep_path)
                return

    if chunk_size:
        base, ext = os.path.splitext(output_path)
        pending_chunks = [
            band_names[i : i + chunk_size]
            for i in range(0, len(band_names), chunk_size)
        ]
        idx = 1
        while pending_chunks:
            chunk = pending_chunks.pop(0)
            out_path = f"{base}_part{idx:02d}{ext or '.tif'}"
            print(f"Downloading bands {chunk[0]}..{chunk[-1]} -> {out_path}")
            try:
                download_image(image.select(chunk), out_path)
                print(f"Saved: {out_path}")
                idx += 1
            except DownloadError as exc:
                can_split = len(chunk) > args.min_chunk_size
                if exc.retriable and can_split:
                    mid = len(chunk) // 2
                    left = chunk[:mid]
                    right = chunk[mid:]
                    print(
                        "Chunk failed with retriable error; splitting "
                        f"{len(chunk)} bands into {len(left)} and {len(right)}.",
                        file=sys.stderr,
                    )
                    pending_chunks = [left, right] + pending_chunks
                    continue
                raise SystemExit(
                    f"Failed downloading band chunk {chunk[0]}..{chunk[-1]}: {exc}"
                )
    else:
        try:
            download_image(image, output_path)
        except DownloadError as exc:
            raise SystemExit(f"Failed downloading image: {exc}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create a nearest-RAWS-station index grid for a GOES GeoTIFF.

Example:
  python3 scripts/raws_nearest_station_grid.py \
    --goes-tif data/ee_exports/August_Complex_2020_GOESEast_MaxConf_part01.tif \
    --stations-json data/raws/august/raws_stations.json \
    --output-dir data/raws/august/grid
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import rasterio

WEB_MERCATOR_R = 6378137.0


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    x = WEB_MERCATOR_R * math.radians(lon)
    lat = max(min(lat, 85.05112878), -85.05112878)
    y = WEB_MERCATOR_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest RAWS station grid")
    parser.add_argument("--goes-tif", required=True, help="GeoTIFF for grid geometry")
    parser.add_argument("--stations-json", required=True, help="raws_stations.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.stations_json, "r", encoding="utf-8") as f:
        stations = json.load(f).get("STATION", [])

    if not stations:
        raise SystemExit("No stations found in stations JSON.")

    with rasterio.open(args.goes_tif) as ds:
        width = ds.width
        height = ds.height
        transform = ds.transform
        crs = ds.crs

    if not crs:
        raise SystemExit("GOES GeoTIFF missing CRS.")
    if crs.to_string() != "EPSG:3857":
        raise SystemExit("This script expects EPSG:3857 GeoTIFF (use download with CRS override).")

    # Station coords in EPSG:3857
    station_ids = []
    station_xy = []
    for s in stations:
        stid = s.get("STID")
        lat = s.get("LATITUDE")
        lon = s.get("LONGITUDE")
        if stid is None or lat is None or lon is None:
            continue
        x, y = lonlat_to_mercator(float(lon), float(lat))
        station_ids.append(stid)
        station_xy.append((x, y))

    if not station_xy:
        raise SystemExit("No valid station coordinates found.")

    station_xy = np.array(station_xy, dtype=np.float64)
    stations_x = station_xy[:, 0]
    stations_y = station_xy[:, 1]

    # Grid cell centers
    x_coords = transform.c + (np.arange(width) + 0.5) * transform.a
    y_coords = transform.f + (np.arange(height) + 0.5) * transform.e

    index_grid = np.zeros((height, width), dtype=np.int32)
    dist_grid = np.zeros((height, width), dtype=np.float32)

    for row, y in enumerate(y_coords):
        dx = x_coords[None, :] - stations_x[:, None]
        dy = y - stations_y[:, None]
        dist2 = dx * dx + dy * dy
        idx = np.argmin(dist2, axis=0)
        index_grid[row, :] = idx
        dist_grid[row, :] = np.sqrt(dist2[idx, np.arange(width)]) / 1000.0

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "raws_station_index.npy"), index_grid)
    np.save(os.path.join(output_dir, "raws_station_distance_km.npy"), dist_grid)

    with open(os.path.join(output_dir, "raws_station_lookup.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "station_ids": station_ids,
                "crs": "EPSG:3857",
                "grid_shape": [height, width],
                "transform": [
                    transform.a,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e,
                    transform.f,
                ],
            },
            f,
            indent=2,
        )

    print(f"Saved grids to: {output_dir}")


if __name__ == "__main__":
    main()

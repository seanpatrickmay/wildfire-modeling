"""Shared utility functions used across wildfire-modeling scripts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def parse_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def normalize_time_str(value: str) -> str:
    dt = parse_iso(value)
    return dt.strftime("%Y-%m-%dT%H:00:00Z")


def affine_from_list(vals: list[float]) -> rasterio.Affine:
    return rasterio.Affine(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


def resample_stack(
    src_stack: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    dst_shape: tuple[int, int],
    dst_transform: rasterio.Affine,
    dst_crs: Any,
) -> np.ndarray:
    bands = src_stack.shape[0]
    dst = np.empty((bands, dst_shape[0], dst_shape[1]), dtype=np.float32)
    for band_idx in range(bands):
        reproject(
            source=src_stack[band_idx],
            destination=dst[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return dst

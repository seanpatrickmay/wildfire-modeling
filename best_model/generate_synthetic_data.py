"""Generate realistic synthetic wildfire training data.

Produces spatiotemporal fire progression data with correlated meteorological
features, terrain, fuel type, and GOES/VIIRS-style fire confidence targets.

Usage:
    python best_model/generate_synthetic_data.py --n-fires 20 --output best_model/data
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter


def generate_terrain(shape: tuple[int, int], seed: int) -> dict[str, np.ndarray]:
    """Generate correlated terrain features (elevation, slope, aspect, TPI)."""
    rng = np.random.default_rng(seed)
    h, w = shape

    # Smooth elevation field via low-frequency noise
    elevation = np.abs(np.fft.irfft2(rng.normal(0, 1, (h, w)), s=(h, w))).astype(np.float32)
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-8)
    elevation = elevation * 2500 + 200  # 200-2700m range

    # Slope from elevation gradient
    dy, dx = np.gradient(elevation)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) / 30.0)).astype(np.float32)

    # Aspect from gradient direction
    aspect = (np.degrees(np.arctan2(-dx, dy)) % 360).astype(np.float32)

    # Topographic Position Index (elevation - local mean)
    tpi = (elevation - uniform_filter(elevation, size=9)).astype(np.float32)

    return {
        "elevation": elevation,
        "slope": slope,
        "aspect": aspect,
        "tpi": tpi,
    }


def generate_fuel(shape: tuple[int, int], elevation: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Generate fuel type and load based on elevation bands."""
    rng = np.random.default_rng(seed)
    h, w = shape

    # Fuel model classes (simplified Scott & Burgan)
    # Low elev → grass/shrub, mid → timber, high → sparse
    fuel_class = np.zeros(shape, dtype=np.int32)
    fuel_class[elevation < 800] = 1    # Grass
    fuel_class[(elevation >= 800) & (elevation < 1500)] = 2   # Shrub
    fuel_class[(elevation >= 1500) & (elevation < 2200)] = 3  # Timber
    fuel_class[elevation >= 2200] = 4  # Sparse/alpine
    # Add noise
    noise = rng.integers(0, 2, shape)
    fuel_class = np.clip(fuel_class + noise - 1, 1, 4).astype(np.int32)

    # Fuel load (tons/acre) correlated with class
    fuel_load_map = {1: 2.0, 2: 5.0, 3: 8.0, 4: 1.0}
    fuel_load = np.vectorize(fuel_load_map.get)(fuel_class).astype(np.float32)
    fuel_load += rng.normal(0, 1, shape).astype(np.float32)
    fuel_load = np.clip(fuel_load, 0.1, 15.0)

    return {
        "fuel_class": fuel_class,
        "fuel_load": fuel_load,
    }


def generate_fire(
    fire_name: str,
    grid_shape: tuple[int, int],
    n_hours: int,
    seed: int,
) -> dict:
    """Generate one fire event with realistic spatiotemporal dynamics."""
    rng = np.random.default_rng(seed)
    h, w = grid_shape

    terrain = generate_terrain(grid_shape, seed)
    fuel = generate_fuel(grid_shape, terrain["elevation"], seed + 1)

    # Meteorological time series (spatially uniform with noise)
    base_temp = rng.uniform(25, 40)  # hot fire weather
    base_wind = rng.uniform(5, 25)
    base_rh = rng.uniform(5, 25)  # low humidity

    hours = []
    fire_mask = np.zeros(grid_shape, dtype=np.float32)
    # Ignition point
    ig_y, ig_x = h // 2 + rng.integers(-h // 6, h // 6), w // 2 + rng.integers(-w // 6, w // 6)
    fire_mask[ig_y, ig_x] = 1.0

    for t in range(n_hours):
        # Evolving weather
        temp = base_temp + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2)
        wind_speed = max(0, base_wind + 5 * np.sin(2 * np.pi * (t - 6) / 24) + rng.normal(0, 3))
        wind_dir = (270 + 30 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 15)) % 360
        rh = max(3, min(95, base_rh + 15 * np.sin(2 * np.pi * (t + 6) / 24) + rng.normal(0, 5)))
        precip = max(0, rng.exponential(0.1) - 0.5) if rng.random() < 0.05 else 0.0

        # Spatial weather fields (smooth variation)
        temp_field = (temp + rng.normal(0, 1, grid_shape) - terrain["elevation"] * 0.006).astype(np.float32)
        wind_field = np.clip(wind_speed + rng.normal(0, 2, grid_shape), 0, 80).astype(np.float32)
        rh_field = np.clip(rh + rng.normal(0, 3, grid_shape), 1, 100).astype(np.float32)
        spfh_field = (rh_field / 100 * 0.02).astype(np.float32)  # approximate specific humidity

        # Wind direction as sin/cos
        wdir_rad = np.deg2rad(wind_dir + rng.normal(0, 10, grid_shape))
        wdir_sin = np.sin(wdir_rad).astype(np.float32)
        wdir_cos = np.cos(wdir_rad).astype(np.float32)

        # Fire spread probability based on physics
        # Higher spread with: wind, slope alignment, low humidity, high fuel load
        wind_alignment = (wdir_cos * np.cos(np.deg2rad(terrain["aspect"])) +
                          wdir_sin * np.sin(np.deg2rad(terrain["aspect"])))
        spread_prob = (
            0.02 * wind_field / 10.0
            + 0.01 * terrain["slope"] / 30.0
            + 0.01 * wind_alignment
            + 0.01 * fuel["fuel_load"] / 8.0
            - 0.02 * rh_field / 50.0
            - 0.05 * precip
        )
        spread_prob = np.clip(spread_prob, 0.001, 0.15).astype(np.float32)

        # Expand fire from active perimeter
        kernel = np.ones((3, 3), dtype=bool)
        perimeter = binary_dilation(fire_mask > 0.5, kernel) & (fire_mask < 0.5)
        new_fire = perimeter & (rng.random(grid_shape) < spread_prob)
        fire_mask = np.where(new_fire, 1.0, fire_mask).astype(np.float32)

        # Confidence = fire_mask with some noise
        confidence = fire_mask * np.clip(0.7 + rng.normal(0, 0.15, grid_shape), 0.1, 1.0)
        confidence = confidence.astype(np.float32)

        hours.append({
            "hour": t,
            "temperature": temp_field,
            "wind_speed": wind_field,
            "wind_dir_sin": wdir_sin,
            "wind_dir_cos": wdir_cos,
            "specific_humidity": spfh_field,
            "precipitation": np.full(grid_shape, precip, dtype=np.float32),
            "relative_humidity": rh_field,
            "confidence_t": confidence,
        })

    return {
        "fire_name": fire_name,
        "grid_shape": grid_shape,
        "n_hours": n_hours,
        "terrain": terrain,
        "fuel": fuel,
        "hours": hours,
    }


def build_samples(fire_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert fire data into feature matrix X and target vector y.

    Uses center-pixel features (no neighbor offsets) for simplicity and speed.
    The neighbor context is captured via max-pooled fire confidence in a 3x3 window.
    """
    terrain = fire_data["terrain"]
    fuel = fire_data["fuel"]
    hours = fire_data["hours"]
    h, w = fire_data["grid_shape"]

    # Precompute static terrain features (interior only, skip 1px border)
    iy = slice(1, h - 1)
    ix = slice(1, w - 1)
    ih, iw = h - 2, w - 2
    aspect_sin = np.sin(np.deg2rad(terrain["aspect"][iy, ix])).ravel()
    aspect_cos = np.cos(np.deg2rad(terrain["aspect"][iy, ix])).ravel()
    elev_flat = terrain["elevation"][iy, ix].ravel()
    slope_flat = terrain["slope"][iy, ix].ravel()
    tpi_flat = terrain["tpi"][iy, ix].ravel()
    fuel_flat = fuel["fuel_load"][iy, ix].ravel()
    n_pixels = ih * iw

    all_X = []
    all_y = []
    rain_accum = np.zeros((h, w), dtype=np.float32)
    decay = 0.5 ** (1.0 / 48)  # 2-day half-life

    for t in range(len(hours) - 1):
        hr = hours[t]
        hr_next = hours[t + 1]

        rain_accum = decay * rain_accum + hr["precipitation"]

        # Max fire confidence in 3x3 neighborhood (captures neighbor context)
        conf_t = hr["confidence_t"]
        padded = np.pad(conf_t, 1, mode="constant", constant_values=0.0)
        max_conf = np.zeros_like(conf_t)
        for dy in range(3):
            for dx in range(3):
                max_conf = np.maximum(max_conf, padded[dy:dy+h, dx:dx+w])

        features = np.column_stack([
            conf_t[iy, ix].ravel(),
            max_conf[iy, ix].ravel(),
            hr["temperature"][iy, ix].ravel(),
            hr["wind_speed"][iy, ix].ravel(),
            hr["wind_dir_sin"][iy, ix].ravel(),
            hr["wind_dir_cos"][iy, ix].ravel(),
            hr["specific_humidity"][iy, ix].ravel(),
            hr["precipitation"][iy, ix].ravel(),
            hr["relative_humidity"][iy, ix].ravel(),
            rain_accum[iy, ix].ravel(),
            elev_flat,
            slope_flat,
            aspect_sin,
            aspect_cos,
            tpi_flat,
            fuel_flat,
        ])

        targets = (hr_next["confidence_t"][iy, ix] > 0.1).astype(np.float32).ravel()
        all_X.append(features)
        all_y.append(targets)

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid]


FEATURE_NAMES = [
    "fire_confidence_center",
    "fire_confidence_max_3x3",
    "temperature",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "specific_humidity",
    "precipitation_1h",
    "relative_humidity",
    "rain_accumulated_72h",
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi",
    "fuel_load",
]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic wildfire data")
    parser.add_argument("--n-fires", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--n-hours", type=int, default=120)
    parser.add_argument("--output", type=str, default="best_model/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    fire_names = [f"fire_{i:02d}" for i in range(args.n_fires)]
    all_X = []
    all_y = []
    fire_labels = []

    for i, name in enumerate(fire_names):
        print(f"Generating {name}...")
        fire = generate_fire(name, (args.grid_size, args.grid_size), args.n_hours, args.seed + i * 100)
        X, y = build_samples(fire)
        all_X.append(X)
        all_y.append(y)
        fire_labels.extend([name] * len(y))
        print(f"  {len(y)} samples, {y.sum():.0f} positives ({100*y.mean():.1f}%)")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    np.save(os.path.join(args.output, "X.npy"), X_all)
    np.save(os.path.join(args.output, "y.npy"), y_all)
    with open(os.path.join(args.output, "fire_labels.json"), "w") as f:
        json.dump(fire_labels, f)
    with open(os.path.join(args.output, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)

    print(f"\nTotal: {len(y_all)} samples, {y_all.sum():.0f} positives ({100*y_all.mean():.1f}%)")
    print(f"Saved to {args.output}/")


if __name__ == "__main__":
    main()

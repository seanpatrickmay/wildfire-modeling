#!/usr/bin/env python3
"""
Aggregate locational regressions across multiple fires.

Config JSON format:
{
  "fires": [
    {
      "name": "Dixie",
      "goes_json": "path/to/GOES.json",
      "rtma_manifest": "path/to/rtma_manifest.json"
    }
  ]
}
"""

import argparse
import json
import math
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def affine_from_list(vals: list) -> rasterio.Affine:
    return rasterio.Affine(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_u_v(wind_speed: np.ndarray, wind_dir_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(wind_dir_deg)
    u = -wind_speed * np.sin(rad)
    v = -wind_speed * np.cos(rad)
    return u, v


def resample_stack(src_stack, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
    bands = src_stack.shape[0]
    dst = np.empty((bands, dst_shape[0], dst_shape[1]), dtype=np.float32)
    for b in range(bands):
        reproject(
            source=src_stack[b],
            destination=dst[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return dst


def parse_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def normalize_time_str(value: str) -> str:
    dt = parse_iso(value)
    return dt.strftime("%Y-%m-%dT%H:00:00Z")


def load_time_steps(goes_meta, goes_conf, goes_start=None):
    goes_time_steps = goes_meta.get("time_steps", [])
    if goes_time_steps and isinstance(goes_time_steps[0], (int, float)):
        if not goes_start:
            raise SystemExit("GOES time_steps are numeric. Provide goes_start in config.")
        start_dt = parse_iso(goes_start)
        goes_time_steps = [
            (start_dt + timedelta(hours=int(i - 1))).strftime("%Y-%m-%dT%H:00:00Z")
            for i in goes_time_steps
        ]
    elif not goes_time_steps and goes_start:
        start_dt = parse_iso(goes_start)
        goes_time_steps = [
            (start_dt + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00:00Z")
            for i in range(goes_conf.shape[0])
        ]
    else:
        goes_time_steps = [normalize_time_str(t) for t in goes_time_steps]
    return goes_time_steps


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean.tolist(), std.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate locational regressions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--neg-ratio", type=int, default=5, help="0 = keep all negatives")
    parser.add_argument("--max-samples-per-fire", type=int, default=150000, help="0 = no cap")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    fires = cfg.get("fires", [])
    if not fires:
        raise SystemExit("Config has no fires.")

    feature_names_spread = ["TMP", "WIND", "SPFH", "ACPC01", "wind_dot", "conf_center", "conf_neighbor"]
    feature_names_continue = ["TMP", "WIND", "SPFH", "ACPC01", "conf_t"]

    X_spread = []
    y_spread = []
    X_continue = []
    y_continue = []

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for fire in fires:
        goes_json_path = fire["goes_json"]
        rtma_manifest_path = fire["rtma_manifest"]
        goes_start = fire.get("goes_start")

        with open(goes_json_path, "r", encoding="utf-8") as f:
            goes_json = json.load(f)
        goes_conf = np.array(goes_json["data"], dtype=np.float32)
        goes_meta = goes_json["metadata"]
        goes_time_steps = load_time_steps(goes_meta, goes_conf, goes_start)

        goes_transform = affine_from_list(goes_meta["geo_transform"])
        goes_shape = (goes_meta["grid_shape"][0], goes_meta["grid_shape"][1])
        goes_crs = goes_meta.get("crs")
        pixel_m = abs(goes_transform.a)

        offset_vectors = {}
        for dy, dx in offsets:
            dx_m = dx * pixel_m
            dy_m = dy * pixel_m
            norm = math.sqrt(dx_m * dx_m + dy_m * dy_m)
            if norm == 0:
                offset_vectors[(dy, dx)] = (0.0, 0.0)
            else:
                offset_vectors[(dy, dx)] = (dx_m / norm, -dy_m / norm)

        with open(rtma_manifest_path, "r", encoding="utf-8") as f:
            rtma_manifest = json.load(f)
        rtma_vars = rtma_manifest["variables"]
        rtma_files = rtma_manifest["files"]
        rtma_time_steps = [normalize_time_str(t) for t in rtma_manifest["time_steps"]]

        goes_time_index = {t: i for i, t in enumerate(goes_time_steps)}
        aligned_times = [t for t in rtma_time_steps if t in goes_time_index]
        if len(aligned_times) < 2:
            print(f"Skip {fire['name']} (no overlap)")
            continue

        parts = list(zip(*[rtma_files[var] for var in rtma_vars]))
        rtma_time_ptr = 0

        # Per-fire sampling limits
        max_spread = args.max_samples_per_fire
        max_continue = args.max_samples_per_fire

        for part_paths in parts:
            rtma_arrays = {}
            rtma_transform = None
            rtma_crs = None
            band_count = None
            for var, path in zip(rtma_vars, part_paths):
                with rasterio.open(path) as ds:
                    if rtma_transform is None:
                        rtma_transform = ds.transform
                        rtma_crs = ds.crs
                        band_count = ds.count
                    rtma_arrays[var] = ds.read().astype("float32")

            if band_count is None:
                continue

            resampled = {}
            for var in rtma_vars:
                resampled[var] = resample_stack(
                    rtma_arrays[var],
                    rtma_transform,
                    rtma_crs,
                    goes_shape,
                    goes_transform,
                    goes_crs,
                )

            u_stack, v_stack = compute_u_v(resampled["WIND"], resampled["WDIR"])

            for local_idx in range(band_count):
                if rtma_time_ptr + local_idx >= len(rtma_time_steps):
                    break
                time_str = rtma_time_steps[rtma_time_ptr + local_idx]
                if time_str not in goes_time_index:
                    continue
                t = goes_time_index[time_str]
                if t + 1 >= goes_conf.shape[0]:
                    continue

                conf_t = goes_conf[t]
                conf_t1 = goes_conf[t + 1]
                fire_t = conf_t >= args.threshold
                fire_t1 = conf_t1 >= args.threshold

                tmp = resampled["TMP"][local_idx]
                wind = resampled["WIND"][local_idx]
                spfh = resampled["SPFH"][local_idx]
                precip = resampled["ACPC01"][local_idx]
                u = u_stack[local_idx]
                v = v_stack[local_idx]

                # Continuation
                if max_continue <= 0 or len(X_continue) < max_continue:
                    mask = fire_t & np.isfinite(tmp) & np.isfinite(wind) & np.isfinite(spfh) & np.isfinite(precip)
                    if mask.any():
                        idxs = np.flatnonzero(mask)
                        labels = fire_t1.ravel()[idxs]
                        pos_idxs = idxs[labels]
                        neg_idxs = idxs[~labels]
                        if args.neg_ratio <= 0:
                            keep_neg = neg_idxs.size
                        else:
                            keep_neg = min(neg_idxs.size, args.neg_ratio * max(len(pos_idxs), 1))
                        if keep_neg < neg_idxs.size:
                            neg_idxs = rng.choice(neg_idxs, size=keep_neg, replace=False)
                        keep_idxs = np.concatenate([pos_idxs, neg_idxs])
                        rng.shuffle(keep_idxs)
                        if max_continue > 0:
                            remaining = max_continue - len(X_continue)
                            keep_idxs = keep_idxs[:remaining]
                        feats = np.stack(
                            [tmp.ravel()[keep_idxs], wind.ravel()[keep_idxs], spfh.ravel()[keep_idxs], precip.ravel()[keep_idxs], conf_t.ravel()[keep_idxs]],
                            axis=1,
                        )
                        X_continue.extend(feats.tolist())
                        y_continue.extend(fire_t1.ravel()[keep_idxs].astype(int).tolist())

                # Spread
                if max_spread <= 0 or len(X_spread) < max_spread:
                    for dy, dx in offsets:
                        y0 = max(0, -dy)
                        y1 = fire_t.shape[0] - max(0, dy)
                        x0 = max(0, -dx)
                        x1 = fire_t.shape[1] - max(0, dx)

                        center = fire_t[y0:y1, x0:x1]
                        if not center.any():
                            continue
                        neighbor_conf_t = conf_t[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                        neighbor_conf_t1 = conf_t1[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                        mask = center & (neighbor_conf_t < args.threshold)
                        if not mask.any():
                            continue
                        label = neighbor_conf_t1 >= args.threshold

                        tmp_n = tmp[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                        wind_n = wind[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                        spfh_n = spfh[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                        precip_n = precip[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                        u_n = u[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                        v_n = v[y0 + dy : y1 + dy, x0 + dx : x1 + dx]

                        dx_unit, dy_unit = offset_vectors[(dy, dx)]
                        wind_dot = u_n * dx_unit + v_n * dy_unit
                        conf_center = conf_t[y0:y1, x0:x1]
                        conf_neighbor = neighbor_conf_t

                        valid = mask & np.isfinite(tmp_n) & np.isfinite(wind_n) & np.isfinite(spfh_n) & np.isfinite(precip_n)
                        if not valid.any():
                            continue
                        idxs = np.flatnonzero(valid)
                        labels = label.ravel()[idxs]
                        pos_idxs = idxs[labels]
                        neg_idxs = idxs[~labels]
                        if args.neg_ratio <= 0:
                            keep_neg = neg_idxs.size
                        else:
                            keep_neg = min(neg_idxs.size, args.neg_ratio * max(len(pos_idxs), 1))
                        if keep_neg < neg_idxs.size:
                            neg_idxs = rng.choice(neg_idxs, size=keep_neg, replace=False)
                        keep_idxs = np.concatenate([pos_idxs, neg_idxs])
                        rng.shuffle(keep_idxs)

                        if max_spread > 0:
                            remaining = max_spread - len(X_spread)
                            if remaining <= 0:
                                break
                            keep_idxs = keep_idxs[:remaining]

                        feats = np.stack(
                            [
                                tmp_n.ravel()[keep_idxs],
                                wind_n.ravel()[keep_idxs],
                                spfh_n.ravel()[keep_idxs],
                                precip_n.ravel()[keep_idxs],
                                wind_dot.ravel()[keep_idxs],
                                conf_center.ravel()[keep_idxs],
                                conf_neighbor.ravel()[keep_idxs],
                            ],
                            axis=1,
                        )
                        X_spread.extend(feats.tolist())
                        y_spread.extend(label.ravel()[keep_idxs].astype(int).tolist())

                if max_spread > 0 and max_continue > 0 and len(X_spread) >= max_spread and len(X_continue) >= max_continue:
                    break

            rtma_time_ptr += band_count

            if max_spread > 0 and max_continue > 0 and len(X_spread) >= max_spread and len(X_continue) >= max_continue:
                break

    # Fit models
    def fit_model(X, y, name, feat_names):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        if len(y) < 10:
            raise SystemExit(f"Not enough samples for {name} regression.")
        X, mean, std = standardize(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(max_iter=200, solver="lbfgs")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        coefs = model.coef_.ravel().tolist()
        intercept = float(model.intercept_[0])
        return {
            "name": name,
            "samples": int(len(y)),
            "positives": int(y.sum()),
            "auc": float(auc),
            "intercept": intercept,
            "coefficients": dict(zip(feat_names, coefs)),
            "feature_mean": dict(zip(feat_names, mean)),
            "feature_std": dict(zip(feat_names, std)),
        }

    spread_result = fit_model(X_spread, y_spread, "spread", feature_names_spread)
    continue_result = fit_model(X_continue, y_continue, "continue", feature_names_continue)

    report = {
        "threshold": args.threshold,
        "spread": spread_result,
        "continue": continue_result,
        "feature_names_spread": feature_names_spread,
        "feature_names_continue": feature_names_continue,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

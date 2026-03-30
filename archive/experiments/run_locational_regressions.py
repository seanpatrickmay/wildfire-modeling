#!/usr/bin/env python3
"""
Run locational spread/continuation logistic regressions using GOES confidence + RTMA.

Inputs:
  --goes-json: GOES confidence JSON (from gofer_confidence_to_json.py)
  --rtma-manifest: RTMA manifest (from ee_download_rtma.py)

Outputs:
  - regression_report.json
  - regression_report.txt
  - probability_maps_spread.json
  - probability_maps_continue.json
"""

import argparse
import json
import math
import os
import random
from typing import List, Tuple
from datetime import datetime, timedelta

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
    # meteorological: direction wind is coming FROM
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


def select_top_hours(goes_conf: np.ndarray, threshold: float, top_n: int) -> List[int]:
    # Count new ignitions per hour (t -> t+1)
    counts = []
    for t in range(goes_conf.shape[0] - 1):
        fire_t = goes_conf[t] >= threshold
        fire_t1 = goes_conf[t + 1] >= threshold
        new_ign = (~fire_t) & fire_t1
        counts.append(new_ign.sum())
    top_idx = np.argsort(counts)[::-1][:top_n]
    return sorted(top_idx.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Locational spread regression")
    parser.add_argument("--goes-json", required=True)
    parser.add_argument("--rtma-manifest", required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--goes-start",
        default=None,
        help="ISO start time for GOES if time_steps are numeric (e.g., 2020-08-16T21:00:00Z)",
    )
    parser.add_argument("--neg-ratio", type=int, default=5, help="0 = keep all negatives")
    parser.add_argument("--max-samples", type=int, default=200000, help="0 = no cap")
    parser.add_argument("--prob-hours", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    with open(args.goes_json, "r", encoding="utf-8") as f:
        goes_json = json.load(f)
    goes_conf = np.array(goes_json["data"], dtype=np.float32)
    goes_meta = goes_json["metadata"]
    goes_time_steps = goes_meta.get("time_steps", [])
    def parse_iso(value: str) -> datetime:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    def normalize_time_str(value: str) -> str:
        dt = parse_iso(value)
        return dt.strftime("%Y-%m-%dT%H:00:00Z")
    if goes_time_steps and isinstance(goes_time_steps[0], (int, float)):
        if not args.goes_start:
            raise SystemExit("GOES time_steps are numeric. Provide --goes-start.")
        start_dt = parse_iso(args.goes_start)
        goes_time_steps = [
            (start_dt + timedelta(hours=int(i - 1))).strftime("%Y-%m-%dT%H:00:00Z")
            for i in goes_time_steps
        ]
    elif not goes_time_steps and args.goes_start:
        start_dt = parse_iso(args.goes_start)
        goes_time_steps = [
            (start_dt + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00:00Z")
            for i in range(goes_conf.shape[0])
        ]
    else:
        # Normalize existing string timestamps (handle +00:00, etc.)
        goes_time_steps = [normalize_time_str(t) for t in goes_time_steps]

    goes_transform = affine_from_list(goes_meta["geo_transform"])
    goes_shape = (goes_meta["grid_shape"][0], goes_meta["grid_shape"][1])
    goes_crs = goes_meta.get("crs")

    with open(args.rtma_manifest, "r", encoding="utf-8") as f:
        rtma_manifest = json.load(f)
    rtma_vars = rtma_manifest["variables"]
    rtma_files = rtma_manifest["files"]
    rtma_time_steps = [normalize_time_str(t) for t in rtma_manifest["time_steps"]]

    # Align time steps (intersection)
    goes_time_index = {t: i for i, t in enumerate(goes_time_steps)}
    aligned_times = [t for t in rtma_time_steps if t in goes_time_index]
    if len(aligned_times) < 2:
        raise SystemExit("No overlapping time steps between GOES and RTMA.")

    # Build mapping from RTMA time index to GOES index
    aligned_goes_indices = [goes_time_index[t] for t in aligned_times]

    # Choose hours for probability maps
    selected_hours = select_top_hours(
        goes_conf[aligned_goes_indices], args.threshold, args.prob_hours
    )

    # Prepare outputs
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Feature lists
    feature_names_spread = ["TMP", "WIND", "SPFH", "ACPC01", "wind_dot", "conf_center", "conf_neighbor"]
    feature_names_continue = ["TMP", "WIND", "SPFH", "ACPC01", "conf_t"]

    X_spread = []
    y_spread = []
    X_continue = []
    y_continue = []

    # Precompute neighbor offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    pixel_m = abs(goes_transform.a)
    offset_vectors = {}
    for dy, dx in offsets:
        dx_m = dx * pixel_m
        dy_m = dy * pixel_m
        norm = math.sqrt(dx_m * dx_m + dy_m * dy_m)
        if norm == 0:
            offset_vectors[(dy, dx)] = (0.0, 0.0)
        else:
            # dx east, dy south -> north component is -dy
            offset_vectors[(dy, dx)] = (dx_m / norm, -dy_m / norm)

    # Iterate over RTMA parts
    # Assume all variables have the same number of parts and band counts.
    parts = list(zip(*[rtma_files[var] for var in rtma_vars]))
    rtma_time_ptr = 0

    for part_idx, part_paths in enumerate(parts, start=1):
        # Read RTMA bands for each variable
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
                arr = ds.read().astype("float32")
                rtma_arrays[var] = arr

        if band_count is None:
            continue

        # Resample to GOES grid
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

        # Compute wind u/v
        u_stack, v_stack = compute_u_v(resampled["WIND"], resampled["WDIR"])

        # Process each hour in this part
        for local_idx in range(band_count):
            if rtma_time_ptr + local_idx >= len(rtma_time_steps):
                break
            time_str = rtma_time_steps[rtma_time_ptr + local_idx]
            if time_str not in goes_time_index:
                continue
            t = goes_time_index[time_str]
            # need t+1 within aligned
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

            # Continuation samples (cells that are burning at t)
            if args.max_samples <= 0 or len(X_continue) < args.max_samples:
                mask = fire_t
                mask &= np.isfinite(tmp) & np.isfinite(wind) & np.isfinite(spfh) & np.isfinite(precip)
                if mask.any():
                    idxs = np.flatnonzero(mask)
                    labels = fire_t1.ravel()[idxs]
                    # subsample to control size
                    if idxs.size > 0:
                        # sample all positives, subsample negatives
                        pos_mask = labels
                        neg_mask = ~labels
                        pos_idxs = idxs[pos_mask]
                        neg_idxs = idxs[neg_mask]
                        if args.neg_ratio <= 0:
                            keep_neg = neg_idxs.size
                        else:
                            keep_neg = min(neg_idxs.size, args.neg_ratio * max(len(pos_idxs), 1))
                        if keep_neg < neg_idxs.size:
                            neg_idxs = rng.choice(neg_idxs, size=keep_neg, replace=False)
                        keep_idxs = np.concatenate([pos_idxs, neg_idxs])
                        # shuffle
                        rng.shuffle(keep_idxs)
                        # truncate if exceeds max
                        if args.max_samples > 0:
                            remaining = args.max_samples - len(X_continue)
                            keep_idxs = keep_idxs[:remaining]
                        # build features
                        tmp_v = tmp.ravel()[keep_idxs]
                        wind_v = wind.ravel()[keep_idxs]
                        spfh_v = spfh.ravel()[keep_idxs]
                        precip_v = precip.ravel()[keep_idxs]
                        conf_v = conf_t.ravel()[keep_idxs]
                        feats = np.stack([tmp_v, wind_v, spfh_v, precip_v, conf_v], axis=1)
                        X_continue.extend(feats.tolist())
                        y_continue.extend(fire_t1.ravel()[keep_idxs].astype(int).tolist())

            # Spread samples (neighbors of burning cells)
            if args.max_samples <= 0 or len(X_spread) < args.max_samples:
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

                    # features at neighbor cell
                    tmp_n = tmp[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                    wind_n = wind[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                    spfh_n = spfh[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                    precip_n = precip[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                    u_n = u[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                    v_n = v[y0 + dy : y1 + dy, x0 + dx : x1 + dx]

                    dx_unit, dy_unit = offset_vectors[(dy, dx)]
                    wind_dot = u_n * dx_unit + v_n * dy_unit

                    conf_center = conf_t[y0:y1, x0:x1]

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

                    if args.max_samples > 0:
                        remaining = args.max_samples - len(X_spread)
                        if remaining <= 0:
                            break
                        keep_idxs = keep_idxs[:remaining]

                    tmp_v = tmp_n.ravel()[keep_idxs]
                    wind_v = wind_n.ravel()[keep_idxs]
                    spfh_v = spfh_n.ravel()[keep_idxs]
                    precip_v = precip_n.ravel()[keep_idxs]
                    wind_dot_v = wind_dot.ravel()[keep_idxs]
                    conf_center_v = conf_center.ravel()[keep_idxs]
                    conf_neighbor_v = neighbor_conf_t.ravel()[keep_idxs]

                    feats = np.stack(
                        [tmp_v, wind_v, spfh_v, precip_v, wind_dot_v, conf_center_v, conf_neighbor_v], axis=1
                    )
                    X_spread.extend(feats.tolist())
                    y_spread.extend(label.ravel()[keep_idxs].astype(int).tolist())

            # Probability maps for selected hours
            if selected_hours and t in selected_hours:
                # We will compute after model fit; store required slices
                pass

        rtma_time_ptr += band_count

        if args.max_samples > 0 and len(X_spread) >= args.max_samples and len(X_continue) >= args.max_samples:
            break

    # Fit logistic regressions
    def fit_model(X, y, name, feat_names):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        if len(y) < 10:
            raise SystemExit(f"Not enough samples for {name} regression.")
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
            "model": model,
        }

    spread_result = fit_model(X_spread, y_spread, "spread", feature_names_spread)
    continue_result = fit_model(X_continue, y_continue, "continue", feature_names_continue)

    # Probability maps for selected hours
    prob_spread_maps = []
    prob_continue_maps = []

    if selected_hours:
        # Re-open RTMA and compute for selected hours
        # We'll recompute using the same resampling approach but only for selected hours.
        selected_set = set(selected_hours)
        rtma_time_ptr = 0
        for part_idx, part_paths in enumerate(parts, start=1):
            # read RTMA
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
                    arr = ds.read().astype("float32")
                    rtma_arrays[var] = arr

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
                if t not in selected_set:
                    continue

                conf_t = goes_conf[t]
                conf_t1 = goes_conf[t + 1]
                fire_t = conf_t >= args.threshold

                tmp = resampled["TMP"][local_idx]
                wind = resampled["WIND"][local_idx]
                spfh = resampled["SPFH"][local_idx]
                precip = resampled["ACPC01"][local_idx]
                u = u_stack[local_idx]
                v = v_stack[local_idx]

                # Continue map
                cont_map = np.full_like(conf_t, np.nan, dtype=np.float32)
                mask = fire_t & np.isfinite(tmp) & np.isfinite(wind) & np.isfinite(spfh) & np.isfinite(precip)
                if mask.any():
                    feats = np.stack(
                        [tmp[mask], wind[mask], spfh[mask], precip[mask], conf_t[mask]], axis=1
                    )
                    logits = continue_result["model"].intercept_[0] + feats @ continue_result["model"].coef_.ravel()
                    cont_map[mask] = sigmoid(logits)
                prob_continue_maps.append({"time": time_str, "data": cont_map.tolist()})

                # Spread map
                spread_map = np.full_like(conf_t, np.nan, dtype=np.float32)
                for dy, dx in offsets:
                    y0 = max(0, -dy)
                    y1 = fire_t.shape[0] - max(0, dy)
                    x0 = max(0, -dx)
                    x1 = fire_t.shape[1] - max(0, dx)

                    center = fire_t[y0:y1, x0:x1]
                    if not center.any():
                        continue

                    neighbor_conf_t = conf_t[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                    mask = center & (neighbor_conf_t < args.threshold)
                    if not mask.any():
                        continue

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

                    feats = np.stack(
                        [
                            tmp_n[valid],
                            wind_n[valid],
                            spfh_n[valid],
                            precip_n[valid],
                            wind_dot[valid],
                            conf_center[valid],
                            conf_neighbor[valid],
                        ],
                        axis=1,
                    )
                    logits = spread_result["model"].intercept_[0] + feats @ spread_result["model"].coef_.ravel()
                    probs = sigmoid(logits)
                    # write max probability for each neighbor cell
                    spread_slice = spread_map[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                    update = np.full_like(spread_slice, np.nan, dtype=np.float32)
                    update[valid] = probs
                    # max combine
                    existing = np.nan_to_num(spread_slice, nan=-1.0)
                    update_num = np.nan_to_num(update, nan=-1.0)
                    spread_map[y0 + dy : y1 + dy, x0 + dx : x1 + dx] = np.maximum(existing, update_num)

                prob_spread_maps.append({"time": time_str, "data": spread_map.tolist()})

            rtma_time_ptr += band_count

    # Build report
    report = {
        "threshold": args.threshold,
        "spread": {
            "samples": spread_result["samples"],
            "positives": spread_result["positives"],
            "auc": spread_result["auc"],
            "coefficients": spread_result["coefficients"],
        },
        "continue": {
            "samples": continue_result["samples"],
            "positives": continue_result["positives"],
            "auc": continue_result["auc"],
            "coefficients": continue_result["coefficients"],
        },
        "selected_hours": [aligned_times[i] for i in selected_hours],
        "feature_names_spread": feature_names_spread,
        "feature_names_continue": feature_names_continue,
    }

    report_path = os.path.join(out_dir, "regression_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    text_path = os.path.join(out_dir, "regression_report.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("Spread regression\n")
        f.write(json.dumps(report["spread"], indent=2))
        f.write("\n\nContinuation regression\n")
        f.write(json.dumps(report["continue"], indent=2))
        f.write("\n\nSelected hours\n")
        f.write("\n".join(report["selected_hours"]))

    if prob_spread_maps:
        prob_spread_path = os.path.join(out_dir, "probability_maps_spread.json")
        with open(prob_spread_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "grid_shape": goes_shape,
                        "geo_transform": goes_meta["geo_transform"],
                        "crs": goes_crs,
                        "threshold": args.threshold,
                        "model": "spread",
                    },
                    "data": prob_spread_maps,
                },
                f,
                indent=2,
            )

    if prob_continue_maps:
        prob_continue_path = os.path.join(out_dir, "probability_maps_continue.json")
        with open(prob_continue_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {
                        "grid_shape": goes_shape,
                        "geo_transform": goes_meta["geo_transform"],
                        "crs": goes_crs,
                        "threshold": args.threshold,
                        "model": "continue",
                    },
                    "data": prob_continue_maps,
                },
                f,
                indent=2,
            )

    print(f"Saved report: {report_path}")
    print(f"Saved report: {text_path}")
    if prob_spread_maps:
        print(f"Saved spread maps: {prob_spread_path}")
    if prob_continue_maps:
        print(f"Saved continue maps: {prob_continue_path}")


if __name__ == "__main__":
    main()

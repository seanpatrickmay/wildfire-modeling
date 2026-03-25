"""Data loader for wildfire-data-pipeline NPZ files.

Serves two consumers:
  1. FireSpreadNet v2 (full-grid model): (T, C, H, W) tensors
  2. XGBoost pixel model: flat feature vectors with 3x3 neighborhood context
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import numpy as np
from numpy import ndarray
from scipy.ndimage import distance_transform_edt


CHANNEL_ORDER: list[str] = [
    "confidence",
    "frp",
    "hourly_ugrd",
    "hourly_vgrd",
    "hourly_gust",
    "hourly_tmp",
    "hourly_dpt",
    "hourly_soil_moisture",
    "daily_erc",
    "daily_bi",
    "daily_fm100",
    "daily_fm1000",
    "daily_vpd",
    "static_slope_deg",
    "static_aspect_sin",
    "static_aspect_cos",
    "static_elevation",
    "static_tpi",
    "static_fuel_load",
    "static_canopy_cover_pct",
    "slow_NDVI",
    "slow_EVI",
    "temporal_hour_sin",
    "temporal_hour_cos",
    "temporal_doy_sin",
    "temporal_doy_cos",
    "validity",
]

WIND_U_CH: int = CHANNEL_ORDER.index("hourly_ugrd")
WIND_V_CH: int = CHANNEL_ORDER.index("hourly_vgrd")

_STATIC_KEYS = [
    "static_slope_deg",
    "static_aspect_sin",
    "static_aspect_cos",
    "static_elevation",
    "static_tpi",
    "static_fuel_load",
    "static_canopy_cover_pct",
]

_SLOW_KEYS = ["slow_NDVI", "slow_EVI"]

_HOURLY_KEYS = [
    "hourly_ugrd",
    "hourly_vgrd",
    "hourly_gust",
    "hourly_tmp",
    "hourly_dpt",
    "hourly_soil_moisture",
]

_DAILY_KEYS = [
    "daily_erc",
    "daily_bi",
    "daily_fm100",
    "daily_fm1000",
    "daily_vpd",
]

_TEMPORAL_KEYS = [
    "temporal_hour_sin",
    "temporal_hour_cos",
    "temporal_doy_sin",
    "temporal_doy_cos",
]


# ─── File I/O ────────────────────────────────────────────────────────────────

def load_fire_data(
    fire_name: str,
    pipeline_dir: str,
) -> tuple[dict, dict, dict, dict]:
    """Load both NPZ files for a fire.

    Returns (fire_arrays, feature_arrays, fire_metadata, feature_metadata).
    """
    base = Path(pipeline_dir) / fire_name
    conf_candidates = sorted(base.glob(f"{fire_name}_*_FusedConf.npz"))
    feat_candidates = sorted(base.glob(f"{fire_name}_*_Features.npz"))
    if not conf_candidates:
        raise FileNotFoundError(f"No FusedConf NPZ found in {base}")
    if not feat_candidates:
        raise FileNotFoundError(f"No Features NPZ found in {base}")
    conf_path = conf_candidates[0]
    feat_path = feat_candidates[0]

    conf_npz = np.load(str(conf_path), allow_pickle=True)
    feat_npz = np.load(str(feat_path), allow_pickle=True)

    fire_arrays: dict[str, ndarray] = {}
    for key in conf_npz.files:
        if key == "_metadata":
            continue
        fire_arrays[key] = conf_npz[key]

    feature_arrays: dict[str, ndarray] = {}
    for key in feat_npz.files:
        if key == "_metadata":
            continue
        feature_arrays[key] = feat_npz[key]

    fire_metadata = json.loads(conf_npz["_metadata"].item())
    feature_metadata = json.loads(feat_npz["_metadata"].item())

    return fire_arrays, feature_arrays, fire_metadata, feature_metadata


# ─── Label Smoothing ─────────────────────────────────────────────────────────

def smooth_labels(
    confidence: ndarray,
    cloud_mask: ndarray | None,
    observation_valid: ndarray | None,
    window: int = 5,
    min_votes: int = 2,
    threshold: float = 0.30,
) -> tuple[ndarray, ndarray]:
    """Cloud-aware temporal smoothing of fire labels.

    Returns (labels, validity_mask) both (T, H, W).
    """
    T, H, W = confidence.shape
    binary = (confidence >= threshold).astype(np.float32)

    if cloud_mask is not None and observation_valid is not None:
        validity_mask = (observation_valid.astype(np.float32)
                         * (1.0 - cloud_mask.astype(np.float32)))
    else:
        validity_mask = np.ones((T, H, W), dtype=np.float32)

    clear_fire = binary * validity_mask

    labels = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        start = max(0, t - window + 1)
        votes = clear_fire[start:t + 1].sum(axis=0)
        clear_hours = validity_mask[start:t + 1].sum(axis=0)
        if cloud_mask is not None and observation_valid is not None:
            labels[t] = (votes >= min_votes).astype(np.float32)
        else:
            labels[t] = (votes > clear_hours / 2.0).astype(np.float32)

    return labels, validity_mask


# ─── Channel Stack Building ──────────────────────────────────────────────────

def _get_array(
    fire_arrays: dict,
    feature_arrays: dict,
    key: str,
    T: int,
    H: int,
    W: int,
) -> ndarray:
    """Retrieve an array by channel key, broadcasting static/slow to (T, H, W)."""
    if key == "confidence":
        arr = fire_arrays.get("data")
    elif key == "frp":
        arr = fire_arrays.get("frp")
    else:
        arr = feature_arrays.get(key)

    if arr is None:
        return np.zeros((T, H, W), dtype=np.float32)

    arr = arr.astype(np.float32)

    if arr.ndim == 2:
        if arr.shape != (H, W):
            out = np.zeros((H, W), dtype=np.float32)
            h_copy = min(arr.shape[0], H)
            w_copy = min(arr.shape[1], W)
            out[:h_copy, :w_copy] = arr[:h_copy, :w_copy]
            arr = out
        arr = np.broadcast_to(arr[np.newaxis, :, :], (T, H, W)).copy()
    elif arr.ndim == 3:
        if arr.shape[1:] != (H, W):
            out = np.zeros((arr.shape[0], H, W), dtype=np.float32)
            h_copy = min(arr.shape[1], H)
            w_copy = min(arr.shape[2], W)
            out[:, :h_copy, :w_copy] = arr[:, :h_copy, :w_copy]
            arr = out
        if arr.shape[0] != T:
            if arr.shape[0] < T:
                padded = np.zeros((T, H, W), dtype=np.float32)
                padded[:arr.shape[0]] = arr
                arr = padded
            else:
                arr = arr[:T]

    return arr


def build_channel_stack(
    fire_arrays: dict,
    feature_arrays: dict,
    pad_h: int | None = None,
    pad_w: int | None = None,
) -> ndarray:
    """Assemble all features into a (T, C, H, W) float32 array."""
    conf = fire_arrays.get("data")
    if conf is None:
        raise ValueError("fire_arrays must contain 'data' (confidence)")

    T, H, W = conf.shape
    C = len(CHANNEL_ORDER)

    if pad_h is None:
        pad_h = ((H + 7) // 8) * 8
    if pad_w is None:
        pad_w = ((W + 7) // 8) * 8

    stack = np.zeros((T, C, pad_h, pad_w), dtype=np.float32)

    obs_valid = fire_arrays.get("observation_valid")
    cloud = fire_arrays.get("cloud_mask")

    for c_idx, key in enumerate(CHANNEL_ORDER):
        if key == "validity":
            if obs_valid is not None and cloud is not None:
                v = obs_valid.astype(np.float32) * (1.0 - cloud.astype(np.float32))
            elif obs_valid is not None:
                v = obs_valid.astype(np.float32)
            else:
                v = np.ones((T, H, W), dtype=np.float32)
            stack[:, c_idx, :H, :W] = np.nan_to_num(v, nan=0.0)
            continue

        arr = _get_array(fire_arrays, feature_arrays, key, T, H, W)
        stack[:, c_idx, :H, :W] = np.nan_to_num(arr, nan=0.0)

    return stack


# ─── Normalization ────────────────────────────────────────────────────────────

def compute_channel_stats(
    stacks: list[ndarray],
    validity_masks: list[ndarray],
) -> tuple[ndarray, ndarray]:
    """Compute per-channel mean and std from list of (T, C, H, W) stacks.

    Only count valid pixels (validity channel > 0).
    """
    C = stacks[0].shape[1]
    validity_ch = CHANNEL_ORDER.index("validity")

    sums = np.zeros(C, dtype=np.float64)
    sq_sums = np.zeros(C, dtype=np.float64)
    counts = np.zeros(C, dtype=np.float64)

    for stack, vmask in zip(stacks, validity_masks):
        if vmask.ndim == 3:
            valid = vmask > 0
        else:
            valid = stack[:, validity_ch, :, :] > 0

        for c in range(C):
            ch_data = stack[:, c, :, :]
            vals = ch_data[valid].astype(np.float64)
            sums[c] += vals.sum()
            sq_sums[c] += (vals ** 2).sum()
            counts[c] += vals.size

    counts = np.maximum(counts, 1)
    means = sums / counts
    stds = np.sqrt(np.maximum(sq_sums / counts - means ** 2, 0.0))
    stds = np.maximum(stds, 1e-8)

    return means.astype(np.float32), stds.astype(np.float32)


def normalize_stack(
    stack: ndarray,
    means: ndarray,
    stds: ndarray,
) -> ndarray:
    """Z-score normalize each channel. Don't normalize the validity channel."""
    out = stack.astype(np.float32, copy=True)
    validity_idx = len(CHANNEL_ORDER) - 1

    m = means.reshape(1, -1, 1, 1)
    s = stds.reshape(1, -1, 1, 1)

    out[:, :validity_idx] = (out[:, :validity_idx] - m[:, :validity_idx]) / s[:, :validity_idx]
    return out


# ─── FireSpreadNet Sequence Generator ─────────────────────────────────────────

def iter_grid_sequences(
    stack: ndarray,
    labels: ndarray,
    validity: ndarray,
    seq_len: int = 6,
) -> Generator[tuple[ndarray, ndarray, ndarray], None, None]:
    """Yield (frames, target, mask) for FireSpreadNet training.

    stack:    (T, C, H, W) normalized channel stack
    labels:   (T, H, W) smoothed binary labels
    validity: (T, H, W) validity mask
    """
    T = stack.shape[0]

    for t in range(seq_len - 1, T - 1):
        frames = stack[t - seq_len + 1: t + 1].astype(np.float32)
        target = labels[t + 1][np.newaxis].astype(np.float32)
        mask = validity[t + 1][np.newaxis].astype(np.float32)
        yield frames, target, mask


# ─── XGBoost Pixel Feature Extraction ─────────────────────────────────────────

def get_pixel_feature_names(seq_len: int = 6, neighborhood: int = 1) -> list[str]:
    """Return feature names matching the order of iter_pixel_samples output."""
    names: list[str] = []

    side = 2 * neighborhood + 1
    for t_offset in range(seq_len):
        for di in range(-neighborhood, neighborhood + 1):
            for dj in range(-neighborhood, neighborhood + 1):
                names.append(f"conf_t-{seq_len - 1 - t_offset}_di{di}_dj{dj}")

    names.append("frp_t0")

    for key in _STATIC_KEYS:
        short = key.replace("static_", "")
        names.append(short)

    for key in _HOURLY_KEYS:
        short = key.replace("hourly_", "")
        names.append(short)

    for key in _DAILY_KEYS:
        short = key.replace("daily_", "")
        names.append(short)

    for key in _SLOW_KEYS:
        short = key.replace("slow_", "")
        names.append(short)

    for key in _TEMPORAL_KEYS:
        short = key.replace("temporal_", "")
        names.append(short)

    names.append("dist_nearest_fire")
    names.append("frac_3x3_fire")

    return names


def _safe_get_2d(arr: ndarray | None, T: int, H: int, W: int) -> ndarray:
    """Get a (T, H, W) or (H, W) array, falling back to zeros. Handles spatial mismatches."""
    if arr is None:
        return np.zeros((T, H, W), dtype=np.float32)
    a = arr.astype(np.float32)
    if a.ndim == 2:
        # Pad/clip spatial dims to match (H, W), then broadcast to (T, H, W)
        out = np.zeros((H, W), dtype=np.float32)
        h_copy = min(a.shape[0], H)
        w_copy = min(a.shape[1], W)
        out[:h_copy, :w_copy] = a[:h_copy, :w_copy]
        a = np.broadcast_to(out[np.newaxis], (T, H, W)).copy()
    elif a.ndim == 3:
        # Pad/clip both temporal and spatial dims
        out = np.zeros((T, H, W), dtype=np.float32)
        t_copy = min(a.shape[0], T)
        h_copy = min(a.shape[1], H)
        w_copy = min(a.shape[2], W)
        out[:t_copy, :h_copy, :w_copy] = a[:t_copy, :h_copy, :w_copy]
        a = out
    return np.nan_to_num(a, nan=0.0)


def iter_pixel_samples(
    fire_arrays: dict,
    feature_arrays: dict,
    labels: ndarray,
    validity: ndarray,
    seq_len: int = 6,
    neighborhood: int = 1,
    label_gap: int = 0,
) -> Generator[tuple[ndarray, float], None, None]:
    """Yield (feature_vector, label) for XGBoost training.

    Parameters
    ----------
    label_gap : int
        Number of timesteps to shift fire-detection features back to avoid
        temporal overlap with the smoothing window used to create ``labels``.
        Set to ``smooth_window - 1`` (e.g. 4 for window=5) to ensure zero
        overlap between confidence-derived features and the label's voting
        window.  Weather/terrain features stay at time ``t`` since they don't
        feed into the label computation.
    """
    conf = fire_arrays.get("data")
    if conf is None:
        return
    T, H, W = conf.shape
    conf = np.nan_to_num(conf.astype(np.float32), nan=0.0)

    frp = _safe_get_2d(fire_arrays.get("frp"), T, H, W)

    static_arrs = [_safe_get_2d(feature_arrays.get(k), T, H, W) for k in _STATIC_KEYS]
    hourly_arrs = [_safe_get_2d(feature_arrays.get(k), T, H, W) for k in _HOURLY_KEYS]
    daily_arrs = [_safe_get_2d(feature_arrays.get(k), T, H, W) for k in _DAILY_KEYS]
    slow_arrs = [_safe_get_2d(feature_arrays.get(k), T, H, W) for k in _SLOW_KEYS]
    temporal_arrs = [_safe_get_2d(feature_arrays.get(k), T, H, W) for k in _TEMPORAL_KEYS]

    side = 2 * neighborhood + 1
    n_neigh = side * side
    n_conf_feat = n_neigh * seq_len
    n_total = n_conf_feat + 1 + len(_STATIC_KEYS) + len(_HOURLY_KEYS) + len(_DAILY_KEYS) + len(_SLOW_KEYS) + len(_TEMPORAL_KEYS) + 2

    conf_padded = np.pad(conf, ((0, 0), (neighborhood, neighborhood), (neighborhood, neighborhood)),
                         mode="constant", constant_values=0.0)

    threshold = 0.30

    for t in range(seq_len - 1 + label_gap, T - 1):
        target_valid = validity[t + 1]
        target_labels = labels[t + 1]

        # Fire-detection features use t_fire to avoid overlap with the
        # smoothing window of labels[t+1].
        t_fire = t - label_gap

        fire_binary_t = (conf[t_fire] >= threshold)
        if fire_binary_t.any():
            inv_fire = ~fire_binary_t
            dist_map = distance_transform_edt(inv_fire).astype(np.float32)
        else:
            dist_map = None

        fire_frac = np.zeros((H, W), dtype=np.float32)
        conf_t_padded = np.pad(conf[t_fire], ((neighborhood, neighborhood), (neighborhood, neighborhood)),
                               mode="constant", constant_values=0.0)
        for di in range(-neighborhood, neighborhood + 1):
            for dj in range(-neighborhood, neighborhood + 1):
                si = neighborhood + di
                sj = neighborhood + dj
                fire_frac += (conf_t_padded[si:si + H, sj:sj + W] >= threshold).astype(np.float32)
        fire_frac /= n_neigh

        valid_i, valid_j = np.where(target_valid > 0)

        for idx in range(len(valid_i)):
            i = valid_i[idx]
            j = valid_j[idx]

            fv = np.empty(n_total, dtype=np.float32)
            pos = 0

            # Confidence neighborhood patches: shifted back by label_gap
            for t_off in range(seq_len):
                t_src = t_fire - seq_len + 1 + t_off
                if t_src < 0:
                    fv[pos:pos + n_neigh] = 0.0
                else:
                    pi = i + neighborhood
                    pj = j + neighborhood
                    patch = conf_padded[t_src, pi - neighborhood:pi + neighborhood + 1,
                                        pj - neighborhood:pj + neighborhood + 1]
                    fv[pos:pos + n_neigh] = patch.ravel()
                pos += n_neigh

            # FRP also shifted (same GOES observation as confidence)
            fv[pos] = frp[t_fire, i, j]
            pos += 1

            for arr in static_arrs:
                fv[pos] = arr[0, i, j]
                pos += 1

            # Weather features stay at time t (independent of label smoothing)
            for arr in hourly_arrs:
                fv[pos] = arr[t, i, j]
                pos += 1

            for arr in daily_arrs:
                fv[pos] = arr[t, i, j]
                pos += 1

            for arr in slow_arrs:
                fv[pos] = arr[0, i, j]
                pos += 1

            for arr in temporal_arrs:
                fv[pos] = arr[t, i, j]
                pos += 1

            if dist_map is not None:
                fv[pos] = dist_map[i, j]
            else:
                fv[pos] = -1.0
            pos += 1

            fv[pos] = fire_frac[i, j]
            pos += 1

            yield fv, float(target_labels[i, j])


# ─── Class Imbalance ──────────────────────────────────────────────────────────

def subsample_negatives(
    X: ndarray,
    y: ndarray,
    ratio: float = 5.0,
    seed: int = 42,
) -> tuple[ndarray, ndarray]:
    """Subsample negative class to achieve target neg:pos ratio."""
    pos_mask = y > 0
    neg_mask = ~pos_mask

    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        return X, y

    n_neg_target = int(n_pos * ratio)
    n_neg_actual = int(neg_mask.sum())

    if n_neg_actual <= n_neg_target:
        return X, y

    rng = np.random.RandomState(seed)
    neg_indices = np.where(neg_mask)[0]
    keep_neg = rng.choice(neg_indices, size=n_neg_target, replace=False)

    pos_indices = np.where(pos_mask)[0]
    keep = np.concatenate([pos_indices, keep_neg])
    keep.sort()

    return X[keep], y[keep]

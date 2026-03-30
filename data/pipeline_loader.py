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


# ═══════════════════════════════════════════════════════════════════════════════
# V3: Pipeline-processed data (prev_fire_state approach — no leakage)
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_ORDER_V3: list[str] = [
    # Pre-shifted fire features (safe as input: prev_fire_state[t] = labels[t-1])
    "prev_fire_state",
    "prev_distance_to_fire",
    "prev_fire_neighborhood",
    # Hourly weather (RTMA)
    "hourly_ugrd",
    "hourly_vgrd",
    "hourly_gust",
    "hourly_tmp",
    "hourly_dpt",
    "hourly_soil_moisture",
    # Daily fire weather (GRIDMET)
    "daily_erc",
    "daily_bi",
    "daily_fm100",
    "daily_fm1000",
    "daily_vpd",
    # Static terrain
    "static_slope_deg",
    "static_aspect_sin",
    "static_aspect_cos",
    "static_elevation",
    "static_tpi",
    "static_fuel_load",
    "static_canopy_cover_pct",
    # Vegetation
    "slow_NDVI",
    "slow_EVI",
    # Temporal encoding
    "temporal_hour_sin",
    "temporal_hour_cos",
    "temporal_doy_sin",
    "temporal_doy_cos",
    # Quality mask
    "validity",
]

WIND_U_CH_V3: int = CHANNEL_ORDER_V3.index("hourly_ugrd")
WIND_V_CH_V3: int = CHANNEL_ORDER_V3.index("hourly_vgrd")


def load_processed_fire_data(
    fire_name: str,
    pipeline_dir: str,
) -> tuple[dict, dict]:
    """Load processed fire labels + feature arrays for the V3 pipeline format.

    Returns (processed_arrays, feature_arrays).
    processed_arrays keys: labels, validity, prev_fire_state, prev_distance_to_fire,
                          prev_fire_neighborhood, loss_weights, fire_change, ...
    feature_arrays keys: hourly_*, daily_*, static_*, slow_*, temporal_*
    """
    base = Path(pipeline_dir) / fire_name

    proc_dir = base / "processed"
    proc_candidates = sorted(proc_dir.glob(f"{fire_name}_*_processed.npz"))
    if not proc_candidates:
        raise FileNotFoundError(f"No processed NPZ in {proc_dir}")
    feat_candidates = sorted(base.glob(f"{fire_name}_*_Features.npz"))
    if not feat_candidates:
        raise FileNotFoundError(f"No Features NPZ in {base}")

    proc_npz = np.load(str(proc_candidates[0]), allow_pickle=True)
    feat_npz = np.load(str(feat_candidates[0]), allow_pickle=True)

    processed: dict[str, ndarray] = {}
    for key in proc_npz.files:
        if key.startswith("_"):
            continue
        processed[key] = proc_npz[key]

    features: dict[str, ndarray] = {}
    for key in feat_npz.files:
        if key == "_metadata":
            continue
        features[key] = feat_npz[key]

    return processed, features


def build_channel_stack_v3(
    processed: dict,
    features: dict,
    pad_h: int | None = None,
    pad_w: int | None = None,
) -> ndarray:
    """Build (T, C, H, W) stack from processed + feature arrays (V3 format).

    Uses prev_fire_state instead of raw confidence — no temporal leakage.
    """
    labels = processed["labels"]
    T, H, W = labels.shape

    if pad_h is None:
        pad_h = max(32, ((H + 15) // 16) * 16)
    if pad_w is None:
        pad_w = max(48, ((W + 15) // 16) * 16)

    C = len(CHANNEL_ORDER_V3)
    stack = np.zeros((T, C, pad_h, pad_w), dtype=np.float32)

    _PROC_KEYS = ["prev_fire_state", "prev_distance_to_fire", "prev_fire_neighborhood", "validity"]

    for c, key in enumerate(CHANNEL_ORDER_V3):
        if key in _PROC_KEYS:
            arr = processed.get(key)
        else:
            arr = features.get(key)

        if arr is None:
            continue

        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)

        if arr.ndim == 2:
            # Static (H, W) → broadcast
            h_clip = min(arr.shape[0], H, pad_h)
            w_clip = min(arr.shape[1], W, pad_w)
            stack[:, c, :h_clip, :w_clip] = arr[:h_clip, :w_clip]
        elif arr.ndim == 3:
            t_clip = min(arr.shape[0], T)
            h_clip = min(arr.shape[1], H, pad_h)
            w_clip = min(arr.shape[2], W, pad_w)
            stack[:t_clip, c, :h_clip, :w_clip] = arr[:t_clip, :h_clip, :w_clip]

    return stack


def iter_grid_sequences_v3(
    stack: ndarray,
    labels: ndarray,
    validity: ndarray,
    loss_weights: ndarray | None = None,
    seq_len: int = 6,
) -> Generator[tuple[ndarray, ndarray, ndarray, ndarray], None, None]:
    """Yield (frames, target, mask, weights) for FireSpreadNet training (V3).

    With pre-shifted fire features, we predict labels[t] from input at time t.
    The stack already contains prev_fire_state[t] = labels[t-1].
    """
    T = stack.shape[0]

    for t in range(seq_len - 1, T):
        frames = stack[t - seq_len + 1: t + 1].astype(np.float32)
        target = labels[t][np.newaxis].astype(np.float32)
        mask = validity[t][np.newaxis].astype(np.float32)
        w = loss_weights[t][np.newaxis].astype(np.float32) if loss_weights is not None else mask
        yield frames, target, mask, w


# ─── V3 XGBoost Pixel Features ───────────────────────────────────────────────

_PIXEL_V3_STATIC_KEYS = _STATIC_KEYS
_PIXEL_V3_HOURLY_KEYS = _HOURLY_KEYS
_PIXEL_V3_DAILY_KEYS = _DAILY_KEYS
_PIXEL_V3_SLOW_KEYS = _SLOW_KEYS
_PIXEL_V3_TEMPORAL_KEYS = _TEMPORAL_KEYS


def get_pixel_feature_names_v3(seq_len: int = 6, neighborhood: int = 1) -> list[str]:
    """Feature names for V3 iter_pixel_samples_v3 output."""
    names: list[str] = []

    side = 2 * neighborhood + 1
    for t_offset in range(seq_len):
        for di in range(-neighborhood, neighborhood + 1):
            for dj in range(-neighborhood, neighborhood + 1):
                names.append(f"prev_fire_t-{seq_len - 1 - t_offset}_di{di}_dj{dj}")

    names.append("prev_distance_to_fire")
    names.append("prev_fire_neighborhood")

    for key in _PIXEL_V3_STATIC_KEYS:
        names.append(key.replace("static_", ""))

    for key in _PIXEL_V3_HOURLY_KEYS:
        names.append(key.replace("hourly_", ""))

    for key in _PIXEL_V3_DAILY_KEYS:
        names.append(key.replace("daily_", ""))

    for key in _PIXEL_V3_SLOW_KEYS:
        names.append(key.replace("slow_", ""))

    for key in _PIXEL_V3_TEMPORAL_KEYS:
        names.append(key.replace("temporal_", ""))

    return names


def iter_pixel_samples_v3(
    processed: dict,
    features: dict,
    seq_len: int = 6,
    neighborhood: int = 1,
) -> Generator[tuple[ndarray, float, float], None, None]:
    """Yield (feature_vector, label, loss_weight) for XGBoost (V3 format).

    Uses prev_fire_state as the fire detection feature (pre-shifted by pipeline).
    No temporal contamination — prev_fire_state[t] = labels[t-1] by construction.
    Target is labels[t].
    """
    labels = processed["labels"]
    validity = processed["validity"]
    prev_fire = processed["prev_fire_state"]
    prev_dist = processed["prev_distance_to_fire"]
    prev_neigh = processed["prev_fire_neighborhood"]
    loss_weights = processed.get("loss_weights")

    T, H, W = labels.shape

    static_arrs = [_safe_get_2d(features.get(k), T, H, W) for k in _PIXEL_V3_STATIC_KEYS]
    hourly_arrs = [_safe_get_2d(features.get(k), T, H, W) for k in _PIXEL_V3_HOURLY_KEYS]
    daily_arrs = [_safe_get_2d(features.get(k), T, H, W) for k in _PIXEL_V3_DAILY_KEYS]
    slow_arrs = [_safe_get_2d(features.get(k), T, H, W) for k in _PIXEL_V3_SLOW_KEYS]
    temporal_arrs = [_safe_get_2d(features.get(k), T, H, W) for k in _PIXEL_V3_TEMPORAL_KEYS]

    side = 2 * neighborhood + 1
    n_neigh = side * side
    n_conf_feat = n_neigh * seq_len
    n_total = (n_conf_feat + 2
               + len(_PIXEL_V3_STATIC_KEYS) + len(_PIXEL_V3_HOURLY_KEYS)
               + len(_PIXEL_V3_DAILY_KEYS) + len(_PIXEL_V3_SLOW_KEYS)
               + len(_PIXEL_V3_TEMPORAL_KEYS))

    prev_fire_padded = np.pad(
        prev_fire, ((0, 0), (neighborhood, neighborhood), (neighborhood, neighborhood)),
        mode="constant", constant_values=0.0,
    )

    for t in range(seq_len - 1, T):
        target_valid = validity[t]
        target_labels = labels[t]
        target_weight = loss_weights[t] if loss_weights is not None else target_valid

        valid_i, valid_j = np.where(target_valid > 0)

        for idx in range(len(valid_i)):
            i = valid_i[idx]
            j = valid_j[idx]

            fv = np.empty(n_total, dtype=np.float32)
            pos = 0

            # prev_fire_state 3x3 neighborhood over seq_len hours
            for t_off in range(seq_len):
                t_src = t - seq_len + 1 + t_off
                if t_src < 0:
                    fv[pos:pos + n_neigh] = 0.0
                else:
                    pi = i + neighborhood
                    pj = j + neighborhood
                    patch = prev_fire_padded[t_src, pi - neighborhood:pi + neighborhood + 1,
                                             pj - neighborhood:pj + neighborhood + 1]
                    fv[pos:pos + n_neigh] = patch.ravel()
                pos += n_neigh

            # Spatial fire context from previous timestep
            fv[pos] = prev_dist[t, i, j]
            pos += 1
            fv[pos] = prev_neigh[t, i, j]
            pos += 1

            # Static terrain
            for arr in static_arrs:
                fv[pos] = arr[0, i, j]
                pos += 1

            # Weather at time t (no leakage — weather doesn't feed into labels)
            for arr in hourly_arrs:
                fv[pos] = arr[t, i, j]
                pos += 1
            for arr in daily_arrs:
                fv[pos] = arr[t, i, j]
                pos += 1

            # Vegetation
            for arr in slow_arrs:
                fv[pos] = arr[0, i, j]
                pos += 1

            # Temporal encoding
            for arr in temporal_arrs:
                fv[pos] = arr[t, i, j]
                pos += 1

            yield fv, float(target_labels[i, j]), float(target_weight[i, j])

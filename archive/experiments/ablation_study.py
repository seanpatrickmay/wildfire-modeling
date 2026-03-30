"""Ablation study for wildfire prediction models.

Systematically removes feature groups, varies temporal window, and compares
label quality to isolate the contribution of each component.

Experiments:
  A. FireSpreadNet channel group ablation (6 configs)
  B. FireSpreadNet temporal window ablation (seq_len = 1, 3, 6, 12)
  C. FireSpreadNet label quality (smoothed vs raw)
  D. XGBoost feature group ablation (6 configs)
  E. XGBoost temporal window ablation (seq_len = 1, 3, 6, 12)
  F. XGBoost label quality (smoothed vs raw)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fire_spread_model_v2 import (
    CombinedLoss,
    FireSpreadNetV2,
    augment_sequence_v2,
)
from scripts.pipeline_data_loader import (
    CHANNEL_ORDER,
    build_channel_stack,
    compute_channel_stats,
    iter_grid_sequences,
    iter_pixel_samples,
    get_pixel_feature_names,
    load_fire_data,
    normalize_stack,
    smooth_labels,
    subsample_negatives,
)

# ─── Config ─────────────────────────────────────────────────────────────────
_CANDIDATE_PIPELINE_DIRS = [
    REPO_ROOT.parent / "wildfire-data-pipeline" / "data",
    Path.home() / "Desktop" / "Current Projects" / "wildfire-data-pipeline" / "data",
]
PIPELINE_DIR: Path | None = None
for _c in _CANDIDATE_PIPELINE_DIRS:
    if _c.resolve().is_dir():
        PIPELINE_DIR = _c.resolve()
        break

CONFIDENCE_THRESHOLD = 0.30
SMOOTH_WINDOW = 5
SMOOTH_MIN_VOTES = 2
MIN_FIRE_PIXELS = 50
TRAIN_FRACTION = 0.7
SPLIT_SEED = 42

# Reduced epochs for ablation (full training used 20)
FSN_EPOCHS = 10
FSN_BATCH_SIZE = 4
FSN_LR = 3e-4
FSN_WEIGHT_DECAY = 1e-4
FSN_GRAD_CLIP = 1.0

XGB_PARAMS = {
    "max_depth": 8, "n_estimators": 300, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "eval_metric": "logloss", "tree_method": "hist", "random_state": 42,
}
NEG_POS_RATIO = 5.0

ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "ablation"

# ─── Channel group definitions ───────────────────────────────────────────────
CHANNEL_GROUPS = {
    "fire_only": ["confidence", "validity"],
    "fire_frp": ["confidence", "frp", "validity"],
    "fire_hourly_weather": [
        "confidence", "frp",
        "hourly_ugrd", "hourly_vgrd", "hourly_gust", "hourly_tmp", "hourly_dpt", "hourly_soil_moisture",
        "validity",
    ],
    "fire_terrain": [
        "confidence", "frp",
        "static_slope_deg", "static_aspect_sin", "static_aspect_cos", "static_elevation", "static_tpi",
        "static_fuel_load", "static_canopy_cover_pct",
        "validity",
    ],
    "fire_all_weather": [
        "confidence", "frp",
        "hourly_ugrd", "hourly_vgrd", "hourly_gust", "hourly_tmp", "hourly_dpt", "hourly_soil_moisture",
        "daily_erc", "daily_bi", "daily_fm100", "daily_fm1000", "daily_vpd",
        "temporal_hour_sin", "temporal_hour_cos", "temporal_doy_sin", "temporal_doy_cos",
        "validity",
    ],
    "full": list(CHANNEL_ORDER),
}

# XGBoost feature group masks (indices into the 81-feature vector)
# Feature layout: 54 neighborhood conf + 1 FRP + 7 static + 6 hourly + 5 daily + 2 slow + 4 temporal + 2 derived = 81
_XGB_GROUPS = {
    "fire_only":          list(range(0, 54)) + [80],                    # neighbor conf + fire_frac
    "fire_frp":           list(range(0, 55)) + [80],                    # + FRP
    "fire_hourly_weather": list(range(0, 55)) + list(range(62, 68)) + [79, 80],  # + hourly weather + derived
    "fire_terrain":       list(range(0, 55)) + list(range(55, 62)) + [79, 80],   # + static
    "fire_all_weather":   list(range(0, 55)) + list(range(62, 79)) + [79, 80],   # + hourly + daily + slow + temporal
    "full":               list(range(81)),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(i) for i in obj]
    return obj


# ─── Fire discovery + split (shared) ────────────────────────────────────────
def discover_and_split(pipeline_dir: Path):
    """Discover fires, filter by pixel count, split into train/test."""
    fires = []
    for subdir in sorted(pipeline_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if list(subdir.glob(f"{name}_*_FusedConf.npz")) and list(subdir.glob(f"{name}_*_Features.npz")):
            fires.append(name)

    usable = []
    fire_data_cache: dict[str, tuple] = {}
    for name in fires:
        fa, feat, fm, featm = load_fire_data(name, str(pipeline_dir))
        labels, validity = smooth_labels(
            fa["data"], fa.get("cloud_mask"), fa.get("observation_valid"),
            window=SMOOTH_WINDOW, min_votes=SMOOTH_MIN_VOTES, threshold=CONFIDENCE_THRESHOLD,
        )
        n_fire = int(labels.sum())
        if n_fire >= MIN_FIRE_PIXELS:
            usable.append(name)
            fire_data_cache[name] = (fa, feat, fm, featm, labels, validity)
            log(f"  {name}: {fa['data'].shape[0]}h, {n_fire} fire pixels")
        else:
            log(f"  {name}: SKIP ({n_fire} fire pixels)")

    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    n_train = max(1, min(int(len(shuffled) * TRAIN_FRACTION), len(shuffled) - 1))
    train_names = sorted(shuffled[:n_train])
    test_names = sorted(shuffled[n_train:])

    return train_names, test_names, fire_data_cache


# ─── FireSpreadNet helpers ───────────────────────────────────────────────────
def compute_metrics_fsn(logits, targets, mask):
    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= 0.5).float()
        valid = mask.bool()
        p = preds[valid].int()
        t = targets[valid].int()
        tp = int(((p == 1) & (t == 1)).sum().item())
        fp = int(((p == 1) & (t == 0)).sum().item())
        fn = int(((p == 0) & (t == 1)).sum().item())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def run_fsn_experiment(
    name: str,
    channel_subset: list[str],
    seq_len: int,
    train_stacks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    test_stacks: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate FireSpreadNet with given channel subset and seq_len."""
    n_ch = len(channel_subset)
    log(f"  FSN experiment: {name} ({n_ch}ch, seq={seq_len})")

    # Extract channel indices from full stacks
    ch_indices = [CHANNEL_ORDER.index(c) for c in channel_subset]

    # Slice channels from preloaded stacks
    train_data = [(s[:, ch_indices], l, v) for s, l, v in train_stacks]
    test_data = [(s[:, ch_indices], l, v) for s, l, v in test_stacks]

    # Compute channel stats from training data
    means, stds = compute_channel_stats(
        [s for s, _, _ in train_data],
        [v for _, _, v in train_data],
    )

    # Normalize
    train_data = [(normalize_stack(s, means, stds), l, v) for s, l, v in train_data]
    test_data = [(normalize_stack(s, means, stds), l, v) for s, l, v in test_data]

    # Find wind channels in subset (if present)
    wind_u_idx = channel_subset.index("hourly_ugrd") if "hourly_ugrd" in channel_subset else -1
    wind_v_idx = channel_subset.index("hourly_vgrd") if "hourly_vgrd" in channel_subset else -1

    model = FireSpreadNetV2(in_channels=n_ch, dropout=0.1).to(device)
    criterion = CombinedLoss(focal_weight=0.5, tversky_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FSN_LR, weight_decay=FSN_WEIGHT_DECAY)

    best_f1 = 0.0
    rng = torch.Generator()

    for epoch in range(1, FSN_EPOCHS + 1):
        # Train
        model.train()
        rng.manual_seed(epoch * 1337)
        batch_f, batch_t, batch_m = [], [], []
        fire_order = list(range(len(train_data)))
        np.random.shuffle(fire_order)

        for fi in fire_order:
            stack, labels, validity = train_data[fi]
            for frames, target, mask in iter_grid_sequences(stack, labels, validity, seq_len):
                f_t = torch.from_numpy(frames)
                t_t = torch.from_numpy(target)
                m_t = torch.from_numpy(mask)

                # Augment with wind correction if wind channels present
                if torch.rand(1, generator=rng).item() > 0.5:
                    f_t = f_t.flip(-1)
                    t_t = t_t.flip(-1)
                    m_t = m_t.flip(-1)
                    if wind_u_idx >= 0:
                        f_t[:, wind_u_idx] = -f_t[:, wind_u_idx]
                if torch.rand(1, generator=rng).item() > 0.5:
                    f_t = f_t.flip(-2)
                    t_t = t_t.flip(-2)
                    m_t = m_t.flip(-2)
                    if wind_v_idx >= 0:
                        f_t[:, wind_v_idx] = -f_t[:, wind_v_idx]

                batch_f.append(f_t)
                batch_t.append(t_t)
                batch_m.append(m_t)

                if len(batch_f) >= FSN_BATCH_SIZE:
                    x = torch.stack(batch_f).to(device)
                    y = torch.stack(batch_t).to(device)
                    m = torch.stack(batch_m).to(device)
                    loss = criterion(model(x), y, m)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), FSN_GRAD_CLIP)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    batch_f, batch_t, batch_m = [], [], []

        if batch_f:
            x = torch.stack(batch_f).to(device)
            y = torch.stack(batch_t).to(device)
            m = torch.stack(batch_m).to(device)
            loss = criterion(model(x), y, m)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), FSN_GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Eval
        model.eval()
        agg = {"tp": 0, "fp": 0, "fn": 0}
        with torch.no_grad():
            for stack, labels, validity in test_data:
                batch_f, batch_t, batch_m = [], [], []
                for frames, target, mask in iter_grid_sequences(stack, labels, validity, seq_len):
                    batch_f.append(torch.from_numpy(frames))
                    batch_t.append(torch.from_numpy(target))
                    batch_m.append(torch.from_numpy(mask))
                    if len(batch_f) >= FSN_BATCH_SIZE:
                        x = torch.stack(batch_f).to(device)
                        y = torch.stack(batch_t).to(device)
                        m = torch.stack(batch_m).to(device)
                        mb = compute_metrics_fsn(model(x), y, m)
                        for k in ("tp", "fp", "fn"):
                            agg[k] += mb[k]
                        batch_f, batch_t, batch_m = [], [], []
                if batch_f:
                    x = torch.stack(batch_f).to(device)
                    y = torch.stack(batch_t).to(device)
                    m = torch.stack(batch_m).to(device)
                    mb = compute_metrics_fsn(model(x), y, m)
                    for k in ("tp", "fp", "fn"):
                        agg[k] += mb[k]

        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_prec = prec
            best_rec = rec
            best_epoch = epoch

    log(f"    -> F1={best_f1:.4f} P={best_prec:.4f} R={best_rec:.4f} (epoch {best_epoch})")
    return {
        "name": name, "model": "FireSpreadNet", "channels": n_ch,
        "channel_names": channel_subset, "seq_len": seq_len,
        "f1": best_f1, "precision": best_prec, "recall": best_rec, "best_epoch": best_epoch,
    }


# ─── XGBoost helpers ─────────────────────────────────────────────────────────
def run_xgb_experiment(
    name: str,
    feature_mask: list[int] | None,
    seq_len: int,
    train_fire_data: dict[str, tuple],
    test_fire_data: dict[str, tuple],
    use_smoothed: bool = True,
) -> dict[str, Any]:
    """Train and evaluate XGBoost with given feature subset and seq_len."""
    import xgboost as xgb
    from sklearn.metrics import f1_score, precision_score, recall_score

    log(f"  XGB experiment: {name} (seq={seq_len})")

    # Collect samples
    X_train_parts, y_train_parts = [], []
    for fname, (fa, feat, fm, featm, labels_smooth, validity_smooth) in train_fire_data.items():
        if use_smoothed:
            labels, validity = labels_smooth, validity_smooth
        else:
            labels = (fa["data"] >= CONFIDENCE_THRESHOLD).astype(np.float32)
            validity = np.ones_like(labels)
        for fv, label in iter_pixel_samples(fa, feat, labels, validity, seq_len=seq_len):
            X_train_parts.append(fv)
            y_train_parts.append(label)

    X_test_parts, y_test_parts = [], []
    for fname, (fa, feat, fm, featm, labels_smooth, validity_smooth) in test_fire_data.items():
        if use_smoothed:
            labels, validity = labels_smooth, validity_smooth
        else:
            labels = (fa["data"] >= CONFIDENCE_THRESHOLD).astype(np.float32)
            validity = np.ones_like(labels)
        for fv, label in iter_pixel_samples(fa, feat, labels, validity, seq_len=seq_len):
            X_test_parts.append(fv)
            y_test_parts.append(label)

    if not X_train_parts or not X_test_parts:
        log(f"    -> NO SAMPLES")
        return {"name": name, "model": "XGBoost", "f1": 0.0, "precision": 0.0, "recall": 0.0}

    X_train = np.array(X_train_parts, dtype=np.float32)
    y_train = np.array(y_train_parts, dtype=np.float32)
    X_test = np.array(X_test_parts, dtype=np.float32)
    y_test = np.array(y_test_parts, dtype=np.float32)

    # Apply feature mask
    if feature_mask is not None:
        mask = [i for i in feature_mask if i < X_train.shape[1]]
        X_train = X_train[:, mask]
        X_test = X_test[:, mask]

    # Subsample
    X_train_sub, y_train_sub = subsample_negatives(X_train, y_train, ratio=NEG_POS_RATIO, seed=42)
    n_pos = int((y_train_sub > 0).sum())
    n_neg = len(y_train_sub) - n_pos
    spw = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw, verbosity=0)
    model.fit(X_train_sub, y_train_sub, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))

    n_features = X_train.shape[1] if feature_mask is None else len(mask)
    log(f"    -> F1={f1:.4f} P={prec:.4f} R={rec:.4f} ({n_features} features, {len(y_train_sub)} train)")
    return {
        "name": name, "model": "XGBoost", "n_features": n_features,
        "seq_len": seq_len, "f1": f1, "precision": prec, "recall": rec,
        "train_samples": len(y_train_sub), "test_samples": len(y_test),
    }


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    if PIPELINE_DIR is None:
        log("ERROR: Pipeline data directory not found")
        sys.exit(1)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log(f"Device: {device}")
    log(f"Pipeline: {PIPELINE_DIR}")

    # Discover and split fires
    log("Discovering fires...")
    train_names, test_names, cache = discover_and_split(PIPELINE_DIR)
    log(f"Train: {train_names}")
    log(f"Test:  {test_names}")

    # Preload FireSpreadNet stacks (full 27 channels)
    max_h, max_w = 0, 0
    for name in train_names + test_names:
        fa = cache[name][0]
        _, h, w = fa["data"].shape
        max_h, max_w = max(max_h, h), max(max_w, w)
    pad_h = ((max_h + 15) // 16) * 16
    pad_w = ((max_w + 15) // 16) * 16
    log(f"Pad dims: {pad_h}x{pad_w}")

    train_stacks_full, test_stacks_full = [], []
    for name in train_names:
        fa, feat, _, _, labels, validity = cache[name]
        stack = build_channel_stack(fa, feat, pad_h, pad_w)
        T, H_data, W_data = fa["data"].shape[0], fa["data"].shape[1], fa["data"].shape[2]
        lab = np.zeros((T, pad_h, pad_w), dtype=np.float32)
        lab[:, :H_data, :W_data] = labels[:T]
        val = np.zeros((T, pad_h, pad_w), dtype=np.float32)
        val[:, :H_data, :W_data] = validity[:T]
        train_stacks_full.append((stack, lab, val))

    for name in test_names:
        fa, feat, _, _, labels, validity = cache[name]
        stack = build_channel_stack(fa, feat, pad_h, pad_w)
        T, H_data, W_data = fa["data"].shape[0], fa["data"].shape[1], fa["data"].shape[2]
        lab = np.zeros((T, pad_h, pad_w), dtype=np.float32)
        lab[:, :H_data, :W_data] = labels[:T]
        val = np.zeros((T, pad_h, pad_w), dtype=np.float32)
        val[:, :H_data, :W_data] = validity[:T]
        test_stacks_full.append((stack, lab, val))

    # Prepare XGBoost fire data dicts
    train_xgb = {n: cache[n] for n in train_names}
    test_xgb = {n: cache[n] for n in test_names}

    all_results: list[dict] = []

    # ═══ A. FireSpreadNet Channel Group Ablation ═══════════════════════════
    log("\n" + "=" * 70)
    log("A. FIRESPREADNET CHANNEL GROUP ABLATION (seq_len=6)")
    log("=" * 70)
    for group_name, channels in CHANNEL_GROUPS.items():
        result = run_fsn_experiment(
            f"fsn_channels_{group_name}", channels, 6,
            train_stacks_full, test_stacks_full, device,
        )
        result["experiment"] = "A_channel_group"
        result["group"] = group_name
        all_results.append(result)
        # Save incrementally
        with open(ANALYSIS_DIR / "ablation_results.json", "w") as f:
            json.dump(_serializable(all_results), f, indent=2)

    # ═══ B. FireSpreadNet Temporal Window Ablation ═════════════════════════
    log("\n" + "=" * 70)
    log("B. FIRESPREADNET TEMPORAL WINDOW ABLATION (full channels)")
    log("=" * 70)
    for seq_len in [1, 3, 6, 12]:
        result = run_fsn_experiment(
            f"fsn_seq{seq_len}", list(CHANNEL_ORDER), seq_len,
            train_stacks_full, test_stacks_full, device,
        )
        result["experiment"] = "B_temporal_window"
        all_results.append(result)
        with open(ANALYSIS_DIR / "ablation_results.json", "w") as f:
            json.dump(_serializable(all_results), f, indent=2)

    # ═══ C. FireSpreadNet Label Quality ════════════════════════════════════
    log("\n" + "=" * 70)
    log("C. FIRESPREADNET LABEL QUALITY (full channels, seq=6)")
    log("=" * 70)
    # Raw labels: just threshold confidence, no smoothing
    train_stacks_raw, test_stacks_raw = [], []
    for name, (stack, _, _) in zip(train_names, train_stacks_full):
        fa = cache[name][0]
        T, H_data, W_data = fa["data"].shape
        raw_labels = (fa["data"] >= CONFIDENCE_THRESHOLD).astype(np.float32)
        lab = np.zeros((T, pad_h, pad_w), dtype=np.float32)
        lab[:, :H_data, :W_data] = raw_labels
        val = np.ones((T, pad_h, pad_w), dtype=np.float32)
        val[:, H_data:, :] = 0
        val[:, :, W_data:] = 0
        train_stacks_raw.append((stack, lab, val))
    for name, (stack, _, _) in zip(test_names, test_stacks_full):
        fa = cache[name][0]
        T, H_data, W_data = fa["data"].shape
        raw_labels = (fa["data"] >= CONFIDENCE_THRESHOLD).astype(np.float32)
        lab = np.zeros((T, pad_h, pad_w), dtype=np.float32)
        lab[:, :H_data, :W_data] = raw_labels
        val = np.ones((T, pad_h, pad_w), dtype=np.float32)
        val[:, H_data:, :] = 0
        val[:, :, W_data:] = 0
        test_stacks_raw.append((stack, lab, val))

    result = run_fsn_experiment(
        "fsn_raw_labels", list(CHANNEL_ORDER), 6,
        train_stacks_raw, test_stacks_raw, device,
    )
    result["experiment"] = "C_label_quality"
    result["label_type"] = "raw"
    all_results.append(result)

    result = run_fsn_experiment(
        "fsn_smoothed_labels", list(CHANNEL_ORDER), 6,
        train_stacks_full, test_stacks_full, device,
    )
    result["experiment"] = "C_label_quality"
    result["label_type"] = "smoothed"
    all_results.append(result)

    with open(ANALYSIS_DIR / "ablation_results.json", "w") as f:
        json.dump(_serializable(all_results), f, indent=2)

    # ═══ D. XGBoost Feature Group Ablation ═════════════════════════════════
    log("\n" + "=" * 70)
    log("D. XGBOOST FEATURE GROUP ABLATION (seq_len=6)")
    log("=" * 70)
    for group_name, mask in _XGB_GROUPS.items():
        result = run_xgb_experiment(
            f"xgb_features_{group_name}", mask, 6,
            train_xgb, test_xgb,
        )
        result["experiment"] = "D_feature_group"
        result["group"] = group_name
        all_results.append(result)
        with open(ANALYSIS_DIR / "ablation_results.json", "w") as f:
            json.dump(_serializable(all_results), f, indent=2)

    # ═══ E. XGBoost Temporal Window Ablation ═══════════════════════════════
    log("\n" + "=" * 70)
    log("E. XGBOOST TEMPORAL WINDOW ABLATION (all features)")
    log("=" * 70)
    for seq_len in [1, 3, 6, 12]:
        result = run_xgb_experiment(
            f"xgb_seq{seq_len}", None, seq_len,
            train_xgb, test_xgb,
        )
        result["experiment"] = "E_temporal_window"
        all_results.append(result)
        with open(ANALYSIS_DIR / "ablation_results.json", "w") as f:
            json.dump(_serializable(all_results), f, indent=2)

    # ═══ F. XGBoost Label Quality ══════════════════════════════════════════
    log("\n" + "=" * 70)
    log("F. XGBOOST LABEL QUALITY (all features, seq=6)")
    log("=" * 70)
    result = run_xgb_experiment(
        "xgb_raw_labels", None, 6,
        train_xgb, test_xgb, use_smoothed=False,
    )
    result["experiment"] = "F_label_quality"
    result["label_type"] = "raw"
    all_results.append(result)

    result = run_xgb_experiment(
        "xgb_smoothed_labels", None, 6,
        train_xgb, test_xgb, use_smoothed=True,
    )
    result["experiment"] = "F_label_quality"
    result["label_type"] = "smoothed"
    all_results.append(result)

    # Save final results
    with open(ANALYSIS_DIR / "ablation_results.json", "w") as f:
        json.dump(_serializable(all_results), f, indent=2)

    # ═══ Print summary table ══════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("ABLATION STUDY RESULTS")
    log("=" * 70)

    log("\nA. FireSpreadNet Channel Groups:")
    log(f"  {'Group':<25} {'Channels':>4}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    log(f"  {'-'*55}")
    for r in all_results:
        if r.get("experiment") == "A_channel_group":
            log(f"  {r['group']:<25} {r['channels']:>4}  {r['f1']:>6.4f}  {r['precision']:>6.4f}  {r['recall']:>6.4f}")

    log("\nB. FireSpreadNet Temporal Window:")
    log(f"  {'SeqLen':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    log(f"  {'-'*30}")
    for r in all_results:
        if r.get("experiment") == "B_temporal_window":
            log(f"  {r['seq_len']:>6}  {r['f1']:>6.4f}  {r['precision']:>6.4f}  {r['recall']:>6.4f}")

    log("\nC. FireSpreadNet Label Quality:")
    log(f"  {'Type':<12}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    log(f"  {'-'*35}")
    for r in all_results:
        if r.get("experiment") == "C_label_quality":
            log(f"  {r.get('label_type','?'):<12}  {r['f1']:>6.4f}  {r['precision']:>6.4f}  {r['recall']:>6.4f}")

    log("\nD. XGBoost Feature Groups:")
    log(f"  {'Group':<25} {'Features':>4}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    log(f"  {'-'*55}")
    for r in all_results:
        if r.get("experiment") == "D_feature_group":
            log(f"  {r.get('group','?'):<25} {r.get('n_features',0):>4}  {r['f1']:>6.4f}  {r['precision']:>6.4f}  {r['recall']:>6.4f}")

    log("\nE. XGBoost Temporal Window:")
    log(f"  {'SeqLen':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    log(f"  {'-'*30}")
    for r in all_results:
        if r.get("experiment") == "E_temporal_window":
            log(f"  {r['seq_len']:>6}  {r['f1']:>6.4f}  {r['precision']:>6.4f}  {r['recall']:>6.4f}")

    log("\nF. XGBoost Label Quality:")
    log(f"  {'Type':<12}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    log(f"  {'-'*35}")
    for r in all_results:
        if r.get("experiment") == "F_label_quality":
            log(f"  {r.get('label_type','?'):<12}  {r['f1']:>6.4f}  {r['precision']:>6.4f}  {r['recall']:>6.4f}")

    log("\n" + "=" * 70)
    log("Ablation study complete. Results saved to:")
    log(f"  {ANALYSIS_DIR / 'ablation_results.json'}")


if __name__ == "__main__":
    main()

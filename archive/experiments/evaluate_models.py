"""Unified evaluation of XGBoost pixel model and FireSpreadNet v2."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
)
from scripts.fire_spread_model_v2 import FireSpreadNetV2

# ─── Config ──────────────────────────────────────────────────────────────────
_CANDIDATE_PIPELINE_DIRS = [
    REPO_ROOT.parent / "wildfire-data-pipeline" / "data",
    REPO_ROOT / ".." / "wildfire-data-pipeline" / "data",
    Path.home() / "Desktop" / "Current Projects" / "wildfire-data-pipeline" / "data",
]

PIPELINE_DIR: Path | None = None
for _candidate in _CANDIDATE_PIPELINE_DIRS:
    _resolved = _candidate.resolve()
    if _resolved.is_dir():
        PIPELINE_DIR = _resolved
        break

FIRE_NAMES = ["Kincade", "Walker"]
SEQ_LEN = 6
CONFIDENCE_THRESHOLD = 0.30
SMOOTH_WINDOW = 5
SMOOTH_MIN_VOTES = 2
CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "v2_evaluation"

BASELINES = {
    "XGBoost v1 (old data)": 0.7436,
    "FireSpreadNet v1 (2ch smoothed)": 0.8374,
    "FireSpreadNet v1 (8ch raw)": 0.7487,
    "GRU+Attention RNN": 0.7480,
    "Logistic Regression": 0.7160,
    "MLP": 0.7086,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── JSON helper ─────────────────────────────────────────────────────────────
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


# ─── Data preparation ────────────────────────────────────────────────────────
def auto_pad_dims(
    fire_names: list[str], pipeline_dir: Path,
) -> tuple[int, int]:
    max_h, max_w = 0, 0
    for name in fire_names:
        fire_arrays, _, _, _ = load_fire_data(name, pipeline_dir)
        _, h, w = fire_arrays["data"].shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    def ceil_to(val: int, multiple: int) -> int:
        return ((val + multiple - 1) // multiple) * multiple

    return ceil_to(max_h, 16), ceil_to(max_w, 16)


def load_and_prepare_fire(
    fire_name: str,
    pipeline_dir: Path,
    pad_h: int,
    pad_w: int,
) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Load a fire, build channel stack, compute smoothed labels.

    Returns (fire_arrays, feat_arrays, stack, labels, validity).
    The raw dicts are returned so XGBoost can use them for pixel extraction.
    """
    fire_arrays, feat_arrays, fire_meta, feat_meta = load_fire_data(
        fire_name, pipeline_dir,
    )
    stack = build_channel_stack(fire_arrays, feat_arrays, pad_h, pad_w)

    confidence = fire_arrays["data"]
    cloud_mask = fire_arrays.get("cloud_mask")
    obs_valid = fire_arrays.get("observation_valid")
    raw_labels, raw_validity = smooth_labels(
        confidence, cloud_mask, obs_valid,
        window=SMOOTH_WINDOW,
        min_votes=SMOOTH_MIN_VOTES,
        threshold=CONFIDENCE_THRESHOLD,
    )

    T = stack.shape[0]
    H_data = raw_labels.shape[1]
    W_data = raw_labels.shape[2]

    labels = np.zeros((T, pad_h, pad_w), dtype=np.float32)
    labels[:, :H_data, :W_data] = raw_labels[:T]

    validity = np.zeros((T, pad_h, pad_w), dtype=np.float32)
    validity[:, :H_data, :W_data] = raw_validity[:T]

    return fire_arrays, feat_arrays, stack, labels, validity


# ─── Pixel-level metrics ─────────────────────────────────────────────────────
def compute_pixel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute F1, precision, recall, and optionally AUC-ROC from flat arrays."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    result: dict[str, float] = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            result["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except (ImportError, ValueError):
            pass

    return result


# ─── XGBoost evaluation ──────────────────────────────────────────────────────
def evaluate_xgboost(
    test_fire_arrays: dict,
    test_feat_arrays: dict,
    test_labels: np.ndarray,
    test_validity: np.ndarray,
    direction_tag: str,
) -> dict[str, Any] | None:
    """Load and evaluate XGBoost v2 on test fire pixels.

    Tries direction-specific checkpoint first, falls back to generic best.
    Returns metrics dict or None if no checkpoint found.
    """
    import xgboost as xgb

    candidates = [
        CHECKPOINT_DIR / f"xgboost_v2_{direction_tag}.json",
        CHECKPOINT_DIR / "xgboost_v2_best.json",
        CHECKPOINT_DIR / "xgboost_v2_best.pkl",
    ]

    model_path: Path | None = None
    for p in candidates:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        log(f"  WARNING: No XGBoost v2 checkpoint found. Tried:")
        for p in candidates:
            log(f"    {p}")
        return None

    log(f"  Loading XGBoost from {model_path.name}")
    model = xgb.XGBClassifier()
    if model_path.suffix == ".pkl":
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model.load_model(str(model_path))

    # Extract the raw labels/validity in the original (unpadded) spatial dims
    # since iter_pixel_samples works on the raw fire_arrays directly
    conf = test_fire_arrays.get("data")
    if conf is None:
        log("  ERROR: test fire_arrays missing 'data'")
        return None
    T, H_data, W_data = conf.shape

    raw_labels = test_labels[:T, :H_data, :W_data]
    raw_validity = test_validity[:T, :H_data, :W_data]

    log("  Collecting test pixel samples...")
    t0 = time.time()
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for fv, label in iter_pixel_samples(
        test_fire_arrays, test_feat_arrays,
        raw_labels, raw_validity,
        seq_len=SEQ_LEN,
    ):
        X_list.append(fv)
        y_list.append(label)

    if not X_list:
        log("  WARNING: No test pixel samples extracted")
        return None

    X_test = np.array(X_list, dtype=np.float32)
    y_test = np.array(y_list, dtype=np.float32)
    collect_time = time.time() - t0
    n_pos = int(y_test.sum())
    n_neg = len(y_test) - n_pos
    log(f"  Collected {len(y_test):,} samples ({n_pos:,} pos, {n_neg:,} neg) in {collect_time:.1f}s")

    log("  Running XGBoost predictions...")
    t0 = time.time()
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.float32)
    pred_time = time.time() - t0
    log(f"  Prediction done in {pred_time:.1f}s")

    metrics = compute_pixel_metrics(y_test, y_pred, y_prob)
    metrics["n_samples"] = len(y_test)
    metrics["n_positive"] = n_pos
    metrics["prediction_time_s"] = pred_time

    return metrics


# ─── FireSpreadNet v2 evaluation ─────────────────────────────────────────────
def evaluate_firespreadnet(
    test_stack: np.ndarray,
    test_labels: np.ndarray,
    test_validity: np.ndarray,
    train_stack: np.ndarray,
    train_validity: np.ndarray,
    direction_tag: str,
    device,
) -> dict[str, Any] | None:
    """Load and evaluate FireSpreadNet v2 on test fire.

    Returns metrics dict (both grid-level and pixel-level) or None if no
    checkpoint found.
    """
    import torch

    # Find checkpoint
    candidates = [
        CHECKPOINT_DIR / f"firespreadnet_v2_{direction_tag}.pt",
        CHECKPOINT_DIR / "firespreadnet_v2_best.pt",
    ]

    ckpt_path: Path | None = None
    for p in candidates:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        log(f"  WARNING: No FireSpreadNet v2 checkpoint found. Tried:")
        for p in candidates:
            log(f"    {p}")
        return None

    log(f"  Loading FireSpreadNet v2 from {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    in_channels = ckpt.get("in_channels", len(CHANNEL_ORDER))

    model = FireSpreadNetV2(in_channels=in_channels, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
        f"train F1={ckpt.get('best_f1', '?')})")

    # Load normalization stats: try saved stats, fall back to computing from
    # training fire
    stats_path = CHECKPOINT_DIR / "firespreadnet_v2_channel_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        means = np.array(stats["means"], dtype=np.float32)
        stds = np.array(stats["stds"], dtype=np.float32)
        log(f"  Loaded channel stats from {stats_path.name}")
    else:
        log("  Computing channel stats from training fire (no saved stats found)...")
        means, stds = compute_channel_stats([train_stack], [train_validity])

    test_norm = normalize_stack(test_stack, means, stds)

    # Run inference
    log("  Running FireSpreadNet v2 inference...")
    t0 = time.time()

    agg_tp, agg_fp, agg_fn, agg_tn = 0, 0, 0, 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    n_sequences = 0

    batch_frames: list[torch.Tensor] = []
    batch_targets: list[torch.Tensor] = []
    batch_masks: list[torch.Tensor] = []
    batch_size = 4

    with torch.no_grad():
        for frames, target, mask in iter_grid_sequences(
            test_norm, test_labels, test_validity, SEQ_LEN,
        ):
            batch_frames.append(torch.from_numpy(frames))
            batch_targets.append(torch.from_numpy(target))
            batch_masks.append(torch.from_numpy(mask))

            if len(batch_frames) >= batch_size:
                x = torch.stack(batch_frames).to(device)
                y = torch.stack(batch_targets).to(device)
                m = torch.stack(batch_masks).to(device)

                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                valid = m.bool()
                p = preds[valid].int()
                t = y[valid].int()
                agg_tp += int(((p == 1) & (t == 1)).sum().item())
                agg_fp += int(((p == 1) & (t == 0)).sum().item())
                agg_fn += int(((p == 0) & (t == 1)).sum().item())
                agg_tn += int(((p == 0) & (t == 0)).sum().item())

                # Store for pixel-level AUC
                for i in range(x.shape[0]):
                    m_i = m[i].bool().cpu().numpy()
                    all_probs.append(probs[i].cpu().numpy()[m_i])
                    all_targets.append(y[i].cpu().numpy()[m_i])
                    all_masks.append(m_i)

                n_sequences += x.shape[0]
                batch_frames, batch_targets, batch_masks = [], [], []

        # Process remaining partial batch
        if batch_frames:
            x = torch.stack(batch_frames).to(device)
            y = torch.stack(batch_targets).to(device)
            m = torch.stack(batch_masks).to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            valid = m.bool()
            p = preds[valid].int()
            t = y[valid].int()
            agg_tp += int(((p == 1) & (t == 1)).sum().item())
            agg_fp += int(((p == 1) & (t == 0)).sum().item())
            agg_fn += int(((p == 0) & (t == 1)).sum().item())
            agg_tn += int(((p == 0) & (t == 0)).sum().item())

            for i in range(x.shape[0]):
                m_i = m[i].bool().cpu().numpy()
                all_probs.append(probs[i].cpu().numpy()[m_i])
                all_targets.append(y[i].cpu().numpy()[m_i])
                all_masks.append(m_i)

            n_sequences += x.shape[0]

    pred_time = time.time() - t0
    log(f"  Inference done: {n_sequences} sequences in {pred_time:.1f}s")

    # Grid-level metrics (aggregated across all sequences)
    prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
    rec = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    result: dict[str, Any] = {
        "tp": agg_tp, "fp": agg_fp, "fn": agg_fn, "tn": agg_tn,
        "precision": prec, "recall": rec, "f1": f1,
        "n_sequences": n_sequences,
        "prediction_time_s": pred_time,
    }

    # Pixel-level AUC from concatenated per-pixel probabilities
    if all_probs:
        flat_probs = np.concatenate(all_probs)
        flat_targets = np.concatenate(all_targets)
        result["n_pixels_evaluated"] = len(flat_probs)
        result["n_pixels_positive"] = int(flat_targets.sum())
        if len(np.unique(flat_targets)) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                result["auc_roc"] = float(roc_auc_score(flat_targets, flat_probs))
            except (ImportError, ValueError):
                pass

    return result


# ─── Comparison table ─────────────────────────────────────────────────────────
def print_comparison(results: dict[str, dict[str, Any]]) -> None:
    print()
    print("=" * 62)
    print("MODEL COMPARISON")
    print("=" * 62)
    print(f"{'Model':<38} |  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print("-" * 62)

    for name, m in results.items():
        if m is not None:
            f1 = m.get("f1", 0.0)
            p = m.get("precision", 0.0)
            r = m.get("recall", 0.0)
            auc = m.get("auc_roc")
            auc_str = f"  AUC={auc:.4f}" if auc is not None else ""
            print(f"{name:<38} | {f1:>6.4f}  {p:>6.4f}  {r:>6.4f}{auc_str}")
        else:
            print(f"{name:<38} |  (checkpoint not found)")

    print("-" * 62)
    print("Previous Baselines:")
    for name, f1 in sorted(BASELINES.items(), key=lambda x: -x[1]):
        print(f"  {name:<36} | {f1:>6.4f}")
    print("=" * 62)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    if PIPELINE_DIR is None:
        log("ERROR: Could not find wildfire-data-pipeline data directory.")
        log("Searched:")
        for c in _CANDIDATE_PIPELINE_DIRS:
            log(f"  {c.resolve()}")
        sys.exit(1)

    log(f"Pipeline data directory: {PIPELINE_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    device = pick_device()
    log(f"Device: {device}")
    log(f"Channels: {len(CHANNEL_ORDER)}")
    log(f"Label smoothing: window={SMOOTH_WINDOW}, "
        f"min_votes={SMOOTH_MIN_VOTES}, threshold={CONFIDENCE_THRESHOLD}")

    # Auto-detect pad dimensions
    log("Detecting spatial dimensions...")
    pad_h, pad_w = auto_pad_dims(FIRE_NAMES, PIPELINE_DIR)
    log(f"Padding: {pad_h} x {pad_w}")

    all_results: dict[str, dict[str, Any] | None] = {}

    # Evaluate in both cross-validation directions
    directions = [
        (FIRE_NAMES[0], FIRE_NAMES[1]),
        (FIRE_NAMES[1], FIRE_NAMES[0]),
    ]

    for train_name, test_name in directions:
        direction = f"{train_name}->{test_name}"
        direction_tag = f"{train_name.lower()}_to_{test_name.lower()}"
        log("")
        log("=" * 60)
        log(f"DIRECTION: {direction}")
        log("=" * 60)

        # Load both fires
        log(f"Loading {train_name} (train)...")
        (train_fire, train_feat,
         train_stack, train_labels, train_validity) = load_and_prepare_fire(
            train_name, PIPELINE_DIR, pad_h, pad_w,
        )
        log(f"  stack: {train_stack.shape}, fire pixels: {train_labels.sum():.0f}")

        log(f"Loading {test_name} (test)...")
        (test_fire, test_feat,
         test_stack, test_labels, test_validity) = load_and_prepare_fire(
            test_name, PIPELINE_DIR, pad_h, pad_w,
        )
        log(f"  stack: {test_stack.shape}, fire pixels: {test_labels.sum():.0f}")

        # ── Evaluate XGBoost ─────────────────────────────────────────────
        log("")
        log(f"--- XGBoost v2 ({direction}) ---")
        xgb_metrics = evaluate_xgboost(
            test_fire, test_feat,
            test_labels, test_validity,
            direction_tag,
        )
        label = f"XGBoost v2 (pixel, {train_name[0]}->{test_name[0]})"
        all_results[label] = xgb_metrics
        if xgb_metrics is not None:
            log(f"  F1={xgb_metrics['f1']:.4f}  "
                f"P={xgb_metrics['precision']:.4f}  "
                f"R={xgb_metrics['recall']:.4f}")
            if "auc_roc" in xgb_metrics:
                log(f"  AUC-ROC={xgb_metrics['auc_roc']:.4f}")

        # ── Evaluate FireSpreadNet v2 ────────────────────────────────────
        log("")
        log(f"--- FireSpreadNet v2 ({direction}) ---")
        fsn_metrics = evaluate_firespreadnet(
            test_stack, test_labels, test_validity,
            train_stack, train_validity,
            direction_tag, device,
        )
        label = f"FireSpreadNet v2 (27ch, {train_name[0]}->{test_name[0]})"
        all_results[label] = fsn_metrics
        if fsn_metrics is not None:
            log(f"  F1={fsn_metrics['f1']:.4f}  "
                f"P={fsn_metrics['precision']:.4f}  "
                f"R={fsn_metrics['recall']:.4f}")
            if "auc_roc" in fsn_metrics:
                log(f"  AUC-ROC={fsn_metrics['auc_roc']:.4f}")

    # ── Print comparison table ────────────────────────────────────────────
    print_comparison(all_results)

    # ── Save results ──────────────────────────────────────────────────────
    save_payload = {
        "config": {
            "fire_names": FIRE_NAMES,
            "seq_len": SEQ_LEN,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "smooth_window": SMOOTH_WINDOW,
            "smooth_min_votes": SMOOTH_MIN_VOTES,
            "n_channels": len(CHANNEL_ORDER),
            "channel_order": list(CHANNEL_ORDER),
            "pad_h": pad_h,
            "pad_w": pad_w,
        },
        "results": {
            k: _serializable(v) if v is not None else None
            for k, v in all_results.items()
        },
        "baselines": BASELINES,
    }

    out_path = ANALYSIS_DIR / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(save_payload, f, indent=2)
    log(f"Results saved to {out_path}")

    log("Evaluation complete.")


if __name__ == "__main__":
    main()

"""Train FireSpreadNet v2 on pipeline data with smoothed labels.

Uses the 27-channel feature stack from wildfire-data-pipeline (fire detection,
weather, terrain, vegetation, temporal encoding) with cloud-aware majority-vote
label smoothing. Cross-validates by training on one fire and testing on the other.
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
    load_fire_data,
    normalize_stack,
    smooth_labels,
)

# ─── Config ─────────────────────────────────────────────────────────────────
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

SEQ_LEN = 6
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
N_EPOCHS = 20
CONFIDENCE_THRESHOLD = 0.30
SMOOTH_WINDOW = 5
SMOOTH_MIN_VOTES = 2
CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "firespreadnet_v2"

# Auto-discover fires that have both FusedConf and Features NPZ
MIN_FIRE_PIXELS = 50  # Skip fires with fewer smoothed fire pixels
TRAIN_FRACTION = 0.7  # 70% of fires for training
SPLIT_SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
) -> dict[str, int | float]:
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        valid = mask.bool()
        p = preds[valid].int()
        t = targets[valid].int()
        tp = int(((p == 1) & (t == 1)).sum().item())
        fp = int(((p == 1) & (t == 0)).sum().item())
        fn = int(((p == 0) & (t == 1)).sum().item())
        tn = int(((p == 0) & (t == 0)).sum().item())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


# ─── Data Preparation ──────────────────────────────────────────────────────
def load_and_prepare_fire(
    fire_name: str,
    pipeline_dir: Path,
    pad_h: int,
    pad_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a fire, build the channel stack, and compute smoothed labels.

    Returns (stack, labels, validity) where:
        stack:    (T, C, pad_h, pad_w) float32
        labels:   (T, pad_h, pad_w)   float32 binary
        validity: (T, pad_h, pad_w)   float32
    """
    fire_arrays, feat_arrays, fire_meta, feat_meta = load_fire_data(fire_name, pipeline_dir)
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

    return stack, labels, validity


def auto_pad_dims(
    fire_names: list[str], pipeline_dir: Path,
) -> tuple[int, int]:
    """Determine pad_h/pad_w from the data, rounding up to nearest multiple of 16.

    The encoder downsamples 3x (by factor 8 total) so spatial dims must be
    divisible by 8. We use 16 for extra safety with future architecture changes.
    """
    max_h, max_w = 0, 0
    for name in fire_names:
        fire_arrays, feat_arrays, _, _ = load_fire_data(name, pipeline_dir)
        data = fire_arrays["data"]
        _, h, w = data.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    def ceil_to(val: int, multiple: int) -> int:
        return ((val + multiple - 1) // multiple) * multiple

    return ceil_to(max_h, 16), ceil_to(max_w, 16)


# ─── Training ───────────────────────────────────────────────────────────────
def train_epoch(
    model: FireSpreadNetV2,
    fire_data_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    """Train one epoch across multiple fires.

    fire_data_list: list of (stack, labels, validity) tuples.
    """
    model.train()
    rng = torch.Generator()
    rng.manual_seed(epoch * 1337)

    total_loss = 0.0
    n_batches = 0
    agg: dict[str, int] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    batch_frames: list[torch.Tensor] = []
    batch_targets: list[torch.Tensor] = []
    batch_masks: list[torch.Tensor] = []

    # Shuffle fire order each epoch
    fire_order = list(range(len(fire_data_list)))
    epoch_rng = np.random.RandomState(epoch * 1337)
    epoch_rng.shuffle(fire_order)

    for fire_idx in fire_order:
        stack, labels, validity = fire_data_list[fire_idx]
        for frames, target, mask in iter_grid_sequences(
            stack, labels, validity, SEQ_LEN,
        ):
            f_t = torch.from_numpy(frames)
            t_t = torch.from_numpy(target)
            m_t = torch.from_numpy(mask)

            f_t, t_t, m_t = augment_sequence_v2(f_t, t_t, m_t, rng)

            batch_frames.append(f_t)
            batch_targets.append(t_t)
            batch_masks.append(m_t)

            if len(batch_frames) >= BATCH_SIZE:
                x = torch.stack(batch_frames).to(device)
                y = torch.stack(batch_targets).to(device)
                m = torch.stack(batch_masks).to(device)

                logits = model(x)
                loss = criterion(logits, y, m)

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                    batch_frames, batch_targets, batch_masks = [], [], []
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item()
                n_batches += 1
                mb = compute_metrics(logits, y, m)
                for k in ("tp", "fp", "fn", "tn"):
                    agg[k] += mb[k]

                batch_frames, batch_targets, batch_masks = [], [], []

    # Process remaining partial batch
    if batch_frames:
        x = torch.stack(batch_frames).to(device)
        y = torch.stack(batch_targets).to(device)
        m = torch.stack(batch_masks).to(device)
        logits = model(x)
        loss = criterion(logits, y, m)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            n_batches += 1
            mb = compute_metrics(logits, y, m)
            for k in ("tp", "fp", "fn", "tn"):
                agg[k] += mb[k]

    avg_loss = total_loss / max(n_batches, 1)
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"loss": avg_loss, "precision": p, "recall": r, "f1": f1, **agg}


def evaluate(
    model: FireSpreadNetV2,
    fire_data_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    criterion: CombinedLoss,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate across multiple fires. No augmentation, torch.no_grad().

    fire_data_list: list of (stack, labels, validity) tuples.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    agg: dict[str, int] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    batch_frames: list[torch.Tensor] = []
    batch_targets: list[torch.Tensor] = []
    batch_masks: list[torch.Tensor] = []

    with torch.no_grad():
        for stack, labels, validity in fire_data_list:
            for frames, target, mask in iter_grid_sequences(
                stack, labels, validity, SEQ_LEN,
            ):
                batch_frames.append(torch.from_numpy(frames))
                batch_targets.append(torch.from_numpy(target))
                batch_masks.append(torch.from_numpy(mask))

                if len(batch_frames) >= BATCH_SIZE:
                    x = torch.stack(batch_frames).to(device)
                    y = torch.stack(batch_targets).to(device)
                    m = torch.stack(batch_masks).to(device)
                    logits = model(x)
                    loss = criterion(logits, y, m)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        n_batches += 1
                    mb = compute_metrics(logits, y, m)
                    for k in ("tp", "fp", "fn", "tn"):
                        agg[k] += mb[k]
                    batch_frames, batch_targets, batch_masks = [], [], []

        if batch_frames:
            x = torch.stack(batch_frames).to(device)
            y = torch.stack(batch_targets).to(device)
            m = torch.stack(batch_masks).to(device)
            logits = model(x)
            loss = criterion(logits, y, m)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1
            mb = compute_metrics(logits, y, m)
            for k in ("tp", "fp", "fn", "tn"):
                agg[k] += mb[k]

    avg_loss = total_loss / max(n_batches, 1)
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"loss": avg_loss, "precision": p, "recall": r, "f1": f1, **agg}


# ─── JSON serializer ────────────────────────────────────────────────────────
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


# ─── Fire Discovery ──────────────────────────────────────────────────────────
def discover_fires(pipeline_dir: Path) -> list[str]:
    """Auto-discover fires that have both FusedConf and Features NPZ files."""
    fires: list[str] = []
    if not pipeline_dir.is_dir():
        return fires
    for subdir in sorted(pipeline_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        has_conf = bool(list(subdir.glob(f"{name}_*_FusedConf.npz")))
        has_feat = bool(list(subdir.glob(f"{name}_*_Features.npz")))
        if has_conf and has_feat:
            fires.append(name)
    return fires


# ─── Main ───────────────────────────────────────────────────────────────────
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

    device = pick_device()
    log(f"Device: {device}")
    log(f"Channels: {len(CHANNEL_ORDER)}")
    log(f"Label smoothing: majority vote, window={SMOOTH_WINDOW}, "
        f"min_votes={SMOOTH_MIN_VOTES}, threshold={CONFIDENCE_THRESHOLD}")

    # 1. Auto-discover available fires
    all_fires = discover_fires(PIPELINE_DIR)
    log(f"Discovered {len(all_fires)} fires: {all_fires}")
    if not all_fires:
        log("ERROR: No fire data found.")
        sys.exit(1)

    # 2. Load each fire, compute smoothed labels, check fire pixel count
    fire_info: list[dict[str, Any]] = []
    for name in all_fires:
        log(f"Loading {name}...")
        fire_arrays, feat_arrays, fire_meta, feat_meta = load_fire_data(name, PIPELINE_DIR)
        confidence = fire_arrays["data"]
        cloud_mask = fire_arrays.get("cloud_mask")
        obs_valid = fire_arrays.get("observation_valid")
        raw_labels, _ = smooth_labels(
            confidence, cloud_mask, obs_valid,
            window=SMOOTH_WINDOW, min_votes=SMOOTH_MIN_VOTES, threshold=CONFIDENCE_THRESHOLD,
        )
        n_fire = int(raw_labels.sum())
        T, H, W = confidence.shape
        log(f"  {name}: {T}h, {H}x{W} grid, {n_fire} smoothed fire pixels")
        if n_fire < MIN_FIRE_PIXELS:
            log(f"  SKIPPING {name} (only {n_fire} fire pixels < {MIN_FIRE_PIXELS})")
            continue
        fire_info.append({"name": name, "n_fire": n_fire, "T": T, "H": H, "W": W})

    if len(fire_info) < 2:
        log(f"ERROR: Need at least 2 usable fires, found {len(fire_info)}.")
        sys.exit(1)

    usable_names = [f["name"] for f in fire_info]
    log(f"Usable fires ({len(usable_names)}): {usable_names}")

    # 3. Split fires into train/test by fire (not temporal)
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(usable_names)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * TRAIN_FRACTION))
    n_train = min(n_train, len(shuffled) - 1)  # Ensure at least 1 test fire
    train_names = sorted(shuffled[:n_train])
    test_names = sorted(shuffled[n_train:])
    log(f"Train fires ({len(train_names)}): {train_names}")
    log(f"Test fires  ({len(test_names)}):  {test_names}")

    # 4. Compute max spatial dims across ALL usable fires, pad to multiple of 16
    max_h = max(f["H"] for f in fire_info)
    max_w = max(f["W"] for f in fire_info)

    def ceil_to(val: int, multiple: int) -> int:
        return ((val + multiple - 1) // multiple) * multiple

    pad_h = ceil_to(max_h, 16)
    pad_w = ceil_to(max_w, 16)
    log(f"Padding: {pad_h} x {pad_w} (max data: {max_h} x {max_w})")

    # 5. Build channel stacks for all fires
    train_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    test_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    train_stacks_raw: list[np.ndarray] = []
    train_validity_raw: list[np.ndarray] = []
    test_fire_names_ordered: list[str] = []

    for name in train_names:
        stack, labels, validity = load_and_prepare_fire(name, PIPELINE_DIR, pad_h, pad_w)
        log(f"  Train: {name} stack={stack.shape}, fire_px={labels.sum():.0f}")
        train_data.append((stack, labels, validity))
        train_stacks_raw.append(stack)
        train_validity_raw.append(validity)

    for name in test_names:
        stack, labels, validity = load_and_prepare_fire(name, PIPELINE_DIR, pad_h, pad_w)
        log(f"  Test:  {name} stack={stack.shape}, fire_px={labels.sum():.0f}")
        test_data.append((stack, labels, validity))
        test_fire_names_ordered.append(name)

    # 6. Compute channel stats from training fires only
    log("Computing channel normalization stats from training fires...")
    means, stds = compute_channel_stats(train_stacks_raw, train_validity_raw)
    log(f"  means range: [{means.min():.4f}, {means.max():.4f}]")
    log(f"  stds range:  [{stds.min():.4f}, {stds.max():.4f}]")

    # Normalize all stacks
    train_data = [
        (normalize_stack(s, means, stds), l, v) for s, l, v in train_data
    ]
    test_data = [
        (normalize_stack(s, means, stds), l, v) for s, l, v in test_data
    ]

    # 7. Create model
    n_channels = len(CHANNEL_ORDER)
    model = FireSpreadNetV2(in_channels=n_channels, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"FireSpreadNetV2: {n_params:,} parameters, {n_channels} input channels")

    criterion = CombinedLoss(focal_weight=0.5, tversky_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1 = 0.0
    history: list[dict[str, Any]] = []

    # 8. Training loop
    for epoch in range(1, N_EPOCHS + 1):
        log("=" * 60)
        log(f"EPOCH {epoch}/{N_EPOCHS}")
        log("=" * 60)

        t0 = time.time()
        train_result = train_epoch(
            model, train_data, criterion, optimizer, device, epoch,
        )
        train_time = time.time() - t0
        log(f"Train: loss={train_result['loss']:.4f}  F1={train_result['f1']:.4f}  "
            f"P={train_result['precision']:.4f}  R={train_result['recall']:.4f}  [{train_time:.0f}s]")

        t0 = time.time()
        test_result = evaluate(model, test_data, criterion, device)
        eval_time = time.time() - t0
        log(f"Test:  loss={test_result['loss']:.4f}  F1={test_result['f1']:.4f}  "
            f"P={test_result['precision']:.4f}  R={test_result['recall']:.4f}  [{eval_time:.0f}s]")

        # Per-fire test metrics
        for i, name in enumerate(test_fire_names_ordered):
            fire_result = evaluate(model, [test_data[i]], criterion, device)
            log(f"  {name}: F1={fire_result['f1']:.4f}  "
                f"P={fire_result['precision']:.4f}  R={fire_result['recall']:.4f}")

        improved = test_result["f1"] > best_f1
        if improved:
            best_f1 = test_result["f1"]
            torch.save({
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "best_f1": best_f1,
                "train_fires": train_names,
                "test_fires": test_names,
                "in_channels": n_channels,
            }, CHECKPOINT_DIR / "firespreadnet_v2_best.pt")
            log(f"*** NEW BEST F1={best_f1:.4f} ***")

        history.append({
            "epoch": epoch,
            "train": {k: v for k, v in train_result.items()},
            "test": {k: v for k, v in test_result.items()},
            "best_f1": best_f1,
            "improved": improved,
            "train_time_s": train_time,
            "eval_time_s": eval_time,
        })

    # 9. Save channel stats
    stats_path = CHECKPOINT_DIR / "firespreadnet_v2_channel_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "means": means.tolist(),
            "stds": stds.tolist(),
            "channel_order": list(CHANNEL_ORDER),
        }, f, indent=2)
    log(f"Channel stats saved to {stats_path}")

    # Save training history with fire-level breakdown
    # Compute final per-fire test metrics
    per_fire_results: dict[str, Any] = {}
    for i, name in enumerate(test_fire_names_ordered):
        fire_result = evaluate(model, [test_data[i]], criterion, device)
        per_fire_results[name] = _serializable(fire_result)

    combined_history = {
        "multi_fire_cv": {
            "train_fires": train_names,
            "test_fires": test_names,
            "best_f1": best_f1,
            "per_fire_test": per_fire_results,
            "epochs": _serializable(history),
        },
        "config": {
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "n_epochs": N_EPOCHS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "smooth_window": SMOOTH_WINDOW, "smooth_min_votes": SMOOTH_MIN_VOTES,
            "pad_h": pad_h, "pad_w": pad_w,
            "n_channels": n_channels, "channel_order": list(CHANNEL_ORDER),
            "min_fire_pixels": MIN_FIRE_PIXELS,
            "train_fraction": TRAIN_FRACTION, "split_seed": SPLIT_SEED,
        },
    }
    history_path = ANALYSIS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)

    log("\n" + "=" * 60)
    log("TRAINING COMPLETE — FireSpreadNet v2 (multi-fire CV)")
    log("=" * 60)
    log(f"Train fires: {train_names}")
    log(f"Test fires:  {test_names}")
    log(f"Best test F1: {best_f1:.4f}")
    for name, res in per_fire_results.items():
        log(f"  {name}: F1={res['f1']:.4f}  P={res['precision']:.4f}  R={res['recall']:.4f}")


if __name__ == "__main__":
    main()

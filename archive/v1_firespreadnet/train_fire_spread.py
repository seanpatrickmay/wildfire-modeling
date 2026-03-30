"""Continuous training loop for FireSpreadNet.

Trains the enhanced ConvGRU-U-Net on full-image fire spread prediction.
Runs indefinitely, evaluating after each epoch and saving improvements.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fire_spread_model import (
    CombinedLoss,
    FireSpreadNet,
    augment_sequence,
    build_frame,
    IN_CHANNELS,
    N_PHYSICAL_CHANNELS,
    PAD_H,
    PAD_W,
)
from scripts.neighbor_cell_logreg import (
    discover_fire_entries,
    find_repo_root,
    iter_aligned_hours_for_fire,
    load_fire_entry_context,
    split_fire_entries,
)
from scripts.neighbor_cell_nn import pick_device

# ─── Config ─────────────────────────────────────────────────────────────────
SEQ_LEN = 6
POSITIVE_THRESHOLD = 0.10  # fire confidence >= 10% = fire pixel
BATCH_SIZE = 4
GRAD_ACCUM = 1
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.10
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7
DISCOUNTED_RAIN_LOOKBACK = 0
DISCOUNTED_RAIN_HALF_LIFE = 0.0

CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "fire_spread"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Channel normalization stats ────────────────────────────────────────────
def compute_channel_stats(
    entries: list[dict], repo_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std from training fires (streaming)."""
    sums = np.zeros(N_PHYSICAL_CHANNELS, dtype=np.float64)
    sq_sums = np.zeros(N_PHYSICAL_CHANNELS, dtype=np.float64)
    counts = np.zeros(N_PHYSICAL_CHANNELS, dtype=np.float64)

    for entry in entries:
        ctx = load_fire_entry_context(entry, repo_root)
        H, W = ctx["goes_shape"]
        n_pixels = H * W

        for t, rtma_hour in iter_aligned_hours_for_fire(
            repo_root, ctx["goes_conf"], ctx["goes_time_index"],
            ctx["rtma_manifest"], ctx["rtma_manifest_path"],
            ctx["goes_shape"], ctx["goes_transform"], ctx["goes_crs"],
            include_discounted_rain=False,
            discounted_rain_lookback_hours=0,
            discounted_rain_half_life_days=0.0,
        ):
            conf = ctx["goes_conf"][t].astype(np.float64)
            tmp = rtma_hour["TMP"].astype(np.float64)
            wind = rtma_hour["WIND"].astype(np.float64)
            wdir_rad = np.deg2rad(rtma_hour["WDIR"].astype(np.float64))
            spfh = rtma_hour["SPFH"].astype(np.float64)
            precip = np.clip(rtma_hour["ACPC01"].astype(np.float64), 0.0, 100.0)

            channels = [conf, tmp, wind, np.sin(wdir_rad), np.cos(wdir_rad), spfh, precip]
            for i, ch in enumerate(channels):
                ch_clean = np.nan_to_num(ch, nan=0.0)
                sums[i] += ch_clean.sum()
                sq_sums[i] += (ch_clean ** 2).sum()
                counts[i] += n_pixels

        log(f"  channel stats: {entry['fire_name']} done")

    means = sums / counts
    stds = np.sqrt(np.maximum(sq_sums / counts - means ** 2, 0.0))
    stds = np.where(stds > 0, stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


# ─── Sequence iteration ─────────────────────────────────────────────────────
def iter_fire_sequences(
    entry: dict, repo_root: Path,
    channel_means: np.ndarray, channel_stds: np.ndarray,
    seq_len: int = SEQ_LEN,
):
    """Yield (frames, target, mask) for contiguous sequences from one fire.

    frames: (T, C, PAD_H, PAD_W) float32
    target: (1, PAD_H, PAD_W) float32 binary
    mask:   (1, PAD_H, PAD_W) float32
    """
    ctx = load_fire_entry_context(entry, repo_root)
    goes_conf = ctx["goes_conf"]
    H, W = ctx["goes_shape"]

    frame_buffer: deque[np.ndarray] = deque(maxlen=seq_len)
    time_buffer: deque[int] = deque(maxlen=seq_len)

    for t, rtma_hour in iter_aligned_hours_for_fire(
        repo_root, goes_conf, ctx["goes_time_index"],
        ctx["rtma_manifest"], ctx["rtma_manifest_path"],
        ctx["goes_shape"], ctx["goes_transform"], ctx["goes_crs"],
        include_discounted_rain=False,
        discounted_rain_lookback_hours=0,
        discounted_rain_half_life_days=0.0,
    ):
        frame = build_frame(goes_conf[t], rtma_hour, (H, W), channel_means, channel_stds)
        frame_buffer.append(frame)
        time_buffer.append(t)

        if len(frame_buffer) < seq_len:
            continue

        # Check contiguity
        times = list(time_buffer)
        if any(times[i + 1] - times[i] != 1 for i in range(len(times) - 1)):
            continue

        last_t = times[-1]
        if last_t + 1 >= goes_conf.shape[0]:
            continue

        # Target: next hour fire confidence binarized
        target_conf = goes_conf[last_t + 1]
        target_binary = (target_conf >= POSITIVE_THRESHOLD).astype(np.float32)
        valid = np.isfinite(target_conf).astype(np.float32)

        target = np.zeros((1, PAD_H, PAD_W), dtype=np.float32)
        target[0, :H, :W] = target_binary

        mask = np.zeros((1, PAD_H, PAD_W), dtype=np.float32)
        mask[0, :H, :W] = valid

        frames = np.stack(list(frame_buffer), axis=0)  # (T, C, PAD_H, PAD_W)
        yield frames, target, mask


# ─── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5):
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        valid = mask.bool()

        p = preds[valid].int()
        t = targets[valid].int()

        tp = int(((p == 1) & (t == 1)).sum().item())
        fp = int(((p == 1) & (t == 0)).sum().item())
        fn = int(((p == 0) & (t == 1)).sum().item())
        tn = int(((p == 0) & (t == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


# ─── Training ───────────────────────────────────────────────────────────────
def train_epoch(
    model: FireSpreadNet,
    entries: list[dict],
    repo_root: Path,
    channel_means: np.ndarray,
    channel_stds: np.ndarray,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epoch: int,
    augment: bool = True,
) -> dict:
    model.train()
    rng = torch.Generator()
    rng.manual_seed(epoch * 1337)

    total_loss = 0.0
    n_batches = 0
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    # Shuffle fire order each epoch
    fire_order = list(range(len(entries)))
    np.random.shuffle(fire_order)

    batch_frames, batch_targets, batch_masks = [], [], []

    for fire_idx in fire_order:
        entry = entries[fire_idx]
        for frames, target, mask in iter_fire_sequences(
            entry, repo_root, channel_means, channel_stds,
        ):
            f_t = torch.from_numpy(frames)
            t_t = torch.from_numpy(target)
            m_t = torch.from_numpy(mask)

            if augment:
                f_t, t_t, m_t = augment_sequence(f_t, t_t, m_t, rng)

            batch_frames.append(f_t)
            batch_targets.append(t_t)
            batch_masks.append(m_t)

            if len(batch_frames) >= BATCH_SIZE:
                x = torch.stack(batch_frames).to(device)    # (B, T, C, H, W)
                y = torch.stack(batch_targets).to(device)    # (B, 1, H, W)
                m = torch.stack(batch_masks).to(device)      # (B, 1, H, W)

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
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                n_batches += 1
                m_batch = compute_metrics(logits, y, m)
                for k in ("tp", "fp", "fn", "tn"):
                    agg[k] += m_batch[k]

                batch_frames, batch_targets, batch_masks = [], [], []

        log(f"  train: {entry['fire_name']} done")

    # Process remaining
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
            m_batch = compute_metrics(logits, y, m)
            for k in ("tp", "fp", "fn", "tn"):
                agg[k] += m_batch[k]

    avg_loss = total_loss / max(n_batches, 1)
    p = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0.0
    r = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {"loss": avg_loss, "precision": p, "recall": r, "f1": f1, **agg}


def evaluate(
    model: FireSpreadNet,
    entries: list[dict],
    repo_root: Path,
    channel_means: np.ndarray,
    channel_stds: np.ndarray,
    criterion: CombinedLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_fire = {}

    with torch.no_grad():
        for entry in entries:
            fire_agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            batch_frames, batch_targets, batch_masks = [], [], []

            for frames, target, mask in iter_fire_sequences(
                entry, repo_root, channel_means, channel_stds,
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
                    total_loss += loss.item()
                    n_batches += 1
                    m_batch = compute_metrics(logits, y, m)
                    for k in ("tp", "fp", "fn", "tn"):
                        agg[k] += m_batch[k]
                        fire_agg[k] += m_batch[k]
                    batch_frames, batch_targets, batch_masks = [], [], []

            if batch_frames:
                x = torch.stack(batch_frames).to(device)
                y = torch.stack(batch_targets).to(device)
                m = torch.stack(batch_masks).to(device)
                logits = model(x)
                loss = criterion(logits, y, m)
                total_loss += loss.item()
                n_batches += 1
                m_batch = compute_metrics(logits, y, m)
                for k in ("tp", "fp", "fn", "tn"):
                    agg[k] += m_batch[k]
                    fire_agg[k] += m_batch[k]

            fp = fire_agg["tp"] / (fire_agg["tp"] + fire_agg["fp"]) if (fire_agg["tp"] + fire_agg["fp"]) > 0 else 0.0
            fr = fire_agg["tp"] / (fire_agg["tp"] + fire_agg["fn"]) if (fire_agg["tp"] + fire_agg["fn"]) > 0 else 0.0
            ff = 2 * fp * fr / (fp + fr) if (fp + fr) > 0 else 0.0
            per_fire[entry["fire_name"]] = {"precision": fp, "recall": fr, "f1": ff, **fire_agg}
            log(f"  eval: {entry['fire_name']:30s} F1={ff:.4f} P={fp:.4f} R={fr:.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    p = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0.0
    r = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {"loss": avg_loss, "precision": p, "recall": r, "f1": f1, **agg, "per_fire": per_fire}


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log(f"Device: {device}")

    # Discover and split
    entries = discover_fire_entries(repo_root)
    train_entries, test_entries = split_fire_entries(
        entries, None, None, TRAIN_FRACTION, SPLIT_SEED,
    )
    log(f"Train: {len(train_entries)} fires, Test: {len(test_entries)} fires")

    # Channel normalization stats
    stats_path = CHECKPOINT_DIR / "fire_spread_channel_stats.json"
    if stats_path.exists():
        log("Loading cached channel stats...")
        with open(stats_path) as f:
            stats = json.load(f)
        channel_means = np.array(stats["means"], dtype=np.float32)
        channel_stds = np.array(stats["stds"], dtype=np.float32)
    else:
        log("Computing channel normalization stats from training fires...")
        channel_means, channel_stds = compute_channel_stats(train_entries, repo_root)
        with open(stats_path, "w") as f:
            json.dump({"means": channel_means.tolist(), "stds": channel_stds.tolist()}, f, indent=2)
        log(f"Channel stats saved to {stats_path}")

    log(f"Channel means: {channel_means}")
    log(f"Channel stds:  {channel_stds}")

    # Model
    model = FireSpreadNet(dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"FireSpreadNet: {n_params:,} parameters")

    criterion = CombinedLoss(focal_weight=0.5, tversky_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Load existing checkpoint if available
    ckpt_path = CHECKPOINT_DIR / "fire_spread_best.pt"
    start_epoch = 1
    best_f1 = 0.0
    if ckpt_path.exists():
        log(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_f1 = ckpt.get("best_f1", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        log(f"Resumed from epoch {start_epoch - 1}, best F1={best_f1:.4f}")

    history = []

    # Run indefinitely
    epoch = start_epoch
    while True:
        log("=" * 60)
        log(f"EPOCH {epoch}")
        log("=" * 60)

        t0 = time.time()
        train_result = train_epoch(
            model, train_entries, repo_root, channel_means, channel_stds,
            criterion, optimizer, None, device, epoch, augment=True,
        )
        train_time = time.time() - t0

        log(f"Train: loss={train_result['loss']:.4f}  F1={train_result['f1']:.4f}  "
            f"P={train_result['precision']:.4f}  R={train_result['recall']:.4f}  [{train_time:.0f}s]")

        # Evaluate on test fires
        log("Evaluating on test fires...")
        t0 = time.time()
        test_result = evaluate(
            model, test_entries, repo_root, channel_means, channel_stds,
            criterion, device,
        )
        eval_time = time.time() - t0

        log(f"Test:  loss={test_result['loss']:.4f}  F1={test_result['f1']:.4f}  "
            f"P={test_result['precision']:.4f}  R={test_result['recall']:.4f}  [{eval_time:.0f}s]")

        improved = test_result["f1"] > best_f1
        if improved:
            best_f1 = test_result["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_f1": best_f1,
                "train_result": {k: v for k, v in train_result.items()},
                "test_result": {k: v for k, v in test_result.items() if k != "per_fire"},
            }, ckpt_path)
            log(f"*** NEW BEST F1={best_f1:.4f} — saved to {ckpt_path} ***")

        epoch_record = {
            "epoch": epoch,
            "train": {k: v for k, v in train_result.items()},
            "test": {k: v for k, v in test_result.items() if k != "per_fire"},
            "test_per_fire": test_result.get("per_fire", {}),
            "best_f1": best_f1,
            "improved": improved,
            "train_time_s": train_time,
            "eval_time_s": eval_time,
        }
        history.append(epoch_record)

        # Save history
        def serializable(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {str(k): serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [serializable(i) for i in obj]
            return obj

        history_path = ANALYSIS_DIR / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(serializable(history), f, indent=2)

        log(f"Epoch {epoch} complete. Best F1={best_f1:.4f}. History saved.")
        epoch += 1


if __name__ == "__main__":
    main()

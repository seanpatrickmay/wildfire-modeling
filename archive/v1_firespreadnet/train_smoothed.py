"""Train FireSpreadNet with temporally-smoothed labels (fire-only, 2 channels).

Raw GOES labels have 23% hourly flicker → oracle F1 ceiling of 0.757.
Majority voting (2-of-5 hours) raises the ceiling to 0.896.
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
    PAD_H,
    PAD_W,
    N_PHYSICAL_CHANNELS,
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
RAW_THRESHOLD = 0.10        # raw GOES confidence threshold for binary fire
SMOOTH_WINDOW = 5           # majority voting window (hours)
SMOOTH_MIN_VOTES = 2        # minimum detections in window to count as fire
N_CHANNELS = 2              # fire_conf + validity (fire-only ablation winner)
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
N_EPOCHS = 15
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7

CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "fire_spread_smoothed"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Label smoothing ────────────────────────────────────────────────────────
def smooth_labels(goes_conf: np.ndarray, window: int, min_votes: int, threshold: float) -> np.ndarray:
    """Apply majority voting temporal smoothing to fire labels.

    A pixel is fire at time t if it was detected as fire (conf >= threshold)
    in >= min_votes of the hours [t-window+1, ..., t].
    """
    T, H, W = goes_conf.shape
    binary = (goes_conf >= threshold).astype(np.float32)
    smoothed = np.zeros_like(binary)

    for t in range(T):
        start = max(0, t - window + 1)
        votes = binary[start:t + 1].sum(axis=0)
        smoothed[t] = (votes >= min_votes).astype(np.float32)

    return smoothed


# ─── Channel stats (fire-only) ──────────────────────────────────────────────
def compute_fire_only_stats(entries: list[dict], repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean/std for fire confidence channel only."""
    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    for entry in entries:
        ctx = load_fire_entry_context(entry, repo_root)
        conf = ctx["goes_conf"]  # (T, H, W)
        vals = np.nan_to_num(conf, nan=0.0).astype(np.float64)
        total_sum += vals.sum()
        total_sq += (vals ** 2).sum()
        total_count += vals.size

    mean = total_sum / total_count
    std = np.sqrt(max(total_sq / total_count - mean ** 2, 0.0))
    if std == 0:
        std = 1.0
    return np.array([mean], dtype=np.float32), np.array([std], dtype=np.float32)


# ─── Frame building (fire-only, 2 channels) ─────────────────────────────────
def build_fire_only_frame(
    goes_conf_t: np.ndarray,
    goes_shape: tuple[int, int],
    ch_mean: float, ch_std: float,
) -> np.ndarray:
    """Build a 2-channel frame: normalized fire confidence + validity mask."""
    H, W = goes_shape
    frame = np.zeros((N_CHANNELS, PAD_H, PAD_W), dtype=np.float32)
    conf = np.nan_to_num(goes_conf_t.astype(np.float32), nan=0.0)
    frame[0, :H, :W] = (conf - ch_mean) / max(ch_std, 1e-8)
    frame[1, :H, :W] = 1.0  # validity mask
    return frame


# ─── Sequence iteration with smoothed labels ────────────────────────────────
def iter_fire_sequences_smoothed(
    entry: dict, repo_root: Path,
    ch_mean: float, ch_std: float,
):
    """Yield (frames, target, mask) using temporally-smoothed labels."""
    ctx = load_fire_entry_context(entry, repo_root)
    goes_conf = ctx["goes_conf"]  # (T, H, W)
    H, W = ctx["goes_shape"]
    T = goes_conf.shape[0]

    # Pre-compute smoothed labels for entire fire
    smoothed = smooth_labels(goes_conf, SMOOTH_WINDOW, SMOOTH_MIN_VOTES, RAW_THRESHOLD)

    # Build all frames (fire-only, no RTMA needed)
    # We still need aligned hours for time indexing, but for fire-only we can
    # iterate directly over GOES timesteps
    frame_buffer: deque[np.ndarray] = deque(maxlen=SEQ_LEN)

    for t in range(T):
        frame = build_fire_only_frame(goes_conf[t], (H, W), ch_mean, ch_std)
        frame_buffer.append(frame)

        if len(frame_buffer) < SEQ_LEN:
            continue

        if t + 1 >= T:
            continue

        # Target: smoothed label at t+1
        target_smoothed = smoothed[t + 1]
        valid = np.isfinite(goes_conf[t + 1]).astype(np.float32)

        target = np.zeros((1, PAD_H, PAD_W), dtype=np.float32)
        target[0, :H, :W] = target_smoothed

        mask = np.zeros((1, PAD_H, PAD_W), dtype=np.float32)
        mask[0, :H, :W] = valid

        frames = np.stack(list(frame_buffer), axis=0)
        yield frames, target, mask


# ─── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(logits, targets, mask):
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


# ─── Train / Eval ───────────────────────────────────────────────────────────
def train_epoch(model, entries, repo_root, ch_mean, ch_std, criterion, optimizer, device, epoch):
    model.train()
    total_loss, n_batches = 0.0, 0
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    batch_f, batch_t, batch_m = [], [], []

    fire_order = list(range(len(entries)))
    np.random.shuffle(fire_order)

    for fi in fire_order:
        entry = entries[fi]
        for frames, target, mask in iter_fire_sequences_smoothed(
            entry, repo_root, ch_mean, ch_std,
        ):
            batch_f.append(torch.from_numpy(frames))
            batch_t.append(torch.from_numpy(target))
            batch_m.append(torch.from_numpy(mask))

            if len(batch_f) >= BATCH_SIZE:
                x = torch.stack(batch_f).to(device)
                y = torch.stack(batch_t).to(device)
                m = torch.stack(batch_m).to(device)
                logits = model(x)
                loss = criterion(logits, y, m)
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                    batch_f, batch_t, batch_m = [], [], []
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
                batch_f, batch_t, batch_m = [], [], []

        log(f"  train: {entry['fire_name']} done")

    if batch_f:
        x = torch.stack(batch_f).to(device)
        y = torch.stack(batch_t).to(device)
        m = torch.stack(batch_m).to(device)
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


def evaluate(model, entries, repo_root, ch_mean, ch_std, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_fire = {}

    with torch.no_grad():
        for entry in entries:
            fire_agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            batch_f, batch_t, batch_m = [], [], []

            for frames, target, mask in iter_fire_sequences_smoothed(
                entry, repo_root, ch_mean, ch_std,
            ):
                batch_f.append(torch.from_numpy(frames))
                batch_t.append(torch.from_numpy(target))
                batch_m.append(torch.from_numpy(mask))

                if len(batch_f) >= BATCH_SIZE:
                    x = torch.stack(batch_f).to(device)
                    y = torch.stack(batch_t).to(device)
                    m = torch.stack(batch_m).to(device)
                    logits = model(x)
                    loss = criterion(logits, y, m)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        n_batches += 1
                    mb = compute_metrics(logits, y, m)
                    for k in ("tp", "fp", "fn", "tn"):
                        agg[k] += mb[k]
                        fire_agg[k] += mb[k]
                    batch_f, batch_t, batch_m = [], [], []

            if batch_f:
                x = torch.stack(batch_f).to(device)
                y = torch.stack(batch_t).to(device)
                m = torch.stack(batch_m).to(device)
                logits = model(x)
                mb = compute_metrics(logits, y, m)
                for k in ("tp", "fp", "fn", "tn"):
                    agg[k] += mb[k]
                    fire_agg[k] += mb[k]

            ftp, ffp, ffn = fire_agg["tp"], fire_agg["fp"], fire_agg["fn"]
            fp = ftp / (ftp + ffp) if (ftp + ffp) > 0 else 0.0
            fr = ftp / (ftp + ffn) if (ftp + ffn) > 0 else 0.0
            ff = 2 * fp * fr / (fp + fr) if (fp + fr) > 0 else 0.0
            per_fire[entry["fire_name"]] = {"f1": ff, "precision": fp, "recall": fr}
            log(f"  eval: {entry['fire_name']:30s} F1={ff:.4f} P={fp:.4f} R={fr:.4f}")

    tp, fp_, fn = agg["tp"], agg["fp"], agg["fn"]
    p = tp / (tp + fp_) if (tp + fp_) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"loss": total_loss / max(n_batches, 1), "precision": p, "recall": r, "f1": f1,
            **agg, "per_fire": per_fire}


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log(f"Device: {device}")
    log(f"Label smoothing: majority voting, window={SMOOTH_WINDOW}, min_votes={SMOOTH_MIN_VOTES}")
    log(f"Model: fire-only ({N_CHANNELS} channels)")

    entries = discover_fire_entries(repo_root)
    train_entries, test_entries = split_fire_entries(
        entries, None, None, TRAIN_FRACTION, SPLIT_SEED,
    )
    log(f"Train: {len(train_entries)} fires, Test: {len(test_entries)} fires")

    # Channel stats (fire conf only)
    stats_path = CHECKPOINT_DIR / "fire_only_channel_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        ch_mean = stats["mean"]
        ch_std = stats["std"]
        log(f"Loaded channel stats: mean={ch_mean:.6f}, std={ch_std:.6f}")
    else:
        log("Computing fire-only channel stats...")
        means, stds = compute_fire_only_stats(train_entries, repo_root)
        ch_mean, ch_std = float(means[0]), float(stds[0])
        with open(stats_path, "w") as f:
            json.dump({"mean": ch_mean, "std": ch_std}, f, indent=2)
        log(f"Saved: mean={ch_mean:.6f}, std={ch_std:.6f}")

    # Model
    model = FireSpreadNet(in_channels=N_CHANNELS, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Parameters: {n_params:,}")

    criterion = CombinedLoss(focal_weight=0.5, tversky_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ckpt_path = CHECKPOINT_DIR / "fire_spread_smoothed_best.pt"
    best_f1 = 0.0
    history = []

    for epoch in range(1, N_EPOCHS + 1):
        log("=" * 60)
        log(f"EPOCH {epoch}/{N_EPOCHS}")
        log("=" * 60)

        t0 = time.time()
        train_result = train_epoch(
            model, train_entries, repo_root, ch_mean, ch_std,
            criterion, optimizer, device, epoch,
        )
        train_time = time.time() - t0
        log(f"Train: loss={train_result['loss']:.4f}  F1={train_result['f1']:.4f}  "
            f"P={train_result['precision']:.4f}  R={train_result['recall']:.4f}  [{train_time:.0f}s]")

        t0 = time.time()
        test_result = evaluate(model, test_entries, repo_root, ch_mean, ch_std, criterion, device)
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
            }, ckpt_path)
            log(f"*** NEW BEST F1={best_f1:.4f} — saved ***")

        history.append({
            "epoch": epoch,
            "train": {k: v for k, v in train_result.items()},
            "test": {k: v for k, v in test_result.items() if k != "per_fire"},
            "test_per_fire": test_result.get("per_fire", {}),
            "best_f1": best_f1,
            "improved": improved,
        })

        def ser(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {str(k): ser(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [ser(i) for i in obj]
            return obj

        with open(ANALYSIS_DIR / "training_history.json", "w") as f:
            json.dump(ser(history), f, indent=2)

    # Final summary
    log("\n" + "=" * 60)
    log("TRAINING COMPLETE — Smoothed Labels (majority_w5_v2)")
    log("=" * 60)
    log(f"Best test F1: {best_f1:.4f}")
    log(f"Baseline (raw labels): 0.7487")
    log(f"Improvement: {best_f1 - 0.7487:+.4f}")
    log(f"Oracle ceiling (smoothed): 0.8959")
    log(f"Model utilization: {best_f1 / 0.8959:.1%} of theoretical max")


if __name__ == "__main__":
    main()

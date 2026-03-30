"""Find optimal classification threshold for FireSpreadNet."""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fire_spread_model import (
    FireSpreadNet,
    build_frame,
    PAD_H, PAD_W,
)
from scripts.neighbor_cell_logreg import (
    discover_fire_entries,
    find_repo_root,
    iter_aligned_hours_for_fire,
    load_fire_entry_context,
    split_fire_entries,
)
from scripts.neighbor_cell_nn import pick_device

SEQ_LEN = 6
POSITIVE_THRESHOLD = 0.10
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7
BATCH_SIZE = 4

CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iter_fire_sequences(entry, repo_root, channel_means, channel_stds):
    ctx = load_fire_entry_context(entry, repo_root)
    goes_conf = ctx["goes_conf"]
    H, W = ctx["goes_shape"]
    frame_buffer: deque[np.ndarray] = deque(maxlen=SEQ_LEN)
    time_buffer: deque[int] = deque(maxlen=SEQ_LEN)

    for t, rtma_hour in iter_aligned_hours_for_fire(
        repo_root, goes_conf, ctx["goes_time_index"],
        ctx["rtma_manifest"], ctx["rtma_manifest_path"],
        ctx["goes_shape"], ctx["goes_transform"], ctx["goes_crs"],
        include_discounted_rain=False,
        discounted_rain_lookback_hours=0, discounted_rain_half_life_days=0.0,
    ):
        frame = build_frame(goes_conf[t], rtma_hour, (H, W), channel_means, channel_stds)
        frame_buffer.append(frame)
        time_buffer.append(t)
        if len(frame_buffer) < SEQ_LEN:
            continue
        times = list(time_buffer)
        if any(times[i+1] - times[i] != 1 for i in range(len(times)-1)):
            continue
        last_t = times[-1]
        if last_t + 1 >= goes_conf.shape[0]:
            continue

        target_conf = goes_conf[last_t + 1]
        target_binary = (target_conf >= POSITIVE_THRESHOLD).astype(np.float32)
        valid = np.isfinite(target_conf).astype(np.float32)

        target = np.zeros((1, PAD_H, PAD_W), dtype=np.float32)
        target[0, :H, :W] = target_binary
        mask = np.zeros((1, PAD_H, PAD_W), dtype=np.float32)
        mask[0, :H, :W] = valid

        frames = np.stack(list(frame_buffer), axis=0)
        yield frames, target, mask


def main():
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    device = pick_device()
    log(f"Device: {device}")

    # Load channel stats
    stats_path = CHECKPOINT_DIR / "fire_spread_channel_stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    channel_means = np.array(stats["means"], dtype=np.float32)
    channel_stds = np.array(stats["stds"], dtype=np.float32)

    # Load model
    ckpt_path = CHECKPOINT_DIR / "fire_spread_best.pt"
    model = FireSpreadNet(dropout=0.1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, best F1={ckpt.get('best_f1', '?')})")

    # Get test fires
    entries = discover_fire_entries(repo_root)
    _, test_entries = split_fire_entries(entries, None, None, TRAIN_FRACTION, SPLIT_SEED)
    log(f"Test fires: {[e['fire_name'] for e in test_entries]}")

    # Collect all predictions
    all_probs = []
    all_targets = []
    per_fire_data = {}

    with torch.no_grad():
        for entry in test_entries:
            fire_probs, fire_targets = [], []
            batch_f, batch_t, batch_m = [], [], []

            for frames, target, mask in iter_fire_sequences(
                entry, repo_root, channel_means, channel_stds,
            ):
                batch_f.append(torch.from_numpy(frames))
                batch_t.append(torch.from_numpy(target))
                batch_m.append(torch.from_numpy(mask))

                if len(batch_f) >= BATCH_SIZE:
                    x = torch.stack(batch_f).to(device)
                    y = torch.stack(batch_t).to(device)
                    m = torch.stack(batch_m).to(device)
                    logits = model(x)
                    probs = torch.sigmoid(logits)

                    valid = m.bool()
                    fire_probs.append(probs[valid].cpu().numpy())
                    fire_targets.append(y[valid].cpu().numpy())
                    batch_f, batch_t, batch_m = [], [], []

            if batch_f:
                x = torch.stack(batch_f).to(device)
                y = torch.stack(batch_t).to(device)
                m = torch.stack(batch_m).to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)
                valid = m.bool()
                fire_probs.append(probs[valid].cpu().numpy())
                fire_targets.append(y[valid].cpu().numpy())

            if fire_probs:
                fp = np.concatenate(fire_probs)
                ft = np.concatenate(fire_targets)
                per_fire_data[entry["fire_name"]] = (fp, ft)
                all_probs.append(fp)
                all_targets.append(ft)
                log(f"  {entry['fire_name']}: {len(fp):,} pixels, {int(ft.sum()):,} positive")

    all_p = np.concatenate(all_probs)
    all_t = np.concatenate(all_targets)
    log(f"Total: {len(all_p):,} pixels, {int(all_t.sum()):,} positive ({100*all_t.mean():.3f}%)")

    # Sweep thresholds
    log("\nThreshold sweep:")
    log(f"{'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'TP':>10} {'FP':>10} {'FN':>10}")
    log("-" * 75)

    best_f1 = -1.0
    best_result = {}
    results = []

    for thresh in np.concatenate([
        np.arange(0.05, 0.50, 0.05),
        np.arange(0.50, 0.95, 0.02),
        np.arange(0.95, 1.00, 0.005),
    ]):
        pred = (all_p >= thresh).astype(np.int32)
        tgt = all_t.astype(np.int32)
        tp = int(((pred == 1) & (tgt == 1)).sum())
        fp = int(((pred == 1) & (tgt == 0)).sum())
        fn = int(((pred == 0) & (tgt == 1)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results.append({"threshold": float(thresh), "f1": f1, "precision": p, "recall": r,
                        "tp": tp, "fp": fp, "fn": fn})

        if f1 > best_f1:
            best_f1 = f1
            best_result = results[-1]

        log(f"{thresh:>10.4f} {f1:>8.4f} {p:>10.4f} {r:>8.4f} {tp:>10,} {fp:>10,} {fn:>10,}")

    log("\n" + "=" * 75)
    log(f"BEST THRESHOLD: {best_result['threshold']:.4f}")
    log(f"  F1:        {best_result['f1']:.4f}")
    log(f"  Precision: {best_result['precision']:.4f}")
    log(f"  Recall:    {best_result['recall']:.4f}")
    log(f"  TP: {best_result['tp']:,}  FP: {best_result['fp']:,}  FN: {best_result['fn']:,}")

    # Per-fire at best threshold
    log(f"\nPer-fire results at threshold={best_result['threshold']:.4f}:")
    log(f"{'Fire':<30} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    log("-" * 60)
    for fire_name, (fp, ft) in per_fire_data.items():
        pred = (fp >= best_result["threshold"]).astype(np.int32)
        tgt = ft.astype(np.int32)
        tp = int(((pred == 1) & (tgt == 1)).sum())
        fpos = int(((pred == 1) & (tgt == 0)).sum())
        fn = int(((pred == 0) & (tgt == 1)).sum())
        p = tp / (tp + fpos) if (tp + fpos) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        log(f"{fire_name:<30} {f1:>8.4f} {p:>10.4f} {r:>8.4f}")

    # Save
    out_path = REPO_ROOT / "data" / "analysis" / "fire_spread" / "threshold_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"best": best_result, "all": results}, f, indent=2)
    log(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

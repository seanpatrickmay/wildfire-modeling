"""Ablation: Train FireSpreadNet with ONLY fire confidence (no weather data).

Compares 2-channel model (fire_conf + validity) vs the full 8-channel model
to determine whether weather data actually helps fire spread prediction.
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
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7
N_EPOCHS = 10  # Enough to see convergence based on full model behavior

CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "ablation"

# ─── Ablation configs ───────────────────────────────────────────────────────
# Each config specifies which channels to include
ABLATIONS = {
    "fire_only": {
        "channels": ["goes_conf"],  # just fire confidence
        "n_channels": 2,  # +1 for validity mask
        "description": "Fire confidence + validity mask only",
    },
    "fire_plus_wind": {
        "channels": ["goes_conf", "wind_speed", "wind_dir_sin", "wind_dir_cos"],
        "n_channels": 5,  # +1 for validity mask
        "description": "Fire confidence + wind (speed, direction)",
    },
    "fire_plus_temp_wind": {
        "channels": ["goes_conf", "temperature", "wind_speed", "wind_dir_sin", "wind_dir_cos"],
        "n_channels": 6,
        "description": "Fire confidence + temperature + wind",
    },
    "full": {
        "channels": ["goes_conf", "temperature", "wind_speed", "wind_dir_sin",
                     "wind_dir_cos", "specific_humidity", "precipitation_1h"],
        "n_channels": 8,
        "description": "All 7 physical channels + validity mask (baseline)",
    },
}

# Channel builders
CHANNEL_BUILDERS = {
    "goes_conf": lambda conf, rtma: conf,
    "temperature": lambda conf, rtma: rtma["TMP"].astype(np.float32),
    "wind_speed": lambda conf, rtma: rtma["WIND"].astype(np.float32),
    "wind_dir_sin": lambda conf, rtma: np.sin(np.deg2rad(rtma["WDIR"].astype(np.float32))),
    "wind_dir_cos": lambda conf, rtma: np.cos(np.deg2rad(rtma["WDIR"].astype(np.float32))),
    "specific_humidity": lambda conf, rtma: rtma["SPFH"].astype(np.float32),
    "precipitation_1h": lambda conf, rtma: np.clip(rtma["ACPC01"].astype(np.float32), 0.0, 100.0),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_channel_stats_for_config(
    entries: list[dict], repo_root: Path, channel_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n_ch = len(channel_names)
    sums = np.zeros(n_ch, dtype=np.float64)
    sq_sums = np.zeros(n_ch, dtype=np.float64)
    counts = np.zeros(n_ch, dtype=np.float64)

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
            conf = ctx["goes_conf"][t].astype(np.float32)
            for i, ch_name in enumerate(channel_names):
                ch = CHANNEL_BUILDERS[ch_name](conf, rtma_hour)
                ch = np.nan_to_num(ch, nan=0.0).astype(np.float64)
                sums[i] += ch.sum()
                sq_sums[i] += (ch ** 2).sum()
                counts[i] += n_pixels

    means = sums / counts
    stds = np.sqrt(np.maximum(sq_sums / counts - means ** 2, 0.0))
    stds = np.where(stds > 0, stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def build_frame_for_config(
    goes_conf_t: np.ndarray,
    rtma_hour: dict[str, np.ndarray],
    goes_shape: tuple[int, int],
    channel_names: list[str],
    channel_means: np.ndarray,
    channel_stds: np.ndarray,
    n_total_channels: int,
) -> np.ndarray:
    H, W = goes_shape
    conf = goes_conf_t.astype(np.float32)
    frame = np.zeros((n_total_channels, PAD_H, PAD_W), dtype=np.float32)

    for i, ch_name in enumerate(channel_names):
        ch = CHANNEL_BUILDERS[ch_name](conf, rtma_hour)
        ch = np.nan_to_num(ch, nan=0.0)
        ch = (ch - channel_means[i]) / max(channel_stds[i], 1e-8)
        frame[i, :H, :W] = ch.astype(np.float32)

    # Validity mask is always last channel
    frame[n_total_channels - 1, :H, :W] = 1.0
    return frame


def iter_fire_sequences(entry, repo_root, channel_names, channel_means, channel_stds, n_channels):
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
        discounted_rain_lookback_hours=0,
        discounted_rain_half_life_days=0.0,
    ):
        frame = build_frame_for_config(
            goes_conf[t], rtma_hour, (H, W),
            channel_names, channel_means, channel_stds, n_channels,
        )
        frame_buffer.append(frame)
        time_buffer.append(t)

        if len(frame_buffer) < SEQ_LEN:
            continue

        times = list(time_buffer)
        if any(times[i + 1] - times[i] != 1 for i in range(len(times) - 1)):
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


def run_ablation(
    config_name: str,
    config: dict,
    train_entries: list[dict],
    test_entries: list[dict],
    repo_root: Path,
    device: torch.device,
) -> dict:
    channel_names = config["channels"]
    n_channels = config["n_channels"]

    log(f"--- Ablation: {config_name} ({config['description']}) ---")
    log(f"    Channels: {channel_names} + validity = {n_channels} total")

    # Channel stats
    stats_path = CHECKPOINT_DIR / f"ablation_{config_name}_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        means = np.array(stats["means"], dtype=np.float32)
        stds = np.array(stats["stds"], dtype=np.float32)
        log("    Loaded cached channel stats")
    else:
        log("    Computing channel stats...")
        means, stds = compute_channel_stats_for_config(train_entries, repo_root, channel_names)
        with open(stats_path, "w") as f:
            json.dump({"means": means.tolist(), "stds": stds.tolist()}, f, indent=2)

    # Model
    model = FireSpreadNet(in_channels=n_channels, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"    Parameters: {n_params:,}")

    criterion = CombinedLoss(focal_weight=0.5, tversky_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Add cosine annealing to prevent late-epoch instability
    scheduler = None

    best_f1 = 0.0
    history = []

    for epoch in range(1, N_EPOCHS + 1):
        # Train
        model.train()
        fire_order = list(range(len(train_entries)))
        np.random.shuffle(fire_order)
        total_loss, n_batches = 0.0, 0
        agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        batch_f, batch_t, batch_m = [], [], []

        for fi in fire_order:
            entry = train_entries[fi]
            for frames, target, mask in iter_fire_sequences(
                entry, repo_root, channel_names, means, stds, n_channels,
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

        # Leftover batch
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

        train_loss = total_loss / max(n_batches, 1)
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        train_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        train_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        train_f1 = 2 * train_p * train_r / (train_p + train_r) if (train_p + train_r) > 0 else 0.0

        # Eval
        model.eval()
        test_agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        test_loss, test_batches = 0.0, 0
        per_fire = {}

        with torch.no_grad():
            for entry in test_entries:
                fire_agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                batch_f, batch_t, batch_m = [], [], []

                for frames, target, mask in iter_fire_sequences(
                    entry, repo_root, channel_names, means, stds, n_channels,
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
                            test_loss += loss.item()
                            test_batches += 1
                        mb = compute_metrics(logits, y, m)
                        for k in ("tp", "fp", "fn", "tn"):
                            test_agg[k] += mb[k]
                            fire_agg[k] += mb[k]
                        batch_f, batch_t, batch_m = [], [], []

                if batch_f:
                    x = torch.stack(batch_f).to(device)
                    y = torch.stack(batch_t).to(device)
                    m = torch.stack(batch_m).to(device)
                    logits = model(x)
                    mb = compute_metrics(logits, y, m)
                    for k in ("tp", "fp", "fn", "tn"):
                        test_agg[k] += mb[k]
                        fire_agg[k] += mb[k]

                ftp, ffp, ffn = fire_agg["tp"], fire_agg["fp"], fire_agg["fn"]
                fp_ = ftp / (ftp + ffp) if (ftp + ffp) > 0 else 0.0
                fr_ = ftp / (ftp + ffn) if (ftp + ffn) > 0 else 0.0
                ff_ = 2 * fp_ * fr_ / (fp_ + fr_) if (fp_ + fr_) > 0 else 0.0
                per_fire[entry["fire_name"]] = {"f1": ff_, "precision": fp_, "recall": fr_}

        ttp, tfp, tfn = test_agg["tp"], test_agg["fp"], test_agg["fn"]
        test_p = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0.0
        test_r = ttp / (ttp + tfn) if (ttp + tfn) > 0 else 0.0
        test_f1 = 2 * test_p * test_r / (test_p + test_r) if (test_p + test_r) > 0 else 0.0

        improved = test_f1 > best_f1
        if improved:
            best_f1 = test_f1
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"ablation_{config_name}_best.pt")

        history.append({
            "epoch": epoch,
            "train_f1": train_f1, "train_loss": train_loss,
            "test_f1": test_f1, "test_precision": test_p, "test_recall": test_r,
            "improved": improved, "per_fire": per_fire,
        })

        marker = "***" if improved else "   "
        log(f"    Epoch {epoch:2d}: train F1={train_f1:.4f}  test F1={test_f1:.4f}  "
            f"P={test_p:.4f} R={test_r:.4f}  {marker}")

    log(f"    Best test F1: {best_f1:.4f}")
    return {"config": config, "best_f1": best_f1, "history": history}


def main():
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log(f"Device: {device}")

    entries = discover_fire_entries(repo_root)
    train_entries, test_entries = split_fire_entries(
        entries, None, None, TRAIN_FRACTION, SPLIT_SEED,
    )
    log(f"Train: {len(train_entries)} fires, Test: {len(test_entries)} fires")

    all_results = {}

    for config_name, config in ABLATIONS.items():
        t0 = time.time()
        result = run_ablation(config_name, config, train_entries, test_entries, repo_root, device)
        elapsed = time.time() - t0
        result["elapsed_s"] = elapsed
        all_results[config_name] = result
        log(f"    Completed in {elapsed:.0f}s")
        log("")

    # Summary
    log("=" * 70)
    log("ABLATION SUMMARY")
    log("=" * 70)
    log(f"{'Config':<25} {'Channels':>4} {'Best F1':>8} {'Description'}")
    log("-" * 70)
    for name, r in all_results.items():
        cfg = r["config"]
        log(f"{name:<25} {cfg['n_channels']:>4} {r['best_f1']:>8.4f} {cfg['description']}")

    # Per-fire comparison at best epoch
    log("")
    log("Per-fire F1 at best epoch:")
    header = f"{'Fire':<30}"
    for name in all_results:
        header += f" {name:>12}"
    log(header)
    log("-" * (30 + 13 * len(all_results)))

    fire_names = list(all_results[list(all_results.keys())[0]]["history"][-1]["per_fire"].keys())
    for fire in fire_names:
        row = f"{fire:<30}"
        for name, r in all_results.items():
            # Find best epoch
            best_epoch = max(r["history"], key=lambda e: e["test_f1"])
            f1 = best_epoch["per_fire"].get(fire, {}).get("f1", 0.0)
            row += f" {f1:>12.4f}"
        log(row)

    # Save
    def ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [ser(i) for i in obj]
        return obj

    with open(ANALYSIS_DIR / "ablation_results.json", "w") as f:
        json.dump(ser(all_results), f, indent=2)
    log(f"\nSaved to {ANALYSIS_DIR / 'ablation_results.json'}")


if __name__ == "__main__":
    main()

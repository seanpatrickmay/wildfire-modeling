"""Train FireSpreadNet v2 on V4 pipeline data (ablation-optimized features).

Uses 38-channel stack with pre-shifted fire features, GPM precipitation,
extended daily weather, PDSI drought, terrain ruggedness, and firebreaks.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.firespreadnet.architecture import (
    CombinedLoss,
    FireSpreadNetV2,
    augment_sequence_v2,
)
from data.pipeline_loader import (
    CHANNEL_ORDER_V4,
    WIND_U_CH_V4,
    WIND_V_CH_V4,
    build_channel_stack_v4,
    compute_channel_stats,
    iter_grid_sequences_v3,
    load_processed_fire_data,
    normalize_stack,
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
CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "firespreadnet_v4"

MIN_FIRE_PIXELS = 50
TRAIN_FRACTION = 0.7
SPLIT_SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def load_and_prepare_fire(
    fire_name: str, pipeline_dir: Path, pad_h: int, pad_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load processed fire, build V3 channel stack.

    Returns (stack, labels, validity, loss_weights) all padded to (T, pad_h, pad_w).
    """
    processed, features = load_processed_fire_data(fire_name, str(pipeline_dir))
    stack = build_channel_stack_v4(processed, features, pad_h, pad_w)

    T, H, W = processed["labels"].shape
    labels_padded = np.zeros((T, pad_h, pad_w), dtype=np.float32)
    labels_padded[:, :H, :W] = processed["labels"]

    validity_padded = np.zeros((T, pad_h, pad_w), dtype=np.float32)
    validity_padded[:, :H, :W] = processed["validity"]

    lw = processed.get("loss_weights")
    lw_padded = np.zeros((T, pad_h, pad_w), dtype=np.float32)
    if lw is not None:
        lw_padded[:, :H, :W] = lw
    else:
        lw_padded = validity_padded.copy()

    return stack, labels_padded, validity_padded, lw_padded


def discover_fires(pipeline_dir: Path) -> list[str]:
    """Discover fires with both processed and Features NPZ files."""
    fires: list[str] = []
    if not pipeline_dir.is_dir():
        return fires
    for subdir in sorted(pipeline_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        proc_dir = subdir / "processed"
        has_proc = bool(list(proc_dir.glob(f"{name}_*_processed.npz"))) if proc_dir.is_dir() else False
        has_feat = bool(list(subdir.glob(f"{name}_*_Features.npz")))
        if has_proc and has_feat:
            fires.append(name)
    return fires


def train_epoch(
    model: FireSpreadNetV2,
    fire_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    rng = torch.Generator()
    rng.manual_seed(epoch * 1337)

    total_loss = 0.0
    n_batches = 0
    agg: dict[str, int] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    batch_f, batch_t, batch_m = [], [], []

    fire_order = list(range(len(fire_data)))
    np.random.RandomState(epoch * 1337).shuffle(fire_order)

    for fire_idx in fire_order:
        stack, labels, validity, loss_weights = fire_data[fire_idx]
        for frames, target, mask, _w in iter_grid_sequences_v3(
            stack, labels, validity, loss_weights, SEQ_LEN,
        ):
            f_t = torch.from_numpy(frames)
            t_t = torch.from_numpy(target)
            m_t = torch.from_numpy(mask)

            f_t, t_t, m_t = augment_sequence_v2(
                f_t, t_t, m_t, rng,
                wind_u_ch=WIND_U_CH_V4, wind_v_ch=WIND_V_CH_V4,
            )

            batch_f.append(f_t)
            batch_t.append(t_t)
            batch_m.append(m_t)

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


def evaluate(
    model: FireSpreadNetV2,
    fire_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    criterion: CombinedLoss,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    agg: dict[str, int] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    batch_f, batch_t, batch_m = [], [], []

    with torch.no_grad():
        for stack, labels, validity, loss_weights in fire_data:
            for frames, target, mask, _w in iter_grid_sequences_v3(
                stack, labels, validity, loss_weights, SEQ_LEN,
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
                    batch_f, batch_t, batch_m = [], [], []

        if batch_f:
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

    avg_loss = total_loss / max(n_batches, 1)
    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"loss": avg_loss, "precision": p, "recall": r, "f1": f1, **agg}


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


def main() -> None:
    if PIPELINE_DIR is None:
        log("ERROR: Could not find wildfire-data-pipeline data directory.")
        sys.exit(1)

    log(f"Pipeline data directory: {PIPELINE_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    log(f"Device: {device}")
    log(f"Channels: {len(CHANNEL_ORDER_V4)} (V4 — ablation-optimized, 38ch)")

    # Discover fires with processed data
    all_fires = discover_fires(PIPELINE_DIR)
    log(f"Discovered {len(all_fires)} fires: {all_fires}")
    if not all_fires:
        log("ERROR: No processed fire data found.")
        sys.exit(1)

    # Check fire pixel counts
    fire_info: list[dict[str, Any]] = []
    for name in all_fires:
        processed, _ = load_processed_fire_data(name, str(PIPELINE_DIR))
        n_fire = int(processed["labels"].sum())
        T, H, W = processed["labels"].shape
        log(f"  {name}: {T}h, {H}x{W}, {n_fire} fire pixels")
        if n_fire < MIN_FIRE_PIXELS:
            log(f"  SKIPPING {name}")
            continue
        fire_info.append({"name": name, "n_fire": n_fire, "T": T, "H": H, "W": W})

    if len(fire_info) < 2:
        log(f"ERROR: Need >= 2 usable fires, found {len(fire_info)}.")
        sys.exit(1)

    usable_names = [f["name"] for f in fire_info]
    log(f"Usable fires ({len(usable_names)}): {usable_names}")

    # Train/test split
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(usable_names)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * TRAIN_FRACTION))
    n_train = min(n_train, len(shuffled) - 1)
    train_names = sorted(shuffled[:n_train])
    test_names = sorted(shuffled[n_train:])
    log(f"Train fires ({len(train_names)}): {train_names}")
    log(f"Test fires  ({len(test_names)}):  {test_names}")

    # Pad dims
    max_h = max(f["H"] for f in fire_info)
    max_w = max(f["W"] for f in fire_info)
    pad_h = ((max_h + 15) // 16) * 16
    pad_w = ((max_w + 15) // 16) * 16
    log(f"Padding: {pad_h}x{pad_w} (max data: {max_h}x{max_w})")

    # Load data — compute channel stats incrementally to avoid double-storing
    log("Loading train fires and computing channel stats...")
    train_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    train_stacks_for_stats: list[np.ndarray] = []
    train_validity_for_stats: list[np.ndarray] = []

    for name in train_names:
        stack, labels, validity, lw = load_and_prepare_fire(name, PIPELINE_DIR, pad_h, pad_w)
        log(f"  Train: {name} stack={stack.shape}, fire_px={labels.sum():.0f}")
        train_data.append((stack, labels, validity, lw))
        train_stacks_for_stats.append(stack)
        train_validity_for_stats.append(validity)

    means, stds = compute_channel_stats(train_stacks_for_stats, train_validity_for_stats)
    del train_stacks_for_stats, train_validity_for_stats  # free references

    # Normalize in-place to avoid doubling memory
    def normalize_inplace(stack: np.ndarray) -> None:
        validity_idx = len(CHANNEL_ORDER_V4) - 1
        m = means.reshape(1, -1, 1, 1)
        s = stds.reshape(1, -1, 1, 1)
        stack[:, :validity_idx] -= m[:, :validity_idx]
        stack[:, :validity_idx] /= s[:, :validity_idx]

    for s, _l, _v, _w in train_data:
        normalize_inplace(s)

    log("Loading test fires...")
    test_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for name in test_names:
        stack, labels, validity, lw = load_and_prepare_fire(name, PIPELINE_DIR, pad_h, pad_w)
        normalize_inplace(stack)
        log(f"  Test:  {name} stack={stack.shape}, fire_px={labels.sum():.0f}")
        test_data.append((stack, labels, validity, lw))

    import gc; gc.collect()
    log(f"Data loaded. {len(train_data)} train + {len(test_data)} test fires.")

    # Model
    n_channels = len(CHANNEL_ORDER_V4)
    model = FireSpreadNetV2(in_channels=n_channels, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"FireSpreadNetV2: {n_params:,} params, {n_channels} channels")

    criterion = CombinedLoss(focal_weight=0.5, tversky_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1 = 0.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, N_EPOCHS + 1):
        log(f"{'='*60}\nEPOCH {epoch}/{N_EPOCHS}\n{'='*60}")

        t0 = time.time()
        tr = train_epoch(model, train_data, criterion, optimizer, device, epoch)
        log(f"Train: loss={tr['loss']:.4f} F1={tr['f1']:.4f} P={tr['precision']:.4f} R={tr['recall']:.4f} [{time.time()-t0:.0f}s]")

        t0 = time.time()
        te = evaluate(model, test_data, criterion, device)
        log(f"Test:  loss={te['loss']:.4f} F1={te['f1']:.4f} P={te['precision']:.4f} R={te['recall']:.4f} [{time.time()-t0:.0f}s]")

        for i, name in enumerate(test_names):
            fr = evaluate(model, [test_data[i]], criterion, device)
            log(f"  {name}: F1={fr['f1']:.4f} P={fr['precision']:.4f} R={fr['recall']:.4f}")

        if te["f1"] > best_f1:
            best_f1 = te["f1"]
            torch.save({
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": epoch, "best_f1": best_f1,
                "train_fires": train_names, "test_fires": test_names,
                "in_channels": n_channels,
                "data_version": "v4_ablation_optimized",
            }, CHECKPOINT_DIR / "firespreadnet_v4_best.pt")
            log(f"*** NEW BEST F1={best_f1:.4f} ***")

        history.append({
            "epoch": epoch,
            "train": {k: v for k, v in tr.items()},
            "test": {k: v for k, v in te.items()},
            "best_f1": best_f1,
        })

    # Save stats and history
    with open(CHECKPOINT_DIR / "firespreadnet_v4_channel_stats.json", "w") as f:
        json.dump({"means": means.tolist(), "stds": stds.tolist(), "channel_order": list(CHANNEL_ORDER_V4)}, f, indent=2)

    per_fire = {}
    for i, name in enumerate(test_names):
        per_fire[name] = _serializable(evaluate(model, [test_data[i]], criterion, device))

    with open(ANALYSIS_DIR / "training_history.json", "w") as f:
        json.dump(_serializable({
            "train_fires": train_names, "test_fires": test_names,
            "best_f1": best_f1, "per_fire_test": per_fire,
            "epochs": history,
            "data_version": "v4_ablation_optimized",
            "config": {"seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "lr": LR,
                       "n_epochs": N_EPOCHS, "n_channels": n_channels,
                       "channel_order": list(CHANNEL_ORDER_V4)},
        }), f, indent=2)

    log(f"\nDONE — FireSpreadNet V4: best F1={best_f1:.4f}")
    for name, res in per_fire.items():
        log(f"  {name}: F1={res['f1']:.4f} P={res['precision']:.4f} R={res['recall']:.4f}")


if __name__ == "__main__":
    main()

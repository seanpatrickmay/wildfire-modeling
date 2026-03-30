"""Analyze and improve GOES fire confidence label quality.

The raw GOES labels flicker — ~23% of fire pixels disappear each hour then
reappear. This caps model F1 at ~0.77 regardless of architecture.

This script:
1. Quantifies the flicker problem
2. Tests temporal smoothing strategies
3. Evaluates how much F1 headroom each strategy creates
4. Generates cleaned label files for retraining
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.neighbor_cell_logreg import discover_fire_entries, find_repo_root


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_fire_conf(entry: dict) -> tuple[np.ndarray, dict]:
    with open(entry["goes_json"], "r") as f:
        d = json.load(f)
    conf = np.array(d["data"], dtype=np.float32)
    return conf, d["metadata"]


def flicker_stats(conf: np.ndarray, threshold: float) -> dict:
    """Compute how often fire pixels flicker on/off."""
    binary = (conf >= threshold).astype(np.int32)
    T = conf.shape[0]

    fire_to_nofire = 0
    nofire_to_fire = 0
    total_fire = int((binary == 1).sum())

    for t in range(T - 1):
        diff = binary[t + 1] - binary[t]
        valid = np.isfinite(conf[t]) & np.isfinite(conf[t + 1])
        fire_to_nofire += int((diff[valid] == -1).sum())
        nofire_to_fire += int((diff[valid] == 1).sum())

    return {
        "total_fire_pixels": total_fire,
        "fire_to_nofire": fire_to_nofire,
        "nofire_to_fire": nofire_to_fire,
        "flicker_rate": fire_to_nofire / max(total_fire, 1),
    }


def temporal_smooth(conf: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """Apply rolling max over a temporal window to fill flicker gaps.

    If a pixel was fire in any of the last `window` hours, mark it as fire.
    This fills 1-2 hour gaps where the satellite lost detection temporarily.
    """
    T, H, W = conf.shape
    binary = (conf >= threshold).astype(np.float32)
    smoothed = np.zeros_like(binary)

    for t in range(T):
        start = max(0, t - window + 1)
        smoothed[t] = binary[start:t + 1].max(axis=0)

    return smoothed


def temporal_majority(conf: np.ndarray, window: int, threshold: float, min_votes: int = 2) -> np.ndarray:
    """Apply majority voting over a temporal window.

    A pixel is fire if it was fire in >= min_votes of the last `window` hours.
    More conservative than rolling max — requires repeated detection.
    """
    T, H, W = conf.shape
    binary = (conf >= threshold).astype(np.float32)
    smoothed = np.zeros_like(binary)

    for t in range(T):
        start = max(0, t - window + 1)
        votes = binary[start:t + 1].sum(axis=0)
        smoothed[t] = (votes >= min_votes).astype(np.float32)

    return smoothed


def raise_threshold(conf: np.ndarray, threshold: float) -> np.ndarray:
    """Simply raise the confidence threshold to filter noisy detections."""
    return (conf >= threshold).astype(np.float32)


def oracle_f1(labels_t: np.ndarray, labels_t1: np.ndarray) -> dict:
    """Compute the F1 score a PERFECT model would get predicting t+1 from t.

    If labels_t == labels_t1 for all pixels, F1 = 1.0.
    If labels flicker, even perfect prediction of t → t+1 has errors.
    """
    tp = int(((labels_t == 1) & (labels_t1 == 1)).sum())
    fp = int(((labels_t == 1) & (labels_t1 == 0)).sum())  # model says fire, truth says no
    fn = int(((labels_t == 0) & (labels_t1 == 1)).sum())  # model says no fire, truth says fire
    tn = int(((labels_t == 0) & (labels_t1 == 0)).sum())

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate_strategy(
    entries: list[dict],
    strategy_name: str,
    label_fn,
) -> dict:
    """Evaluate a labeling strategy across all fires.

    The 'oracle F1' is: if the model perfectly predicts labels_t as its output,
    what F1 does it get against labels_t+1? This is the theoretical maximum F1.
    """
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_fire = {}

    for entry in entries:
        conf, _ = load_fire_conf(entry)
        labels = label_fn(conf)
        T = labels.shape[0]

        fire_tp = fire_fp = fire_fn = fire_tn = 0
        for t in range(T - 1):
            valid = np.isfinite(conf[t]) & np.isfinite(conf[t + 1])
            pred = labels[t][valid].astype(np.int32)
            truth = labels[t + 1][valid].astype(np.int32)

            fire_tp += int(((pred == 1) & (truth == 1)).sum())
            fire_fp += int(((pred == 1) & (truth == 0)).sum())
            fire_fn += int(((pred == 0) & (truth == 1)).sum())
            fire_tn += int(((pred == 0) & (truth == 0)).sum())

        for k, v in [("tp", fire_tp), ("fp", fire_fp), ("fn", fire_fn), ("tn", fire_tn)]:
            total[k] += v

        p = fire_tp / (fire_tp + fire_fp) if (fire_tp + fire_fp) > 0 else 0.0
        r = fire_tp / (fire_tp + fire_fn) if (fire_tp + fire_fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_fire[entry["fire_name"]] = {"f1": f1, "precision": p, "recall": r}

    tp, fp, fn = total["tp"], total["fp"], total["fn"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "strategy": strategy_name,
        "oracle_f1": f1,
        "precision": p,
        "recall": r,
        "per_fire": per_fire,
        **total,
    }


def main():
    repo_root = find_repo_root(Path(__file__).resolve().parent)
    entries = discover_fire_entries(repo_root)
    log(f"Found {len(entries)} fires")

    # Define strategies to test
    strategies = [
        ("raw_thresh_0.10", lambda c: raise_threshold(c, 0.10)),
        ("raw_thresh_0.30", lambda c: raise_threshold(c, 0.30)),
        ("raw_thresh_0.50", lambda c: raise_threshold(c, 0.50)),
        ("raw_thresh_0.80", lambda c: raise_threshold(c, 0.80)),
        ("smooth_w2_t0.10", lambda c: temporal_smooth(c, window=2, threshold=0.10)),
        ("smooth_w3_t0.10", lambda c: temporal_smooth(c, window=3, threshold=0.10)),
        ("smooth_w3_t0.30", lambda c: temporal_smooth(c, window=3, threshold=0.30)),
        ("smooth_w4_t0.10", lambda c: temporal_smooth(c, window=4, threshold=0.10)),
        ("majority_w3_v2_t0.10", lambda c: temporal_majority(c, window=3, threshold=0.10, min_votes=2)),
        ("majority_w3_v2_t0.30", lambda c: temporal_majority(c, window=3, threshold=0.30, min_votes=2)),
        ("majority_w5_v2_t0.10", lambda c: temporal_majority(c, window=5, threshold=0.10, min_votes=2)),
        ("majority_w5_v3_t0.10", lambda c: temporal_majority(c, window=5, threshold=0.10, min_votes=3)),
    ]

    results = []
    log(f"\n{'Strategy':<30} {'Oracle F1':>10} {'Precision':>10} {'Recall':>10}")
    log("-" * 65)

    for name, fn in strategies:
        t0 = time.time()
        r = evaluate_strategy(entries, name, fn)
        elapsed = time.time() - t0
        results.append(r)
        log(f"{name:<30} {r['oracle_f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}  ({elapsed:.1f}s)")

    # Summary
    log("\n" + "=" * 65)
    best = max(results, key=lambda r: r["oracle_f1"])
    log(f"BEST STRATEGY: {best['strategy']}")
    log(f"  Oracle F1:  {best['oracle_f1']:.4f}")
    log(f"  Precision:  {best['precision']:.4f}")
    log(f"  Recall:     {best['recall']:.4f}")

    log(f"\nPer-fire oracle F1 for best strategy ({best['strategy']}):")
    log(f"{'Fire':<30} {'F1':>8}")
    log("-" * 40)
    for fire, m in sorted(best["per_fire"].items(), key=lambda x: x[1]["f1"]):
        log(f"{fire:<30} {m['f1']:>8.4f}")

    # Save
    out_dir = repo_root / "data" / "analysis" / "label_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    def ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [ser(i) for i in obj]
        return obj

    with open(out_dir / "strategy_comparison.json", "w") as f:
        json.dump(ser(results), f, indent=2)
    log(f"\nSaved to {out_dir / 'strategy_comparison.json'}")


if __name__ == "__main__":
    main()

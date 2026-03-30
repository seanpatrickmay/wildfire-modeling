"""Retrain XGBoost + MLP on sentinel-cleaned data, build ensemble, evaluate.

The ACPC01 sentinel fix (clamp to [0, 100]) is in neighbor_cell_logreg.py.
This script retrains neighborhood-based models with clean data and evaluates
them alongside a simple averaging ensemble.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.neighbor_cell_logreg import (
    build_feature_schema,
    discover_fire_entries,
    find_repo_root,
    fit_zscore_normalizer,
    iter_fire_hour_samples,
    split_fire_entries,
)
from scripts.neighbor_cell_nn import (
    FeedForwardMLP,
    MLPConfig,
    build_mlp,
    count_model_parameters,
    pick_device,
    predict_probabilities,
    set_seed,
)

# ─── Constants ──────────────────────────────────────────────────────────────
POSITIVE_THRESHOLD = 0.5
DISCOUNTED_RAIN_LOOKBACK_HOURS = 720
DISCOUNTED_RAIN_HALF_LIFE_DAYS = 7.0
NEGATIVE_SUBSAMPLE_RATE = 0.03
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7

CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "retrain_clean"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def collect_samples(entries, repo_root, schema, normalizer, subsample_neg=None, rng=None):
    X_parts, y_parts = [], []
    for entry in entries:
        for fire_name, t, X_hour, y_hour in iter_fire_hour_samples(
            entry, repo_root, schema, POSITIVE_THRESHOLD,
            discounted_rain_lookback_hours=DISCOUNTED_RAIN_LOOKBACK_HOURS,
            discounted_rain_half_life_days=DISCOUNTED_RAIN_HALF_LIFE_DAYS,
        ):
            if subsample_neg is not None and rng is not None:
                pos_mask = y_hour == 1
                neg_mask = ~pos_mask
                neg_keep = rng.random(int(neg_mask.sum())) < subsample_neg
                keep = np.zeros(len(y_hour), dtype=bool)
                keep[pos_mask] = True
                keep[np.where(neg_mask)[0][neg_keep]] = True
                X_hour, y_hour = X_hour[keep], y_hour[keep]
            if normalizer is not None:
                X_hour = normalizer.transform(X_hour).astype(np.float32)
            X_parts.append(X_hour)
            y_parts.append(y_hour)
        log(f"  collected: {fire_name}")
    return np.concatenate(X_parts), np.concatenate(y_parts)


def collect_per_fire(entries, repo_root, schema, normalizer):
    result = {}
    for entry in entries:
        xs, ys = [], []
        for _, _, X_h, y_h in iter_fire_hour_samples(
            entry, repo_root, schema, POSITIVE_THRESHOLD,
            discounted_rain_lookback_hours=DISCOUNTED_RAIN_LOOKBACK_HOURS,
            discounted_rain_half_life_days=DISCOUNTED_RAIN_HALF_LIFE_DAYS,
        ):
            xs.append(normalizer.transform(X_h).astype(np.float32))
            ys.append(y_h)
        if xs:
            result[entry["fire_name"]] = (np.concatenate(xs), np.concatenate(ys))
        log(f"  per-fire collected: {entry['fire_name']}")
    return result


def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_t = y_true.astype(np.int32)
    tp = int(((y_pred == 1) & (y_t == 1)).sum())
    fp = int(((y_pred == 1) & (y_t == 0)).sum())
    fn = int(((y_pred == 0) & (y_t == 1)).sum())
    tn = int(((y_pred == 0) & (y_t == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {"threshold": threshold, "accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(y_true, y_prob, n=200):
    best_f1, best = -1.0, {}
    for t in np.linspace(0.01, 0.999, n):
        m = metrics(y_true, y_prob, t)
        if m["f1"] > best_f1:
            best_f1, best = m["f1"], m
    return best


def train_mlp_from_arrays(X_train, y_train, schema, config, device):
    """Train MLP directly from arrays — no re-reading from disk."""
    set_seed(config.seed)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / max(pos, 1)

    model = build_mlp(schema, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    n = len(y_train)
    for epoch in range(1, config.epochs + 1):
        model.train()
        indices = np.random.permutation(n)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n, config.batch_size):
            end = min(n, start + config.batch_size)
            idx = indices[start:end]
            xb = torch.from_numpy(X_train[idx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y_train[idx]).to(device=device, dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        log(f"  MLP epoch {epoch}/{config.epochs}: loss={epoch_loss/max(n_batches,1):.4f}")

    return model


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
    log(f"Test fires: {[e['fire_name'] for e in test_entries]}")

    schema = build_feature_schema(include_discounted_rain=True, include_ndvi=False)
    log(f"Features: {schema.n_features}")

    # Fit normalizer (one pass through train fires)
    log("Fitting normalizer...")
    normalizer = fit_zscore_normalizer(
        train_entries, repo_root, schema, POSITIVE_THRESHOLD,
        enabled=True,
        discounted_rain_lookback_hours=DISCOUNTED_RAIN_LOOKBACK_HOURS,
        discounted_rain_half_life_days=DISCOUNTED_RAIN_HALF_LIFE_DAYS,
    )
    log(f"Normalizer: {normalizer.samples_used} samples")

    # Collect training data (one pass, 3% neg subsample)
    rng = np.random.default_rng(SPLIT_SEED)
    log("Collecting training samples...")
    X_train, y_train = collect_samples(
        train_entries, repo_root, schema, normalizer,
        subsample_neg=NEGATIVE_SUBSAMPLE_RATE, rng=rng,
    )
    log(f"Train: {X_train.shape[0]:,} samples, {int(y_train.sum()):,} pos ({100*y_train.mean():.2f}%)")

    # Collect test data (one pass, no subsampling)
    log("Collecting test samples...")
    X_test, y_test = collect_samples(test_entries, repo_root, schema, normalizer)
    log(f"Test: {X_test.shape[0]:,} samples, {int(y_test.sum()):,} pos ({100*y_test.mean():.2f}%)")

    results = {}

    # ──── XGBoost ────
    log("=" * 60)
    log("Training XGBoost (500 trees, depth 8)...")
    t0 = time.time()
    pos, neg = int(y_train.sum()), int(len(y_train) - y_train.sum())
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        scale_pos_weight=neg / max(pos, 1),
        tree_method="hist", eval_metric="logloss",
        random_state=SPLIT_SEED, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, verbose=True)
    xgb_time = time.time() - t0
    log(f"XGBoost trained in {xgb_time:.1f}s")

    log("Evaluating XGBoost...")
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_m05 = metrics(y_test, xgb_probs, 0.5)
    xgb_best = best_threshold(y_test, xgb_probs)
    results["xgboost"] = {"at_0.5": xgb_m05, "best": xgb_best, "time_s": xgb_time}
    log(f"XGB @0.5:  F1={xgb_m05['f1']:.4f}  P={xgb_m05['precision']:.4f}  R={xgb_m05['recall']:.4f}")
    log(f"XGB best:  F1={xgb_best['f1']:.4f}  @thresh={xgb_best['threshold']:.4f}")

    # ──── MLP (from arrays, no re-reading) ────
    log("=" * 60)
    config = MLPConfig()
    log(f"Training MLP (256→128→64, {config.epochs} epochs, from arrays)...")
    t0 = time.time()
    mlp_model = train_mlp_from_arrays(X_train, y_train, schema, config, device)
    mlp_time = time.time() - t0
    log(f"MLP trained in {mlp_time:.1f}s ({count_model_parameters(mlp_model)} params)")

    log("Evaluating MLP...")
    mlp_probs = predict_probabilities(mlp_model, X_test, batch_size=config.batch_size, device=str(device))
    mlp_m05 = metrics(y_test, mlp_probs, 0.5)
    mlp_best = best_threshold(y_test, mlp_probs)
    results["mlp"] = {"at_0.5": mlp_m05, "best": mlp_best, "time_s": mlp_time}
    log(f"MLP @0.5:  F1={mlp_m05['f1']:.4f}  P={mlp_m05['precision']:.4f}  R={mlp_m05['recall']:.4f}")
    log(f"MLP best:  F1={mlp_best['f1']:.4f}  @thresh={mlp_best['threshold']:.4f}")

    # ──── Ensemble (XGBoost + MLP average) ────
    log("=" * 60)
    log("Ensemble (XGBoost + MLP simple average)...")
    ens_probs = (xgb_probs + mlp_probs) / 2.0
    ens_m05 = metrics(y_test, ens_probs, 0.5)
    ens_best = best_threshold(y_test, ens_probs)
    results["ensemble"] = {"at_0.5": ens_m05, "best": ens_best, "method": "avg(xgb,mlp)"}
    log(f"ENS @0.5:  F1={ens_m05['f1']:.4f}  P={ens_m05['precision']:.4f}  R={ens_m05['recall']:.4f}")
    log(f"ENS best:  F1={ens_best['f1']:.4f}  @thresh={ens_best['threshold']:.4f}")

    # ──── Per-fire breakdown ────
    log("=" * 60)
    log("Per-fire evaluation...")
    per_fire = collect_per_fire(test_entries, repo_root, schema, normalizer)
    pf_results = {}
    for fire_name, (X_f, y_f) in per_fire.items():
        xp = xgb_model.predict_proba(X_f)[:, 1]
        mp = predict_probabilities(mlp_model, X_f, batch_size=config.batch_size, device=str(device))
        ep = (xp + mp) / 2.0
        pf_results[fire_name] = {
            "n": len(y_f), "pos": int(y_f.sum()),
            "xgb": metrics(y_f, xp), "mlp": metrics(y_f, mp), "ens": metrics(y_f, ep),
            "xgb_best": best_threshold(y_f, xp), "mlp_best": best_threshold(y_f, mp),
            "ens_best": best_threshold(y_f, ep),
        }
        log(f"  {fire_name:30s}  XGB={pf_results[fire_name]['xgb']['f1']:.4f}  "
            f"MLP={pf_results[fire_name]['mlp']['f1']:.4f}  "
            f"ENS={pf_results[fire_name]['ens']['f1']:.4f}")
    results["per_fire"] = pf_results

    # ──── Summary ────
    log("=" * 60)
    log("SUMMARY — Clean data (ACPC01 clamped [0,100])")
    log(f"{'Model':<12} {'F1@0.5':>8} {'Prec':>8} {'Recall':>8} | {'BestF1':>8} {'@Thresh':>8}")
    log("-" * 60)
    for name, key in [("XGBoost", "xgboost"), ("MLP", "mlp"), ("Ensemble", "ensemble")]:
        r = results[key]
        a, b = r["at_0.5"], r["best"]
        log(f"{name:<12} {a['f1']:>8.4f} {a['precision']:>8.4f} {a['recall']:>8.4f} | {b['f1']:>8.4f} {b['threshold']:>8.4f}")

    # Save
    def serializable(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [serializable(i) for i in obj]
        return obj

    report_path = ANALYSIS_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(serializable(results), f, indent=2)
    log(f"Saved: {report_path}")

    xgb_model.save_model(str(CHECKPOINT_DIR / "xgboost_clean.json"))
    torch.save(mlp_model.state_dict(), str(CHECKPOINT_DIR / "mlp_clean.pt"))

    norm_meta = {"mean": normalizer.mean.tolist(), "std": normalizer.std.tolist(),
                 "std_safe": normalizer.std_safe.tolist(), "samples_used": normalizer.samples_used}
    with open(CHECKPOINT_DIR / "normalizer_clean_meta.json", "w") as f:
        json.dump(norm_meta, f, indent=2)
    log("Checkpoints saved. Done.")


if __name__ == "__main__":
    main()

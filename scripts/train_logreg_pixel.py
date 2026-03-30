"""Logistic Regression baseline for fire prediction using V3 pipeline data.

Uses prev_fire_state (pre-shifted by pipeline) — no temporal leakage.
Same feature set as XGBoost V3 for fair comparison.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline_data_loader import (
    load_processed_fire_data,
    iter_pixel_samples_v3,
    get_pixel_feature_names_v3,
    subsample_negatives,
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
NEG_POS_RATIO = 5.0
SEED = 42

MIN_FIRE_PIXELS = 50
TRAIN_FRACTION = 0.7
SPLIT_SEED = 42

CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "logreg_v3"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def collect_samples(fire_name: str, pipeline_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    log(f"Loading {fire_name} (V3 processed)...")
    processed, features = load_processed_fire_data(fire_name, str(pipeline_dir))

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    count = 0

    for fv, label, _weight in iter_pixel_samples_v3(
        processed, features, seq_len=SEQ_LEN,
    ):
        X_list.append(fv)
        y_list.append(label)
        count += 1
        if count % 50_000 == 0:
            n_pos = sum(1 for v in y_list if v > 0)
            log(f"  ...{count:,} samples (pos: {n_pos:,})")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    n_pos = int((y > 0).sum())
    n_neg = int((y == 0).sum())
    log(f"  Samples: {len(y):,} (pos: {n_pos:,}, neg: {n_neg:,})")
    return X, y


def discover_fires(pipeline_dir: Path) -> list[str]:
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


def fbeta_from_counts(tp, fp, fn, beta=2.0):
    d = (1 + beta**2) * tp + beta**2 * fn + fp
    return (1 + beta**2) * tp / d if d > 0 else 0.0


def main() -> None:
    if PIPELINE_DIR is None:
        log("ERROR: Could not find wildfire-data-pipeline data directory.")
        sys.exit(1)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    except ImportError:
        log("ERROR: scikit-learn not installed.")
        sys.exit(1)

    log(f"Pipeline data directory: {PIPELINE_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover fires
    all_fires = discover_fires(PIPELINE_DIR)
    log(f"Discovered {len(all_fires)} fires: {all_fires}")

    # Filter by fire pixels
    usable_fires: list[str] = []
    for name in all_fires:
        processed, _ = load_processed_fire_data(name, str(PIPELINE_DIR))
        n_fire = int(processed["labels"].sum())
        T, H, W = processed["labels"].shape
        log(f"  {name}: {T}h, {H}x{W}, {n_fire} fire pixels")
        if n_fire < MIN_FIRE_PIXELS:
            log(f"  SKIPPING {name}")
            continue
        usable_fires.append(name)

    if len(usable_fires) < 2:
        log(f"ERROR: Need >= 2 usable fires.")
        sys.exit(1)

    log(f"Usable fires ({len(usable_fires)}): {usable_fires}")

    # Train/test split
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(usable_fires)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * TRAIN_FRACTION))
    n_train = min(n_train, len(shuffled) - 1)
    train_names = sorted(shuffled[:n_train])
    test_names = sorted(shuffled[n_train:])
    log(f"Train fires ({len(train_names)}): {train_names}")
    log(f"Test fires  ({len(test_names)}):  {test_names}")

    # Collect samples
    X_train_parts, y_train_parts = [], []
    for name in train_names:
        X_f, y_f = collect_samples(name, PIPELINE_DIR)
        X_train_parts.append(X_f)
        y_train_parts.append(y_f)
    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    log(f"Total train: {len(y_train):,} (pos: {int((y_train > 0).sum()):,})")

    test_fire_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    X_test_parts, y_test_parts = [], []
    for name in test_names:
        X_f, y_f = collect_samples(name, PIPELINE_DIR)
        test_fire_data[name] = (X_f, y_f)
        X_test_parts.append(X_f)
        y_test_parts.append(y_f)
    X_test = np.concatenate(X_test_parts)
    y_test = np.concatenate(y_test_parts)
    log(f"Total test:  {len(y_test):,} (pos: {int((y_test > 0).sum()):,})")

    # Subsample negatives
    X_train_sub, y_train_sub = subsample_negatives(X_train, y_train, ratio=NEG_POS_RATIO, seed=SEED)
    n_pos_sub = int((y_train_sub > 0).sum())
    n_neg_sub = int((y_train_sub == 0).sum())
    log(f"After subsampling: {len(y_train_sub):,} (pos: {n_pos_sub:,}, neg: {n_neg_sub:,})")

    # Standardize features
    log("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_test_scaled = scaler.transform(X_test)

    # Replace NaN/inf from scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Train
    log("Training Logistic Regression...")
    t0 = time.time()
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=SEED,
        n_jobs=1,
    )
    model.fit(X_train_scaled, y_train_sub)
    train_time = time.time() - t0
    log(f"Training took {train_time:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f2 = fbeta_from_counts(tp, fp, fn)
    auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) == 2 else 0.0

    log(f"Overall: F1={f1:.4f}  F2={f2:.4f}  P={p:.4f}  R={r:.4f}  AUC={auc:.4f}")

    # Per-fire
    per_fire: dict[str, Any] = {}
    for name in test_names:
        X_f, y_f = test_fire_data[name]
        X_f_scaled = np.nan_to_num(scaler.transform(X_f), nan=0.0, posinf=0.0, neginf=0.0)
        y_fp = model.predict(X_f_scaled)
        y_fpr = model.predict_proba(X_f_scaled)[:, 1]
        tp_f = int(((y_fp == 1) & (y_f == 1)).sum())
        fp_f = int(((y_fp == 1) & (y_f == 0)).sum())
        fn_f = int(((y_fp == 0) & (y_f == 1)).sum())
        p_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        r_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        f1_f = 2 * p_f * r_f / (p_f + r_f) if (p_f + r_f) > 0 else 0
        f2_f = fbeta_from_counts(tp_f, fp_f, fn_f)
        auc_f = float(roc_auc_score(y_f, y_fpr)) if len(np.unique(y_f)) == 2 else 0.0
        n_pos_f = int((y_f > 0).sum())
        log(f"  {name}: F1={f1_f:.4f} F2={f2_f:.4f} P={p_f:.4f} R={r_f:.4f} AUC={auc_f:.4f} (pos={n_pos_f:,})")
        per_fire[name] = {
            "f1": f1_f, "f2": f2_f, "precision": p_f, "recall": r_f,
            "auc_roc": auc_f, "n_pos": n_pos_f,
            "tp": tp_f, "fp": fp_f, "fn": fn_f,
        }

    # Feature importance (coefficients)
    feature_names = get_pixel_feature_names_v3(seq_len=SEQ_LEN)
    coefs = np.abs(model.coef_[0])
    sorted_idx = np.argsort(coefs)[::-1]
    log("Top 10 features (by |coefficient|):")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        nm = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        log(f"  {rank}. {nm} (|coef|={coefs[idx]:.4f})")

    # Save
    with open(CHECKPOINT_DIR / "logreg_v3_best.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    with open(ANALYSIS_DIR / "results.json", "w") as f:
        json.dump(_serializable({
            "f1": f1, "f2": f2, "precision": p, "recall": r, "auc_roc": auc,
            "tp": tp, "fp": fp, "fn": fn,
            "train_fires": train_names, "test_fires": test_names,
            "per_fire_test": per_fire,
            "train_samples": len(y_train_sub), "test_samples": len(y_test),
            "train_time_s": train_time,
            "data_version": "v3_prev_fire_state",
        }), f, indent=2)

    log(f"\nDONE — Logistic Regression V3: F1={f1:.4f} F2={f2:.4f} P={p:.4f} R={r:.4f}")


if __name__ == "__main__":
    main()

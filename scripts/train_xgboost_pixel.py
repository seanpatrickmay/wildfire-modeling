"""Single-pixel XGBoost for fire prediction using V3 pipeline data.

Uses prev_fire_state (pre-shifted by pipeline) — no temporal leakage by construction.
Target is labels[t], features include prev_fire_state[t] = labels[t-1].
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
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "xgboost_v3"

XGB_PARAMS = {
    "max_depth": 8,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": SEED,
}


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
    """Load a fire and collect all pixel samples."""
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
        if count % 10_000 == 0:
            n_pos = sum(1 for v in y_list if v > 0)
            log(f"  ...{count:,} samples (pos: {n_pos:,})")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    n_pos = int((y > 0).sum())
    n_neg = int((y == 0).sum())
    log(f"  Samples: {len(y):,} (pos: {n_pos:,}, neg: {n_neg:,})")
    return X, y


def discover_fires(pipeline_dir: Path) -> list[str]:
    """Auto-discover fires that have both processed and Features NPZ files."""
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


def main() -> None:
    if PIPELINE_DIR is None:
        log("ERROR: Could not find wildfire-data-pipeline data directory.")
        sys.exit(1)

    try:
        import xgboost as xgb
    except ImportError:
        log("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    except ImportError:
        log("ERROR: scikit-learn not installed.")
        sys.exit(1)

    log(f"Pipeline data directory: {PIPELINE_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover fires with processed data
    all_fires = discover_fires(PIPELINE_DIR)
    log(f"Discovered {len(all_fires)} fires with processed data: {all_fires}")
    if not all_fires:
        log("ERROR: No processed fire data. Run pipeline processing first.")
        sys.exit(1)

    # Filter by fire pixel count
    usable_fires: list[str] = []
    for name in all_fires:
        processed, _ = load_processed_fire_data(name, str(PIPELINE_DIR))
        n_fire = int(processed["labels"].sum())
        T, H, W = processed["labels"].shape
        log(f"  {name}: {T}h, {H}x{W}, {n_fire} fire pixels")
        if n_fire < MIN_FIRE_PIXELS:
            log(f"  SKIPPING {name} ({n_fire} < {MIN_FIRE_PIXELS})")
            continue
        usable_fires.append(name)

    if len(usable_fires) < 2:
        log(f"ERROR: Need >= 2 usable fires, found {len(usable_fires)}.")
        sys.exit(1)

    log(f"Usable fires ({len(usable_fires)}): {usable_fires}")

    # Train/test split by fire
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

    scale_pos_weight = n_neg_sub / max(n_pos_sub, 1)

    # Train
    log("Training XGBoost (V3 — no leakage)...")
    t0 = time.time()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train_sub, y_train_sub, eval_set=[(X_test, y_test)], verbose=50)
    train_time = time.time() - t0
    log(f"Training took {train_time:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    auc_roc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) == 2 else 0.0

    log(f"Overall: F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  AUC={auc_roc:.4f}")

    # Per-fire
    per_fire: dict[str, Any] = {}
    for name in test_names:
        X_f, y_f = test_fire_data[name]
        y_fp = model.predict(X_f)
        y_fpr = model.predict_proba(X_f)[:, 1]
        f1_f = float(f1_score(y_f, y_fp, zero_division=0))
        prec_f = float(precision_score(y_f, y_fp, zero_division=0))
        rec_f = float(recall_score(y_f, y_fp, zero_division=0))
        auc_f = float(roc_auc_score(y_f, y_fpr)) if len(np.unique(y_f)) == 2 else 0.0
        n_pos_f = int((y_f > 0).sum())
        log(f"  {name}: F1={f1_f:.4f} P={prec_f:.4f} R={rec_f:.4f} AUC={auc_f:.4f} (pos={n_pos_f:,})")
        per_fire[name] = {"f1": f1_f, "precision": prec_f, "recall": rec_f, "auc_roc": auc_f, "n_pos": n_pos_f}

    # Feature importance
    feature_names = get_pixel_feature_names_v3(seq_len=SEQ_LEN)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    log("Top 10 features:")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        nm = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        log(f"  {rank}. {nm} (gain={importances[idx]:.4f})")

    # Save
    with open(CHECKPOINT_DIR / "xgboost_v3_best.pkl", "wb") as f:
        pickle.dump(model, f)

    all_feat_imp = {feature_names[i]: float(importances[i]) for i in sorted_idx if i < len(feature_names)}
    with open(ANALYSIS_DIR / "results.json", "w") as f:
        json.dump(_serializable({
            "f1": f1, "precision": precision, "recall": recall, "auc_roc": auc_roc,
            "train_fires": train_names, "test_fires": test_names,
            "per_fire_test": per_fire,
            "train_samples": len(y_train_sub), "test_samples": len(y_test),
            "train_time_s": train_time, "feature_importance": all_feat_imp,
            "data_version": "v3_prev_fire_state",
        }), f, indent=2)

    log(f"\nDONE — XGBoost V3: F1={f1:.4f} P={precision:.4f} R={recall:.4f}")


if __name__ == "__main__":
    main()

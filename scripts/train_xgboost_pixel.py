"""Single-pixel XGBoost for fire prediction using pipeline data."""
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
    load_fire_data,
    smooth_labels,
    iter_pixel_samples,
    get_pixel_feature_names,
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
CONFIDENCE_THRESHOLD = 0.30
SMOOTH_WINDOW = 5
SMOOTH_MIN_VOTES = 2
NEG_POS_RATIO = 5.0
SEED = 42

# Auto-discover fires that have both FusedConf and Features NPZ
MIN_FIRE_PIXELS = 50  # Skip fires with fewer smoothed fire pixels
TRAIN_FRACTION = 0.7  # 70% of fires for training
SPLIT_SEED = 42

CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "xgboost_v2"

# XGBoost hyperparameters
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


# ─── Sample Collection ──────────────────────────────────────────────────────
def collect_samples(
    fire_name: str,
    pipeline_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a fire and collect all pixel samples into arrays."""
    log(f"Loading {fire_name}...")
    fire_arrays, feat_arrays, fire_meta, feat_meta = load_fire_data(
        fire_name, str(pipeline_dir),
    )

    confidence = fire_arrays["data"]
    cloud_mask = fire_arrays.get("cloud_mask")
    obs_valid = fire_arrays.get("observation_valid")

    labels, validity = smooth_labels(
        confidence, cloud_mask, obs_valid,
        window=SMOOTH_WINDOW,
        min_votes=SMOOTH_MIN_VOTES,
        threshold=CONFIDENCE_THRESHOLD,
    )

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    count = 0

    for fv, label in iter_pixel_samples(
        fire_arrays, feat_arrays, labels, validity, seq_len=SEQ_LEN,
    ):
        X_list.append(fv)
        y_list.append(label)
        count += 1
        if count % 10_000 == 0:
            n_pos = sum(1 for v in y_list if v > 0)
            log(f"  ...{count:,} samples collected (pos: {n_pos:,})")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    n_pos = int((y > 0).sum())
    n_neg = int((y == 0).sum())
    log(f"  Samples: {len(y):,} (pos: {n_pos:,}, neg: {n_neg:,})")

    return X, y


# ─── Training ───────────────────────────────────────────────────────────────
def train_and_evaluate(
    train_name: str,
    test_name: str,
    pipeline_dir: Path,
) -> dict[str, Any]:
    """Train on one fire, test on the other. Returns results dict."""
    try:
        import xgboost as xgb
    except ImportError:
        log("ERROR: xgboost is not installed. Run: pip install xgboost")
        sys.exit(1)

    try:
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            classification_report,
        )
    except ImportError:
        log("ERROR: scikit-learn is not installed. Run: pip install scikit-learn")
        sys.exit(1)

    direction = f"{train_name} -> {test_name}"

    # Collect train samples
    X_train, y_train = collect_samples(train_name, pipeline_dir)

    # Collect test samples
    X_test, y_test = collect_samples(test_name, pipeline_dir)

    # Subsample negatives for training
    X_train_sub, y_train_sub = subsample_negatives(
        X_train, y_train, ratio=NEG_POS_RATIO, seed=SEED,
    )
    n_pos_sub = int((y_train_sub > 0).sum())
    n_neg_sub = int((y_train_sub == 0).sum())
    log(f"  After subsampling: {len(y_train_sub):,} (pos: {n_pos_sub:,}, neg: {n_neg_sub:,})")

    # Compute scale_pos_weight from remaining imbalance
    if n_pos_sub > 0:
        scale_pos_weight = n_neg_sub / n_pos_sub
    else:
        scale_pos_weight = 1.0
    log(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    # Train XGBoost
    log(f"Training XGBoost ({direction})...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    train_time = time.time() - t0
    log(f"  Training took {train_time:.1f}s")

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    auc_roc = float(roc_auc_score(y_test, y_prob))

    log(f"Test results ({test_name}):")
    log(f"  F1: {f1:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")
    log(f"  AUC-ROC: {auc_roc:.4f}")

    # Feature importance (gain-based)
    feature_names = get_pixel_feature_names(seq_len=SEQ_LEN)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    log("Top 10 features:")
    top_features = []
    for rank, idx in enumerate(sorted_indices[:10], 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        gain = float(importances[idx])
        log(f"  {rank}. {name} (gain={gain:.4f})")
        top_features.append({"rank": rank, "name": name, "gain": gain})

    # Full feature importance
    all_features = {}
    for idx in sorted_indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        all_features[name] = float(importances[idx])

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "direction": direction,
        "train_fire": train_name,
        "test_fire": test_name,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
        "train_samples": len(y_train_sub),
        "train_pos": n_pos_sub,
        "train_neg": n_neg_sub,
        "test_samples": len(y_test),
        "test_pos": int((y_test > 0).sum()),
        "test_neg": int((y_test == 0).sum()),
        "scale_pos_weight": scale_pos_weight,
        "train_time_s": train_time,
        "top_features": top_features,
        "feature_importance": all_features,
        "classification_report": _serializable(report),
        "model": model,
    }


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

    try:
        import xgboost as xgb
    except ImportError:
        log("ERROR: xgboost is not installed. Run: pip install xgboost")
        sys.exit(1)
    try:
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, roc_auc_score, classification_report,
        )
    except ImportError:
        log("ERROR: scikit-learn is not installed. Run: pip install scikit-learn")
        sys.exit(1)

    log(f"Pipeline data directory: {PIPELINE_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Sequence length: {SEQ_LEN}")
    log(f"XGBoost params: {XGB_PARAMS}")

    # 1. Auto-discover available fires
    all_fires = discover_fires(PIPELINE_DIR)
    log(f"Discovered {len(all_fires)} fires: {all_fires}")
    if not all_fires:
        log("ERROR: No fire data found.")
        sys.exit(1)

    # 2. Load each fire, compute smoothed labels, check fire pixel count
    usable_fires: list[str] = []
    for name in all_fires:
        log(f"Checking {name}...")
        fire_arrays, feat_arrays, fire_meta, feat_meta = load_fire_data(name, str(PIPELINE_DIR))
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
        usable_fires.append(name)

    if len(usable_fires) < 2:
        log(f"ERROR: Need at least 2 usable fires, found {len(usable_fires)}.")
        sys.exit(1)

    log(f"Usable fires ({len(usable_fires)}): {usable_fires}")

    # 3. Split into train/test fires
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(usable_fires)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * TRAIN_FRACTION))
    n_train = min(n_train, len(shuffled) - 1)  # Ensure at least 1 test fire
    train_names = sorted(shuffled[:n_train])
    test_names = sorted(shuffled[n_train:])
    log(f"Train fires ({len(train_names)}): {train_names}")
    log(f"Test fires  ({len(test_names)}):  {test_names}")

    # 4. Collect pixel samples from ALL training fires
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    for name in train_names:
        X_fire, y_fire = collect_samples(name, PIPELINE_DIR)
        X_train_parts.append(X_fire)
        y_train_parts.append(y_fire)
    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    log(f"Total train: {len(y_train):,} samples (pos: {int((y_train > 0).sum()):,})")

    # 5. Collect pixel samples from ALL test fires (keep per-fire arrays for breakdown)
    test_fire_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    X_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []
    for name in test_names:
        X_fire, y_fire = collect_samples(name, PIPELINE_DIR)
        test_fire_data[name] = (X_fire, y_fire)
        X_test_parts.append(X_fire)
        y_test_parts.append(y_fire)
    X_test = np.concatenate(X_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)
    log(f"Total test:  {len(y_test):,} samples (pos: {int((y_test > 0).sum()):,})")

    if (y_train > 0).sum() == 0:
        log("ERROR: No positive training samples. Cannot train.")
        sys.exit(1)

    # 6. Subsample negatives from training set
    X_train_sub, y_train_sub = subsample_negatives(X_train, y_train, ratio=NEG_POS_RATIO, seed=SEED)
    n_pos_sub = int((y_train_sub > 0).sum())
    n_neg_sub = int((y_train_sub == 0).sum())
    log(f"After subsampling: {len(y_train_sub):,} (pos: {n_pos_sub:,}, neg: {n_neg_sub:,})")

    scale_pos_weight = n_neg_sub / max(n_pos_sub, 1)
    log(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # 7. Train XGBoost
    log("Training XGBoost...")
    t0 = time.time()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train_sub, y_train_sub, eval_set=[(X_test, y_test)], verbose=50)
    train_time = time.time() - t0
    log(f"Training took {train_time:.1f}s")

    # 8. Evaluate on combined test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    auc_roc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) == 2 else 0.0

    log(f"Overall test results:")
    log(f"  F1: {f1:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}")
    log(f"  AUC-ROC: {auc_roc:.4f}")

    # 9. Print per-fire test metrics
    per_fire_results: dict[str, Any] = {}
    log("Per-fire test results:")
    for name in test_names:
        X_f, y_f = test_fire_data[name]
        if len(y_f) == 0:
            log(f"  {name}: no samples")
            per_fire_results[name] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "auc_roc": 0.0, "n_samples": 0}
            continue
        y_f_pred = model.predict(X_f)
        y_f_prob = model.predict_proba(X_f)[:, 1]
        f1_f = float(f1_score(y_f, y_f_pred, zero_division=0))
        prec_f = float(precision_score(y_f, y_f_pred, zero_division=0))
        rec_f = float(recall_score(y_f, y_f_pred, zero_division=0))
        auc_f = float(roc_auc_score(y_f, y_f_prob)) if len(np.unique(y_f)) == 2 else 0.0
        n_pos_f = int((y_f > 0).sum())
        log(f"  {name}: F1={f1_f:.4f}  P={prec_f:.4f}  R={rec_f:.4f}  AUC={auc_f:.4f}  "
            f"(samples={len(y_f):,}, pos={n_pos_f:,})")
        per_fire_results[name] = {
            "f1": f1_f, "precision": prec_f, "recall": rec_f, "auc_roc": auc_f,
            "n_samples": len(y_f), "n_pos": n_pos_f,
        }

    # Feature importance
    feature_names = get_pixel_feature_names(seq_len=SEQ_LEN)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    log("Top 10 features:")
    for rank, idx in enumerate(sorted_indices[:10], 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        log(f"  {rank}. {name} (gain={importances[idx]:.4f})")

    # Save model
    model_path = CHECKPOINT_DIR / "xgboost_v2_best.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log(f"Model saved to {model_path}")

    # Save results with fire-level breakdown
    all_features = {feature_names[i]: float(importances[i]) for i in sorted_indices if i < len(feature_names)}
    results_path = ANALYSIS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(_serializable({
            "f1": f1, "precision": precision, "recall": recall, "auc_roc": auc_roc,
            "train_fires": train_names, "test_fires": test_names,
            "per_fire_test": per_fire_results,
            "train_samples": len(y_train_sub), "test_samples": len(y_test),
            "train_time_s": train_time, "feature_importance": all_features,
            "config": {
                "seq_len": SEQ_LEN, "xgb_params": XGB_PARAMS,
                "min_fire_pixels": MIN_FIRE_PIXELS,
                "train_fraction": TRAIN_FRACTION, "split_seed": SPLIT_SEED,
            },
        }), f, indent=2)

    log("\n" + "=" * 60)
    log("TRAINING COMPLETE — XGBoost v2 Pixel Model (multi-fire CV)")
    log("=" * 60)
    log(f"Train fires: {train_names}")
    log(f"Test fires:  {test_names}")
    log(f"F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  AUC={auc_roc:.4f}")
    for name, res in per_fire_results.items():
        log(f"  {name}: F1={res['f1']:.4f}  P={res['precision']:.4f}  R={res['recall']:.4f}")


if __name__ == "__main__":
    main()

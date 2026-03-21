"""Train an ensemble wildfire prediction model.

Trains LightGBM, XGBoost, and Logistic Regression on fire-holdout splits,
tunes thresholds on validation fires, and reports per-fire test metrics.

Usage:
    python best_model/train_ensemble.py --data best_model/data --output best_model/results
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def fire_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    fire_labels: list[str],
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Split data by fire event (no spatial/temporal leakage)."""
    rng = np.random.default_rng(seed)
    fires = sorted(set(fire_labels))
    rng.shuffle(fires)

    n_train = max(1, int(len(fires) * train_fraction))
    n_val = max(1, int(len(fires) * val_fraction))
    train_fires = set(fires[:n_train])
    val_fires = set(fires[n_train:n_train + n_val])
    test_fires = set(fires[n_train + n_val:])
    if not test_fires:
        test_fires = val_fires  # fallback

    labels_arr = np.array(fire_labels)
    train_mask = np.isin(labels_arr, list(train_fires))
    val_mask = np.isin(labels_arr, list(val_fires))
    test_mask = np.isin(labels_arr, list(test_fires))

    return {
        "X_train": X[train_mask], "y_train": y[train_mask],
        "X_val": X[val_mask], "y_val": y[val_mask],
        "X_test": X[test_mask], "y_test": y[test_mask],
        "train_fires": sorted(train_fires),
        "val_fires": sorted(val_fires),
        "test_fires": sorted(test_fires),
        "test_labels": labels_arr[test_mask],
    }


def train_lightgbm(X_train, y_train, X_val, y_val, seed=42):
    """Train LightGBM with early stopping."""
    pos = y_train.sum()
    neg = len(y_train) - pos
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_jobs": -1,
        "seed": seed,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": neg / max(pos, 1),
    }
    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, seed=42):
    """Train XGBoost with early stopping."""
    pos = y_train.sum()
    neg = len(y_train) - pos
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "nthread": -1,
        "seed": seed,
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": neg / max(pos, 1),
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return model


def train_logistic(X_train, y_train, scaler):
    """Train L2-regularized logistic regression on scaled features."""
    X_scaled = scaler.transform(X_train)
    model = LogisticRegression(
        penalty="l2", C=1.0, max_iter=500,
        class_weight="balanced", solver="lbfgs",
    )
    model.fit(X_scaled, y_train)
    return model


def predict_ensemble(models, X, scaler, weights=None):
    """Average predicted probabilities across models."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    probs = np.zeros(len(X), dtype=np.float64)
    for (name, model), w in zip(models, weights):
        if name == "lgbm":
            p = model.predict(X)
        elif name == "xgb":
            p = model.predict(xgb.DMatrix(X))
        elif name == "lr":
            p = model.predict_proba(scaler.transform(X))[:, 1]
        else:
            raise ValueError(f"Unknown model: {name}")
        probs += w * p
    return probs


def find_best_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    best_idx = np.argmax(f1)
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    return float(thresholds[best_idx]), float(f1[best_idx])


def evaluate(y_true, y_prob, threshold):
    """Compute classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "n_samples": len(y_true),
        "n_positives": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }


def per_fire_metrics(y_true, y_prob, labels, threshold):
    """Compute metrics per fire event."""
    results = {}
    for fire in sorted(set(labels)):
        mask = labels == fire
        if mask.sum() < 10:
            continue
        results[fire] = evaluate(y_true[mask], y_prob[mask], threshold)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train ensemble wildfire model")
    parser.add_argument("--data", default="best_model/data")
    parser.add_argument("--output", default="best_model/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    print("Loading data...")
    X = np.load(os.path.join(args.data, "X.npy"))
    y = np.load(os.path.join(args.data, "y.npy"))
    with open(os.path.join(args.data, "fire_labels.json")) as f:
        fire_labels = json.load(f)
    with open(os.path.join(args.data, "feature_names.json")) as f:
        feature_names = json.load(f)

    print(f"  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  {y.sum():.0f} positives ({100*y.mean():.1f}%)")
    print(f"  {len(set(fire_labels))} fires")

    # Split
    split = fire_holdout_split(X, y, fire_labels, seed=args.seed)
    print(f"\nSplit:")
    print(f"  Train: {len(split['train_fires'])} fires, {len(split['y_train'])} samples ({100*split['y_train'].mean():.1f}% pos)")
    print(f"  Val:   {len(split['val_fires'])} fires, {len(split['y_val'])} samples ({100*split['y_val'].mean():.1f}% pos)")
    print(f"  Test:  {len(split['test_fires'])} fires, {len(split['y_test'])} samples ({100*split['y_test'].mean():.1f}% pos)")

    # Scale for logistic regression
    scaler = StandardScaler()
    scaler.fit(split["X_train"])

    # Train models
    print("\nTraining LightGBM...")
    t0 = time.time()
    lgbm_model = train_lightgbm(split["X_train"], split["y_train"], split["X_val"], split["y_val"], args.seed)
    print(f"  Done in {time.time()-t0:.1f}s ({lgbm_model.best_iteration} rounds)")

    print("Training XGBoost...")
    t0 = time.time()
    xgb_model = train_xgboost(split["X_train"], split["y_train"], split["X_val"], split["y_val"], args.seed)
    print(f"  Done in {time.time()-t0:.1f}s ({xgb_model.best_iteration} rounds)")

    print("Training Logistic Regression...")
    t0 = time.time()
    lr_model = train_logistic(split["X_train"], split["y_train"], scaler)
    print(f"  Done in {time.time()-t0:.1f}s")

    models = [("lgbm", lgbm_model), ("xgb", xgb_model), ("lr", lr_model)]

    # Predict on validation set to find optimal threshold
    print("\nTuning threshold on validation set...")
    val_probs = predict_ensemble(models, split["X_val"], scaler)
    best_thresh, best_f1 = find_best_threshold(split["y_val"], val_probs)
    print(f"  Best threshold: {best_thresh:.3f} (F1={best_f1:.3f})")

    # Also find per-model thresholds for comparison
    model_results = {}
    for name, model in models:
        if name == "lgbm":
            p = model.predict(split["X_val"])
        elif name == "xgb":
            p = model.predict(xgb.DMatrix(split["X_val"]))
        elif name == "lr":
            p = model.predict_proba(scaler.transform(split["X_val"]))[:, 1]
        t, f = find_best_threshold(split["y_val"], p)
        model_results[name] = {"val_threshold": t, "val_f1": f}
        print(f"  {name}: threshold={t:.3f}, F1={f:.3f}")

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    test_probs = predict_ensemble(models, split["X_test"], scaler)
    test_metrics = evaluate(split["y_test"], test_probs, best_thresh)
    print(f"  Ensemble F1:        {test_metrics['f1']:.4f}")
    print(f"  Ensemble Precision: {test_metrics['precision']:.4f}")
    print(f"  Ensemble Recall:    {test_metrics['recall']:.4f}")
    print(f"  Ensemble ROC-AUC:   {test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] else "  ROC-AUC: N/A")

    # Per-model test metrics
    for name, model in models:
        if name == "lgbm":
            p = model.predict(split["X_test"])
        elif name == "xgb":
            p = model.predict(xgb.DMatrix(split["X_test"]))
        elif name == "lr":
            p = model.predict_proba(scaler.transform(split["X_test"]))[:, 1]
        m = evaluate(split["y_test"], p, model_results[name]["val_threshold"])
        model_results[name]["test"] = m
        print(f"  {name:6s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  AUC={m['roc_auc']:.4f}" if m['roc_auc'] else f"  {name:6s} F1={m['f1']:.4f}")

    # Per-fire breakdown
    print("\nPer-fire test metrics (ensemble):")
    fire_metrics = per_fire_metrics(split["y_test"], test_probs, split["test_labels"], best_thresh)
    for fire, m in fire_metrics.items():
        print(f"  {fire:20s}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  pos_rate={m['positive_rate']:.3f}  n={m['n_samples']}")

    # Feature importance
    lgbm_importance = dict(zip(feature_names, lgbm_model.feature_importance("gain").tolist()))
    sorted_features = sorted(lgbm_importance.items(), key=lambda x: -x[1])
    print("\nTop features (LightGBM gain):")
    for fname, imp in sorted_features[:10]:
        print(f"  {fname:30s}  {imp:.1f}")

    # Save results
    report = {
        "split": {
            "train_fires": split["train_fires"],
            "val_fires": split["val_fires"],
            "test_fires": split["test_fires"],
            "train_samples": len(split["y_train"]),
            "val_samples": len(split["y_val"]),
            "test_samples": len(split["y_test"]),
        },
        "ensemble": {
            "threshold": best_thresh,
            "test_metrics": test_metrics,
        },
        "per_model": model_results,
        "per_fire": fire_metrics,
        "feature_importance_lgbm": lgbm_importance,
        "feature_names": feature_names,
    }

    with open(os.path.join(args.output, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    lgbm_model.save_model(os.path.join(args.output, "lgbm_model.txt"))
    xgb_model.save_model(os.path.join(args.output, "xgb_model.json"))
    with open(os.path.join(args.output, "lr_model.pkl"), "wb") as f:
        pickle.dump({"model": lr_model, "scaler": scaler}, f)

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()

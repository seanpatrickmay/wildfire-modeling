# XGBoost Feature Ablation Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the optimal feature set for XGBoost pixel classification by running a staged ablation study across all pipeline features, optimizing for F2 score.

**Architecture:** A single `models/xgboost/ablation.py` script defines all ~25 experiment configs and runs them sequentially. It reuses `data/pipeline_loader.py` for data loading but accepts configurable feature key lists and training options (soft labels, sample weights). Results are saved per-run to JSON and a summary table is printed at the end.

**Tech Stack:** XGBoost, scikit-learn, NumPy. No new dependencies.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `models/xgboost/ablation.py` | Ablation runner — defines experiment configs, trains XGBoost, computes F2/F1/P/R per fire, saves results |
| `data/pipeline_loader.py` | Modify `iter_pixel_samples_v3` and `get_pixel_feature_names_v3` to accept optional feature key overrides |
| `data/analysis/ablation_v4/` | Output directory for per-run JSON results + summary |

---

### Task 1: Make pipeline_loader feature keys configurable

**Files:**
- Modify: `data/pipeline_loader.py:723-849`

Currently `iter_pixel_samples_v3` and `get_pixel_feature_names_v3` hardcode `_PIXEL_V3_STATIC_KEYS`, `_PIXEL_V3_HOURLY_KEYS`, etc. We need them to accept overrides so the ablation script can control the feature set.

- [ ] **Step 1: Add optional key override params to `get_pixel_feature_names_v3`**

In `data/pipeline_loader.py`, change the function signature and body:

```python
def get_pixel_feature_names_v3(
    seq_len: int = 6,
    neighborhood: int = 1,
    static_keys: list[str] | None = None,
    hourly_keys: list[str] | None = None,
    daily_keys: list[str] | None = None,
    slow_keys: list[str] | None = None,
    temporal_keys: list[str] | None = None,
    extra_proc_keys: list[str] | None = None,
) -> list[str]:
    """Feature names for V3 iter_pixel_samples_v3 output."""
    _static = static_keys if static_keys is not None else _PIXEL_V3_STATIC_KEYS
    _hourly = hourly_keys if hourly_keys is not None else _PIXEL_V3_HOURLY_KEYS
    _daily = daily_keys if daily_keys is not None else _PIXEL_V3_DAILY_KEYS
    _slow = slow_keys if slow_keys is not None else _PIXEL_V3_SLOW_KEYS
    _temporal = temporal_keys if temporal_keys is not None else _PIXEL_V3_TEMPORAL_KEYS
    _extra_proc = extra_proc_keys or []

    names: list[str] = []

    side = 2 * neighborhood + 1
    for t_offset in range(seq_len):
        for di in range(-neighborhood, neighborhood + 1):
            for dj in range(-neighborhood, neighborhood + 1):
                names.append(f"prev_fire_t-{seq_len - 1 - t_offset}_di{di}_dj{dj}")

    names.append("prev_distance_to_fire")
    names.append("prev_fire_neighborhood")

    for key in _extra_proc:
        names.append(key)

    for key in _static:
        names.append(key.replace("static_", ""))

    for key in _hourly:
        names.append(key.replace("hourly_", ""))

    for key in _daily:
        names.append(key.replace("daily_", ""))

    for key in _slow:
        names.append(key.replace("slow_", ""))

    for key in _temporal:
        names.append(key.replace("temporal_", ""))

    return names
```

- [ ] **Step 2: Add same override params to `iter_pixel_samples_v3`**

Change signature to accept the same optional key lists plus `extra_proc_keys` and `use_soft_labels`:

```python
def iter_pixel_samples_v3(
    processed: dict,
    features: dict,
    seq_len: int = 6,
    neighborhood: int = 1,
    static_keys: list[str] | None = None,
    hourly_keys: list[str] | None = None,
    daily_keys: list[str] | None = None,
    slow_keys: list[str] | None = None,
    temporal_keys: list[str] | None = None,
    extra_proc_keys: list[str] | None = None,
    use_soft_labels: bool = False,
) -> Generator[tuple[ndarray, float, float], None, None]:
```

Inside the function body, resolve defaults:

```python
    _static = static_keys if static_keys is not None else _PIXEL_V3_STATIC_KEYS
    _hourly = hourly_keys if hourly_keys is not None else _PIXEL_V3_HOURLY_KEYS
    _daily = daily_keys if daily_keys is not None else _PIXEL_V3_DAILY_KEYS
    _slow = slow_keys if slow_keys is not None else _PIXEL_V3_SLOW_KEYS
    _temporal = temporal_keys if temporal_keys is not None else _PIXEL_V3_TEMPORAL_KEYS
    _extra_proc = extra_proc_keys or []
```

Replace all references to `_PIXEL_V3_*` with the local `_static`, `_hourly`, etc.

Change target selection:

```python
    target_labels = processed["soft_labels" if use_soft_labels else "labels"][t]
```

After `prev_neigh` features, add extra processed keys:

```python
            # Extra processed features (e.g., is_smoke, btd_fire_smoke)
            for key in _extra_proc:
                arr = processed.get(key)
                if arr is not None and arr.ndim == 3:
                    fv[pos] = arr[t, i, j]
                elif arr is not None and arr.ndim == 2:
                    fv[pos] = arr[i, j]
                else:
                    fv[pos] = 0.0
                pos += 1
```

Update `n_total` to include `len(_extra_proc)`:

```python
    n_total = (n_conf_feat + 2 + len(_extra_proc)
               + len(_static) + len(_hourly)
               + len(_daily) + len(_slow)
               + len(_temporal))
```

- [ ] **Step 3: Verify existing callers still work (no args = same behavior)**

Run: `python3 -c "import sys; sys.path.insert(0,'.'); from data.pipeline_loader import get_pixel_feature_names_v3; print(len(get_pixel_feature_names_v3())); assert len(get_pixel_feature_names_v3()) == 80"`

Expected: prints `80`, no assertion error.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/ -v`
Expected: 45 passed

- [ ] **Step 5: Commit**

```bash
git add data/pipeline_loader.py
git commit -m "Make pipeline_loader V3 feature keys configurable for ablation"
```

---

### Task 2: Write ablation runner script

**Files:**
- Create: `models/xgboost/ablation.py`

- [ ] **Step 1: Create the ablation script**

Write `models/xgboost/ablation.py` with this structure:

```python
"""Staged XGBoost feature ablation study.

Optimizes for F2 score. Runs ~25 experiments in 6 stages,
each building on the best config from the previous stage.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.pipeline_loader import (
    load_processed_fire_data,
    iter_pixel_samples_v3,
    get_pixel_feature_names_v3,
    subsample_negatives,
    _PIXEL_V3_STATIC_KEYS,
    _PIXEL_V3_HOURLY_KEYS,
    _PIXEL_V3_DAILY_KEYS,
    _PIXEL_V3_SLOW_KEYS,
    _PIXEL_V3_TEMPORAL_KEYS,
)

# ─── Config ─────────────────────────────────────────────────────────────────

_CANDIDATE_PIPELINE_DIRS = [
    REPO_ROOT.parent / "wildfire-data-pipeline" / "data",
    Path.home() / "Desktop" / "Current Projects" / "wildfire-data-pipeline" / "data",
]

PIPELINE_DIR: Path | None = None
for _c in _CANDIDATE_PIPELINE_DIRS:
    if _c.resolve().is_dir():
        PIPELINE_DIR = _c.resolve()
        break

SEQ_LEN = 6
NEG_POS_RATIO = 5.0
SEED = 42
MIN_FIRE_PIXELS = 50
TRAIN_FRACTION = 0.7
SPLIT_SEED = 42

ANALYSIS_DIR = REPO_ROOT / "data" / "analysis" / "ablation_v4"

XGB_PARAMS = {
    "max_depth": 8,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "random_state": SEED,
    "tree_method": "hist",
    "n_jobs": 1,
    "verbosity": 0,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fbeta(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    d = (1 + beta**2) * tp + beta**2 * fn + fp
    return (1 + beta**2) * tp / d if d > 0 else 0.0


# ─── Experiment Config ──────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    stage: int
    static_keys: list[str] = field(default_factory=lambda: list(_PIXEL_V3_STATIC_KEYS))
    hourly_keys: list[str] = field(default_factory=lambda: list(_PIXEL_V3_HOURLY_KEYS))
    daily_keys: list[str] = field(default_factory=lambda: list(_PIXEL_V3_DAILY_KEYS))
    slow_keys: list[str] = field(default_factory=lambda: list(_PIXEL_V3_SLOW_KEYS))
    temporal_keys: list[str] = field(default_factory=lambda: list(_PIXEL_V3_TEMPORAL_KEYS))
    extra_proc_keys: list[str] = field(default_factory=list)
    use_soft_labels: bool = False
    use_loss_weights: bool = False


# ─── Stage definitions ──────────────────────────────────────────────────────

def build_stages() -> list[list[ExperimentConfig]]:
    """Return list of stages, each a list of experiments."""
    baseline = ExperimentConfig(name="s0_baseline", stage=0)

    # Stage 1: Training signal improvements
    s1a = ExperimentConfig(name="s1a_soft_labels", stage=1, use_soft_labels=True)
    s1b = ExperimentConfig(name="s1b_loss_weights", stage=1, use_loss_weights=True)
    s1c = ExperimentConfig(name="s1c_soft_plus_weights", stage=1,
                           use_soft_labels=True, use_loss_weights=True)

    # Stage 2: Precipitation + smoke (built on best of stage 1 dynamically)
    # These are templates — the runner will copy the best stage 1 config into them
    s2a = ExperimentConfig(name="s2a_era5_precip", stage=2,
                           hourly_keys=list(_PIXEL_V3_HOURLY_KEYS) + ["hourly_precipitation_m"])
    s2b = ExperimentConfig(name="s2b_gpm_precip", stage=2,
                           hourly_keys=list(_PIXEL_V3_HOURLY_KEYS) + ["hourly_gpm_precipitation_mmhr"])
    s2c = ExperimentConfig(name="s2c_both_precip", stage=2,
                           hourly_keys=list(_PIXEL_V3_HOURLY_KEYS) + ["hourly_precipitation_m", "hourly_gpm_precipitation_mmhr"])
    s2d = ExperimentConfig(name="s2d_smoke", stage=2, extra_proc_keys=["is_smoke", "btd_fire_smoke"])
    s2e = ExperimentConfig(name="s2e_precip_plus_smoke", stage=2,
                           hourly_keys=list(_PIXEL_V3_HOURLY_KEYS) + ["hourly_precipitation_m", "hourly_gpm_precipitation_mmhr"],
                           extra_proc_keys=["is_smoke", "btd_fire_smoke"])

    # Stage 3: Drought & fuel moisture
    s3a = ExperimentConfig(name="s3a_pdsi", stage=3,
                           slow_keys=list(_PIXEL_V3_SLOW_KEYS) + ["slow_pdsi"])
    s3b = ExperimentConfig(name="s3b_eddi", stage=3,
                           slow_keys=list(_PIXEL_V3_SLOW_KEYS) + ["slow_eddi_14d", "slow_eddi_30d"])
    s3c = ExperimentConfig(name="s3c_ndwi", stage=3,
                           slow_keys=list(_PIXEL_V3_SLOW_KEYS) + ["slow_ndwi"])
    s3d = ExperimentConfig(name="s3d_smoke_aerosol", stage=3,
                           slow_keys=list(_PIXEL_V3_SLOW_KEYS) + ["slow_smoke_aerosol_index"])
    s3e = ExperimentConfig(name="s3e_all_slow", stage=3,
                           slow_keys=list(_PIXEL_V3_SLOW_KEYS) + [
                               "slow_pdsi", "slow_eddi_14d", "slow_eddi_30d",
                               "slow_ndwi", "slow_smoke_aerosol_index"])

    # Stage 4: Extended daily weather
    s4a = ExperimentConfig(name="s4a_humidity", stage=4,
                           daily_keys=list(_PIXEL_V3_DAILY_KEYS) + ["daily_rmin", "daily_rmax"])
    s4b = ExperimentConfig(name="s4b_daily_wind", stage=4,
                           daily_keys=list(_PIXEL_V3_DAILY_KEYS) + ["daily_vs", "daily_th"])
    s4c = ExperimentConfig(name="s4c_lst", stage=4,
                           daily_keys=list(_PIXEL_V3_DAILY_KEYS) + ["daily_lst_day_k", "daily_lst_night_k"])
    s4d = ExperimentConfig(name="s4d_all_daily", stage=4,
                           daily_keys=list(_PIXEL_V3_DAILY_KEYS) + [
                               "daily_rmin", "daily_rmax", "daily_vs", "daily_th",
                               "daily_lst_day_k", "daily_lst_night_k"])

    # Stage 5: Static terrain & land cover
    s5a = ExperimentConfig(name="s5a_terrain_ext", stage=5,
                           static_keys=list(_PIXEL_V3_STATIC_KEYS) + [
                               "static_terrain_ruggedness", "static_is_firebreak"])
    s5b = ExperimentConfig(name="s5b_landfire", stage=5,
                           static_keys=list(_PIXEL_V3_STATIC_KEYS) + [
                               "static_vegetation_type", "static_vegetation_cover",
                               "static_vegetation_height"])
    s5c = ExperimentConfig(name="s5c_fire_history", stage=5,
                           static_keys=list(_PIXEL_V3_STATIC_KEYS) + [
                               "static_years_since_burn", "static_burn_count"])
    s5d = ExperimentConfig(name="s5d_wui", stage=5,
                           static_keys=list(_PIXEL_V3_STATIC_KEYS) + [
                               "static_population", "static_built_up", "static_has_road"])
    s5e = ExperimentConfig(name="s5e_all_static", stage=5,
                           static_keys=list(_PIXEL_V3_STATIC_KEYS) + [
                               "static_terrain_ruggedness", "static_is_firebreak",
                               "static_vegetation_type", "static_vegetation_cover",
                               "static_vegetation_height",
                               "static_years_since_burn", "static_burn_count",
                               "static_population", "static_built_up", "static_has_road"])

    return [
        [baseline],
        [s1a, s1b, s1c],
        [s2a, s2b, s2c, s2d, s2e],
        [s3a, s3b, s3c, s3d, s3e],
        [s4a, s4b, s4c, s4d],
        [s5a, s5b, s5c, s5d, s5e],
    ]


# ─── Data Loading ───────────────────────────────────────────────────────────

def discover_fires(pipeline_dir: Path) -> list[str]:
    fires: list[str] = []
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


def split_fires(fires: list[str]) -> tuple[list[str], list[str]]:
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = list(fires)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * TRAIN_FRACTION))
    n_train = min(n_train, len(shuffled) - 1)
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def collect_samples(
    fire_name: str, pipeline_dir: Path, cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, weights)."""
    processed, features = load_processed_fire_data(fire_name, str(pipeline_dir))

    X_list, y_list, w_list = [], [], []
    for fv, label, weight in iter_pixel_samples_v3(
        processed, features, seq_len=SEQ_LEN,
        static_keys=cfg.static_keys, hourly_keys=cfg.hourly_keys,
        daily_keys=cfg.daily_keys, slow_keys=cfg.slow_keys,
        temporal_keys=cfg.temporal_keys, extra_proc_keys=cfg.extra_proc_keys,
        use_soft_labels=cfg.use_soft_labels,
    ):
        X_list.append(fv)
        y_list.append(label)
        w_list.append(weight)

    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(w_list, dtype=np.float32))


# ─── Single Experiment ──────────────────────────────────────────────────────

def run_experiment(
    cfg: ExperimentConfig,
    train_names: list[str],
    test_names: list[str],
    pipeline_dir: Path,
) -> dict[str, Any]:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    log(f"--- {cfg.name} (stage {cfg.stage}) ---")
    n_features = len(get_pixel_feature_names_v3(
        seq_len=SEQ_LEN,
        static_keys=cfg.static_keys, hourly_keys=cfg.hourly_keys,
        daily_keys=cfg.daily_keys, slow_keys=cfg.slow_keys,
        temporal_keys=cfg.temporal_keys, extra_proc_keys=cfg.extra_proc_keys,
    ))
    log(f"  Features: {n_features}")

    # Collect training samples
    X_parts, y_parts, w_parts = [], [], []
    for name in train_names:
        X_f, y_f, w_f = collect_samples(name, pipeline_dir, cfg)
        X_parts.append(X_f)
        y_parts.append(y_f)
        w_parts.append(w_f)
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    w_train = np.concatenate(w_parts)

    # For soft labels, binarize y for XGBoost classification
    # but keep soft_labels for potential regression use
    y_train_binary = (y_train > 0.5).astype(np.float32)

    # Subsample negatives
    pos_mask = y_train_binary > 0
    n_pos = int(pos_mask.sum())
    n_neg_want = int(n_pos * NEG_POS_RATIO)
    neg_idx = np.where(~pos_mask)[0]
    if len(neg_idx) > n_neg_want:
        rng = np.random.RandomState(SEED)
        neg_idx = rng.choice(neg_idx, n_neg_want, replace=False)
    pos_idx = np.where(pos_mask)[0]
    keep = np.sort(np.concatenate([pos_idx, neg_idx]))

    X_sub = X_train[keep]
    y_sub = y_train_binary[keep]
    w_sub = w_train[keep]

    n_pos_sub = int(y_sub.sum())
    n_neg_sub = len(y_sub) - n_pos_sub
    scale_pos_weight = n_neg_sub / max(n_pos_sub, 1)

    # Train
    sample_weight = w_sub if cfg.use_loss_weights else None
    t0 = time.time()
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_sub, y_sub, sample_weight=sample_weight, verbose=False)
    train_time = time.time() - t0
    log(f"  Train: {train_time:.1f}s ({len(y_sub):,} samples, pos={n_pos_sub:,})")

    # Evaluate per fire
    per_fire = {}
    all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0
    all_y, all_prob = [], []

    for name in test_names:
        X_f, y_f, _ = collect_samples(name, pipeline_dir, cfg)
        y_f_binary = (y_f > 0.5).astype(np.float32)
        y_pred = model.predict(X_f)
        y_prob = model.predict_proba(X_f)[:, 1]

        tp = int(((y_pred == 1) & (y_f_binary == 1)).sum())
        fp = int(((y_pred == 1) & (y_f_binary == 0)).sum())
        fn = int(((y_pred == 0) & (y_f_binary == 1)).sum())
        tn = int(((y_pred == 0) & (y_f_binary == 0)).sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f2 = fbeta(tp, fp, fn, beta=2.0)
        auc = float(roc_auc_score(y_f_binary, y_prob)) if len(np.unique(y_f_binary)) == 2 else 0.0

        per_fire[name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "precision": p, "recall": r, "f1": f1, "f2": f2, "auc": auc}
        all_tp += tp; all_fp += fp; all_fn += fn; all_tn += tn
        all_y.extend(y_f_binary.tolist())
        all_prob.extend(y_prob.tolist())

    # Aggregate
    p_all = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    r_all = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0
    f2_all = fbeta(all_tp, all_fp, all_fn, beta=2.0)
    auc_all = float(roc_auc_score(all_y, all_prob)) if len(set(all_y)) == 2 else 0.0

    log(f"  F2={f2_all:.4f}  F1={f1_all:.4f}  P={p_all:.4f}  R={r_all:.4f}  AUC={auc_all:.4f}")

    # Feature importance top 10
    feature_names = get_pixel_feature_names_v3(
        seq_len=SEQ_LEN,
        static_keys=cfg.static_keys, hourly_keys=cfg.hourly_keys,
        daily_keys=cfg.daily_keys, slow_keys=cfg.slow_keys,
        temporal_keys=cfg.temporal_keys, extra_proc_keys=cfg.extra_proc_keys,
    )
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [(feature_names[i], float(importances[i])) for i in top_idx if i < len(feature_names)]

    return {
        "name": cfg.name, "stage": cfg.stage,
        "n_features": n_features, "train_time_s": train_time,
        "f2": f2_all, "f1": f1_all, "precision": p_all, "recall": r_all, "auc": auc_all,
        "tp": all_tp, "fp": all_fp, "fn": all_fn, "tn": all_tn,
        "per_fire": per_fire, "top_features": top_features,
        "config": asdict(cfg),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def _apply_best(cfg: ExperimentConfig, best: ExperimentConfig) -> ExperimentConfig:
    """Inherit training signal settings from the best config of the previous stage."""
    cfg.use_soft_labels = best.use_soft_labels
    cfg.use_loss_weights = best.use_loss_weights
    # Inherit feature additions from ALL prior best stages
    # For keys that are still at default, inherit from best
    # For keys that have new additions (this stage's test), keep them
    # This is done by checking if cfg's keys are the baseline defaults
    for attr in ("extra_proc_keys",):
        current = getattr(cfg, attr)
        best_val = getattr(best, attr)
        if not current and best_val:
            setattr(cfg, attr, list(best_val))
        elif current and best_val:
            merged = list(best_val)
            for k in current:
                if k not in merged:
                    merged.append(k)
            setattr(cfg, attr, merged)

    for attr, default_keys in [
        ("hourly_keys", list(_PIXEL_V3_HOURLY_KEYS)),
        ("daily_keys", list(_PIXEL_V3_DAILY_KEYS)),
        ("slow_keys", list(_PIXEL_V3_SLOW_KEYS)),
        ("static_keys", list(_PIXEL_V3_STATIC_KEYS)),
    ]:
        current = getattr(cfg, attr)
        best_val = getattr(best, attr)
        # If this experiment adds new keys beyond default, merge with best
        current_additions = [k for k in current if k not in default_keys]
        if best_val != default_keys:
            base = list(best_val)
        else:
            base = list(default_keys)
        for k in current_additions:
            if k not in base:
                base.append(k)
        setattr(cfg, attr, base)

    return cfg


def main() -> None:
    if PIPELINE_DIR is None:
        log("ERROR: Could not find wildfire-data-pipeline data directory.")
        sys.exit(1)

    import xgboost  # fail fast if missing

    log(f"Pipeline: {PIPELINE_DIR}")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    all_fires = discover_fires(PIPELINE_DIR)
    usable = []
    for name in all_fires:
        processed, _ = load_processed_fire_data(name, str(PIPELINE_DIR))
        n_fire = int(processed["labels"].sum())
        if n_fire >= MIN_FIRE_PIXELS:
            usable.append(name)
            log(f"  {name}: {n_fire} fire pixels")
        else:
            log(f"  SKIP {name}: {n_fire} < {MIN_FIRE_PIXELS}")

    train_names, test_names = split_fires(usable)
    log(f"Train ({len(train_names)}): {train_names}")
    log(f"Test  ({len(test_names)}):  {test_names}")

    stages = build_stages()
    all_results: list[dict] = []
    best_per_stage: dict[int, ExperimentConfig] = {}

    for stage_idx, experiments in enumerate(stages):
        log(f"\n{'='*60}")
        log(f"STAGE {stage_idx}")
        log(f"{'='*60}")

        stage_results = []
        for cfg in experiments:
            if stage_idx > 0:
                prev_best = best_per_stage[stage_idx - 1]
                cfg = _apply_best(cfg, prev_best)
            result = run_experiment(cfg, train_names, test_names, PIPELINE_DIR)
            stage_results.append(result)
            all_results.append(result)

            # Save per-experiment result
            with open(ANALYSIS_DIR / f"{cfg.name}.json", "w") as f:
                json.dump(result, f, indent=2)

        # Pick best by F2
        best_result = max(stage_results, key=lambda r: r["f2"])
        best_cfg = experiments[stage_results.index(best_result)]
        if stage_idx > 0:
            best_cfg = _apply_best(best_cfg, best_per_stage[stage_idx - 1])
        best_per_stage[stage_idx] = best_cfg
        log(f"\n  BEST stage {stage_idx}: {best_result['name']} "
            f"F2={best_result['f2']:.4f} F1={best_result['f1']:.4f}")

    # Stage 6: Confirmation — re-run the final best
    log(f"\n{'='*60}")
    log("STAGE 6: CONFIRMATION (final best feature set)")
    log(f"{'='*60}")
    final_cfg = ExperimentConfig(
        name="s6_final_best", stage=6,
        **{k: v for k, v in asdict(best_per_stage[5]).items() if k not in ("name", "stage")},
    )
    final_result = run_experiment(final_cfg, train_names, test_names, PIPELINE_DIR)
    all_results.append(final_result)
    with open(ANALYSIS_DIR / "s6_final_best.json", "w") as f:
        json.dump(final_result, f, indent=2)

    # Summary table
    log(f"\n{'='*60}")
    log("ABLATION SUMMARY (sorted by F2)")
    log(f"{'='*60}")
    log(f"{'Name':<30} {'Stage':>5} {'F2':>7} {'F1':>7} {'P':>7} {'R':>7} {'AUC':>7} {'#feat':>6}")
    log("-" * 85)
    for r in sorted(all_results, key=lambda x: -x["f2"]):
        log(f"{r['name']:<30} {r['stage']:>5} {r['f2']:>7.4f} {r['f1']:>7.4f} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['auc']:>7.4f} {r['n_features']:>6}")

    # Save full summary
    with open(ANALYSIS_DIR / "summary.json", "w") as f:
        json.dump({
            "train_fires": train_names,
            "test_fires": test_names,
            "results": all_results,
            "best_per_stage": {str(k): asdict(v) for k, v in best_per_stage.items()},
            "final_best": final_result,
        }, f, indent=2)

    log(f"\nResults saved to {ANALYSIS_DIR}/")
    log(f"FINAL BEST: {final_result['name']} F2={final_result['f2']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it compiles**

Run: `python3 -c "import py_compile; py_compile.compile('models/xgboost/ablation.py', doraise=True); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add models/xgboost/ablation.py
git commit -m "Add staged XGBoost ablation study script (F2-optimized, 6 stages)"
```

---

### Task 3: Run the ablation study

**Files:**
- Run: `models/xgboost/ablation.py`
- Output: `data/analysis/ablation_v4/*.json`

- [ ] **Step 1: Run the full ablation**

Run: `cd /Users/seanmay/Desktop/Current\ Projects/wildfire-prediction && python3 models/xgboost/ablation.py`

This will take ~30-60 minutes. Watch for errors in the first experiment (stage 0 baseline). If it completes the baseline successfully, the rest will follow the same pattern.

Expected output: ~25 JSON files in `data/analysis/ablation_v4/` plus a `summary.json` with the full ranking.

- [ ] **Step 2: Verify results exist**

Run: `ls data/analysis/ablation_v4/ | wc -l && cat data/analysis/ablation_v4/summary.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Runs: {len(d[\"results\"])}'); print(f'Final F2: {d[\"final_best\"][\"f2\"]:.4f}')"`

- [ ] **Step 3: Commit results**

```bash
git add models/xgboost/ablation.py
git commit -m "Complete ablation study: document final best feature set"
```

---

### Task 4: Push and update PR

- [ ] **Step 1: Push**

```bash
git push origin main
```

# Wildfire Models v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-download fire data using the new `wildfire-data-pipeline` (Kincade + Walker fires with cloud masking and full feature stack), then train two models: a single-pixel XGBoost classifier and a full-grid FireSpreadNet v2.

**Architecture:** Data flows from the pipeline as NPZ files (fire detection + features), through a shared data loader that supports both per-pixel extraction (XGBoost) and spatial-temporal windowing (FireSpreadNet). Both models predict fire at t+1 from 6-hour lookback windows. Train on one fire, test on the other (cross-validated).

**Tech Stack:** Python 3, PyTorch (FireSpreadNet), XGBoost, NumPy, scikit-learn (metrics), wildfire-data-pipeline (data download)

---

## File Structure

```
scripts/
  pipeline_data_loader.py    — Loads NPZ data from new pipeline, provides both pixel-level and grid-level interfaces
  train_xgboost_pixel.py     — Single-pixel XGBoost: feature extraction, training, evaluation
  fire_spread_model_v2.py    — FireSpreadNet v2: dynamic in_channels from pipeline features
  train_firespreadnet_v2.py  — Full-grid training loop with expanded channels and cloud masking
  evaluate_models.py         — Cross-fire evaluation comparing both models
```

---

### Task 1: Download Pipeline Data

**Files:**
- Read: `/Users/seanmay/Desktop/Current Projects/wildfire-data-pipeline/config/fires.json`
- Output: `wildfire-data-pipeline/data/Kincade/` and `wildfire-data-pipeline/data/Walker/`

- [ ] **Step 1: Set up pipeline environment**

```bash
cd "/Users/seanmay/Desktop/Current Projects/wildfire-data-pipeline"
cat .env.example
# Create .env with GEE project
```

- [ ] **Step 2: Download Kincade fire data + features**

```bash
cd "/Users/seanmay/Desktop/Current Projects/wildfire-data-pipeline"
uv run wildfire download Kincade --format npz
```

Expected: `data/Kincade/Kincade_2019_FusedConf.npz` and `data/Kincade/Kincade_2019_Features.npz`

- [ ] **Step 3: Download Walker fire data + features**

```bash
cd "/Users/seanmay/Desktop/Current Projects/wildfire-data-pipeline"
uv run wildfire download Walker --format npz
```

Expected: `data/Walker/Walker_2019_FusedConf.npz` and `data/Walker/Walker_2019_Features.npz`

- [ ] **Step 4: Process labels for both fires**

```bash
uv run wildfire process data/Kincade/Kincade_2019_FusedConf.npz
uv run wildfire process data/Walker/Walker_2019_FusedConf.npz
```

Expected: Processed NPZ files with smoothed labels and validity masks in `processed/` subdirectories

- [ ] **Step 5: Validate data quality**

```bash
uv run wildfire validate data/Kincade/Kincade_2019_FusedConf.npz
uv run wildfire validate data/Walker/Walker_2019_FusedConf.npz
```

- [ ] **Step 6: Inspect downloaded data shapes**

```python
# Quick inspection script to verify array shapes and feature channels
import numpy as np
for fire in ["Kincade", "Walker"]:
    base = f"../wildfire-data-pipeline/data/{fire}"
    conf = np.load(f"{base}/{fire}_2019_FusedConf.npz", allow_pickle=False)
    feat = np.load(f"{base}/{fire}_2019_Features.npz", allow_pickle=False)
    print(f"\n{fire} fire data:")
    for k in conf.files:
        if k != "_metadata":
            print(f"  {k}: {conf[k].shape}")
    print(f"\n{fire} features:")
    for k in feat.files:
        if k != "_metadata":
            print(f"  {k}: {feat[k].shape}")
```

---

### Task 2: Build Pipeline Data Loader

**Files:**
- Create: `scripts/pipeline_data_loader.py`
- Test: Run inline validation after writing

This module is the bridge between the pipeline's NPZ format and both models. It provides:
- `load_fire_data()`: Load both fire detection and feature NPZ files
- `build_channel_stack()`: Assemble all features into (T, C, H, W) tensor
- `compute_channel_stats()`: Z-score statistics across training fires
- `iter_grid_sequences()`: Yield (frames, target, mask) for FireSpreadNet
- `iter_pixel_samples()`: Yield (feature_vector, label) for XGBoost
- `smooth_labels()`: Majority vote temporal smoothing with cloud awareness

- [ ] **Step 1: Write the data loader module**

```python
# scripts/pipeline_data_loader.py
# Key functions:
# 1. load_fire_data(fire_name, pipeline_dir) -> (fire_arrays, feature_arrays, fire_meta, feat_meta)
# 2. build_channel_stack(fire_arrays, feature_arrays, pad_h, pad_w) -> (T, C, H, W) ndarray
# 3. smooth_labels(confidence, cloud_mask, obs_valid, window, min_votes, threshold) -> labels
# 4. compute_channel_stats(channel_stacks: list) -> (means, stds) per channel
# 5. iter_grid_sequences(stack, labels, validity, seq_len, ch_means, ch_stds) -> yields (frames, target, mask)
# 6. iter_pixel_samples(fire_arrays, feature_arrays, labels, validity, seq_len) -> yields (features, label)
```

Key design decisions:
- Channel order: fire confidence, FRP, hourly weather (ugrd, vgrd, gust, tmp, dpt, soil_moisture), daily weather (erc, bi, fm100, fm1000, vpd), static terrain (slope, aspect_sin, aspect_cos, elevation, tpi), slow (NDVI, EVI), temporal (hour_sin, hour_cos, doy_sin, doy_cos), validity mask
- Static/slow features broadcast to (T, H, W) matching hourly cadence
- Daily features already expanded to hourly in pipeline output
- Cloud-aware label smoothing: only count non-cloudy hours in vote window

- [ ] **Step 2: Verify data loader works with downloaded data**

```bash
cd "/Users/seanmay/Desktop/Current Projects/wildfire-prediction"
python3 -c "
from scripts.pipeline_data_loader import load_fire_data, build_channel_stack, smooth_labels
fire, feat, fm, featm = load_fire_data('Kincade', '../wildfire-data-pipeline/data')
stack = build_channel_stack(fire, feat)
print(f'Channel stack shape: {stack.shape}')
labels = smooth_labels(fire['data'], fire.get('cloud_mask'), fire.get('observation_valid'))
print(f'Labels shape: {labels.shape}, fire pixels: {labels.sum():.0f}')
"
```

Expected: `Channel stack shape: (160, ~25, H, W)`, labels with fire pixel counts

- [ ] **Step 3: Commit**

```bash
git add scripts/pipeline_data_loader.py
git commit -m "feat: add pipeline data loader for new NPZ format"
```

---

### Task 3: Train Single-Pixel XGBoost Model

**Files:**
- Create: `scripts/train_xgboost_pixel.py`
- Depends on: `scripts/pipeline_data_loader.py`

- [ ] **Step 1: Write XGBoost training script**

Feature extraction per pixel (i, j) at time t:
- 3x3 neighborhood fire state at t-5..t (9 cells x 6 timesteps = 54 features)
- FRP for center pixel at t (1 feature)
- Static features for center pixel: slope, aspect_sin, aspect_cos, elevation, tpi, fuel_load, canopy_cover (7 features)
- Hourly weather at center: ugrd, vgrd, gust, tmp, dpt, soil_moisture (6 features)
- Daily fire weather at center: erc, bi, fm100, fm1000, vpd (5 features)
- Slow features: NDVI, EVI (2 features)
- Temporal: hour_sin, hour_cos, doy_sin, doy_cos (4 features)
- Derived: distance to nearest fire pixel, fraction of 3x3 on fire (2 features)

Total: ~81 features per sample

Training:
- Subsample negative class (99%+ of pixels are no-fire) to ratio ~5:1 neg:pos
- `scale_pos_weight` for remaining imbalance
- Only train on valid (non-cloudy) pixels
- Train on Kincade, eval on Walker; then swap for cross-validation
- Hyperparameters: max_depth=8, n_estimators=500, learning_rate=0.05, subsample=0.8

- [ ] **Step 2: Run XGBoost training**

```bash
python3 scripts/train_xgboost_pixel.py
```

Expected: F1, precision, recall printed per-fire. Feature importance saved.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_xgboost_pixel.py
git commit -m "feat: add single-pixel XGBoost model with pipeline features"
```

---

### Task 4: Build FireSpreadNet v2 (Multi-Channel)

**Files:**
- Create: `scripts/fire_spread_model_v2.py`
- Reference: `scripts/fire_spread_model.py` (original architecture)

- [ ] **Step 1: Write FireSpreadNet v2**

Changes from v1:
- `in_channels` is now dynamic (set from pipeline feature count, ~25 channels)
- Keep proven ConvGRU U-Net + CBAM architecture
- Update wind direction channel indices for new channel ordering (ugrd/vgrd instead of sin/cos)
- Augmentation: flip ugrd/vgrd signs for H/V flips (not wind_dir_sin/cos)
- Add channel group info for potential group normalization

Architecture stays the same:
- 3-level encoder (32 -> 64 -> 128) + bottleneck (256) + decoder with skip connections
- ConvGRU at bottleneck for temporal processing
- CBAM attention on skip connections
- Output: (B, 1, H, W) fire probability logits

- [ ] **Step 2: Verify model forward pass**

```bash
python3 -c "
import torch
from scripts.fire_spread_model_v2 import FireSpreadNetV2
model = FireSpreadNetV2(in_channels=25)
x = torch.randn(2, 6, 25, 112, 192)
out = model(x)
print(f'Output shape: {out.shape}')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

Expected: `Output shape: torch.Size([2, 1, 112, 192])`, ~2.5M parameters

- [ ] **Step 3: Commit**

```bash
git add scripts/fire_spread_model_v2.py
git commit -m "feat: add FireSpreadNet v2 with dynamic multi-channel input"
```

---

### Task 5: Train FireSpreadNet v2

**Files:**
- Create: `scripts/train_firespreadnet_v2.py`
- Depends on: `scripts/fire_spread_model_v2.py`, `scripts/pipeline_data_loader.py`

- [ ] **Step 1: Write training script**

Training config:
- SEQ_LEN = 6
- BATCH_SIZE = 4
- LR = 3e-4, weight_decay = 1e-4
- Loss: CombinedLoss (Focal + Focal Tversky), same proven config
- Cloud-masked loss: multiply loss mask by validity * (1 - cloud_mask)
- Label smoothing: majority vote (2-of-5) on confidence >= 0.30
- Gradient clipping = 1.0
- Epochs = 20
- Save best checkpoint by test F1

Data flow:
1. Load both fires from pipeline NPZ
2. Build channel stacks for each
3. Compute z-score stats from training fire only
4. Iterate sequences with `iter_grid_sequences()`
5. Augment with flips (correcting wind u/v components)

- [ ] **Step 2: Run training**

```bash
python3 scripts/train_firespreadnet_v2.py
```

Expected: Epoch-by-epoch F1 scores, best checkpoint saved to `data/checkpoints/`

- [ ] **Step 3: Commit**

```bash
git add scripts/train_firespreadnet_v2.py
git commit -m "feat: add FireSpreadNet v2 training with pipeline data"
```

---

### Task 6: Cross-Fire Evaluation

**Files:**
- Create: `scripts/evaluate_models.py`

- [ ] **Step 1: Write unified evaluation script**

Evaluates both models on both fires (train-on-A/test-on-B and vice versa):
- Load XGBoost model and FireSpreadNet v2 checkpoint
- Run inference on held-out fire
- Compute F1, precision, recall, per-fire metrics
- Compare with previous best (F1=0.837 smoothed, F1=0.744 XGBoost)
- Save results JSON to `data/analysis/v2_evaluation/`

- [ ] **Step 2: Run evaluation**

```bash
python3 scripts/evaluate_models.py
```

- [ ] **Step 3: Commit all results**

```bash
git add scripts/evaluate_models.py data/analysis/
git commit -m "feat: add cross-fire evaluation for v2 models"
```

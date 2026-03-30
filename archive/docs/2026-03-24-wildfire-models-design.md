# Wildfire Prediction Models — Re-download & Retrain

**Date**: 2026-03-24
**Status**: Approved

## Goal

Re-download fire data using the `wildfire-data-pipeline` repository (Kincade + Walker), then build two models:
1. **Single-pixel model** — predicts whether an individual pixel is on fire at t+1
2. **Full-grid model** — predicts the entire spatial grid at t+1

## Data Pipeline

**Source**: `wildfire-data-pipeline` (neighboring repo)
**Fires**: Kincade (2019, 160h, 77K acres), Walker (2019, 295h, 54K acres)
**Improvements over old GOFER pipeline**:
- DQF cloud masking (clouds = "unknown", not "no fire")
- Multi-source GOES fusion with MAX aggregation
- FDCC 5-min CONUS cadence
- Rich feature stack: terrain, fire weather, NDVI, FRP

### Data Format (per fire NPZ)

| Array | Shape | Description |
|-------|-------|-------------|
| `data` | (T, H, W) | Fire confidence [0, 1] |
| `observation_valid` | (T, H, W) | Valid observation mask |
| `cloud_mask` | (T, H, W) | Cloud flag |
| `frp` | (T, H, W) | Fire Radiative Power (MW) |
| `static_*` | (H, W) | Terrain, fuel, WUI (17 bands) |
| `slow_*` | (H, W) | Pre-fire NDVI, EVI, drought |
| `hourly_*` | (T, H, W) | RTMA wind/temp, soil moisture |
| `daily_*` | (T, H, W) | GRIDMET fire weather, LST |
| `temporal_*` | (T, H, W) | Hour/day-of-year sin/cos |

### Label Processing

- Confidence threshold: 0.30
- Temporal smoothing: majority vote (2-of-5 hour window)
- Cloud-aware persistence: forward-fill through <=3h cloud gaps
- Validity mask excludes cloudy pixels from training loss

## Model 1: Single-Pixel XGBoost

**Task**: Binary classification — will pixel (i, j) be on fire at time t+1?

**Feature vector per sample**:
- Center pixel + 8 neighbors (3x3 window) fire state at times t-5..t (6 steps)
- FRP for 3x3 window at current timestep
- Static features for center pixel: slope, aspect, TPI, elevation, fuel load, canopy cover, vegetation type/height, road proximity, population density
- Hourly weather at center: wind u/v, gust, temperature, dewpoint, soil moisture
- Daily fire weather at center: ERC, BI, FM100, FM1000, VPD
- Slow features at center: NDVI, EVI, PDSI, EDDI
- Temporal encoding: hour-of-day sin/cos, day-of-year sin/cos

**Estimated feature count**: ~120-150 features per sample

**Training**:
- Train on Kincade, test on Walker (and vice versa for cross-validation)
- Handle class imbalance with `scale_pos_weight`
- Only train on valid (non-cloudy) pixels

## Model 2: Full-Grid FireSpreadNet

**Task**: Semantic segmentation — predict fire probability for entire grid at t+1

**Architecture**: ConvGRU U-Net with CBAM attention (proven at F1=0.837)
- 3-level encoder (32->64->128) + bottleneck (256) + decoder with skip connections
- Input: 6 consecutive hours, multi-channel
- Output: (H, W) fire probability map

**Input channels** (expanded from previous 2 to full feature stack):
- Fire confidence + validity mask (2 ch)
- FRP (1 ch)
- Hourly weather: wind u/v, gust, temp, dewpoint, soil moisture (6 ch)
- Static terrain: slope, aspect, elevation, TPI (4 ch, broadcast across time)
- Vegetation: NDVI, canopy cover (2 ch, broadcast)
- Fire weather: ERC, BI (2 ch, broadcast daily values)

**Total**: ~17 input channels

**Training**:
- Loss: Focal Loss + Focal Tversky (proven combination)
- Cloud-masked loss: zero weight for cloudy pixels via validity mask
- Optimizer: AdamW, LR=3e-4
- Data augmentation: random H/V flips with wind direction correction
- Train on Kincade, test on Walker (cross-val)

## Evaluation

Both models evaluated on:
- **F1 score** (primary metric)
- **Precision** and **Recall**
- **Per-fire metrics** (cross-validation between Kincade/Walker)
- Comparison with previous best (F1=0.837 smoothed, F1=0.744 XGBoost)

## File Structure

```
scripts/
  download_pipeline_data.py   — Orchestrates data download + processing
  build_pixel_dataset.py      — Converts NPZ -> per-pixel feature vectors
  train_xgboost_pixel.py      — Single-pixel XGBoost training
  train_firespreadnet_v2.py   — Full-grid FireSpreadNet with expanded channels
  fire_spread_model_v2.py     — Updated FireSpreadNet architecture (multi-channel)
  evaluate_models.py          — Unified evaluation for both models
```

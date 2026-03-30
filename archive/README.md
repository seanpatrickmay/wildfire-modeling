# Archive

Superseded and experimental code preserved for reference. The active codebase lives in `models/` and `data/`. Git history preserves full context for everything here.

## v1_firespreadnet/

Original FireSpreadNet (8-channel, sin/cos wind encoding).

- `fire_spread_model.py` — ConvGRU-U-Net with CBAM attention, focal+Tversky loss
- `train_fire_spread.py` — Training loop for v1 (8 channels: fire conf, FRP, wind dir/speed, temp, humidity)
- `train_smoothed.py` — Variant with 2-of-5 majority-vote label smoothing (best v1 result: **F1=0.837**)
- `convgru_model.py` — Lightweight ConvGRU extracted from notebook
- `rnn_model.py` — GRU+Attention RNN (F1=0.748)
- `threshold_sweep.py` — Decision threshold optimization (finding: model outputs near-binary, threshold tuning irrelevant)

**Key finding:** Weather features (temp, wind, humidity) contributed nothing at GOES 2km resolution — fire-only (2 channels) matched the full 8-channel model. Label smoothing was the breakthrough (F1 0.749 → 0.837).

**Superseded by:** `models/firespreadnet/` (v2 architecture on V3 pipeline data with pre-shifted fire features)

## experiments/

- `ablation_study.py` — Systematic ablation of channel groups, temporal window, label quality for FireSpreadNet v2 + XGBoost. Used V2 data API.
- `ablation_fire_only.py` — Proved weather channels add no value at 2km resolution
- `evaluate_models.py` — Unified evaluation harness for XGBoost + FireSpreadNet v2 (V2 API)
- `label_quality_analysis.py` — Quantified GOES label flicker: 23% hourly flip rate, oracle F1 ceiling 0.757 (raw) vs 0.896 (smoothed)
- `retrain_with_clean_data.py` — Retrained neighborhood models after ACPC01 sentinel cleanup (9999 → clamped to [0,100])
- `precompute_predictions.py` — Cached predictions for logreg, MLP, XGBoost, RNN, ConvGRU on 8 test fires
- `run_locational_regressions.py` — Per-fire spread/continuation logistic regressions (earliest experiment)

**Key findings:**
- Raw GOES labels have a 23% hourly flicker rate → need smoothing or pre-processing
- Weather features don't help at 2km resolution (ablation confirmed across architectures)
- ACPC01 precipitation sentinels (9999) caused training artifacts — now clamped

## neighbor_cell/

Original pixel-level approach using 3x3 neighborhood features from raw GeoTIFFs.

- `neighbor_cell_logreg.py` — SGD logistic regression with feature schema, z-score normalization. Core utility that 8+ scripts depended on.
- `neighbor_cell_nn.py` — Companion feedforward MLP (F1=0.482)

**Progression:** LogReg → MLP (F1=0.482) → XGBoost (F1=0.461) → FireSpreadNet (F1=0.749 → 0.837 smoothed)

**Superseded by:** V3 pipeline data loader (`data/pipeline_loader.py`) which loads pre-processed NPZ files from the wildfire-data-pipeline repo.

## notebooks/

Jupyter notebooks documenting the model development progression.

| Notebook | Model | Key Result |
|----------|-------|------------|
| `neighbor_cell_confidence_regression.ipynb` | Logistic regression (single + multi-fire) | Baseline approach |
| `xgb_fire_holdout.ipynb` | XGBoost on held-out fires | F1=0.461 |
| `xgb_fire_holdout_ndvi.ipynb` | XGBoost + NDVI features | Marginal improvement |
| `mlp_fire_holdout.ipynb` | MLP neural network | F1=0.482 |
| `rnn_fire_holdout.ipynb` | GRU+Attention RNN | F1=0.748 |
| `convgru_fire_holdout.ipynb` | ConvGRU U-Net | F1=0.749 |

Also includes: `neighbor_cell_confidence_regression.pdf` (rendered report)

## etl/

Download and data conversion scripts. **The data pipeline now lives in a separate repository** (`wildfire-data-pipeline`). These scripts handled the original data acquisition workflow.

- `download_all_28_fires.py` — Orchestrated full pipeline for 28 GOFER fires
- `ee_download_confidence_stack.py` — GOES fire confidence from Earth Engine
- `ee_download_rtma.py` — RTMA meteorology aligned to GOES grid
- `download_gofer_zenodo.py` — GOFER products from Zenodo
- `download_ndvi.py` — MODIS NDVI resampled to GOES grid
- `gofer_confidence_to_json.py` — GeoTIFF → JSON conversion
- `rtma_to_json.py` — RTMA normalization to JSON
- `firms_viirs_to_json.py` — FIRMS VIIRS detections → gridded JSON
- `synoptic_raws_fetch.py` — RAWS station data from Synoptic API
- `raws_normalize.py` — RAWS z-score normalization
- `raws_nearest_station_grid.py` — Nearest-RAWS-station index grid
- `build_multires_manifest.py` — Multi-resolution covariate manifest builder
- `validate_multires_manifest.py` — Manifest schema validator
- `README.md` — Original script documentation

## docs/

Design documents and planning materials.

- `higher_resolution_data_options.md` — Comparative analysis of VIIRS, HLS, Sentinel, PRISM, MRMS, etc.
- `multires_data_contract.md` — Versioned manifest specification for multi-resolution data
- `examples/multires_config.example.json` — Sample configuration
- `README.md` — Original docs overview

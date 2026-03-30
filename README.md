# Wildfire Prediction

Machine learning models for predicting wildfire spread using GOES satellite fire detection data, weather, terrain, and vegetation features.

## Frontier Models

### FireSpreadNet (full-image prediction)
ConvGRU-U-Net with CBAM attention that predicts the entire next fire spread image from a sequence of frames. Uses 28-channel input (pre-shifted fire state, weather, terrain, vegetation, temporal encoding).

- Architecture: `models/firespreadnet/architecture.py`
- Training: `models/firespreadnet/train.py`

### XGBoost (single-pixel prediction)
Gradient-boosted tree classifier that predicts fire/no-fire per pixel using neighborhood context, weather, and terrain features.

- Training: `models/xgboost/train.py`
- Logistic regression baseline: `models/xgboost/train_logreg.py`

Both models use the **V3 pipeline data format** where `prev_fire_state[t] = labels[t-1]`, eliminating temporal leakage by construction.

## Data Pipeline

Training data is produced by the [wildfire-data-pipeline](https://github.com/seanpatrickmay/wildfire-data-pipeline) repository, which handles:
- GOES fire detection with DQF cloud masking
- Multi-source fusion (GOES + VIIRS + MODIS)
- GRIDMET fire weather indices
- 3DEP terrain features
- Pre-shifted fire state computation

The data loader (`data/pipeline_loader.py`) reads the pipeline's processed NPZ output files.

## Webapp

Interactive fire spread prediction viewer with time playback.

- Frontend: React + TypeScript (`webapp/frontend/`)
- Backend: FastAPI serving predictions (`webapp/backend/`)

## Repository Layout

```
models/
  firespreadnet/    ConvGRU-U-Net architecture + training
  xgboost/          XGBoost + logistic regression training
data/
  pipeline_loader.py  V3 data loader for pipeline NPZ files
webapp/
  frontend/         React visualization app
  backend/          FastAPI prediction server
tests/              Unit tests
archive/            Superseded experiments and legacy code (see archive/README.md)
```

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, XGBoost, scikit-learn, NumPy, FastAPI.

## Archive

Previous model versions, experiments, and ETL scripts are preserved in `archive/` with a detailed summary of findings. See [`archive/README.md`](archive/README.md).

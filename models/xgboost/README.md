# XGBoost Pixel Classifier

Gradient-boosted decision tree ensemble for per-pixel wildfire spread prediction, with a logistic regression baseline.

## Architecture

- **Model**: XGBClassifier (gradient-boosted trees)
- **Task**: Per-pixel binary classification (fire / no-fire)
- **Features**: 80 inputs -- 3x3 neighborhood fire history (6 hours), distance-to-fire, terrain, weather, vegetation, temporal encoding
- **Hyperparameters**: max_depth=8, n_estimators=500, learning_rate=0.05

## Training

| Parameter           | Value                                 |
|---------------------|---------------------------------------|
| Data pipeline       | V3 (prev_fire_state, no temporal leakage) |
| Negative subsampling | 5:1 ratio                            |
| Class balancing     | scale_pos_weight auto-computed        |
| Train fires         | 8                                     |
| Test fires          | 4 (held-out)                          |

## Performance (V3)

| Model   | F1    | Precision | Recall | AUC   |
|---------|-------|-----------|--------|-------|
| XGBoost | 0.884 | 0.822     | 0.956  | 0.995 |
| LogReg  | 0.832 | 0.731     | 0.966  | 0.994 |

Per-fire XGBoost F1:

| Fire       | F1    |
|------------|-------|
| CampFire   | 0.869 |
| CreekFire  | 0.887 |
| DolanFire  | 0.877 |
| GlassFire  | 0.898 |

Top features by importance:

| Feature                    | Importance |
|----------------------------|------------|
| Center pixel fire state    | 37.5%      |
| Previous hour center       | 23.4%      |
| Fire neighborhood density  | 23.0%      |

## Logistic Regression Baseline

- Logistic regression with balanced class weights
- StandardScaler preprocessing
- Serves as a linear baseline for comparison against XGBoost

## Files

- `train.py` -- XGBoost training
- `train_logreg.py` -- Logistic regression baseline
- `analysis.ipynb` -- Comparison notebook

## Usage

```python
import pickle

xgb_model = pickle.load(open("data/checkpoints/xgboost_v3_best.pkl", "rb"))
y_pred = xgb_model.predict(X_features)  # X_features: (N, 80) float32
y_prob = xgb_model.predict_proba(X_features)[:, 1]
```

# FireSpreadNet V2

ConvGRU-U-Net with CBAM attention for satellite-based wildfire spread prediction.

## Architecture

- **Backbone**: 3-level encoder-decoder U-Net with skip connections
- **Temporal modeling**: ConvGRU aggregation at each encoder level
- **Attention**: CBAM (Channel + Spatial) attention modules
- **Loss**: Combined Focal + Tversky loss, balanced for fire class rarity
- **Input**: `(B, T, 28, H, W)` multi-channel sequences
- **Output**: `(B, 1, H, W)` fire probability map

The 28 input channels (defined in `CHANNEL_ORDER_V3` in `data/pipeline_loader.py`) include prev_fire_state (pre-shifted labels), weather variables, terrain features, vegetation indices, and temporal encoding.

## Training

| Parameter        | Value                          |
|------------------|--------------------------------|
| Data pipeline    | V3 (prev_fire_state labels)    |
| Sequence length  | 6 hours                        |
| Batch size       | 4                              |
| Optimizer        | AdamW                          |
| Learning rate    | 3e-4                           |
| Epochs           | 20                             |
| Augmentation     | Random flip with wind vector correction |
| Train fires      | 8                              |
| Test fires       | 4 (held-out)                   |

## Performance (V3)

**Best test F1: 0.929** (epoch 14)

| Metric    | Value |
|-----------|-------|
| Precision | ~0.93 |
| Recall    | ~0.92 |

Per-fire F1 scores:

| Fire       | F1    |
|------------|-------|
| CampFire   | 0.937 |
| CreekFire  | 0.929 |
| DolanFire  | 0.923 |
| GlassFire  | 0.924 |

Remarkably consistent across fires (0.923--0.937 range).

## Files

- `architecture.py` -- Model definition
- `train.py` -- Training script
- `analysis.ipynb` -- Evaluation notebook

## Usage (Inference)

```python
import torch
from models.firespreadnet.architecture import FireSpreadNetV2

ckpt = torch.load("data/checkpoints/firespreadnet_v3_best.pt", map_location="cpu", weights_only=False)
model = FireSpreadNetV2(in_channels=28, dropout=0.1)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# input_seq: (1, 6, 28, H, W) normalized float32
with torch.no_grad():
    logits = model(input_seq)
    probs = torch.sigmoid(logits)  # (1, 1, H, W)
```

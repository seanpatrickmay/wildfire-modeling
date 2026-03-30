"""Enhanced ConvGRU-U-Net for full-image fire spread prediction.

Improvements over the original ConvGRU:
- CBAM (Channel + Spatial) attention on skip connections
- GELU activation instead of ReLU
- Spatial dropout in encoder
- Focal + Focal Tversky loss for extreme class imbalance
- Data augmentation (flips, 90-degree rotations with wind direction correction)
- Multi-level temporal processing option
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_H = 112
PAD_W = 192
IN_CHANNELS = 8
N_PHYSICAL_CHANNELS = 7

# Wind direction channel indices (in the 8-channel frame)
WDIR_SIN_CH = 3
WDIR_COS_CH = 4


# ─── Attention Modules ─────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        attn = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# ─── Building Blocks ───────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        down = self.pool(features)
        return features, down


class DecoderLevel(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn = CBAM(skip_ch)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        skip = self.attn(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvGRUCell(nn.Module):
    def __init__(self, input_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(input_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad)
        self.candidate = nn.Conv2d(input_ch + hidden_ch, hidden_ch, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.gates(combined))
        r, z = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * candidate


# ─── Main Model ────────────────────────────────────────────────────────────
class FireSpreadNet(nn.Module):
    """Enhanced ConvGRU-U-Net with CBAM attention for fire spread prediction.

    Input:  (B, T, C=8, H, W)  — T frames of 8-channel fire/weather grids
    Output: (B, 1, H, W)       — logits for next-hour fire probability
    """
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        encoder_channels: tuple[int, ...] = (32, 64, 128),
        bottleneck_ch: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        enc = list(encoder_channels)

        self.enc0 = EncoderLevel(in_channels, enc[0], dropout=dropout)
        self.enc1 = EncoderLevel(enc[0], enc[1], dropout=dropout)
        self.enc2 = EncoderLevel(enc[1], enc[2], dropout=dropout)

        self.bottleneck = ConvBlock(enc[2], bottleneck_ch)
        self.conv_gru = ConvGRUCell(bottleneck_ch, bottleneck_ch)

        self.dec2 = DecoderLevel(bottleneck_ch, enc[2], enc[2])
        self.dec1 = DecoderLevel(enc[2], enc[1], enc[1])
        self.dec0 = DecoderLevel(enc[1], enc[0], enc[0])

        self.out_conv = nn.Conv2d(enc[0], 1, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x_seq.shape

        h_h, h_w = H // 8, W // 8
        h = torch.zeros(B, self.conv_gru.hidden_ch, h_h, h_w,
                        device=x_seq.device, dtype=x_seq.dtype)

        for t in range(T):
            frame = x_seq[:, t]
            skip0, d0 = self.enc0(frame)
            skip1, d1 = self.enc1(d0)
            skip2, d2 = self.enc2(d1)
            bn = self.bottleneck(d2)
            h = self.conv_gru(bn, h)

        # Decode using last frame's skip connections (with attention)
        up2 = self.dec2(h, skip2)
        up1 = self.dec1(up2, skip1)
        up0 = self.dec0(up1, skip0)
        return self.out_conv(up0)


# ─── Loss Functions ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * ((1 - pt) ** self.gamma) * bce
        return (focal * mask).sum() / mask.sum().clamp(min=1)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight (higher = penalize missed fires more)
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = (probs * mask).flatten(1)
        t = (targets * mask).flatten(1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        # Only compute for samples that have at least one positive pixel
        has_pos = (t.sum(1) > 0)
        if not has_pos.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        tp = tp[has_pos]
        fp = fp[has_pos]
        fn = fn[has_pos]
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky).clamp(min=1e-7) ** self.gamma
        return focal_tversky.mean()


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight: float = 0.5, tversky_weight: float = 0.5) -> None:
        super().__init__()
        self.focal = FocalLoss(alpha=0.85, gamma=2.0)
        self.tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.fw = focal_weight
        self.tw = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(logits, targets, mask) + self.tw * self.tversky(logits, targets, mask)


# ─── Data Augmentation ──────────────────────────────────────────────────────
def augment_sequence(
    frames: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply random flips and 90-degree rotations with wind direction correction.

    frames: (T, C, H, W)
    target: (1, H, W)
    mask:   (1, H, W)
    """
    # Random horizontal flip
    if torch.rand(1, generator=rng).item() > 0.5:
        frames = frames.flip(-1)  # flip W
        target = target.flip(-1)
        mask = mask.flip(-1)
        # Flip wind sin (E-W component changes sign for horizontal flip)
        frames[:, WDIR_SIN_CH] = -frames[:, WDIR_SIN_CH]

    # Random vertical flip
    if torch.rand(1, generator=rng).item() > 0.5:
        frames = frames.flip(-2)  # flip H
        target = target.flip(-2)
        mask = mask.flip(-2)
        # Flip wind cos (N-S component changes sign)
        frames[:, WDIR_COS_CH] = -frames[:, WDIR_COS_CH]

    # Skip 90-degree rotation: PAD_H != PAD_W so rotation changes shape.
    # Flips already provide 4x augmentation and preserve dimensions.

    return frames, target, mask


# ─── Frame Building ─────────────────────────────────────────────────────────
def build_frame(
    goes_conf_t: np.ndarray,
    rtma_hour: dict[str, np.ndarray],
    goes_shape: tuple[int, int],
    channel_means: np.ndarray,
    channel_stds: np.ndarray,
    pad_h: int = PAD_H,
    pad_w: int = PAD_W,
) -> np.ndarray:
    H, W = goes_shape
    conf = goes_conf_t.astype(np.float32)
    tmp = rtma_hour["TMP"].astype(np.float32)
    wind = rtma_hour["WIND"].astype(np.float32)
    wdir_rad = np.deg2rad(rtma_hour["WDIR"].astype(np.float32))
    wdir_sin = np.sin(wdir_rad)
    wdir_cos = np.cos(wdir_rad)
    spfh = rtma_hour["SPFH"].astype(np.float32)
    precip = np.clip(rtma_hour["ACPC01"].astype(np.float32), 0.0, 100.0)

    raw = [conf, tmp, wind, wdir_sin, wdir_cos, spfh, precip]
    frame = np.zeros((IN_CHANNELS, pad_h, pad_w), dtype=np.float32)
    for i, ch in enumerate(raw):
        ch = np.nan_to_num(ch, nan=0.0)
        ch = (ch - channel_means[i]) / max(channel_stds[i], 1e-8)
        frame[i, :H, :W] = ch.astype(np.float32)
    frame[N_PHYSICAL_CHANNELS, :H, :W] = 1.0  # validity mask
    return frame

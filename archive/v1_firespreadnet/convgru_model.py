"""ConvGRU U-Net model classes extracted from docs/convgru_fire_holdout.ipynb."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Default architecture constants
PAD_H = 112
PAD_W = 192
IN_CHANNELS = 8  # 7 physical + 1 validity mask
N_PHYSICAL_CHANNELS = 7

CHANNEL_NAMES = [
    "goes_conf", "temperature", "wind_speed",
    "wind_dir_sin", "wind_dir_cos", "specific_humidity",
    "precipitation_1h",
]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderLevel(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        down = self.pool(features)
        return features, down


class DecoderLevel(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvGRUCell(nn.Module):
    def __init__(self, input_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(input_ch + hidden_ch, 2 * hidden_ch, kernel_size, padding=pad, bias=True)
        self.candidate = nn.Conv2d(input_ch + hidden_ch, hidden_ch, kernel_size, padding=pad, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=1)
        gates = torch.sigmoid(self.gates(combined))
        r, z = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * candidate
        return h_new


class ConvGRUUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        encoder_channels: tuple[int, ...] = (32, 64, 128),
        bottleneck_ch: int = 256,
    ) -> None:
        super().__init__()
        enc_chs = list(encoder_channels)

        self.enc0 = EncoderLevel(in_channels, enc_chs[0])
        self.enc1 = EncoderLevel(enc_chs[0], enc_chs[1])
        self.enc2 = EncoderLevel(enc_chs[1], enc_chs[2])

        self.bottleneck = ConvBlock(enc_chs[2], bottleneck_ch)
        self.conv_gru = ConvGRUCell(bottleneck_ch, bottleneck_ch, kernel_size=3)

        self.dec2 = DecoderLevel(bottleneck_ch, enc_chs[2], enc_chs[2])
        self.dec1 = DecoderLevel(enc_chs[2], enc_chs[1], enc_chs[1])
        self.dec0 = DecoderLevel(enc_chs[1], enc_chs[0], enc_chs[0])

        self.out_conv = nn.Conv2d(enc_chs[0], 1, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (B, T, C, H, W) -> logits (B, 1, H, W)"""
        B, T, C, H, W = x_seq.shape

        h_height = H // 8
        h_width = W // 8
        h = torch.zeros(B, self.conv_gru.hidden_ch, h_height, h_width,
                        device=x_seq.device, dtype=x_seq.dtype)

        last_skip0 = None
        last_skip1 = None
        last_skip2 = None

        for t_idx in range(T):
            frame = x_seq[:, t_idx]

            skip0, down0 = self.enc0(frame)
            skip1, down1 = self.enc1(down0)
            skip2, down2 = self.enc2(down1)

            bottleneck = self.bottleneck(down2)
            h = self.conv_gru(bottleneck, h)

            if t_idx == T - 1:
                last_skip0 = skip0
                last_skip1 = skip1
                last_skip2 = skip2

        up2 = self.dec2(h, last_skip2)
        up1 = self.dec1(up2, last_skip1)
        up0 = self.dec0(up1, last_skip0)

        logits = self.out_conv(up0)
        return logits


def build_frame(
    goes_conf_t: np.ndarray,
    rtma_hour: dict[str, np.ndarray],
    goes_shape: tuple[int, int],
    channel_means: np.ndarray,
    channel_stds: np.ndarray,
    pad_h: int = PAD_H,
    pad_w: int = PAD_W,
) -> np.ndarray:
    """Build an 8-channel padded frame for one timestep.
    Returns: (C, pad_h, pad_w) float32 array.
    """
    H, W = goes_shape
    conf = goes_conf_t.astype(np.float32)
    tmp = rtma_hour["TMP"].astype(np.float32)
    wind = rtma_hour["WIND"].astype(np.float32)
    wdir_rad = np.deg2rad(rtma_hour["WDIR"].astype(np.float32))
    wdir_sin = np.sin(wdir_rad)
    wdir_cos = np.cos(wdir_rad)
    spfh = rtma_hour["SPFH"].astype(np.float32)
    precip = rtma_hour["ACPC01"].astype(np.float32)

    raw_channels = [conf, tmp, wind, wdir_sin, wdir_cos, spfh, precip]

    norm_channels = []
    for i, ch in enumerate(raw_channels):
        ch = np.nan_to_num(ch, nan=0.0)
        ch = (ch - channel_means[i]) / channel_stds[i]
        norm_channels.append(ch.astype(np.float32))

    validity = np.ones((H, W), dtype=np.float32)

    frame = np.zeros((IN_CHANNELS, pad_h, pad_w), dtype=np.float32)
    for i, ch in enumerate(norm_channels):
        frame[i, :H, :W] = ch
    frame[N_PHYSICAL_CHANNELS, :H, :W] = validity

    return frame

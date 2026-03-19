"""Pre-compute model predictions for the 8 held-out test fires.

Generates data/predictions/{fire_name}/{model_name}.npz for each fire × model.
Each .npz contains:
  - probs: float16 array of shape (T, H, W) — per-pixel fire probability
  - time_indices: int array of GOES time indices that were predicted

Usage:
    python scripts/precompute_predictions.py [--models logreg,mlp,xgboost,rnn,convgru]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.neighbor_cell_logreg import (
    build_feature_schema,
    build_hour_samples,
    discover_fire_entries,
    find_repo_root,
    iter_aligned_hours_for_fire,
    load_fire_entry_context,
    select_fire_entries,
    ZScoreNormalizer,
)

TEST_FIRES = [
    "August_Complex",
    "Beckwourth_Complex",
    "CZU_Lightning_Complex",
    "Dolan",
    "LNU_Lightning_Complex",
    "McCash",
    "Walker",
    "Windy",
]

POSITIVE_THRESHOLD = 0.10

# Per-pixel model iteration kwargs
PIXELWISE_ITERATION_KWARGS = {
    "discounted_rain_lookback_hours": 24 * 30,
    "discounted_rain_half_life_days": 7.0,
}

# ConvGRU iteration kwargs (no discounted rain)
CONVGRU_ITERATION_KWARGS = {
    "discounted_rain_lookback_hours": 0,
    "discounted_rain_half_life_days": 0.0,
}


def load_normalizer_from_checkpoint(ckpt: dict) -> ZScoreNormalizer:
    return ZScoreNormalizer(
        enabled=True,
        mean=np.array(ckpt["normalizer_mean"], dtype=np.float64),
        std=np.array(ckpt["normalizer_std"], dtype=np.float64),
        std_safe=np.where(
            np.array(ckpt["normalizer_std"], dtype=np.float64) > 0,
            np.array(ckpt["normalizer_std"], dtype=np.float64),
            1.0,
        ),
        samples_used=0,
        zero_std_feature_count=0,
    )


def precompute_logreg(entries: list[dict], repo_root: Path, ckpt_dir: Path, out_dir: Path) -> None:
    import joblib

    model_path = ckpt_dir / "logreg.joblib"
    meta_path = ckpt_dir / "logreg_meta.json"
    if not model_path.exists():
        print(f"  SKIP logreg: {model_path} not found")
        return

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    normalizer = load_normalizer_from_checkpoint(meta)
    schema = build_feature_schema(include_discounted_rain=meta.get("include_discounted_rain", True))

    for entry in entries:
        fire_name = entry["fire_name"]
        print(f"  logreg -> {fire_name}")
        ctx = load_fire_entry_context(entry, repo_root)
        H, W = ctx["goes_shape"]
        inner_h, inner_w = H - 2, W - 2

        probs_list = []
        time_indices = []

        for t, rtma_hour in iter_aligned_hours_for_fire(
            repo_root, ctx["goes_conf"], ctx["goes_time_index"],
            ctx["rtma_manifest"], ctx["rtma_manifest_path"],
            ctx["goes_shape"], ctx["goes_transform"], ctx["goes_crs"],
            include_discounted_rain=schema.include_discounted_rain,
            **PIXELWISE_ITERATION_KWARGS,
        ):
            X_hour, _ = build_hour_samples(schema, ctx["goes_conf"][t], ctx["goes_conf"][t + 1], rtma_hour)
            if X_hour.shape[0] == 0:
                continue

            X_norm = normalizer.transform(X_hour)
            prob = model.predict_proba(X_norm)[:, 1]

            prob_grid = np.full((H, W), np.nan, dtype=np.float32)
            prob_grid[1:-1, 1:-1] = prob.reshape(inner_h, inner_w).astype(np.float32)
            probs_list.append(prob_grid)
            time_indices.append(t)

        if probs_list:
            fire_out = out_dir / fire_name
            fire_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fire_out / "logreg.npz",
                probs=np.array(probs_list, dtype=np.float16),
                time_indices=np.array(time_indices, dtype=np.int32),
            )
            print(f"    saved {len(probs_list)} frames")


def precompute_mlp(entries: list[dict], repo_root: Path, ckpt_dir: Path, out_dir: Path) -> None:
    import torch
    from scripts.neighbor_cell_nn import FeedForwardMLP, MLPConfig, predict_probabilities

    ckpt_path = ckpt_dir / "mlp.pt"
    if not ckpt_path.exists():
        print(f"  SKIP mlp: {ckpt_path} not found")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    normalizer = load_normalizer_from_checkpoint(ckpt)
    schema = build_feature_schema(include_discounted_rain=ckpt.get("include_discounted_rain", True))

    config = MLPConfig(**ckpt["config"])
    model = FeedForwardMLP(
        in_dim=schema.n_features,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        use_batchnorm=config.use_batchnorm,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = "cpu"

    for entry in entries:
        fire_name = entry["fire_name"]
        print(f"  mlp -> {fire_name}")
        ctx = load_fire_entry_context(entry, repo_root)
        H, W = ctx["goes_shape"]
        inner_h, inner_w = H - 2, W - 2

        probs_list = []
        time_indices = []

        for t, rtma_hour in iter_aligned_hours_for_fire(
            repo_root, ctx["goes_conf"], ctx["goes_time_index"],
            ctx["rtma_manifest"], ctx["rtma_manifest_path"],
            ctx["goes_shape"], ctx["goes_transform"], ctx["goes_crs"],
            include_discounted_rain=schema.include_discounted_rain,
            **PIXELWISE_ITERATION_KWARGS,
        ):
            X_hour, _ = build_hour_samples(schema, ctx["goes_conf"][t], ctx["goes_conf"][t + 1], rtma_hour)
            if X_hour.shape[0] == 0:
                continue

            X_norm = normalizer.transform(X_hour)
            prob = predict_probabilities(model, X_norm, batch_size=8192, device=device)

            prob_grid = np.full((H, W), np.nan, dtype=np.float32)
            prob_grid[1:-1, 1:-1] = prob.reshape(inner_h, inner_w).astype(np.float32)
            probs_list.append(prob_grid)
            time_indices.append(t)

        if probs_list:
            fire_out = out_dir / fire_name
            fire_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fire_out / "mlp.npz",
                probs=np.array(probs_list, dtype=np.float16),
                time_indices=np.array(time_indices, dtype=np.int32),
            )
            print(f"    saved {len(probs_list)} frames")


def precompute_xgboost(entries: list[dict], repo_root: Path, ckpt_dir: Path, out_dir: Path) -> None:
    import xgboost as xgb

    model_path = ckpt_dir / "xgboost.json"
    meta_path = ckpt_dir / "xgboost_meta.json"
    if not model_path.exists():
        print(f"  SKIP xgboost: {model_path} not found")
        return

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)

    normalizer = load_normalizer_from_checkpoint(meta)
    schema = build_feature_schema(include_discounted_rain=meta.get("include_discounted_rain", True))

    for entry in entries:
        fire_name = entry["fire_name"]
        print(f"  xgboost -> {fire_name}")
        ctx = load_fire_entry_context(entry, repo_root)
        H, W = ctx["goes_shape"]
        inner_h, inner_w = H - 2, W - 2

        probs_list = []
        time_indices = []

        for t, rtma_hour in iter_aligned_hours_for_fire(
            repo_root, ctx["goes_conf"], ctx["goes_time_index"],
            ctx["rtma_manifest"], ctx["rtma_manifest_path"],
            ctx["goes_shape"], ctx["goes_transform"], ctx["goes_crs"],
            include_discounted_rain=schema.include_discounted_rain,
            **PIXELWISE_ITERATION_KWARGS,
        ):
            X_hour, _ = build_hour_samples(schema, ctx["goes_conf"][t], ctx["goes_conf"][t + 1], rtma_hour)
            if X_hour.shape[0] == 0:
                continue

            X_norm = normalizer.transform(X_hour).astype(np.float32)
            prob = model.predict_proba(X_norm)[:, 1]

            prob_grid = np.full((H, W), np.nan, dtype=np.float32)
            prob_grid[1:-1, 1:-1] = prob.reshape(inner_h, inner_w).astype(np.float32)
            probs_list.append(prob_grid)
            time_indices.append(t)

        if probs_list:
            fire_out = out_dir / fire_name
            fire_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fire_out / "xgboost.npz",
                probs=np.array(probs_list, dtype=np.float16),
                time_indices=np.array(time_indices, dtype=np.int32),
            )
            print(f"    saved {len(probs_list)} frames")


def precompute_rnn(entries: list[dict], repo_root: Path, ckpt_dir: Path, out_dir: Path) -> None:
    import torch
    from scripts.rnn_model import FireGRU, add_temporal_deltas
    from scripts.neighbor_cell_logreg import iter_fire_hour_samples

    ckpt_path = ckpt_dir / "rnn_gru_attention.pt"
    if not ckpt_path.exists():
        print(f"  SKIP rnn: {ckpt_path} not found")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    normalizer = load_normalizer_from_checkpoint(ckpt)
    schema = build_feature_schema(include_discounted_rain=ckpt.get("include_discounted_rain", True))

    config = ckpt["config"]
    seq_len = config["seq_len"]
    include_deltas = config.get("include_deltas", True)
    input_dim = schema.n_features * (2 if include_deltas else 1)

    model = FireGRU(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = "cpu"

    for entry in entries:
        fire_name = entry["fire_name"]
        print(f"  rnn -> {fire_name}")
        ctx = load_fire_entry_context(entry, repo_root)
        H, W = ctx["goes_shape"]
        inner_h, inner_w = H - 2, W - 2

        probs_list = []
        time_indices = []

        # Build sequences from aligned hours
        buffer: list[tuple[int, np.ndarray, np.ndarray]] = []

        for _, t, X_hour, y_hour in iter_fire_hour_samples(
            entry, repo_root, schema, POSITIVE_THRESHOLD,
            **PIXELWISE_ITERATION_KWARGS,
        ):
            X_norm = normalizer.transform(X_hour).astype(np.float32, copy=False)
            y_bin = y_hour.astype(np.float32, copy=False)
            buffer.append((t, X_norm, y_bin))
            if len(buffer) > seq_len:
                buffer.pop(0)
            if len(buffer) == seq_len:
                t_vals = [b[0] for b in buffer]
                if t_vals[-1] - t_vals[0] != seq_len - 1:
                    continue
                n_pix = buffer[0][1].shape[0]
                if not all(b[1].shape[0] == n_pix for b in buffer):
                    continue

                X_seq = np.stack([b[1] for b in buffer], axis=1)  # (N, T, F)
                if include_deltas:
                    X_seq = add_temporal_deltas(X_seq)

                with torch.no_grad():
                    xb = torch.from_numpy(X_seq).to(device=device, dtype=torch.float32)
                    logits = model(xb)
                    prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()

                prob_grid = np.full((H, W), np.nan, dtype=np.float32)
                prob_grid[1:-1, 1:-1] = prob.reshape(inner_h, inner_w).astype(np.float32)
                probs_list.append(prob_grid)
                time_indices.append(t_vals[-1])

        if probs_list:
            fire_out = out_dir / fire_name
            fire_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fire_out / "rnn.npz",
                probs=np.array(probs_list, dtype=np.float16),
                time_indices=np.array(time_indices, dtype=np.int32),
            )
            print(f"    saved {len(probs_list)} frames")


def precompute_convgru(entries: list[dict], repo_root: Path, ckpt_dir: Path, out_dir: Path) -> None:
    import torch
    from scripts.convgru_model import ConvGRUUNet, build_frame, PAD_H, PAD_W

    ckpt_path = ckpt_dir / "convgru_unet.pt"
    if not ckpt_path.exists():
        print(f"  SKIP convgru: {ckpt_path} not found")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    channel_means = np.array(ckpt["channel_means"], dtype=np.float64)
    channel_stds = np.array(ckpt["channel_stds"], dtype=np.float64)

    config = ckpt["config"]
    seq_len = config["seq_len"]

    model = ConvGRUUNet(
        in_channels=config.get("in_channels", 8),
        encoder_channels=tuple(config.get("encoder_channels", [32, 64, 128])),
        bottleneck_ch=config.get("bottleneck_channels", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = "cpu"

    for entry in entries:
        fire_name = entry["fire_name"]
        print(f"  convgru -> {fire_name}")
        ctx = load_fire_entry_context(entry, repo_root)
        goes_conf = ctx["goes_conf"]
        goes_shape = ctx["goes_shape"]
        H, W = goes_shape

        probs_list = []
        time_indices = []

        buffer: list[tuple[int, np.ndarray]] = []

        for t, rtma_hour in iter_aligned_hours_for_fire(
            repo_root, goes_conf, ctx["goes_time_index"],
            ctx["rtma_manifest"], ctx["rtma_manifest_path"],
            goes_shape, ctx["goes_transform"], ctx["goes_crs"],
            include_discounted_rain=False,
            **CONVGRU_ITERATION_KWARGS,
        ):
            frame = build_frame(goes_conf[t], rtma_hour, goes_shape, channel_means, channel_stds)
            buffer.append((t, frame))
            if len(buffer) > seq_len:
                buffer.pop(0)
            if len(buffer) == seq_len:
                t_vals = [b[0] for b in buffer]
                if t_vals[-1] - t_vals[0] != seq_len - 1:
                    continue

                X_seq = np.stack([b[1] for b in buffer], axis=0)  # (T, C, H, W)
                X_batch = torch.from_numpy(X_seq).unsqueeze(0).to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    logits = model(X_batch)  # (1, 1, PAD_H, PAD_W)
                    prob_padded = torch.sigmoid(logits).squeeze().cpu().numpy()

                prob_grid = prob_padded[:H, :W].astype(np.float32)
                probs_list.append(prob_grid)
                time_indices.append(t_vals[-1])

        if probs_list:
            fire_out = out_dir / fire_name
            fire_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fire_out / "convgru.npz",
                probs=np.array(probs_list, dtype=np.float16),
                time_indices=np.array(time_indices, dtype=np.int32),
            )
            print(f"    saved {len(probs_list)} frames")


MODEL_RUNNERS = {
    "logreg": precompute_logreg,
    "mlp": precompute_mlp,
    "xgboost": precompute_xgboost,
    "rnn": precompute_rnn,
    "convgru": precompute_convgru,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute predictions for test fires")
    parser.add_argument("--models", type=str, default="logreg,mlp,xgboost,rnn,convgru",
                        help="Comma-separated list of models to run")
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve().parent)
    ckpt_dir = repo_root / "data" / "checkpoints"
    out_dir = repo_root / "data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_entries = discover_fire_entries(repo_root)
    test_entries = select_fire_entries(all_entries, TEST_FIRES)
    print(f"Test fires: {[e['fire_name'] for e in test_entries]}")

    models_to_run = [m.strip() for m in args.models.split(",")]

    for model_name in models_to_run:
        runner = MODEL_RUNNERS.get(model_name)
        if runner is None:
            print(f"Unknown model: {model_name}")
            continue
        print(f"\n=== {model_name} ===")
        runner(test_entries, repo_root, ckpt_dir, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

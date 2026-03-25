"""FastAPI backend serving ground truth frames, metadata, and recursive
FireSpreadNet v2 predictions from the wildfire-data-pipeline NPZ files."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fire_spread_model_v2 import FireSpreadNetV2
from scripts.pipeline_data_loader import (
    CHANNEL_ORDER,
    build_channel_stack,
    load_fire_data,
    normalize_stack,
    smooth_labels,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Wildfire Prediction Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPELINE_DATA_DIR = REPO_ROOT.parent / "wildfire-data-pipeline" / "data"
CHECKPOINT_PATH = REPO_ROOT / "data" / "checkpoints" / "firespreadnet_v2_best.pt"
CHANNEL_STATS_PATH = REPO_ROOT / "data" / "checkpoints" / "firespreadnet_v2_channel_stats.json"

# Smoothing parameters (must match training)
SMOOTH_WINDOW = 5
SMOOTH_MIN_VOTES = 2
SMOOTH_THRESHOLD = 0.30

SEQ_LEN = 6  # temporal window the model expects

# Confidence channel index in the 27-channel stack
CONF_CH = CHANNEL_ORDER.index("confidence")

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------
_fire_cache: dict[str, dict[str, Any]] = {}
_model: FireSpreadNetV2 | None = None
_channel_means: np.ndarray | None = None
_channel_stds: np.ndarray | None = None

# Cache for recursive predictions keyed by (fire_name, start)
_prediction_cache: dict[tuple[str, int], list[np.ndarray]] = {}


# ---------------------------------------------------------------------------
# Helpers — model loading
# ---------------------------------------------------------------------------
def _load_model() -> FireSpreadNetV2:
    global _model
    if _model is not None:
        return _model
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=True)
    model = FireSpreadNetV2(in_channels=checkpoint["in_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model
    return _model


def _load_channel_stats() -> tuple[np.ndarray, np.ndarray]:
    global _channel_means, _channel_stds
    if _channel_means is not None and _channel_stds is not None:
        return _channel_means, _channel_stds
    with open(CHANNEL_STATS_PATH, "r") as f:
        stats = json.load(f)
    _channel_means = np.array(stats["means"], dtype=np.float32)
    _channel_stds = np.array(stats["stds"], dtype=np.float32)
    return _channel_means, _channel_stds


# ---------------------------------------------------------------------------
# Helpers — fire data loading & caching
# ---------------------------------------------------------------------------
def _discover_fires() -> list[str]:
    """Return sorted list of fire names available in the pipeline data dir."""
    if not PIPELINE_DATA_DIR.exists():
        return []
    fires: list[str] = []
    for child in sorted(PIPELINE_DATA_DIR.iterdir()):
        if not child.is_dir():
            continue
        conf_files = list(child.glob(f"{child.name}_*_FusedConf.npz"))
        feat_files = list(child.glob(f"{child.name}_*_Features.npz"))
        if conf_files and feat_files:
            fires.append(child.name)
    return fires


def _load_fire(fire_name: str) -> dict[str, Any]:
    """Load and cache all data for a single fire."""
    if fire_name in _fire_cache:
        return _fire_cache[fire_name]

    fire_arrays, feature_arrays, fire_meta, feat_meta = load_fire_data(
        fire_name, str(PIPELINE_DATA_DIR),
    )

    conf = fire_arrays["data"]  # (T, H, W)
    T, H, W = conf.shape

    # Compute padded dimensions (must match training)
    pad_h = max(32, ((H + 15) // 16) * 16)
    pad_w = max(48, ((W + 15) // 16) * 16)

    # Build smoothed labels
    cloud_mask = fire_arrays.get("cloud_mask")
    obs_valid = fire_arrays.get("observation_valid")
    labels, validity = smooth_labels(
        conf,
        cloud_mask,
        obs_valid,
        window=SMOOTH_WINDOW,
        min_votes=SMOOTH_MIN_VOTES,
        threshold=SMOOTH_THRESHOLD,
    )

    # Build channel stack (T, C, pad_h, pad_w)
    stack = build_channel_stack(fire_arrays, feature_arrays, pad_h=pad_h, pad_w=pad_w)

    # Normalize
    means, stds = _load_channel_stats()
    norm_stack = normalize_stack(stack, means, stds)

    # Count smoothed fire pixels per timestep
    smoothed_fire_pixels = [int(labels[t].sum()) for t in range(T)]

    result: dict[str, Any] = {
        "fire_arrays": fire_arrays,
        "feature_arrays": feature_arrays,
        "fire_meta": fire_meta,
        "feat_meta": feat_meta,
        "labels": labels,
        "validity": validity,
        "norm_stack": norm_stack,
        "T": T,
        "H": H,
        "W": W,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "smoothed_fire_pixels": smoothed_fire_pixels,
    }
    _fire_cache[fire_name] = result
    return result


def _get_fire_or_404(name: str) -> dict[str, Any]:
    available = _discover_fires()
    if name not in available:
        raise HTTPException(404, f"Fire not found: {name}")
    return _load_fire(name)


# ---------------------------------------------------------------------------
# Helpers — recursive prediction
# ---------------------------------------------------------------------------
def _run_recursive_predictions(
    fire_name: str,
    start: int,
    steps: int,
) -> list[np.ndarray]:
    """Run FireSpreadNet v2 recursively from `start` for `steps` iterations.

    Returns a list of `steps` numpy arrays, each (H, W) float32 sigmoid
    probabilities (unpadded).

    Step 0: model([start-5 .. start])      → prediction at start+1
    Step 1: replace conf at start+1 with pred[0],
            model([start-4 .. start+1])     → prediction at start+2
    ...
    """
    cache_key = (fire_name, start)
    if cache_key in _prediction_cache:
        cached = _prediction_cache[cache_key]
        if len(cached) >= steps:
            return cached[:steps]
        # Need more steps — recompute from scratch
        # (simpler than extending; the model is fast on CPU for single grids)

    fire = _get_fire_or_404(fire_name)
    norm_stack = fire["norm_stack"]  # (T, C, pad_h, pad_w)
    T, H, W = fire["T"], fire["H"], fire["W"]

    if start < SEQ_LEN - 1:
        raise HTTPException(400, f"start must be >= {SEQ_LEN - 1} (need {SEQ_LEN} frames of history)")
    if start + steps >= T:
        raise HTTPException(400, f"start+steps ({start + steps}) exceeds fire length ({T})")

    model = _load_model()

    # Make a mutable copy of the stack so we can inject predictions
    working_stack = norm_stack.copy()

    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for step_idx in range(steps):
            # Window of SEQ_LEN frames ending at (start + step_idx)
            t_end = start + step_idx
            t_start = t_end - SEQ_LEN + 1
            frames = working_stack[t_start: t_end + 1]  # (SEQ_LEN, C, pad_h, pad_w)

            # Model expects (B, T, C, H, W)
            x = torch.from_numpy(frames).unsqueeze(0).float()
            logits = model(x)  # (1, 1, pad_h, pad_w)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).numpy()  # (pad_h, pad_w)

            # Store unpadded prediction
            pred_unpadded = probs[:H, :W].copy()
            predictions.append(pred_unpadded)

            # Inject prediction into the working stack for the next iteration.
            # The prediction is for time (t_end + 1). Replace the confidence
            # channel at that timestep with the predicted probabilities.
            next_t = t_end + 1
            if next_t < T:
                # Normalize the prediction the same way confidence was normalized
                means, stds = _load_channel_stats()
                conf_mean = float(means[CONF_CH])
                conf_std = float(stds[CONF_CH])

                # The raw prediction is a probability [0,1] — normalize it
                # using the same stats as the confidence channel
                padded_pred = np.zeros((fire["pad_h"], fire["pad_w"]), dtype=np.float32)
                padded_pred[:H, :W] = pred_unpadded
                working_stack[next_t, CONF_CH] = (padded_pred - conf_mean) / conf_std

    _prediction_cache[cache_key] = predictions
    return predictions


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup() -> None:
    # Eagerly load model + stats so first request is faster
    try:
        _load_model()
        _load_channel_stats()
    except Exception as exc:
        print(f"Warning: could not pre-load model: {exc}")

    # Pre-discover fires (but don't pre-load all data — that could be large)
    fires = _discover_fires()
    print(f"Pipeline data dir: {PIPELINE_DATA_DIR}")
    print(f"Found {len(fires)} fires: {fires}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/fires")
def list_fires() -> list[dict[str, Any]]:
    """Return list of available fires with metadata."""
    fires = _discover_fires()
    results: list[dict[str, Any]] = []
    for name in fires:
        try:
            fire = _load_fire(name)
        except Exception as exc:
            print(f"Warning: failed to load {name}: {exc}")
            continue
        results.append({
            "name": name,
            "grid_shape": [fire["H"], fire["W"]],
            "n_hours": fire["T"],
            "smoothed_fire_pixels": fire["smoothed_fire_pixels"],
        })
    return results


@app.get("/api/fires/{name}/meta")
def fire_meta(name: str) -> dict[str, Any]:
    """Return fire metadata."""
    fire = _get_fire_or_404(name)
    return {
        "name": name,
        "n_hours": fire["T"],
        "grid_shape": [fire["H"], fire["W"]],
        "smoothed_fire_pixels": fire["smoothed_fire_pixels"],
    }


@app.get("/api/fires/{name}/frame/{t}")
def fire_frame(name: str, t: int) -> Response:
    """Return ground truth (smoothed labels) at time t as binary float32."""
    fire = _get_fire_or_404(name)
    labels = fire["labels"]  # (T, H, W)
    T, H, W = fire["T"], fire["H"], fire["W"]

    if t < 0 or t >= T:
        raise HTTPException(400, f"Time index {t} out of range [0, {T})")

    frame = labels[t].astype(np.float32)

    return Response(
        content=frame.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Rows": str(H),
            "X-Cols": str(W),
            "X-Time-Index": str(t),
        },
    )


@app.get("/api/fires/{name}/progression")
def fire_progression(
    name: str,
    start: int = Query(..., description="Start timestep"),
    steps: int = Query(6, description="Number of prediction steps"),
) -> dict[str, Any]:
    """Return metadata for a progression comparison window.

    actual:    fire pixel counts for times start .. start+steps
    predicted: fire pixel counts for times start+1 .. start+steps
    """
    fire = _get_fire_or_404(name)
    T, H, W = fire["T"], fire["H"], fire["W"]
    labels = fire["labels"]

    if start < SEQ_LEN - 1:
        raise HTTPException(400, f"start must be >= {SEQ_LEN - 1}")
    if start + steps >= T:
        raise HTTPException(400, f"start+steps ({start + steps}) >= fire length ({T})")

    # Actual fire pixel counts from smoothed labels
    actual = []
    for t in range(start, start + steps + 1):
        actual.append({"t": t, "fire_pixels": int(labels[t].sum())})

    # Run recursive prediction
    predictions = _run_recursive_predictions(name, start, steps)

    predicted = []
    for step_idx, pred in enumerate(predictions):
        t = start + 1 + step_idx
        fire_pixels = int((pred >= 0.5).sum())
        predicted.append({"t": t, "fire_pixels": fire_pixels})

    return {
        "actual": actual,
        "predicted": predicted,
        "grid_shape": [H, W],
        "start": start,
        "steps": steps,
    }


@app.get("/api/fires/{name}/progression/actual/{t}")
def progression_actual_frame(name: str, t: int) -> Response:
    """Return actual (smoothed) fire frame at time t as binary float32."""
    fire = _get_fire_or_404(name)
    labels = fire["labels"]
    T, H, W = fire["T"], fire["H"], fire["W"]

    if t < 0 or t >= T:
        raise HTTPException(400, f"Time index {t} out of range [0, {T})")

    frame = labels[t].astype(np.float32)

    return Response(
        content=frame.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Rows": str(H),
            "X-Cols": str(W),
            "X-Time-Index": str(t),
        },
    )


@app.get("/api/fires/{name}/progression/predicted")
def progression_predicted_frame(
    name: str,
    start: int = Query(..., description="Start timestep for recursive prediction"),
    step: int = Query(..., description="0-indexed prediction step offset from start"),
) -> Response:
    """Run FireSpreadNet v2 recursively and return the predicted frame.

    step=0 → model run on [start-5..start], returns prediction for start+1
    step=N → recursive prediction for start+N+1
    """
    fire = _get_fire_or_404(name)
    H, W = fire["H"], fire["W"]

    needed_steps = step + 1
    predictions = _run_recursive_predictions(name, start, needed_steps)

    frame = predictions[step].astype(np.float32)

    return Response(
        content=frame.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Rows": str(H),
            "X-Cols": str(W),
            "X-Time-Index": str(start + step + 1),
        },
    )

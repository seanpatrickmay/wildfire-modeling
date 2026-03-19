"""FastAPI backend serving ground truth frames and pre-computed predictions."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.neighbor_cell_logreg import (
    discover_fire_entries,
    load_goes_times,
    parse_iso,
)

app = FastAPI(title="Wildfire Prediction Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

MODEL_NAMES = ["logreg", "mlp", "xgboost", "rnn", "convgru"]

# In-memory cache for GOES data
_goes_cache: dict[str, dict[str, Any]] = {}
_fire_entries: list[dict[str, Any]] = []


def _load_goes_data(fire_name: str) -> dict[str, Any]:
    if fire_name in _goes_cache:
        return _goes_cache[fire_name]

    entry = next((e for e in _fire_entries if e["fire_name"] == fire_name), None)
    if entry is None:
        raise HTTPException(404, f"Fire not found: {fire_name}")

    with open(entry["goes_json"], "r") as f:
        goes_json = json.load(f)

    goes_conf = np.array(goes_json["data"], dtype=np.float32)
    meta = goes_json["metadata"]
    time_steps = load_goes_times(meta, goes_conf)

    result = {
        "goes_conf": goes_conf,
        "metadata": meta,
        "time_steps": time_steps,
    }
    _goes_cache[fire_name] = result
    return result


@lru_cache(maxsize=64)
def _load_prediction(fire_name: str, model_name: str) -> dict[str, np.ndarray] | None:
    pred_path = REPO_ROOT / "data" / "predictions" / fire_name / f"{model_name}.npz"
    if not pred_path.exists():
        return None
    data = np.load(pred_path)
    return {
        "probs": np.array(data["probs"], dtype=np.float32),
        "time_indices": np.array(data["time_indices"], dtype=np.int32),
    }


@app.on_event("startup")
def startup() -> None:
    global _fire_entries
    all_entries = discover_fire_entries(REPO_ROOT)
    _fire_entries = [e for e in all_entries if e["fire_name"] in TEST_FIRES]
    # Pre-load GOES data
    for entry in _fire_entries:
        try:
            _load_goes_data(entry["fire_name"])
        except Exception as exc:
            print(f"Warning: failed to load {entry['fire_name']}: {exc}")


@app.get("/api/fires")
def list_fires() -> list[dict[str, Any]]:
    results = []
    for entry in _fire_entries:
        fire_name = entry["fire_name"]
        goes = _load_goes_data(fire_name)
        meta = goes["metadata"]

        available_models = []
        for model_name in MODEL_NAMES:
            pred_path = REPO_ROOT / "data" / "predictions" / fire_name / f"{model_name}.npz"
            if pred_path.exists():
                available_models.append(model_name)

        results.append({
            "name": fire_name,
            "grid_shape": meta["grid_shape"],
            "timestep_count": len(goes["time_steps"]),
            "available_models": available_models,
        })
    return results


@app.get("/api/fires/{name}/meta")
def fire_meta(name: str) -> dict[str, Any]:
    goes = _load_goes_data(name)
    meta = goes["metadata"]

    available_models = []
    for model_name in MODEL_NAMES:
        pred = _load_prediction(name, model_name)
        if pred is not None:
            available_models.append({
                "name": model_name,
                "frame_count": len(pred["time_indices"]),
                "time_indices": pred["time_indices"].tolist(),
            })

    return {
        "name": name,
        "grid_shape": meta["grid_shape"],
        "crs": meta.get("crs"),
        "geo_transform": meta.get("geo_transform"),
        "time_steps": goes["time_steps"],
        "available_models": available_models,
    }


@app.get("/api/fires/{name}/frame/{t}")
def fire_frame(name: str, t: int) -> Response:
    goes = _load_goes_data(name)
    conf = goes["goes_conf"]
    if t < 0 or t >= conf.shape[0]:
        raise HTTPException(400, f"Time index {t} out of range [0, {conf.shape[0]})")

    frame = conf[t].astype(np.float32)
    rows, cols = frame.shape

    return Response(
        content=frame.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Rows": str(rows),
            "X-Cols": str(cols),
            "X-Time-Index": str(t),
            "X-Time-Step": goes["time_steps"][t] if t < len(goes["time_steps"]) else "",
        },
    )


@app.get("/api/fires/{name}/prediction/{model}/{t}")
def prediction_frame(name: str, model: str, t: int) -> Response:
    pred = _load_prediction(name, model)
    if pred is None:
        raise HTTPException(404, f"No predictions for {name}/{model}")

    indices = pred["time_indices"]
    match_idx = np.where(indices == t)[0]
    if len(match_idx) == 0:
        raise HTTPException(404, f"No prediction at time index {t} for {name}/{model}")

    frame = pred["probs"][match_idx[0]].astype(np.float32)
    rows, cols = frame.shape

    goes = _load_goes_data(name)

    return Response(
        content=frame.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Rows": str(rows),
            "X-Cols": str(cols),
            "X-Time-Index": str(t),
            "X-Model": model,
            "X-Time-Step": goes["time_steps"][t] if t < len(goes["time_steps"]) else "",
        },
    )

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from sklearn.linear_model import SGDClassifier


CELL_OFFSETS: list[tuple[str, int, int]] = [
    ("c", 0, 0),
    ("nw", -1, -1),
    ("n", -1, 0),
    ("ne", -1, 1),
    ("w", 0, -1),
    ("e", 0, 1),
    ("sw", 1, -1),
    ("s", 1, 0),
    ("se", 1, 1),
]

BASE_VAR_ORDER: list[str] = [
    "fire_confidence",
    "temperature",
    "wind_speed",
    "specific_humidity",
    "precipitation_1h",
    "wind_direction_sin",
    "wind_direction_cos",
]

DISCOUNTED_RAIN_FEATURE_NAME = "discounted_rain_30d"
RTMA_VARS_REQUIRED = ["TMP", "WIND", "WDIR", "SPFH", "ACPC01"]


@dataclass(frozen=True)
class FeatureSchema:
    feature_names: list[str]
    var_order: list[str]
    n_features: int
    include_discounted_rain: bool


@dataclass(frozen=True)
class ZScoreNormalizer:
    enabled: bool
    mean: np.ndarray
    std: np.ndarray
    std_safe: np.ndarray
    samples_used: int
    zero_std_feature_count: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        X64 = X.astype(np.float64, copy=False)
        if not self.enabled:
            return X64
        return (X64 - self.mean) / self.std_safe

    @property
    def method(self) -> str:
        return "zscore_from_train_fires" if self.enabled else "none"


@dataclass(frozen=True)
class TrainingArtifacts:
    model: SGDClassifier
    intercept: float
    coef_map: dict[str, float]


def parse_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def normalize_time_str(value: str) -> str:
    dt = parse_iso(value)
    return dt.strftime("%Y-%m-%dT%H:00:00Z")


def affine_from_list(vals: list[float]) -> rasterio.Affine:
    return rasterio.Affine(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


def find_repo_root(start: Path) -> Path:
    for path in [start] + list(start.parents):
        if (path / "data").exists() and (path / "scripts").exists() and (path / "docs").exists():
            return path
    raise FileNotFoundError("Could not find repo root containing data/, scripts/, docs/.")


def load_goes_times(goes_meta: dict[str, Any], goes_conf: np.ndarray) -> list[str]:
    goes_time_steps = goes_meta.get("time_steps", [])
    goes_start = goes_meta.get("start_time")

    if goes_time_steps and isinstance(goes_time_steps[0], (int, float)):
        if not goes_start:
            raise ValueError("GOES time_steps are numeric and metadata.start_time is missing.")
        start_dt = parse_iso(goes_start)
        goes_time_steps = [
            (start_dt + timedelta(hours=int(item) - 1)).strftime("%Y-%m-%dT%H:00:00Z")
            for item in goes_time_steps
        ]
    elif not goes_time_steps and goes_start:
        start_dt = parse_iso(goes_start)
        goes_time_steps = [
            (start_dt + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00:00Z")
            for i in range(goes_conf.shape[0])
        ]
    else:
        goes_time_steps = [normalize_time_str(str(item)) for item in goes_time_steps]

    if not goes_time_steps:
        raise ValueError("GOES metadata has no usable time_steps.")

    return goes_time_steps


def discover_fire_entries(repo_root: Path) -> list[dict[str, Any]]:
    base = repo_root / "data" / "multi_fire"
    if not base.exists():
        raise FileNotFoundError(f"Missing multi-fire directory: {base}")

    entries: list[dict[str, Any]] = []
    for fire_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        goes_candidates = sorted(fire_dir.glob("*GOES*json"))
        manifest_path = fire_dir / "rtma" / "rtma_manifest.json"
        if not goes_candidates or not manifest_path.exists():
            continue
        entries.append(
            {
                "fire_name": fire_dir.name,
                "goes_json": goes_candidates[0],
                "rtma_manifest": manifest_path,
            }
        )
    return entries


def select_fire_entries(entries: list[dict[str, Any]], fire_selection: str | list[str] | None) -> list[dict[str, Any]]:
    if fire_selection is None or fire_selection == "all":
        return entries

    if not isinstance(fire_selection, (list, tuple, set)):
        raise ValueError('FIRE_SELECTION must be "all" or a list/tuple/set of fire names.')

    wanted = {str(item) for item in fire_selection}
    selected = [entry for entry in entries if entry["fire_name"] in wanted]
    found = {entry["fire_name"] for entry in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown fire names in FIRE_SELECTION: {missing}")
    return selected


def split_fire_entries(
    entries: list[dict[str, Any]],
    train_fires: str | list[str] | None,
    test_fires: str | list[str] | None,
    train_fraction: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(entries) < 2:
        raise ValueError("Need at least 2 selected fires for fire-level train/test split.")

    names = [entry["fire_name"] for entry in entries]
    name_set = set(names)

    def normalize_group(value: str | list[str] | None) -> str | list[str]:
        if value is None or value == "auto":
            return "auto"
        if not isinstance(value, (list, tuple, set)):
            raise ValueError("TRAIN_FIRES/TEST_FIRES must be 'auto' or list/tuple/set of fire names.")
        normalized = [str(item) for item in value]
        unknown = sorted(set(normalized) - name_set)
        if unknown:
            raise ValueError(f"Unknown fire names in train/test split: {unknown}")
        return normalized

    train_group = normalize_group(train_fires)
    test_group = normalize_group(test_fires)

    if train_group == "auto" and test_group == "auto":
        if not (0.0 < train_fraction < 1.0):
            raise ValueError("FIRE_TRAIN_FRACTION must be between 0 and 1.")
        rng = np.random.default_rng(split_seed)
        perm_names = list(np.array(names)[rng.permutation(len(names))])
        n_train = int(round(train_fraction * len(perm_names)))
        n_train = max(1, min(len(perm_names) - 1, n_train))
        train_names = set(perm_names[:n_train])
        test_names = set(perm_names[n_train:])
    elif train_group == "auto":
        test_names = set(test_group)
        train_names = set(names) - test_names
    elif test_group == "auto":
        train_names = set(train_group)
        test_names = set(names) - train_names
    else:
        train_names = set(train_group)
        test_names = set(test_group)

    overlap = sorted(train_names & test_names)
    if overlap:
        raise ValueError(f"Train/test fire sets overlap: {overlap}")
    if not train_names:
        raise ValueError("Train fire set is empty.")
    if not test_names:
        raise ValueError("Test fire set is empty.")

    train_entries = [entry for entry in entries if entry["fire_name"] in train_names]
    test_entries = [entry for entry in entries if entry["fire_name"] in test_names]
    return train_entries, test_entries


def split_validation_fire_entries(
    train_fire_entries: list[dict[str, Any]],
    validation_fires: str | list[str],
    validation_fraction: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_names = [entry["fire_name"] for entry in train_fire_entries]
    if len(train_names) < 2:
        raise RuntimeError("Need at least 2 train fires to make a validation split.")

    if validation_fires == "auto":
        rng = np.random.default_rng(split_seed)
        perm = list(np.array(train_names)[rng.permutation(len(train_names))])
        n_val = max(1, int(round(len(perm) * validation_fraction)))
        n_val = min(n_val, len(perm) - 1)
        val_name_set = set(perm[:n_val])
    else:
        if not isinstance(validation_fires, (list, tuple, set)):
            raise ValueError('VALIDATION_FIRES must be "auto" or list/tuple/set of fire names.')
        val_name_set = {str(item) for item in validation_fires}

    inner_train_entries = [entry for entry in train_fire_entries if entry["fire_name"] not in val_name_set]
    val_entries = [entry for entry in train_fire_entries if entry["fire_name"] in val_name_set]

    if not inner_train_entries or not val_entries:
        raise RuntimeError("Validation split produced empty inner-train or validation fire set.")

    return inner_train_entries, val_entries


def build_feature_schema(include_discounted_rain: bool = True) -> FeatureSchema:
    var_order = list(BASE_VAR_ORDER)
    if include_discounted_rain:
        var_order.insert(5, DISCOUNTED_RAIN_FEATURE_NAME)

    feature_names: list[str] = []
    for offset_name, _, _ in CELL_OFFSETS:
        for variable_name in var_order:
            feature_names.append(f"{variable_name}_{offset_name}")

    return FeatureSchema(
        feature_names=feature_names,
        var_order=var_order,
        n_features=len(feature_names),
        include_discounted_rain=include_discounted_rain,
    )


def to_binary_target(y_continuous: np.ndarray, threshold: float) -> np.ndarray:
    return (y_continuous >= threshold).astype(np.int32)


def resolve_manifest_file_path(path_str: str, repo_root: Path, manifest_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.exists():
        return path

    parts = path.parts
    if "data" in parts:
        idx = parts.index("data")
        candidate = repo_root.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate

    candidate = (manifest_dir / path_str).resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not resolve RTMA part path: {path_str}")


def resample_stack(
    src_stack: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs: Any,
    dst_shape: tuple[int, int],
    dst_transform: rasterio.Affine,
    dst_crs: Any,
) -> np.ndarray:
    bands = src_stack.shape[0]
    dst = np.empty((bands, dst_shape[0], dst_shape[1]), dtype=np.float32)
    for band_idx in range(bands):
        reproject(
            source=src_stack[band_idx],
            destination=dst[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return dst


def iter_resampled_rtma_hours(
    repo_root: Path,
    rtma_manifest: dict[str, Any],
    rtma_manifest_path: Path,
    goes_shape: tuple[int, int],
    goes_transform: rasterio.Affine,
    goes_crs: Any,
) -> tuple[str, dict[str, np.ndarray]]:
    rtma_vars = rtma_manifest["variables"]
    for required_var in RTMA_VARS_REQUIRED:
        if required_var not in rtma_vars:
            raise KeyError(f"RTMA manifest missing required variable: {required_var}")

    manifest_dir = rtma_manifest_path.parent
    rtma_files = rtma_manifest["files"]
    resolved_files = {
        variable: [resolve_manifest_file_path(path, repo_root, manifest_dir) for path in rtma_files[variable]]
        for variable in rtma_vars
    }

    n_parts = len(resolved_files[rtma_vars[0]])
    for variable in rtma_vars:
        if len(resolved_files[variable]) != n_parts:
            raise ValueError("RTMA variable file lists do not have equal part counts.")

    parts = list(zip(*[resolved_files[variable] for variable in rtma_vars]))
    rtma_time_steps = [normalize_time_str(str(item)) for item in rtma_manifest["time_steps"]]
    rtma_time_ptr = 0

    for part_paths in parts:
        rtma_arrays: dict[str, np.ndarray] = {}
        rtma_transform = None
        rtma_crs = None
        band_count = None

        for variable, part_path in zip(rtma_vars, part_paths):
            with rasterio.open(part_path) as dataset:
                if rtma_transform is None:
                    rtma_transform = dataset.transform
                    rtma_crs = dataset.crs
                    band_count = dataset.count
                rtma_arrays[variable] = dataset.read().astype(np.float32)

        if band_count is None or rtma_transform is None:
            continue

        resampled = {
            variable: resample_stack(
                rtma_arrays[variable],
                rtma_transform,
                rtma_crs,
                goes_shape,
                goes_transform,
                goes_crs,
            )
            for variable in rtma_vars
        }

        for local_idx in range(band_count):
            global_idx = rtma_time_ptr + local_idx
            if global_idx >= len(rtma_time_steps):
                break

            time_str = rtma_time_steps[global_idx]
            hour_payload = {variable: resampled[variable][local_idx] for variable in RTMA_VARS_REQUIRED}
            yield time_str, hour_payload

        rtma_time_ptr += band_count


def build_discounted_rain_state(
    antecedent_state: np.ndarray,
    precip_history: deque[np.ndarray],
    current_precip: np.ndarray,
    *,
    decay_per_hour: float,
    expiry_factor: float,
    lookback_hours: int,
) -> np.ndarray:
    next_state = (decay_per_hour * antecedent_state) + (decay_per_hour * current_precip)
    if len(precip_history) >= lookback_hours:
        expired_precip = precip_history.popleft()
        next_state = next_state - (expiry_factor * expired_precip)
    precip_history.append(current_precip)
    return next_state


def build_hour_samples(
    feature_schema: FeatureSchema,
    conf_t: np.ndarray,
    conf_t1: np.ndarray,
    rtma_hour: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = conf_t.shape
    if height < 3 or width < 3:
        return np.empty((0, feature_schema.n_features), dtype=np.float64), np.empty((0,), dtype=np.float64)

    discounted_rain = rtma_hour.get(DISCOUNTED_RAIN_FEATURE_NAME)
    if feature_schema.include_discounted_rain and discounted_rain is None:
        raise KeyError(f"Missing {DISCOUNTED_RAIN_FEATURE_NAME} in RTMA hour payload.")

    y = conf_t1[1:-1, 1:-1].astype(np.float64)
    feature_blocks: list[np.ndarray] = []

    for _, dy, dx in CELL_OFFSETS:
        ys = slice(1 + dy, height - 1 + dy)
        xs = slice(1 + dx, width - 1 + dx)

        go_cell = conf_t[ys, xs].astype(np.float64)
        tmp_cell = rtma_hour["TMP"][ys, xs].astype(np.float64)
        wind_cell = rtma_hour["WIND"][ys, xs].astype(np.float64)
        spfh_cell = rtma_hour["SPFH"][ys, xs].astype(np.float64)
        precip_cell = rtma_hour["ACPC01"][ys, xs].astype(np.float64)
        wdir_deg_cell = rtma_hour["WDIR"][ys, xs].astype(np.float64)
        wdir_rad_cell = np.deg2rad(wdir_deg_cell)
        wdir_sin_cell = np.sin(wdir_rad_cell)
        wdir_cos_cell = np.cos(wdir_rad_cell)

        feature_blocks.extend(
            [
                go_cell,
                tmp_cell,
                wind_cell,
                spfh_cell,
                precip_cell,
            ]
        )
        if feature_schema.include_discounted_rain:
            feature_blocks.append(discounted_rain[ys, xs].astype(np.float64))
        feature_blocks.extend([wdir_sin_cell, wdir_cos_cell])

    X = np.stack(feature_blocks, axis=-1).reshape(-1, feature_schema.n_features)
    y = y.reshape(-1)

    valid = np.isfinite(y)
    valid &= np.isfinite(X).all(axis=1)

    if not valid.any():
        return np.empty((0, feature_schema.n_features), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return X[valid], y[valid]


def iter_aligned_hours_for_fire(
    repo_root: Path,
    goes_conf: np.ndarray,
    goes_time_index: dict[str, int],
    rtma_manifest: dict[str, Any],
    rtma_manifest_path: Path,
    goes_shape: tuple[int, int],
    goes_transform: rasterio.Affine,
    goes_crs: Any,
    *,
    include_discounted_rain: bool,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> tuple[int, dict[str, np.ndarray]]:
    antecedent_state = None
    precip_history: deque[np.ndarray] = deque()
    decay_per_hour = None
    expiry_factor = None

    if include_discounted_rain:
        if discounted_rain_lookback_hours <= 0:
            raise ValueError("discounted_rain_lookback_hours must be > 0 when discounted rain is enabled.")
        if discounted_rain_half_life_days <= 0:
            raise ValueError("discounted_rain_half_life_days must be > 0 when discounted rain is enabled.")
        half_life_hours = discounted_rain_half_life_days * 24.0
        decay_per_hour = float(0.5 ** (1.0 / half_life_hours))
        expiry_factor = float(decay_per_hour ** (discounted_rain_lookback_hours + 1))

    for time_str, rtma_hour in iter_resampled_rtma_hours(
        repo_root,
        rtma_manifest,
        rtma_manifest_path,
        goes_shape,
        goes_transform,
        goes_crs,
    ):
        if include_discounted_rain:
            current_precip = np.nan_to_num(rtma_hour["ACPC01"], nan=0.0).astype(np.float32, copy=False)
            if antecedent_state is None:
                antecedent_state = np.zeros_like(current_precip, dtype=np.float32)
            rtma_hour = dict(rtma_hour)
            rtma_hour[DISCOUNTED_RAIN_FEATURE_NAME] = antecedent_state.copy()

        if time_str in goes_time_index:
            t = goes_time_index[time_str]
            if t + 1 < goes_conf.shape[0]:
                yield t, rtma_hour

        if include_discounted_rain:
            antecedent_state = build_discounted_rain_state(
                antecedent_state,
                precip_history,
                current_precip,
                decay_per_hour=decay_per_hour,
                expiry_factor=expiry_factor,
                lookback_hours=discounted_rain_lookback_hours,
            )


def load_fire_entry_context(entry: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    with Path(entry["goes_json"]).open("r", encoding="utf-8") as goes_file:
        goes_json = json.load(goes_file)
    with Path(entry["rtma_manifest"]).open("r", encoding="utf-8") as rtma_file:
        rtma_manifest = json.load(rtma_file)

    goes_conf = np.array(goes_json["data"], dtype=np.float32)
    goes_meta = goes_json["metadata"]
    goes_transform = affine_from_list(goes_meta["geo_transform"])
    goes_crs = goes_meta.get("crs")
    goes_shape = tuple(goes_meta["grid_shape"])
    goes_times = load_goes_times(goes_meta, goes_conf)
    goes_time_index = {time_str: idx for idx, time_str in enumerate(goes_times)}

    return {
        "repo_root": repo_root,
        "fire_name": entry["fire_name"],
        "goes_conf": goes_conf,
        "goes_meta": goes_meta,
        "goes_transform": goes_transform,
        "goes_crs": goes_crs,
        "goes_shape": goes_shape,
        "goes_time_index": goes_time_index,
        "rtma_manifest": rtma_manifest,
        "rtma_manifest_path": Path(entry["rtma_manifest"]),
    }


def iter_fire_hour_samples(
    entry: dict[str, Any],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    *,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> tuple[str, int, np.ndarray, np.ndarray]:
    context = load_fire_entry_context(entry, repo_root)

    for t, rtma_hour in iter_aligned_hours_for_fire(
        repo_root,
        context["goes_conf"],
        context["goes_time_index"],
        context["rtma_manifest"],
        context["rtma_manifest_path"],
        context["goes_shape"],
        context["goes_transform"],
        context["goes_crs"],
        include_discounted_rain=feature_schema.include_discounted_rain,
        discounted_rain_lookback_hours=discounted_rain_lookback_hours,
        discounted_rain_half_life_days=discounted_rain_half_life_days,
    ):
        X_hour, y_hour_cont = build_hour_samples(feature_schema, context["goes_conf"][t], context["goes_conf"][t + 1], rtma_hour)
        if X_hour.shape[0] == 0:
            continue
        y_hour = to_binary_target(y_hour_cont, positive_threshold)
        yield context["fire_name"], t, X_hour, y_hour


def collect_entry_stats(
    entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    *,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> dict[str, int]:
    stats = {
        "samples": 0,
        "hours": 0,
        "positives": 0,
        "negatives": 0,
    }

    for entry in entries:
        for _, _, X_hour, y_hour in iter_fire_hour_samples(
            entry,
            repo_root,
            feature_schema,
            positive_threshold,
            discounted_rain_lookback_hours=discounted_rain_lookback_hours,
            discounted_rain_half_life_days=discounted_rain_half_life_days,
        ):
            stats["hours"] += 1
            stats["samples"] += int(X_hour.shape[0])
            positives = int(y_hour.sum())
            stats["positives"] += positives
            stats["negatives"] += int(y_hour.shape[0] - positives)

    return stats


def collect_dataset_stats(
    train_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    *,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> dict[str, Any]:
    train_stats = collect_entry_stats(
        train_entries,
        repo_root,
        feature_schema,
        positive_threshold,
        discounted_rain_lookback_hours=discounted_rain_lookback_hours,
        discounted_rain_half_life_days=discounted_rain_half_life_days,
    )
    test_stats = collect_entry_stats(
        test_entries,
        repo_root,
        feature_schema,
        positive_threshold,
        discounted_rain_lookback_hours=discounted_rain_lookback_hours,
        discounted_rain_half_life_days=discounted_rain_half_life_days,
    )

    total_samples = train_stats["samples"] + test_stats["samples"]
    result = {
        "train": train_stats,
        "test": test_stats,
        "total_samples": total_samples,
        "hours_used": train_stats["hours"] + test_stats["hours"],
        "train_positive_rate": (train_stats["positives"] / train_stats["samples"]) if train_stats["samples"] else None,
        "test_positive_rate": (test_stats["positives"] / test_stats["samples"]) if test_stats["samples"] else None,
    }
    return result


def make_identity_normalizer(feature_schema: FeatureSchema) -> ZScoreNormalizer:
    zeros = np.zeros(feature_schema.n_features, dtype=np.float64)
    ones = np.ones(feature_schema.n_features, dtype=np.float64)
    return ZScoreNormalizer(
        enabled=False,
        mean=zeros,
        std=ones,
        std_safe=ones,
        samples_used=0,
        zero_std_feature_count=0,
    )


def fit_zscore_normalizer(
    entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    *,
    enabled: bool,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> ZScoreNormalizer:
    if not enabled:
        return make_identity_normalizer(feature_schema)

    feature_sum = np.zeros(feature_schema.n_features, dtype=np.float64)
    feature_sq_sum = np.zeros(feature_schema.n_features, dtype=np.float64)
    n_samples = 0

    for entry in entries:
        for _, _, X_hour, _ in iter_fire_hour_samples(
            entry,
            repo_root,
            feature_schema,
            positive_threshold,
            discounted_rain_lookback_hours=discounted_rain_lookback_hours,
            discounted_rain_half_life_days=discounted_rain_half_life_days,
        ):
            X_block = X_hour.astype(np.float64, copy=False)
            feature_sum += X_block.sum(axis=0)
            feature_sq_sum += np.square(X_block).sum(axis=0)
            n_samples += int(X_block.shape[0])

    if n_samples == 0:
        raise RuntimeError("No training samples available for normalization stats.")

    feature_mean = feature_sum / n_samples
    feature_var = np.maximum((feature_sq_sum / n_samples) - np.square(feature_mean), 0.0)
    feature_std = np.sqrt(feature_var)
    feature_std_safe = np.where(feature_std > 0.0, feature_std, 1.0)
    zero_std_feature_count = int((feature_std == 0.0).sum())

    return ZScoreNormalizer(
        enabled=True,
        mean=feature_mean,
        std=feature_std,
        std_safe=feature_std_safe,
        samples_used=n_samples,
        zero_std_feature_count=zero_std_feature_count,
    )


def train_logistic_regression(
    entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    normalizer: ZScoreNormalizer,
    *,
    alpha: float = 1e-4,
    random_state: int | None = None,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> TrainingArtifacts:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=1,
        tol=None,
        random_state=random_state,
    )
    classes = np.array([0, 1], dtype=np.int32)
    trained = False

    for entry in entries:
        for _, _, X_hour, y_hour in iter_fire_hour_samples(
            entry,
            repo_root,
            feature_schema,
            positive_threshold,
            discounted_rain_lookback_hours=discounted_rain_lookback_hours,
            discounted_rain_half_life_days=discounted_rain_half_life_days,
        ):
            X_train = normalizer.transform(X_hour)
            y_train = y_hour.astype(np.int32, copy=False)
            if not trained:
                model.partial_fit(X_train, y_train, classes=classes)
                trained = True
            else:
                model.partial_fit(X_train, y_train)

    if not trained:
        raise RuntimeError("Model did not receive training data from the supplied fire entries.")

    intercept = float(model.intercept_[0])
    coef_map = {name: float(value) for name, value in zip(feature_schema.feature_names, model.coef_.ravel())}
    return TrainingArtifacts(model=model, intercept=intercept, coef_map=coef_map)


def evaluate_fixed_threshold(
    model: SGDClassifier,
    entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    normalizer: ZScoreNormalizer,
    probability_threshold: float,
    *,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    n_eval = 0
    correct = 0

    for entry in entries:
        for _, _, X_hour, y_hour in iter_fire_hour_samples(
            entry,
            repo_root,
            feature_schema,
            positive_threshold,
            discounted_rain_lookback_hours=discounted_rain_lookback_hours,
            discounted_rain_half_life_days=discounted_rain_half_life_days,
        ):
            X_eval = normalizer.transform(X_hour)
            y_eval = y_hour.astype(np.int32, copy=False)
            prob = model.predict_proba(X_eval)[:, 1]
            y_hat = (prob >= probability_threshold).astype(np.int32)

            correct += int((y_hat == y_eval).sum())
            n_eval += int(y_eval.shape[0])
            tp += int(((y_hat == 1) & (y_eval == 1)).sum())
            fp += int(((y_hat == 1) & (y_eval == 0)).sum())
            fn += int(((y_hat == 0) & (y_eval == 1)).sum())
            tn += int(((y_hat == 0) & (y_eval == 0)).sum())

    if n_eval == 0:
        raise RuntimeError("No valid evaluation samples in the supplied fire entries.")

    return {
        "count": n_eval,
        "accuracy_overall": float(correct / n_eval),
        "positive_accuracy": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        "negative_accuracy": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, *, default: float) -> np.ndarray:
    result = np.full_like(numerator, fill_value=default, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def compute_pr_curve(
    model: SGDClassifier,
    entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    normalizer: ZScoreNormalizer,
    thresholds: np.ndarray,
    *,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> dict[str, Any]:
    tp = np.zeros(thresholds.shape[0], dtype=np.int64)
    fp = np.zeros(thresholds.shape[0], dtype=np.int64)
    total_pos = 0
    total_neg = 0

    for entry in entries:
        for _, _, X_hour, y_hour in iter_fire_hour_samples(
            entry,
            repo_root,
            feature_schema,
            positive_threshold,
            discounted_rain_lookback_hours=discounted_rain_lookback_hours,
            discounted_rain_half_life_days=discounted_rain_half_life_days,
        ):
            X_eval = normalizer.transform(X_hour)
            y_eval = y_hour.astype(np.int32, copy=False)

            prob = model.predict_proba(X_eval)[:, 1]
            pos = y_eval == 1
            total_pos += int(pos.sum())
            total_neg += int((~pos).sum())

            pred = prob[:, None] >= thresholds[None, :]
            tp += (pred & pos[:, None]).sum(axis=0).astype(np.int64)
            fp += (pred & (~pos)[:, None]).sum(axis=0).astype(np.int64)

    if total_pos == 0:
        raise RuntimeError("No positive samples in the supplied fire entries; cannot compute precision/recall.")

    fn = total_pos - tp
    tn = total_neg - fp
    precision = safe_divide(tp, tp + fp, default=1.0)
    recall = safe_divide(tp, np.full_like(tp, total_pos), default=0.0)
    f1 = safe_divide(2.0 * precision * recall, precision + recall, default=0.0)

    df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    )
    best = df.iloc[int(df["f1"].idxmax())]
    baseline = total_pos / (total_pos + total_neg) if (total_pos + total_neg) > 0 else None

    return {
        "df": df,
        "best": best,
        "baseline": baseline,
        "total_pos": int(total_pos),
        "total_neg": int(total_neg),
    }


def threshold_transfer_to_entries(
    model: SGDClassifier,
    entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    normalizer: ZScoreNormalizer,
    probability_threshold: float,
    *,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> dict[str, Any]:
    metrics = evaluate_fixed_threshold(
        model,
        entries,
        repo_root,
        feature_schema,
        positive_threshold,
        normalizer,
        probability_threshold,
        discounted_rain_lookback_hours=discounted_rain_lookback_hours,
        discounted_rain_half_life_days=discounted_rain_half_life_days,
    )

    precision = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 1.0
    recall = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "train_selected_threshold": float(probability_threshold),
        "test_accuracy": metrics["accuracy_overall"],
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_tp": metrics["tp"],
        "test_fp": metrics["fp"],
        "test_fn": metrics["fn"],
        "test_tn": metrics["tn"],
    }


def coefficients_to_dataframe(coef_map: dict[str, float], *, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef_rows = [
        {
            "feature": feature,
            "coef": coef,
            "odds_ratio": float(np.exp(coef)),
            "abs_coef": abs(coef),
        }
        for feature, coef in coef_map.items()
    ]
    coef_df = pd.DataFrame(coef_rows).sort_values("abs_coef", ascending=False)
    coef_top = coef_df.head(top_n).drop(columns=["abs_coef"])
    return coef_df, coef_top


def format_feature_normalization_label(normalizer: ZScoreNormalizer) -> str:
    return normalizer.method


def build_summary_df(
    *,
    fire_entries: list[dict[str, Any]],
    train_fire_entries: list[dict[str, Any]],
    test_fire_entries: list[dict[str, Any]],
    positive_threshold: float,
    classification_prob_threshold: float,
    dataset_stats: dict[str, Any] | None,
    metrics_test: dict[str, Any] | None,
    intercept: float | None,
    normalizer: ZScoreNormalizer,
) -> pd.DataFrame:
    train_stats = dataset_stats["train"] if dataset_stats is not None else {"samples": 0, "hours": 0}
    test_stats = dataset_stats["test"] if dataset_stats is not None else {"samples": 0, "hours": 0}

    row = {
        "model": "logistic_regression",
        "target": "center_confidence_t+1_binary",
        "fires_used_count": len(fire_entries),
        "fires_used": [entry["fire_name"] for entry in fire_entries],
        "train_fires_count": len(train_fire_entries),
        "test_fires_count": len(test_fire_entries),
        "train_fires": [entry["fire_name"] for entry in train_fire_entries],
        "test_fires": [entry["fire_name"] for entry in test_fire_entries],
        "positive_threshold": positive_threshold,
        "total_samples": int(dataset_stats["total_samples"]) if dataset_stats is not None else 0,
        "train_samples": int(train_stats["samples"]),
        "test_samples": int(test_stats["samples"]),
        "hours_used": int(dataset_stats["hours_used"]) if dataset_stats is not None else 0,
        "train_hours": int(train_stats["hours"]),
        "test_hours": int(test_stats["hours"]),
        "train_positive_rate": dataset_stats["train_positive_rate"] if dataset_stats is not None else None,
        "test_positive_rate": dataset_stats["test_positive_rate"] if dataset_stats is not None else None,
        "test_accuracy_overall": metrics_test["accuracy_overall"] if metrics_test is not None else None,
        "test_positive_accuracy": metrics_test["positive_accuracy"] if metrics_test is not None else None,
        "test_negative_accuracy": metrics_test["negative_accuracy"] if metrics_test is not None else None,
        "tp": int(metrics_test["tp"]) if metrics_test is not None else 0,
        "fp": int(metrics_test["fp"]) if metrics_test is not None else 0,
        "fn": int(metrics_test["fn"]) if metrics_test is not None else 0,
        "tn": int(metrics_test["tn"]) if metrics_test is not None else 0,
        "classification_prob_threshold": classification_prob_threshold,
        "feature_normalization": format_feature_normalization_label(normalizer),
        "intercept": intercept,
    }
    return pd.DataFrame([row])


def build_confusion_breakdown_df(metrics_test: dict[str, Any] | None) -> pd.DataFrame:
    row = {
        "true_positives": int(metrics_test["tp"]) if metrics_test is not None else 0,
        "false_positives": int(metrics_test["fp"]) if metrics_test is not None else 0,
        "false_negatives": int(metrics_test["fn"]) if metrics_test is not None else 0,
        "true_negatives": int(metrics_test["tn"]) if metrics_test is not None else 0,
    }
    return pd.DataFrame([row])


def build_report(
    *,
    fire_entries: list[dict[str, Any]],
    train_fire_entries: list[dict[str, Any]],
    test_fire_entries: list[dict[str, Any]],
    feature_schema: FeatureSchema,
    positive_threshold: float,
    classification_prob_threshold: float,
    fire_train_fraction: float,
    fire_split_seed: int,
    normalizer: ZScoreNormalizer,
    dataset_stats: dict[str, Any] | None,
    metrics_test: dict[str, Any] | None,
    intercept: float | None,
    coef_map: dict[str, float],
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> dict[str, Any]:
    train_stats = dataset_stats["train"] if dataset_stats is not None else {"samples": 0, "hours": 0, "positives": 0, "negatives": 0}
    test_stats = dataset_stats["test"] if dataset_stats is not None else {"samples": 0, "hours": 0, "positives": 0, "negatives": 0}

    report = {
        "model": "logistic_regression",
        "target": "center_confidence_t_plus_1_binary",
        "fires_used": [entry["fire_name"] for entry in fire_entries],
        "train_fires": [entry["fire_name"] for entry in train_fire_entries],
        "test_fires": [entry["fire_name"] for entry in test_fire_entries],
        "thresholds": {
            "positive_confidence": positive_threshold,
            "classification_probability": classification_prob_threshold,
        },
        "split": {
            "method": "fire_holdout",
            "train_fire_count": len(train_fire_entries),
            "test_fire_count": len(test_fire_entries),
            "train_fire_fraction_target": fire_train_fraction,
            "split_seed": fire_split_seed,
        },
        "feature_order": feature_schema.feature_names,
        "feature_engineering": {
            "include_discounted_rain_30d": feature_schema.include_discounted_rain,
            "discounted_rain_lookback_hours": int(discounted_rain_lookback_hours) if feature_schema.include_discounted_rain else 0,
            "discounted_rain_half_life_days": float(discounted_rain_half_life_days) if feature_schema.include_discounted_rain else None,
        },
        "feature_normalization": {
            "enabled": normalizer.enabled,
            "method": normalizer.method,
            "samples_used": int(normalizer.samples_used),
            "zero_std_feature_count": int(normalizer.zero_std_feature_count),
        },
        "metrics_test": {
            "test_accuracy_overall": metrics_test["accuracy_overall"] if metrics_test is not None else None,
            "test_positive_accuracy": metrics_test["positive_accuracy"] if metrics_test is not None else None,
            "test_negative_accuracy": metrics_test["negative_accuracy"] if metrics_test is not None else None,
            "tp": int(metrics_test["tp"]) if metrics_test is not None else 0,
            "fp": int(metrics_test["fp"]) if metrics_test is not None else 0,
            "fn": int(metrics_test["fn"]) if metrics_test is not None else 0,
            "tn": int(metrics_test["tn"]) if metrics_test is not None else 0,
        },
        "class_balance": {
            "train_positive_rate": dataset_stats["train_positive_rate"] if dataset_stats is not None else None,
            "test_positive_rate": dataset_stats["test_positive_rate"] if dataset_stats is not None else None,
            "train_positives": int(train_stats["positives"]),
            "train_negatives": int(train_stats["negatives"]),
            "test_positives": int(test_stats["positives"]),
            "test_negatives": int(test_stats["negatives"]),
        },
        "coefficients": {
            "intercept": intercept,
            "values": coef_map,
        },
        "data": {
            "total_samples": int(dataset_stats["total_samples"]) if dataset_stats is not None else 0,
            "train_samples": int(train_stats["samples"]),
            "test_samples": int(test_stats["samples"]),
            "hours_used": int(dataset_stats["hours_used"]) if dataset_stats is not None else 0,
            "train_hours": int(train_stats["hours"]),
            "test_hours": int(test_stats["hours"]),
        },
    }
    return report

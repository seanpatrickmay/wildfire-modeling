from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.neighbor_cell_logreg import (
    FeatureSchema,
    ZScoreNormalizer,
    format_feature_normalization_label,
    iter_fire_hour_samples,
    safe_divide,
)


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    use_batchnorm: bool = True
    epochs: int = 3
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1337


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    avg_loss: float
    accuracy: float
    positive_accuracy: float | None
    negative_accuracy: float | None
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class NNTrainingArtifacts:
    model: nn.Module
    config: MLPConfig
    device: str
    pos_weight: float
    history: list[EpochMetrics] = field(default_factory=list)
    parameter_count: int = 0


class FeedForwardMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], dropout: float, use_batchnorm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        requested = prefer.lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_model_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def build_mlp(feature_schema: FeatureSchema, config: MLPConfig) -> FeedForwardMLP:
    return FeedForwardMLP(
        in_dim=feature_schema.n_features,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        use_batchnorm=config.use_batchnorm,
    )


def history_to_records(history: Sequence[EpochMetrics]) -> list[dict[str, Any]]:
    return [asdict(item) for item in history]


def history_to_dataframe(history: Sequence[EpochMetrics]) -> pd.DataFrame:
    return pd.DataFrame(history_to_records(history))


def _iter_batches(n_rows: int, batch_size: int):
    for start in range(0, n_rows, batch_size):
        end = min(n_rows, start + batch_size)
        yield start, end


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, pd.Series):
        return _to_builtin(value.to_dict())
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _compute_pos_weight(
    entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    *,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
) -> tuple[float, dict[str, int]]:
    positives = 0
    negatives = 0
    samples = 0
    hours = 0

    for entry in entries:
        for _, _, _, y_hour in iter_fire_hour_samples(
            entry,
            repo_root,
            feature_schema,
            positive_threshold,
            discounted_rain_lookback_hours=discounted_rain_lookback_hours,
            discounted_rain_half_life_days=discounted_rain_half_life_days,
        ):
            hours += 1
            positives += int(y_hour.sum())
            negatives += int(y_hour.shape[0] - y_hour.sum())
            samples += int(y_hour.shape[0])

    if samples == 0:
        raise RuntimeError("No training samples available for neural-network training.")

    pos_weight = float(negatives / max(positives, 1))
    return pos_weight, {
        "samples": samples,
        "hours": hours,
        "positives": positives,
        "negatives": negatives,
    }


def train_feedforward_network(
    entries: list[dict[str, Any]],
    repo_root: Path,
    feature_schema: FeatureSchema,
    positive_threshold: float,
    normalizer: ZScoreNormalizer,
    classification_prob_threshold: float,
    config: MLPConfig,
    *,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
    device: torch.device | None = None,
) -> NNTrainingArtifacts:
    set_seed(config.seed)
    train_device = device or pick_device()
    pos_weight_value, train_stats = _compute_pos_weight(
        entries,
        repo_root,
        feature_schema,
        positive_threshold,
        discounted_rain_lookback_hours=discounted_rain_lookback_hours,
        discounted_rain_half_life_days=discounted_rain_half_life_days,
    )

    model = build_mlp(feature_schema, config).to(train_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=train_device)
    )

    history: list[EpochMetrics] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        tp = fp = fn = tn = 0

        for entry in entries:
            for _, _, X_hour, y_hour in iter_fire_hour_samples(
                entry,
                repo_root,
                feature_schema,
                positive_threshold,
                discounted_rain_lookback_hours=discounted_rain_lookback_hours,
                discounted_rain_half_life_days=discounted_rain_half_life_days,
            ):
                X_train = normalizer.transform(X_hour).astype(np.float32, copy=False)
                y_train = y_hour.astype(np.float32, copy=False)

                for start, end in _iter_batches(X_train.shape[0], config.batch_size):
                    xb = torch.from_numpy(X_train[start:end]).to(device=train_device, dtype=torch.float32)
                    yb = torch.from_numpy(y_train[start:end]).to(device=train_device, dtype=torch.float32).unsqueeze(1)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.detach().cpu())
                    epoch_batches += 1

                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        pred = (probs >= classification_prob_threshold).to(torch.int32)
                        truth = yb.to(torch.int32)
                        tp += int(((pred == 1) & (truth == 1)).sum().cpu())
                        fp += int(((pred == 1) & (truth == 0)).sum().cpu())
                        fn += int(((pred == 0) & (truth == 1)).sum().cpu())
                        tn += int(((pred == 0) & (truth == 0)).sum().cpu())

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        positive_accuracy = (tp / (tp + fn)) if (tp + fn) > 0 else None
        negative_accuracy = (tn / (tn + fp)) if (tn + fp) > 0 else None
        history.append(
            EpochMetrics(
                epoch=epoch,
                avg_loss=(epoch_loss / max(epoch_batches, 1)),
                accuracy=float(accuracy),
                positive_accuracy=float(positive_accuracy) if positive_accuracy is not None else None,
                negative_accuracy=float(negative_accuracy) if negative_accuracy is not None else None,
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
                tn=int(tn),
            )
        )

    if not history:
        raise RuntimeError("Training completed without producing any epoch history.")

    return NNTrainingArtifacts(
        model=model,
        config=config,
        device=str(train_device),
        pos_weight=pos_weight_value,
        history=history,
        parameter_count=count_model_parameters(model),
    )


def predict_probabilities(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    probs_batches: list[np.ndarray] = []
    device_obj = torch.device(device)

    model.eval()
    with torch.no_grad():
        for start, end in _iter_batches(X.shape[0], batch_size):
            xb = torch.from_numpy(X[start:end].astype(np.float32, copy=False)).to(device=device_obj, dtype=torch.float32)
            logits = model(xb)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            probs_batches.append(probs)

    if not probs_batches:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(probs_batches, axis=0)


def evaluate_fixed_threshold(
    training_artifacts: NNTrainingArtifacts,
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
            probs = predict_probabilities(
                training_artifacts.model,
                X_eval,
                batch_size=training_artifacts.config.batch_size,
                device=training_artifacts.device,
            )
            y_hat = (probs >= probability_threshold).astype(np.int32)

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


def compute_pr_curve(
    training_artifacts: NNTrainingArtifacts,
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
            probs = predict_probabilities(
                training_artifacts.model,
                X_eval,
                batch_size=training_artifacts.config.batch_size,
                device=training_artifacts.device,
            )
            pos = y_eval == 1
            total_pos += int(pos.sum())
            total_neg += int((~pos).sum())

            pred = probs[:, None] >= thresholds[None, :]
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
    training_artifacts: NNTrainingArtifacts,
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
        training_artifacts,
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


def build_nn_summary_df(
    *,
    fire_entries: list[dict[str, Any]],
    train_fire_entries: list[dict[str, Any]],
    test_fire_entries: list[dict[str, Any]],
    positive_threshold: float,
    classification_prob_threshold: float,
    dataset_stats: dict[str, Any] | None,
    metrics_test: dict[str, Any] | None,
    normalizer: ZScoreNormalizer,
    training_artifacts: NNTrainingArtifacts | None,
) -> pd.DataFrame:
    train_stats = dataset_stats["train"] if dataset_stats is not None else {"samples": 0, "hours": 0}
    test_stats = dataset_stats["test"] if dataset_stats is not None else {"samples": 0, "hours": 0}
    config = training_artifacts.config if training_artifacts is not None else None

    row = {
        "model": "mlp_pytorch",
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
        "device": training_artifacts.device if training_artifacts is not None else None,
        "pos_weight": training_artifacts.pos_weight if training_artifacts is not None else None,
        "epochs": config.epochs if config is not None else None,
        "batch_size": config.batch_size if config is not None else None,
        "learning_rate": config.learning_rate if config is not None else None,
        "weight_decay": config.weight_decay if config is not None else None,
        "hidden_dims": list(config.hidden_dims) if config is not None else None,
        "dropout": config.dropout if config is not None else None,
        "use_batchnorm": config.use_batchnorm if config is not None else None,
        "parameter_count": training_artifacts.parameter_count if training_artifacts is not None else None,
    }
    return pd.DataFrame([row])


def build_nn_report(
    *,
    fire_entries: list[dict[str, Any]],
    train_fire_entries: list[dict[str, Any]],
    test_fire_entries: list[dict[str, Any]],
    val_fire_entries: list[dict[str, Any]],
    feature_schema: FeatureSchema,
    positive_threshold: float,
    classification_prob_threshold: float,
    fire_train_fraction: float,
    fire_split_seed: int,
    normalizer: ZScoreNormalizer,
    dataset_stats: dict[str, Any] | None,
    metrics_test: dict[str, Any] | None,
    training_artifacts: NNTrainingArtifacts | None,
    discounted_rain_lookback_hours: int,
    discounted_rain_half_life_days: float,
    validation_threshold_df: pd.DataFrame | None,
    validation_threshold_best: dict[str, Any] | None,
    validation_threshold_value: float | None,
    train_threshold_test_metrics: dict[str, Any] | None,
    test_pr_df: pd.DataFrame | None,
    test_pr_best: dict[str, Any] | None,
    test_pr_baseline: float | None,
) -> dict[str, Any]:
    train_stats = dataset_stats["train"] if dataset_stats is not None else {"samples": 0, "hours": 0, "positives": 0, "negatives": 0}
    test_stats = dataset_stats["test"] if dataset_stats is not None else {"samples": 0, "hours": 0, "positives": 0, "negatives": 0}
    config = training_artifacts.config if training_artifacts is not None else None
    history_records = history_to_records(training_artifacts.history) if training_artifacts is not None else []

    report = {
        "model": "mlp_pytorch",
        "target": "center_confidence_t_plus_1_binary",
        "fires_used": [entry["fire_name"] for entry in fire_entries],
        "train_fires": [entry["fire_name"] for entry in train_fire_entries],
        "test_fires": [entry["fire_name"] for entry in test_fire_entries],
        "validation_fires": [entry["fire_name"] for entry in val_fire_entries],
        "thresholds": {
            "positive_confidence": positive_threshold,
            "classification_probability": classification_prob_threshold,
            "validation_selected_probability": validation_threshold_value,
        },
        "split": {
            "method": "fire_holdout",
            "train_fire_count": len(train_fire_entries),
            "test_fire_count": len(test_fire_entries),
            "validation_fire_count": len(val_fire_entries),
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
        "training": {
            "device": training_artifacts.device if training_artifacts is not None else None,
            "pos_weight": training_artifacts.pos_weight if training_artifacts is not None else None,
            "parameter_count": training_artifacts.parameter_count if training_artifacts is not None else None,
            "config": {
                "hidden_dims": list(config.hidden_dims) if config is not None else None,
                "dropout": config.dropout if config is not None else None,
                "use_batchnorm": config.use_batchnorm if config is not None else None,
                "epochs": config.epochs if config is not None else None,
                "batch_size": config.batch_size if config is not None else None,
                "learning_rate": config.learning_rate if config is not None else None,
                "weight_decay": config.weight_decay if config is not None else None,
                "seed": config.seed if config is not None else None,
            },
            "history": history_records,
        },
        "validation_threshold_selection": {
            "best_threshold_by_f1": validation_threshold_value,
            "best_row": validation_threshold_best,
            "top_by_f1": (
                validation_threshold_df.sort_values("f1", ascending=False).head(12).to_dict(orient="records")
                if validation_threshold_df is not None
                else []
            ),
        },
        "train_threshold_transfer": train_threshold_test_metrics,
        "test_pr_curve": {
            "baseline_positive_rate": test_pr_baseline,
            "best_row": test_pr_best,
            "top_by_f1": (
                test_pr_df.sort_values("f1", ascending=False).head(12).to_dict(orient="records")
                if test_pr_df is not None
                else []
            ),
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
    return _to_builtin(report)


__all__ = [
    "EpochMetrics",
    "MLPConfig",
    "NNTrainingArtifacts",
    "build_mlp",
    "build_nn_report",
    "build_nn_summary_df",
    "compute_pr_curve",
    "count_model_parameters",
    "evaluate_fixed_threshold",
    "history_to_dataframe",
    "history_to_records",
    "pick_device",
    "predict_probabilities",
    "set_seed",
    "threshold_transfer_to_entries",
    "train_feedforward_network",
]

"""Nearest-centroid baseline over precomputed feature bundles."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qif_attribution.eval.metrics import (
    accuracy,
    balanced_accuracy,
    confusion_matrix,
    macro_f1,
)
from qif_attribution.models import build_classifier

SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class SplitEvaluation:
    split: str
    count: int
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    confusion_matrix: dict[str, dict[str, int]]


@dataclass(frozen=True)
class BaselineReport:
    model: str
    params: dict
    label_field: str
    feature_dim: int
    train_count: int
    labels: list[str]
    evaluations: list[SplitEvaluation]


def load_feature_index(path: Path) -> list[dict[str, str]]:
    """Read the row-aligned index.csv emitted next to features.npy."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_split_map(path: Path) -> dict[str, str]:
    """Read a split assignments CSV into an image_id -> split mapping."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "image_id" not in fields or "split" not in fields:
            raise ValueError("splits CSV must have image_id and split columns")
        return {row["image_id"]: row["split"] for row in reader}


def run_baseline(
    features: np.ndarray,
    index: list[dict[str, str]],
    split_map: dict[str, str],
    *,
    label_field: str = "generator_id",
    model: str = "nearest-centroid",
    metric: str = "cosine",
    k: int = 5,
    max_iter: int = 500,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
) -> BaselineReport:
    """Fit the chosen classifier on the train split and score each split."""

    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if features.shape[0] != len(index):
        raise ValueError("features and index rows must align")

    missing = [row["image_id"] for row in index if row["image_id"] not in split_map]
    if missing:
        raise KeyError(
            f"{len(missing)} feature rows have no split assignment "
            f"(first missing: {missing[0]})"
        )

    by_split: dict[str, list[int]] = {}
    for position, row in enumerate(index):
        by_split.setdefault(split_map[row["image_id"]], []).append(position)

    train_positions = by_split.get("train", [])
    if not train_positions:
        raise ValueError("no training rows found in split assignments")

    train_labels = [index[pos][label_field] for pos in train_positions]
    classifier, params = build_classifier(
        model, metric=metric, k=k, max_iter=max_iter, learning_rate=learning_rate, l2=l2
    )
    classifier.fit(features[train_positions], train_labels)

    ordered_splits = [split for split in SPLIT_ORDER if split in by_split]
    ordered_splits += [split for split in sorted(by_split) if split not in SPLIT_ORDER]

    evaluations: list[SplitEvaluation] = []
    for split in ordered_splits:
        positions = by_split[split]
        y_true = [index[pos][label_field] for pos in positions]
        y_pred = classifier.predict(features[positions])
        evaluations.append(
            SplitEvaluation(
                split=split,
                count=len(positions),
                accuracy=accuracy(y_true, y_pred),
                macro_f1=macro_f1(y_true, y_pred),
                balanced_accuracy=balanced_accuracy(y_true, y_pred),
                confusion_matrix=confusion_matrix(y_true, y_pred),
            )
        )

    return BaselineReport(
        model=model,
        params=params,
        label_field=label_field,
        feature_dim=int(features.shape[1]),
        train_count=len(train_positions),
        labels=sorted(set(train_labels)),
        evaluations=evaluations,
    )


def run_baseline_from_paths(
    *,
    feature_dir: Path,
    splits_path: Path,
    label_field: str = "generator_id",
    model: str = "nearest-centroid",
    metric: str = "cosine",
    k: int = 5,
    max_iter: int = 500,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
) -> BaselineReport:
    """Load a feature bundle plus split CSV from disk and run the baseline."""

    features = np.load(feature_dir / "features.npy")
    index = load_feature_index(feature_dir / "index.csv")
    split_map = load_split_map(splits_path)
    return run_baseline(
        features,
        index,
        split_map,
        label_field=label_field,
        model=model,
        metric=metric,
        k=k,
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2=l2,
    )


def report_to_dict(report: BaselineReport) -> dict:
    return {
        "model": report.model,
        "params": report.params,
        "label_field": report.label_field,
        "feature_dim": report.feature_dim,
        "train_count": report.train_count,
        "labels": report.labels,
        "splits": {
            ev.split: {
                "count": ev.count,
                "accuracy": ev.accuracy,
                "macro_f1": ev.macro_f1,
                "balanced_accuracy": ev.balanced_accuracy,
                "confusion_matrix": ev.confusion_matrix,
            }
            for ev in report.evaluations
        },
    }


def write_baseline_report(path: Path, report: BaselineReport) -> dict:
    payload = report_to_dict(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def format_baseline_report(report: BaselineReport) -> str:
    param_str = ", ".join(f"{key}={value}" for key, value in report.params.items())
    descriptor = f"{report.model} {param_str}".strip()
    lines = [
        f"baseline ({descriptor}, label={report.label_field}, "
        f"dim={report.feature_dim}, train={report.train_count})",
        f"labels: {', '.join(report.labels)}",
    ]
    for ev in report.evaluations:
        lines.append(
            f"{ev.split}: n={ev.count} acc={ev.accuracy:.3f} "
            f"macro_f1={ev.macro_f1:.3f} bal_acc={ev.balanced_accuracy:.3f}"
        )
    return "\n".join(lines)

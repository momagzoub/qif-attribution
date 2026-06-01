"""Small dependency-light classification metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check_lengths(y_true, y_pred)
    if not y_true:
        return 0.0
    return sum(true == pred for true, pred in zip(y_true, y_pred, strict=True)) / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check_lengths(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    return sum(_f1_for_label(y_true, y_pred, label) for label in labels) / len(labels)


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check_lengths(y_true, y_pred)
    labels = sorted(set(y_true))
    if not labels:
        return 0.0
    recalls = []
    for label in labels:
        positives = sum(true == label for true in y_true)
        true_positives = sum(
            true == label and pred == label for true, pred in zip(y_true, y_pred, strict=True)
        )
        recalls.append(true_positives / positives if positives else 0.0)
    return sum(recalls) / len(recalls)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, int]]:
    _check_lengths(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {true: {pred: 0 for pred in labels} for true in labels}
    for true, pred in zip(y_true, y_pred, strict=True):
        matrix[true][pred] += 1
    return matrix


def _f1_for_label(y_true: Sequence[str], y_pred: Sequence[str], label: str) -> float:
    counts = Counter(
        "tp" if true == label and pred == label else
        "fp" if true != label and pred == label else
        "fn" if true == label and pred != label else
        "tn"
        for true, pred in zip(y_true, y_pred, strict=True)
    )
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _check_lengths(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

"""Model training and attribution methods."""

from __future__ import annotations

from typing import Any

from qif_attribution.models.contrastive_pairs import (
    ContrastivePair,
    build_attribution_pairs,
    relation_counts,
)
from qif_attribution.models.knn import KNearestNeighborsClassifier
from qif_attribution.models.logreg import LogisticRegressionClassifier
from qif_attribution.models.nearest_centroid import NearestCentroidClassifier
from qif_attribution.models.quantum_discriminator import QuantumStateDiscriminator
from qif_attribution.models.scaling import StandardScaler, Whitener

MODELS = ("nearest-centroid", "knn", "logreg")


def build_classifier(
    model: str,
    *,
    metric: str = "cosine",
    k: int = 5,
    max_iter: int = 500,
    learning_rate: float = 0.5,
    l2: float = 1e-3,
) -> tuple[Any, dict]:
    """Construct a classifier and the parameter dict recorded in the report."""

    if model == "nearest-centroid":
        return NearestCentroidClassifier(metric=metric), {"metric": metric}
    if model == "knn":
        return KNearestNeighborsClassifier(k=k, metric=metric), {"k": k, "metric": metric}
    if model == "logreg":
        return (
            LogisticRegressionClassifier(
                max_iter=max_iter, learning_rate=learning_rate, l2=l2
            ),
            {"max_iter": max_iter, "learning_rate": learning_rate, "l2": l2},
        )
    raise ValueError(f"unknown model: {model!r} (choose from {', '.join(MODELS)})")


__all__ = [
    "MODELS",
    "ContrastivePair",
    "KNearestNeighborsClassifier",
    "LogisticRegressionClassifier",
    "NearestCentroidClassifier",
    "QuantumStateDiscriminator",
    "StandardScaler",
    "Whitener",
    "build_attribution_pairs",
    "build_classifier",
    "relation_counts",
]

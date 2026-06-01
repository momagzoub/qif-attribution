"""A k-nearest-neighbors baseline over precomputed embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from qif_attribution.models.scaling import StandardScaler

DistanceMetric = Literal["cosine", "euclidean"]


@dataclass
class KNearestNeighborsClassifier:
    """Majority-vote k-NN with optional z-score standardization.

    Euclidean distance is computed on standardized features by default so no
    single high-variance descriptor dominates; cosine uses row-normalized
    vectors. Ties in the vote are broken toward the nearer neighbor.
    """

    k: int = 5
    metric: DistanceMetric = "euclidean"
    standardize: bool = True

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> KNearestNeighborsClassifier:
        array = _as_2d(embeddings)
        if array.shape[0] != len(labels):
            raise ValueError("number of embeddings must match number of labels")
        if not labels:
            raise ValueError("labels cannot be empty")
        if self.k < 1:
            raise ValueError("k must be >= 1")

        self.classes_ = sorted(set(labels))
        self._labels = list(labels)
        self._scaler = StandardScaler().fit(array) if self.standardize else None
        self._train = self._prepare(array)
        return self

    def predict(self, embeddings: np.ndarray) -> list[str]:
        self._check_is_fit()
        query = self._prepare(_as_2d(embeddings))
        neighbors = min(self.k, self._train.shape[0])
        distances = self._distances(query)
        predictions: list[str] = []
        for row in distances:
            order = np.argsort(row, kind="stable")[:neighbors]
            votes: dict[str, list[int]] = {}
            for rank, index in enumerate(order):
                label = self._labels[int(index)]
                if label not in votes:
                    votes[label] = [0, rank]
                votes[label][0] += 1
            winner = min(votes.items(), key=lambda item: (-item[1][0], item[1][1]))[0]
            predictions.append(winner)
        return predictions

    def _prepare(self, array: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            array = self._scaler.transform(array)
        if self.metric == "cosine":
            return _row_normalize(array)
        if self.metric == "euclidean":
            return array
        raise ValueError(f"unsupported metric: {self.metric}")

    def _distances(self, query: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return 1.0 - query @ self._train.T
        gram = query @ self._train.T
        query_sq = np.sum(query**2, axis=1, keepdims=True)
        train_sq = np.sum(self._train**2, axis=1, keepdims=True).T
        return np.sqrt(np.maximum(query_sq + train_sq - 2.0 * gram, 0.0))

    def _check_is_fit(self) -> None:
        if not hasattr(self, "_train") or not hasattr(self, "classes_"):
            raise RuntimeError("classifier must be fit before predict")


def _as_2d(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    return array


def _row_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return array / norms

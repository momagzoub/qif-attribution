"""A small baseline classifier for precomputed image embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DistanceMetric = Literal["cosine", "euclidean"]


@dataclass
class NearestCentroidClassifier:
    metric: DistanceMetric = "cosine"

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> NearestCentroidClassifier:
        array = _as_2d(embeddings)
        if array.shape[0] != len(labels):
            raise ValueError("number of embeddings must match number of labels")
        if not labels:
            raise ValueError("labels cannot be empty")

        self.labels_ = sorted(set(labels))
        centroids = []
        for label in self.labels_:
            members = array[[idx for idx, item in enumerate(labels) if item == label]]
            centroids.append(np.mean(members, axis=0))
        self.centroids_ = np.vstack(centroids)
        if self.metric == "cosine":
            self.centroids_ = _row_normalize(self.centroids_)
        return self

    def predict(self, embeddings: np.ndarray) -> list[str]:
        self._check_is_fit()
        array = _as_2d(embeddings)
        if self.metric == "cosine":
            scores = _row_normalize(array) @ self.centroids_.T
            indices = np.argmax(scores, axis=1)
        elif self.metric == "euclidean":
            distances = np.linalg.norm(array[:, None, :] - self.centroids_[None, :, :], axis=2)
            indices = np.argmin(distances, axis=1)
        else:
            raise ValueError(f"unsupported metric: {self.metric}")
        return [self.labels_[int(index)] for index in indices]

    def _check_is_fit(self) -> None:
        if not hasattr(self, "centroids_") or not hasattr(self, "labels_"):
            raise RuntimeError("classifier must be fit before predict")


def _as_2d(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    return array


def _row_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embedding cannot be normalized for cosine distance")
    return array / norms

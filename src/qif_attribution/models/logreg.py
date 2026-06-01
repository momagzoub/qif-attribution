"""Multinomial logistic regression trained with deterministic gradient descent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qif_attribution.models.scaling import StandardScaler


@dataclass
class LogisticRegressionClassifier:
    """Softmax regression with z-score standardization and L2 regularization.

    Training is full-batch gradient descent from a zero initialization, so fits
    are deterministic and reproducible. A bias column is appended after
    standardization and is left unregularized.
    """

    max_iter: int = 500
    learning_rate: float = 0.5
    l2: float = 1e-3
    standardize: bool = True

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> LogisticRegressionClassifier:
        array = _as_2d(embeddings)
        if array.shape[0] != len(labels):
            raise ValueError("number of embeddings must match number of labels")
        if not labels:
            raise ValueError("labels cannot be empty")

        self.classes_ = sorted(set(labels))
        class_index = {label: idx for idx, label in enumerate(self.classes_)}
        targets = _one_hot([class_index[label] for label in labels], len(self.classes_))

        self._scaler = StandardScaler().fit(array) if self.standardize else None
        design = self._design(array)
        n_samples = design.shape[0]
        weights = np.zeros((design.shape[1], len(self.classes_)), dtype=np.float64)
        # Do not regularize the trailing bias row.
        reg_mask = np.ones((design.shape[1], 1), dtype=np.float64)
        reg_mask[-1, 0] = 0.0

        for _ in range(self.max_iter):
            probabilities = _softmax(design @ weights)
            gradient = design.T @ (probabilities - targets) / n_samples
            gradient += self.l2 * weights * reg_mask
            weights -= self.learning_rate * gradient

        self.weights_ = weights
        return self

    def predict(self, embeddings: np.ndarray) -> list[str]:
        if not hasattr(self, "weights_") or not hasattr(self, "classes_"):
            raise RuntimeError("classifier must be fit before predict")
        logits = self._design(_as_2d(embeddings)) @ self.weights_
        return [self.classes_[int(index)] for index in np.argmax(logits, axis=1)]

    def _design(self, array: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            array = self._scaler.transform(array)
        bias = np.ones((array.shape[0], 1), dtype=np.float64)
        return np.hstack([array, bias])


def _as_2d(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    return array


def _one_hot(indices: list[int], num_classes: int) -> np.ndarray:
    targets = np.zeros((len(indices), num_classes), dtype=np.float64)
    targets[np.arange(len(indices)), indices] = 1.0
    return targets


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

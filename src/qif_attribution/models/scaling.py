"""Feature standardization shared by the distance- and gradient-based models."""

from __future__ import annotations

import numpy as np


class StandardScaler:
    """Z-score standardization with a zero-variance guard.

    Fit statistics are computed on the training matrix only; constant columns
    keep unit scale so they map to zeros rather than dividing by zero.
    """

    def fit(self, features: np.ndarray) -> StandardScaler:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("features must be a 2D array")
        self.mean_ = array.mean(axis=0)
        std = array.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float64)
        return (array - self.mean_) / self.std_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


class Whitener:
    """PCA whitening via SVD, fit on training data only.

    Centers the data, rotates onto its principal axes, and rescales each axis
    to unit variance, so a Euclidean metric treats decorrelated directions
    equally instead of being dominated by a few correlated, high-variance
    feature blocks. A small ``eps`` floors tiny singular values to avoid
    amplifying noise directions. Fit statistics come from the training matrix
    only; ``transform`` applies the same rotation and scale to new rows.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def fit(self, features: np.ndarray) -> Whitener:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("features must be a 2D array")
        self.mean_ = array.mean(axis=0)
        centered = array - self.mean_
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        denom = float(max(array.shape[0] - 1, 1))
        std = singular / np.sqrt(denom)
        self.components_ = vt
        self.scale_ = 1.0 / np.maximum(std, self.eps)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float64)
        return ((array - self.mean_) @ self.components_.T) * self.scale_

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

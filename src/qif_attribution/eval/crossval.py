"""Group-aware K-fold cross-validation with confidence intervals.

Folds partition *groups* (e.g. ``prompt_id``) so that no group's images
straddle the train/test boundary, matching the leakage-safe single-split
policy. Fold scores are NOT independent (their training sets overlap), so the
reported interval treats them as independent and is therefore optimistic --
read it as a spread indicator, not a strict frequentist guarantee. The raw
per-fold scores are always returned so the variability is visible.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from qif_attribution.eval.metrics import accuracy, balanced_accuracy, macro_f1

# Two-sided 95% Student-t critical values by degrees of freedom (n_splits - 1).
# Covers n_splits 2..16; larger fold counts fall back to the normal quantile.
_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
}
_Z_975 = 1.96

_METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


class Estimator(Protocol):
    def fit(self, features: np.ndarray, labels: list[str]) -> Any: ...

    def predict(self, features: np.ndarray) -> list[str]: ...


def group_kfold(groups: Sequence[str], n_splits: int, *, seed: int = 1729) -> list[list[int]]:
    """Partition row indices into ``n_splits`` folds, keeping groups intact.

    Groups are ordered by a deterministic hash of ``f"{seed}:{group}"`` and
    dealt round-robin into folds, which keeps fold sizes close to balanced.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[str(group)].append(index)
    unique = sorted(members, key=lambda value: _stable_hash(f"{seed}:{value}"))
    if n_splits > len(unique):
        raise ValueError(f"n_splits={n_splits} exceeds the number of groups ({len(unique)})")
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for position, group in enumerate(unique):
        folds[position % n_splits].extend(members[group])
    return folds


@dataclass(frozen=True)
class CrossValResult:
    """Per-fold scores from a cross-validation run, plus summary helpers."""

    n_splits: int
    fold_accuracy: tuple[float, ...]
    fold_macro_f1: tuple[float, ...]
    fold_balanced_accuracy: tuple[float, ...]

    def summary(self, metric: str = "accuracy") -> dict[str, float]:
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}")
        values = np.asarray(getattr(self, f"fold_{metric}"), dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        se = std / float(np.sqrt(values.size)) if values.size else 0.0
        crit = _critical_value(values.size - 1)
        return {
            "mean": mean,
            "std": std,
            "se": se,
            "ci_low": mean - crit * se,
            "ci_high": mean + crit * se,
            "min": float(values.min()) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
        }


def cross_validate(
    features: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[str],
    estimator_factory: Callable[[], Estimator],
    *,
    n_splits: int = 5,
    seed: int = 1729,
) -> CrossValResult:
    """Run group-aware K-fold CV, fitting a fresh estimator on each fold.

    ``estimator_factory`` returns an unfitted object exposing ``fit(X, y)`` and
    ``predict(X)``; building it per fold keeps any learned state (scalers,
    whiteners, weights) confined to that fold's training rows.
    """

    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("features must be a 2D array")
    if not (array.shape[0] == len(labels) == len(groups)):
        raise ValueError("features, labels, and groups must align")

    labels = list(labels)
    folds = group_kfold(groups, n_splits, seed=seed)
    fold_accuracy: list[float] = []
    fold_macro_f1: list[float] = []
    fold_balanced_accuracy: list[float] = []
    for test_idx in folds:
        held = set(test_idx)
        train_idx = [i for i in range(len(labels)) if i not in held]
        estimator = estimator_factory()
        estimator.fit(array[train_idx], [labels[i] for i in train_idx])
        y_pred = estimator.predict(array[test_idx])
        y_true = [labels[i] for i in test_idx]
        fold_accuracy.append(accuracy(y_true, y_pred))
        fold_macro_f1.append(macro_f1(y_true, y_pred))
        fold_balanced_accuracy.append(balanced_accuracy(y_true, y_pred))
    return CrossValResult(
        n_splits=n_splits,
        fold_accuracy=tuple(fold_accuracy),
        fold_macro_f1=tuple(fold_macro_f1),
        fold_balanced_accuracy=tuple(fold_balanced_accuracy),
    )


def paired_difference_ci(
    values_a: Sequence[float], values_b: Sequence[float]
) -> dict[str, float | bool]:
    """95% t-CI for the per-fold paired difference ``a - b``.

    Fold scores from ``cross_validate`` runs that share groups, seed, and
    ``n_splits`` use *identical* folds, so they are paired by held-out rows and
    a paired test is valid. A CI that straddles 0 means the accuracy gap is not
    distinguishable from fold-to-fold noise -- the honest way to report two
    configs whose marginal CIs overlap. Inherits the same optimism caveat as
    ``CrossValResult.summary`` (overlapping training sets break independence).
    """

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired inputs must have the same shape")
    if a.size < 2:
        raise ValueError("need at least two folds for a paired interval")
    diff = a - b
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(diff.size))
    crit = _critical_value(diff.size - 1)
    low, high = mean - crit * se, mean + crit * se
    return {
        "mean_diff": mean,
        "ci_low": low,
        "ci_high": high,
        "significant": bool(low > 0.0 or high < 0.0),
    }


def _critical_value(dof: int) -> float:
    if dof <= 0:
        return 0.0
    return _T_975.get(dof, _Z_975)


def _stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

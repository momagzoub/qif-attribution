import numpy as np
import pytest

from qif_attribution.eval.crossval import (
    CrossValResult,
    cross_validate,
    group_kfold,
    paired_difference_ci,
)
from qif_attribution.models.knn import KNearestNeighborsClassifier


def test_group_kfold_partitions_all_rows_without_splitting_groups() -> None:
    groups = [f"g{i // 2}" for i in range(20)]  # 10 groups, 2 rows each
    folds = group_kfold(groups, n_splits=5, seed=1)

    flat = sorted(idx for fold in folds for idx in fold)
    assert flat == list(range(20))  # disjoint and covers every row

    fold_of = {idx: f for f, fold in enumerate(folds) for idx in fold}
    for group in set(groups):
        members = [i for i, value in enumerate(groups) if value == group]
        assert len({fold_of[i] for i in members}) == 1  # group stays in one fold


def test_group_kfold_is_deterministic_and_validates() -> None:
    groups = [f"g{i}" for i in range(6)]

    assert group_kfold(groups, 3, seed=7) == group_kfold(groups, 3, seed=7)
    with pytest.raises(ValueError, match="n_splits must be"):
        group_kfold(groups, 1)
    with pytest.raises(ValueError, match="exceeds the number of groups"):
        group_kfold(groups, 7)


def test_cross_validate_on_separable_clusters() -> None:
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0.0, 0.0], scale=0.05, size=(20, 2))
    cluster_b = rng.normal(loc=[5.0, 5.0], scale=0.05, size=(20, 2))
    features = np.vstack([cluster_a, cluster_b])
    labels = ["a"] * 20 + ["b"] * 20
    groups = [f"ga{i}" for i in range(20)] + [f"gb{i}" for i in range(20)]

    result = cross_validate(
        features,
        labels,
        groups,
        lambda: KNearestNeighborsClassifier(k=1, metric="euclidean"),
        n_splits=5,
        seed=1,
    )

    assert isinstance(result, CrossValResult)
    assert len(result.fold_accuracy) == 5
    summary = result.summary("accuracy")
    assert summary["mean"] == 1.0  # clusters are trivially separable
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]


def test_cross_validate_validates_alignment() -> None:
    with pytest.raises(ValueError, match="align"):
        cross_validate(
            np.zeros((3, 2)),
            ["a", "b"],
            ["g", "g", "g"],
            lambda: KNearestNeighborsClassifier(),
            n_splits=2,
        )


def test_paired_difference_ci_flags_overlapping_configs_as_insignificant() -> None:
    # A tiny, noisy edge (b barely ahead, signs flip across folds) must NOT be
    # called significant -- this is the multiple-comparison guard the grid needs.
    a = [0.80, 0.84, 0.79, 0.86, 0.82]
    b = [0.82, 0.81, 0.83, 0.80, 0.84]
    result = paired_difference_ci(a, b)

    assert result["ci_low"] < 0.0 < result["ci_high"]
    assert result["significant"] is False


def test_paired_difference_ci_flags_consistent_gap_as_significant() -> None:
    a = [0.90, 0.91, 0.89, 0.92, 0.90]
    b = [0.80, 0.81, 0.79, 0.82, 0.80]
    result = paired_difference_ci(a, b)

    assert np.isclose(result["mean_diff"], 0.10, atol=1e-9)
    assert result["ci_low"] > 0.0
    assert result["significant"] is True


def test_paired_difference_ci_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        paired_difference_ci([0.1, 0.2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="at least two folds"):
        paired_difference_ci([0.1], [0.2])

import numpy as np
import pytest

from qif_attribution.models import build_classifier
from qif_attribution.models.knn import KNearestNeighborsClassifier
from qif_attribution.models.logreg import LogisticRegressionClassifier
from qif_attribution.models.scaling import StandardScaler, Whitener

TRAIN = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
TRAIN_LABELS = ["sd15", "sd15", "openjourney_v4", "openjourney_v4"]
QUERY = np.array([[1.5, 0.1], [0.1, 1.5]])
QUERY_LABELS = ["sd15", "openjourney_v4"]


def test_standard_scaler_centers_and_scales() -> None:
    data = np.array([[1.0, 10.0, 5.0], [3.0, 30.0, 5.0]])

    scaler = StandardScaler().fit(data)
    transformed = scaler.transform(data)

    assert np.allclose(transformed.mean(axis=0), 0.0)
    # Column 2 is constant: guarded to map to zeros, not NaN/inf.
    assert np.all(np.isfinite(transformed))
    assert np.allclose(transformed[:, 2], 0.0)


def test_whitener_decorrelates_to_identity_covariance() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 2))
    # Induce correlation and unequal scales, then offset the mean.
    data = base @ np.array([[3.0, 1.5], [0.0, 0.5]]).T + np.array([10.0, -4.0])

    whitened = Whitener().fit_transform(data)

    assert np.allclose(whitened.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(np.cov(whitened, rowvar=False), np.eye(2), atol=1e-6)
    assert np.all(np.isfinite(whitened))


def test_whitener_uses_fit_stats_for_new_rows() -> None:
    rng = np.random.default_rng(1)
    whitener = Whitener().fit(rng.normal(size=(200, 3)))
    test = rng.normal(size=(5, 3))

    first = whitener.transform(test)

    assert first.shape == (5, 3)
    assert np.allclose(first, whitener.transform(test))  # deterministic, train-fit stats


def test_knn_separates_clusters_both_metrics() -> None:
    for metric in ("cosine", "euclidean"):
        model = KNearestNeighborsClassifier(k=1, metric=metric).fit(TRAIN, TRAIN_LABELS)
        assert model.predict(QUERY) == QUERY_LABELS


def test_knn_validates_inputs() -> None:
    with pytest.raises(ValueError, match="k must be"):
        KNearestNeighborsClassifier(k=0).fit(TRAIN, TRAIN_LABELS)
    with pytest.raises(ValueError, match="match number of labels"):
        KNearestNeighborsClassifier().fit(TRAIN, TRAIN_LABELS[:-1])
    with pytest.raises(RuntimeError, match="must be fit"):
        KNearestNeighborsClassifier().predict(QUERY)


def test_logreg_separates_clusters_and_is_deterministic() -> None:
    first = LogisticRegressionClassifier(max_iter=300).fit(TRAIN, TRAIN_LABELS)
    second = LogisticRegressionClassifier(max_iter=300).fit(TRAIN, TRAIN_LABELS)

    assert first.predict(QUERY) == QUERY_LABELS
    assert np.allclose(first.weights_, second.weights_)


def test_logreg_validates_inputs() -> None:
    with pytest.raises(ValueError, match="match number of labels"):
        LogisticRegressionClassifier().fit(TRAIN, TRAIN_LABELS[:-1])
    with pytest.raises(RuntimeError, match="must be fit"):
        LogisticRegressionClassifier().predict(QUERY)


def test_build_classifier_returns_estimator_and_params() -> None:
    centroid, centroid_params = build_classifier("nearest-centroid", metric="euclidean")
    knn, knn_params = build_classifier("knn", k=3)
    logreg, logreg_params = build_classifier("logreg", l2=0.01)

    assert centroid_params == {"metric": "euclidean"}
    assert knn_params == {"k": 3, "metric": "cosine"}
    assert logreg_params["l2"] == 0.01
    assert isinstance(knn, KNearestNeighborsClassifier)
    assert isinstance(logreg, LogisticRegressionClassifier)
    assert hasattr(centroid, "fit")

    with pytest.raises(ValueError, match="unknown model"):
        build_classifier("forest")

import numpy as np
import pytest

from qif_attribution.models.quantum_discriminator import RULES, QuantumStateDiscriminator


def _flatten(matrices: list[np.ndarray]) -> np.ndarray:
    return np.vstack([m.reshape(-1) for m in matrices])


def _toy_problem() -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Two well-separated diagonal-state classes on 2x2 density matrices.

    Class A concentrates on |0>, class B on |1>; the reference (mean) states are
    diag([0.875, 0.125]) and diag([0.125, 0.875]). Queries lean the same way but
    are not identical to any training matrix, so every rule must generalize.
    """

    train = _flatten(
        [
            np.diag([0.85, 0.15]),
            np.diag([0.90, 0.10]),
            np.diag([0.15, 0.85]),
            np.diag([0.10, 0.90]),
        ]
    )
    labels = ["A", "A", "B", "B"]
    queries = _flatten([np.diag([0.80, 0.20]), np.diag([0.20, 0.80])])
    expected = ["A", "B"]
    return train, labels, queries, expected


@pytest.mark.parametrize("rule", RULES)
def test_each_rule_separates_two_classes(rule: str) -> None:
    train, labels, queries, expected = _toy_problem()

    model = QuantumStateDiscriminator(dim=2, rule=rule).fit(train, labels)

    assert model.predict(queries) == expected


def test_fit_records_references_and_priors() -> None:
    train, labels, _, _ = _toy_problem()

    model = QuantumStateDiscriminator(dim=2).fit(train, labels)

    assert model.classes_ == ["A", "B"]
    assert np.isclose(model.priors_["A"], 0.5)
    assert np.isclose(model.priors_["B"], 0.5)
    assert np.allclose(model.references_["A"], np.diag([0.875, 0.125]))
    assert np.allclose(model.references_["B"], np.diag([0.125, 0.875]))


def test_pgm_builds_a_resolving_povm() -> None:
    train, labels, _, _ = _toy_problem()

    model = QuantumStateDiscriminator(dim=2, rule="pgm").fit(train, labels)

    # For equiprobable classes the POVM elements sum to the identity.
    total = sum(model.povm_.values())
    assert np.allclose(total, np.eye(2))
    assert all(np.allclose(e, e.T) for e in model.povm_.values())


def test_regularized_pgm_still_resolves_povm() -> None:
    train, labels, _, _ = _toy_problem()

    # A relative spectral floor must not break the resolution of identity on a
    # full-rank rho_bar (the floor never triggers when all eigenvalues are large).
    model = QuantumStateDiscriminator(dim=2, rule="pgm", pgm_rcond=1e-2).fit(train, labels)

    total = sum(model.povm_.values())
    assert np.allclose(total, np.eye(2))


@pytest.mark.parametrize("rule", RULES)
def test_multi_prototype_fits_and_predicts(rule: str) -> None:
    train, labels, queries, expected = _toy_problem()

    model = QuantumStateDiscriminator(dim=2, rule=rule, n_prototypes=2).fit(train, labels)

    # Two training matrices per class -> at most two prototypes per class.
    assert all(1 <= len(protos) <= 2 for protos in model.prototypes_.values())
    # Prototype weights over all classes form a distribution (sum to 1).
    total_weight = sum(w for protos in model.prototypes_.values() for w, _ in protos)
    assert np.isclose(total_weight, 1.0)
    assert model.predict(queries) == expected


def test_single_prototype_matches_class_mean() -> None:
    train, labels, _, _ = _toy_problem()

    model = QuantumStateDiscriminator(dim=2, n_prototypes=1).fit(train, labels)

    # With one prototype, the prototype is exactly the class-mean reference.
    for cls in model.classes_:
        (weight, matrix), = model.prototypes_[cls]
        assert np.allclose(matrix, model.references_[cls])
        assert np.isclose(weight, model.priors_[cls])


def test_unknown_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown rule"):
        QuantumStateDiscriminator(dim=2, rule="helstrom")


def test_non_positive_n_prototypes_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_prototypes must be"):
        QuantumStateDiscriminator(dim=2, n_prototypes=0)


def test_non_positive_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        QuantumStateDiscriminator(dim=0)


def test_label_count_mismatch_is_rejected() -> None:
    train, labels, _, _ = _toy_problem()

    with pytest.raises(ValueError, match="must match number of labels"):
        QuantumStateDiscriminator(dim=2).fit(train, labels[:-1])


def test_predict_before_fit_is_rejected() -> None:
    _, _, queries, _ = _toy_problem()

    with pytest.raises(RuntimeError, match="must be fit before predict"):
        QuantumStateDiscriminator(dim=2).predict(queries)

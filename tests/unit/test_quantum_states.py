import numpy as np

from qif_attribution.quantum.states import (
    class_density_matrix,
    fidelity_pure,
    pure_density_matrix,
    purity,
    trace_distance,
    von_neumann_entropy,
)


def test_pure_density_matrix_has_trace_one_and_purity_one() -> None:
    rho = pure_density_matrix(np.array([1.0, 0.0]))

    assert np.isclose(np.trace(rho), 1.0)
    assert np.isclose(purity(rho), 1.0)
    assert np.isclose(von_neumann_entropy(rho), 0.0)


def test_orthogonal_states_have_zero_fidelity_and_trace_distance_one() -> None:
    rho = pure_density_matrix(np.array([1.0, 0.0]))
    sigma = pure_density_matrix(np.array([0.0, 1.0]))

    assert np.isclose(fidelity_pure(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)
    assert np.isclose(trace_distance(rho, sigma), 1.0)


def test_class_density_matrix_mixes_states() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])

    rho = class_density_matrix(vectors)

    assert np.isclose(np.trace(rho), 1.0)
    assert np.isclose(purity(rho), 0.5)
    assert np.isclose(von_neumann_entropy(rho), 1.0)

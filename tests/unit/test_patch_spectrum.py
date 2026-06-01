import numpy as np
import pytest

from qif_attribution.features.patch_spectrum import (
    DENSITY_SCALAR_FIELDS,
    _box_blur,
    density_matrix_descriptors,
    density_matrix_eigenvalues,
    density_matrix_from_image,
    density_matrix_signature,
    density_signature_dim,
    state_fidelity,
    trace_distance,
)


def test_density_signature_dim() -> None:
    assert density_signature_dim(16) == len(DENSITY_SCALAR_FIELDS) + 16
    assert density_signature_dim(0) == len(DENSITY_SCALAR_FIELDS)
    with pytest.raises(ValueError):
        density_signature_dim(-1)


def test_density_matrix_is_valid_density_operator() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    rho = density_matrix_from_image(image, patch_size=4)

    assert rho.shape == (16, 16)
    assert np.allclose(rho, rho.T)
    assert np.isclose(np.trace(rho), 1.0)
    eigenvalues = np.linalg.eigvalsh(rho)
    assert eigenvalues.min() > -1e-9


def test_density_matrix_solid_image_is_maximally_mixed() -> None:
    image = np.full((16, 16, 3), 120, dtype=np.uint8)

    rho = density_matrix_from_image(image, patch_size=4)

    assert np.allclose(rho, np.eye(16) / 16)


def test_density_matrix_eigenvalues_are_a_distribution() -> None:
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)

    eigenvalues = density_matrix_eigenvalues(density_matrix_from_image(image, patch_size=4))

    assert np.isclose(eigenvalues.sum(), 1.0)
    assert np.all(eigenvalues >= 0.0)
    assert np.all(np.diff(eigenvalues) <= 1e-12)  # descending


def test_descriptors_for_maximally_mixed_state() -> None:
    eigenvalues = np.full(16, 1.0 / 16)

    descriptors = density_matrix_descriptors(eigenvalues)

    assert np.isclose(descriptors["purity"], 1.0 / 16)
    assert np.isclose(descriptors["entropy_norm"], 1.0)
    assert np.isclose(descriptors["effective_rank_frac"], 1.0)
    assert np.isclose(descriptors["trace_distance_to_mixed"], 0.0)
    assert np.isclose(descriptors["fidelity_to_mixed"], 1.0)


def test_descriptors_for_pure_state() -> None:
    eigenvalues = np.zeros(16)
    eigenvalues[0] = 1.0

    descriptors = density_matrix_descriptors(eigenvalues)

    assert np.isclose(descriptors["purity"], 1.0)
    assert np.isclose(descriptors["entropy_norm"], 0.0)
    assert np.isclose(descriptors["effective_rank_frac"], 1.0 / 16)
    assert np.isclose(descriptors["trace_distance_to_mixed"], 15.0 / 16)
    assert np.isclose(descriptors["fidelity_to_mixed"], 1.0 / 16)


def test_signature_length_and_finiteness() -> None:
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    signature = density_matrix_signature(image, patch_size=4, top_eigenvalues=6)

    assert signature.shape == (density_signature_dim(6),)
    assert np.all(np.isfinite(signature))


def test_signature_separates_distinct_textures() -> None:
    rows = np.tile(np.linspace(0, 255, 32, dtype=np.uint8)[:, None], (1, 32))
    vertical = np.repeat(rows[:, :, None], 3, axis=2)
    checker = np.indices((32, 32)).sum(axis=0) % 2 * 255
    checker = np.repeat(checker.astype(np.uint8)[:, :, None], 3, axis=2)

    sig_vertical = density_matrix_signature(vertical, patch_size=8)
    sig_checker = density_matrix_signature(checker, patch_size=8)

    assert not np.allclose(sig_vertical, sig_checker)


def test_box_blur_of_constant_is_constant() -> None:
    plane = np.full((10, 12), 0.37)

    assert np.allclose(_box_blur(plane, 2), 0.37)
    assert np.allclose(_box_blur(plane, 0), plane)  # radius < 1 is a no-op


def test_highpass_of_solid_image_is_maximally_mixed() -> None:
    image = np.full((16, 16, 3), 120, dtype=np.uint8)

    rho = density_matrix_from_image(image, patch_size=4, highpass=True)

    assert np.allclose(rho, np.eye(16) / 16)


def test_color_density_matrix_shape_and_validity() -> None:
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    rho = density_matrix_from_image(image, patch_size=4, color=True)

    assert rho.shape == (48, 48)  # 3 channels * 4 * 4
    assert np.allclose(rho, rho.T)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.linalg.eigvalsh(rho).min() > -1e-9


def test_color_and_highpass_keep_signature_length() -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    for color in (False, True):
        for highpass in (False, True):
            signature = density_matrix_signature(
                image,
                patch_size=4,
                top_eigenvalues=6,
                color=color,
                highpass=highpass,
            )
            assert signature.shape == (density_signature_dim(6),)
            assert np.all(np.isfinite(signature))


def test_highpass_changes_signature() -> None:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    base = density_matrix_signature(image, patch_size=4, top_eigenvalues=6)
    high = density_matrix_signature(image, patch_size=4, top_eigenvalues=6, highpass=True)

    assert not np.allclose(base, high)


def test_state_fidelity_bounds_and_symmetry() -> None:
    rho = np.diag([0.7, 0.2, 0.1])
    sigma = np.diag([0.2, 0.3, 0.5])

    # Identical states: fidelity 1; orthogonal pure states: fidelity 0.
    assert np.isclose(state_fidelity(rho, rho), 1.0)
    assert np.isclose(state_fidelity(np.diag([1.0, 0.0]), np.diag([0.0, 1.0])), 0.0)
    # Symmetric, and within [0, 1] for mixed states.
    assert np.isclose(state_fidelity(rho, sigma), state_fidelity(sigma, rho))
    assert 0.0 <= state_fidelity(rho, sigma) <= 1.0


def test_state_fidelity_matches_classical_for_diagonal_states() -> None:
    # For commuting (diagonal) states, F reduces to (sum_i sqrt(p_i q_i))^2.
    p = np.array([0.8, 0.2])
    q = np.array([0.875, 0.125])
    expected = float(np.sum(np.sqrt(p * q)) ** 2)

    assert np.isclose(state_fidelity(np.diag(p), np.diag(q)), expected)


def test_trace_distance_bounds_and_symmetry() -> None:
    rho = np.diag([0.7, 0.2, 0.1])
    sigma = np.diag([0.2, 0.3, 0.5])

    # Identical states: distance 0; orthogonal pure states: distance 1.
    assert np.isclose(trace_distance(rho, rho), 0.0)
    assert np.isclose(trace_distance(np.diag([1.0, 0.0]), np.diag([0.0, 1.0])), 1.0)
    # Symmetric, and equals the classical total-variation distance when diagonal.
    assert np.isclose(trace_distance(rho, sigma), trace_distance(sigma, rho))
    assert np.isclose(trace_distance(rho, sigma), 0.5 * np.sum(np.abs(np.diag(rho - sigma))))

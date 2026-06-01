"""Quantum-state metrics for normalized vision embeddings.

These functions are mathematically correct quantum-information quantities, but on
real-valued vision embeddings they reduce to classical second-moment statistics,
so treat the "quantum" naming as descriptive language, not a claim of advantage:

- ``fidelity_pure(a, b) = |<a|b>|^2`` is the **squared cosine similarity** of the
  two L2-normalized vectors -- no information a cosine kernel lacks.
- ``class_density_matrix`` is the mean of ``|psi><psi|``, i.e. the **uncentered
  second-moment (correlation) matrix** of the normalized class embeddings.
- ``purity`` (``Tr rho^2``) and ``von_neumann_entropy`` are the **participation
  ratio** and **spectral entropy** of that matrix -- ordinary covariance-spectrum
  diagnostics, equivalent to a PCA / kernel-eigenvalue analysis.

The class-level audit in ``scripts/analysis/quantum_embedding_audit.py`` confirms
empirically that the trace distance between class density matrices ranks generator
confusability identically to plain class-centroid distance (Spearman = 1.0). See
the README "Limitations" section.
"""

from __future__ import annotations

import numpy as np


def normalize_state(vector: np.ndarray) -> np.ndarray:
    """Normalize an embedding so it can be treated as a pure quantum state."""

    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return state / norm


def pure_density_matrix(vector: np.ndarray) -> np.ndarray:
    """Return rho = |psi><psi| for an embedding vector."""

    state = normalize_state(vector)
    return np.outer(state, np.conjugate(state))


def class_density_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the mixed state formed by averaging pure states for a class."""

    array = np.asarray(vectors)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    rho = np.mean([pure_density_matrix(vector) for vector in array], axis=0)
    return _renormalize_density(rho)


def fidelity_pure(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Pure-state fidelity |<psi_a|psi_b>|^2."""

    state_a = normalize_state(vector_a)
    state_b = normalize_state(vector_b)
    return float(np.abs(np.vdot(state_a, state_b)) ** 2)


def fidelity_mixed(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity between two density matrices."""

    rho_matrix = _as_density(rho)
    sigma_matrix = _as_density(sigma)
    sqrt_rho = _matrix_sqrt_psd(rho_matrix)
    middle = sqrt_rho @ sigma_matrix @ sqrt_rho
    sqrt_middle = _matrix_sqrt_psd(middle)
    value = np.real(np.trace(sqrt_middle)) ** 2
    return float(np.clip(value, 0.0, 1.0))


def purity(rho: np.ndarray) -> float:
    """Return Tr(rho^2)."""

    matrix = _as_density(rho)
    return float(np.real(np.trace(matrix @ matrix)))


def von_neumann_entropy(rho: np.ndarray, *, base: float = 2.0, tolerance: float = 1e-12) -> float:
    """Return -Tr(rho log rho), with eigenvalues clipped for numerical safety."""

    matrix = _as_density(rho)
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.clip(np.real(eigenvalues), 0.0, 1.0)
    eigenvalues = eigenvalues[eigenvalues > tolerance]
    if eigenvalues.size == 0:
        return 0.0
    logs = np.log(eigenvalues) / np.log(base)
    return float(-np.sum(eigenvalues * logs))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return 0.5 * ||rho - sigma||_1 for Hermitian density matrices."""

    diff = _as_density(rho) - _as_density(sigma)
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return float(0.5 * np.sum(singular_values))


def _as_density(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("density matrix must be square")
    if not np.allclose(array, np.conjugate(array.T), atol=1e-8):
        raise ValueError("density matrix must be Hermitian")
    trace = np.trace(array)
    if not np.isclose(trace, 1.0, atol=1e-8):
        raise ValueError("density matrix must have trace 1")
    return array


def _renormalize_density(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix)
    if np.isclose(trace, 0.0):
        raise ValueError("density matrix has zero trace")
    return matrix / trace


def _matrix_sqrt_psd(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    hermitian = (matrix + np.conjugate(matrix.T)) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    eigenvalues = np.clip(np.real(eigenvalues), 0.0, None)
    eigenvalues[eigenvalues < tolerance] = 0.0
    return (eigenvectors * np.sqrt(eigenvalues)) @ np.conjugate(eigenvectors.T)

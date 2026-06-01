"""Density-matrix (quantum-state) image signatures.

Each image is turned into an ensemble of "pure states": small patches are
mean-removed (local DC) and L2-normalized so every patch is a unit vector
|x>. The image is then summarized by the even mixture

    rho = (1 / n) * sum_i |x_i><x_i|,

a Hermitian, positive-semidefinite, trace-1 operator -- a valid density
matrix. Its eigenvalue spectrum yields standard quantum-state descriptors
(purity, von Neumann entropy, fidelity / trace distance to the maximally
mixed state) that capture how concentrated vs. spread the image's local
texture modes are. These are intended to expose generator fingerprints that
a radial power spectrum misses.

Honest framing: this is a **patch covariance / second-moment descriptor** in
quantum-information language, not quantum mechanics. ``rho`` is (up to scaling)
the correlation matrix of mean-removed, L2-normalized patches; its eigenvalues
are that matrix's spectrum, and purity / entropy / effective rank are the
standard participation-ratio and spectral-entropy summaries of it. The
quantum-state framing is a re-description, and we make no claim of quantum
advantage -- the equivalent classical pipeline is patch-PCA / region-covariance
texture analysis. This module was renamed from ``features/quantum.py`` to
``features/patch_spectrum.py`` to name it for what it is; see README
"Limitations". (Genuinely quantum-information code lives in the ``quantum/``
subpackage and ``models/quantum_discriminator.py``.)
"""

from __future__ import annotations

import numpy as np

# Scalar descriptors emitted before the eigenvalue tail. All are normalized to
# be O(1) and (mostly) dimension-independent so they compose with the raw
# eigenvalues under both cosine and standardized classifiers.
DENSITY_SCALAR_FIELDS = (
    "purity",
    "entropy_norm",
    "effective_rank_frac",
    "trace_distance_to_mixed",
    "fidelity_to_mixed",
)


def density_signature_dim(top_eigenvalues: int) -> int:
    """Length of the density-matrix signature for a given eigenvalue-tail size."""

    if top_eigenvalues < 0:
        raise ValueError("top_eigenvalues must be non-negative")
    return len(DENSITY_SCALAR_FIELDS) + top_eigenvalues


def density_matrix_from_image(
    image: np.ndarray,
    *,
    patch_size: int = 8,
    color: bool = False,
    highpass: bool = False,
    highpass_radius: int = 2,
) -> np.ndarray:
    """Build the patch-ensemble density matrix for an image.

    Returns a (d, d) Hermitian PSD matrix with unit trace. ``d`` is
    ``patch_size ** 2`` for grayscale and ``3 * patch_size ** 2`` when
    ``color`` stacks the RGB channels into each patch vector. With
    ``highpass`` each channel plane is replaced by its residual after a box
    blur, isolating the high-frequency content where generator fingerprints
    tend to live. A perfectly flat image maps to the maximally mixed state.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    planes = _to_planes(image, color=color)
    if highpass:
        planes = [plane - _box_blur(plane, highpass_radius) for plane in planes]
    patches = np.concatenate([_patches(plane, patch_size) for plane in planes], axis=1)
    dim = patches.shape[1]
    patches = patches - patches.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(patches, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        return np.eye(dim) / dim
    unit = patches[valid] / norms[valid][:, None]
    return (unit.T @ unit) / unit.shape[0]


def density_matrix_eigenvalues(rho: np.ndarray) -> np.ndarray:
    """Return the density matrix eigenvalues, clipped non-negative and descending."""

    values = np.linalg.eigvalsh(rho)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        dim = rho.shape[0]
        return np.full(dim, 1.0 / dim)
    return (values / total)[::-1]


def density_matrix_descriptors(eigenvalues: np.ndarray) -> dict[str, float]:
    """Compute quantum-state descriptors from a (descending) eigenvalue spectrum."""

    values = np.asarray(eigenvalues, dtype=np.float64)
    dim = int(values.shape[0])
    nonzero = values[values > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    log_dim = float(np.log(dim)) if dim > 1 else 1.0
    uniform = 1.0 / dim
    return {
        "purity": float(np.sum(values**2)),
        "entropy_norm": entropy / log_dim,
        "effective_rank_frac": float(np.exp(entropy)) / dim,
        "trace_distance_to_mixed": float(0.5 * np.sum(np.abs(values - uniform))),
        "fidelity_to_mixed": float(np.sum(np.sqrt(values)) ** 2) / dim,
    }


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root of a (numerically) PSD symmetric matrix."""

    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2 in [0, 1].

    1.0 for identical states, 0.0 for states with orthogonal support. Symmetric
    in its arguments for valid density matrices.
    """

    sqrt_rho = _psd_sqrt(rho)
    inner = _symmetrize(sqrt_rho @ sigma @ sqrt_rho)
    values = np.clip(np.linalg.eigvalsh(inner), 0.0, None)
    return float(np.sum(np.sqrt(values)) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance T = (1/2) ||rho - sigma||_1 in [0, 1] for density matrices.

    0.0 for identical states, 1.0 for states with orthogonal support.
    """

    values = np.linalg.eigvalsh(_symmetrize(rho - sigma))
    return float(0.5 * np.sum(np.abs(values)))


def density_matrix_signature(
    image: np.ndarray,
    *,
    patch_size: int = 8,
    top_eigenvalues: int = 16,
    color: bool = False,
    highpass: bool = False,
    highpass_radius: int = 2,
) -> np.ndarray:
    """Concatenate scalar quantum descriptors with the largest eigenvalues."""

    eigenvalues = density_matrix_eigenvalues(
        density_matrix_from_image(
            image,
            patch_size=patch_size,
            color=color,
            highpass=highpass,
            highpass_radius=highpass_radius,
        )
    )
    descriptors = density_matrix_descriptors(eigenvalues)
    scalars = np.array([descriptors[name] for name in DENSITY_SCALAR_FIELDS], dtype=np.float64)
    head = _fit_length(eigenvalues, top_eigenvalues)
    return np.concatenate([scalars, head])


def _fit_length(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    out = np.zeros(length, dtype=np.float64)
    take = min(length, int(values.shape[0]))
    out[:take] = values[:take]
    return out


def _patches(gray: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = gray.shape
    n_rows = height // patch_size
    n_cols = width // patch_size
    if n_rows == 0 or n_cols == 0:
        flat = gray.reshape(-1)
        dim = patch_size * patch_size
        patch = np.zeros(dim, dtype=np.float64)
        patch[: min(dim, flat.shape[0])] = flat[:dim]
        return patch[None, :]
    cropped = gray[: n_rows * patch_size, : n_cols * patch_size]
    blocks = cropped.reshape(n_rows, patch_size, n_cols, patch_size)
    return blocks.transpose(0, 2, 1, 3).reshape(n_rows * n_cols, patch_size * patch_size)


def _to_planes(image: np.ndarray, *, color: bool) -> list[np.ndarray]:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return [_scale01(array)]
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        rgb = _scale01(array[:, :, :3])
        if color:
            return [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        return [gray]
    raise ValueError("image must be a grayscale, RGB, or RGBA array")


def _scale01(array: np.ndarray) -> np.ndarray:
    if array.max(initial=0.0) > 1.0:
        return array / 255.0
    return array


def _box_blur(plane: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return plane
    return _moving_average(_moving_average(plane, radius, axis=1), radius, axis=0)


def _moving_average(array: np.ndarray, radius: int, axis: int) -> np.ndarray:
    window = 2 * radius + 1
    pad = [(radius, radius) if ax == axis else (0, 0) for ax in range(array.ndim)]
    padded = np.pad(array, pad, mode="reflect")
    cumsum = np.cumsum(padded, axis=axis)
    prefix_shape = list(cumsum.shape)
    prefix_shape[axis] = 1
    cumsum = np.concatenate([np.zeros(prefix_shape), cumsum], axis=axis)
    length = array.shape[axis]
    upper = np.take(cumsum, range(window, window + length), axis=axis)
    lower = np.take(cumsum, range(0, length), axis=axis)
    return (upper - lower) / window

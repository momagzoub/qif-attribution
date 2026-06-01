"""Frequency-domain signatures for lightweight fingerprint baselines."""

from __future__ import annotations

import numpy as np


def radial_fft_signature(
    image: np.ndarray,
    bins: int = 64,
    include_phase: bool = True,
) -> np.ndarray:
    """Compute a radial FFT signature from a grayscale or RGB image.

    The signature averages amplitude, and optionally circular phase concentration,
    across annuli centered at the Fourier origin. It is intentionally simple and
    deterministic so it can act as an auditable forensic baseline.
    """

    if bins <= 0:
        raise ValueError("bins must be positive")
    gray = _to_gray(image)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    amplitude = np.log1p(np.abs(spectrum))
    phase = np.angle(spectrum)

    height, width = gray.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height / 2) ** 2 + (xx - width / 2) ** 2)
    radius = radius / radius.max()
    bin_index = np.minimum((radius * bins).astype(int), bins - 1)

    amp_profile = np.zeros(bins, dtype=np.float64)
    phase_profile = np.zeros(bins, dtype=np.float64)
    for idx in range(bins):
        mask = bin_index == idx
        if not np.any(mask):
            continue
        amp_profile[idx] = float(np.mean(amplitude[mask]))
        if include_phase:
            unit_phase = np.exp(1j * phase[mask])
            phase_profile[idx] = float(np.abs(np.mean(unit_phase)))

    if include_phase:
        return np.concatenate([amp_profile, phase_profile])
    return amp_profile


def _to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return _normalize(array)
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        rgb = array[:, :, :3]
        gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
        return _normalize(gray)
    raise ValueError("image must be a grayscale, RGB, or RGBA array")


def _normalize(array: np.ndarray) -> np.ndarray:
    if array.max(initial=0.0) > 1.0:
        return array / 255.0
    return array

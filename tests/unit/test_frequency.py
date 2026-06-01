import numpy as np

from qif_attribution.features.frequency import radial_fft_signature


def test_radial_fft_signature_shape_with_phase() -> None:
    image = np.zeros((16, 16), dtype=np.float64)
    image[4:12, 4:12] = 1.0

    signature = radial_fft_signature(image, bins=8, include_phase=True)

    assert signature.shape == (16,)
    assert np.all(np.isfinite(signature))


def test_radial_fft_signature_shape_without_phase() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    signature = radial_fft_signature(image, bins=4, include_phase=False)

    assert signature.shape == (4,)

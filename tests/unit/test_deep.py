"""Unit tests for the deep-backbone feature path.

These never load a model or touch the network: they only exercise the
device-selection logic, the backbone registry, and module constants. The whole
module is skipped when the optional ``[deep]`` extra (torch/torchvision) is not
installed.
"""

import pytest

torch = pytest.importorskip("torch")

from qif_attribution.features.deep import (  # noqa: E402
    BACKBONES,
    RESNET50_DIM,
    extract_backbone_features,
    resolve_device,
)


def test_resnet_dim_constant() -> None:
    assert RESNET50_DIM == 2048


def test_backbone_registry_has_expected_specs() -> None:
    assert BACKBONES["resnet50"].dim == 2048
    assert BACKBONES["convnext_large"].dim == 1536
    # Every spec names a torchvision factory and a weights enum to look up.
    for name, spec in BACKBONES.items():
        assert spec.ctor and spec.weights_enum and spec.default_weights, name


def test_extract_backbone_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="unknown backbone"):
        extract_backbone_features([], backbone="not_a_real_net", image_root=".")


def test_resolve_device_honors_explicit_override() -> None:
    # An explicit device is returned verbatim, even if unavailable on this host.
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device(None) == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device(None) == "cpu"

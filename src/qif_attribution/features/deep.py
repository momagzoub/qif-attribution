"""Frozen deep-backbone embeddings -- learned-feature ceiling baselines.

This is the only feature path that depends on torch/torchvision; install it
with ``pip install -e '.[deep]'``. The imports are deferred to call time so the
rest of the package keeps importing with numpy alone. Each image is mapped to
the global-average-pooled penultimate activation of an ImageNet-pretrained
backbone (the classifier head is replaced with identity) -- the standard "deep
features" transfer-learning setup, and the same recipe used in the original
internship attribution work.

Backbones live in a small registry so we can compare representations without
touching the extraction loop. ResNet-50 (2048-d) is the historical baseline;
ConvNeXt-Large (1536-d) is a stronger, more modern convnet; ViT-L/16 and
Swin-V2-B are transformer alternatives for robustness checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from qif_attribution.data.manifest import ManifestRecord
from qif_attribution.features.extract import INDEX_FIELDS, FeatureExtractionResult

if TYPE_CHECKING:
    import torch

RESNET50_DIM = 2048


@dataclass(frozen=True)
class BackboneSpec:
    """How to build one torchvision backbone and strip its classifier head."""

    ctor: str  # torchvision.models factory name, e.g. "resnet50"
    weights_enum: str  # torchvision.models weights-enum class, e.g. "ResNet50_Weights"
    default_weights: str  # member name on that enum, e.g. "IMAGENET1K_V2"
    dim: int  # penultimate feature width after the head is removed
    strip: Callable[[Any, Any], None]  # (model, identity_module) -> None


def _strip_fc(model: Any, identity: Any) -> None:
    model.fc = identity  # resnet / regnet


def _strip_convnext(model: Any, identity: Any) -> None:
    model.classifier[2] = identity  # Sequential(LayerNorm2d, Flatten, Linear)


def _strip_vit(model: Any, identity: Any) -> None:
    model.heads.head = identity


def _strip_swin(model: Any, identity: Any) -> None:
    model.head = identity


BACKBONES: dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec("resnet50", "ResNet50_Weights", "IMAGENET1K_V2", 2048, _strip_fc),
    "convnext_large": BackboneSpec(
        "convnext_large", "ConvNeXt_Large_Weights", "IMAGENET1K_V1", 1536, _strip_convnext
    ),
    "vit_l_16": BackboneSpec("vit_l_16", "ViT_L_16_Weights", "IMAGENET1K_V1", 1024, _strip_vit),
    "swin_v2_b": BackboneSpec("swin_v2_b", "Swin_V2_B_Weights", "IMAGENET1K_V1", 1024, _strip_swin),
}


def resolve_device(device: str | None = None) -> str:
    """Pick an accelerator: explicit override, else cuda > mps > cpu."""

    import torch

    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_backbone_features(
    records: list[ManifestRecord],
    *,
    backbone: str = "resnet50",
    image_root: Path,
    weights: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
    limit: int | None = None,
    skip_missing: bool = False,
) -> FeatureExtractionResult:
    """Embed each manifest image with a frozen, pretrained backbone.

    Images are streamed through the network in batches so peak memory stays
    bounded. The classifier head is replaced with identity, so the output is the
    backbone's pooled penultimate feature vector (width depends on ``backbone``).
    """

    import torch
    from PIL import Image
    from torchvision import models as tv_models

    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    spec = BACKBONES[backbone]
    resolved_weights = weights or spec.default_weights

    weight_enum = getattr(tv_models, spec.weights_enum)[resolved_weights]
    transform = weight_enum.transforms()
    resolved = resolve_device(device)
    model = getattr(tv_models, spec.ctor)(weights=weight_enum)
    spec.strip(model, torch.nn.Identity())
    model.eval().to(resolved)

    selected = records[:limit] if limit is not None else records
    index: list[dict[str, str]] = []
    missing: list[str] = []
    chunks: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        stacked = torch.stack(batch).to(resolved)
        with torch.no_grad():
            embeddings = model(stacked)
        chunks.append(embeddings.cpu().numpy().astype(np.float64))
        batch.clear()

    for record in selected:
        image_path = image_root / record.relative_path
        if not image_path.exists():
            if skip_missing:
                missing.append(record.relative_path)
                continue
            raise FileNotFoundError(f"missing image for {record.image_id}: {image_path}")
        with Image.open(image_path) as handle:
            batch.append(transform(handle.convert("RGB")))
        index.append({name: getattr(record, name) for name in INDEX_FIELDS})
        if len(batch) >= batch_size:
            flush()
    flush()

    features = np.vstack(chunks) if chunks else np.empty((0, spec.dim), dtype=np.float64)
    return FeatureExtractionResult(
        features=features,
        index=index,
        bins=0,
        include_phase=False,
        missing=missing,
        feature_kind=backbone,
        params={"backbone": backbone, "weights": resolved_weights, "dim": spec.dim},
    )


def extract_resnet_features(
    records: list[ManifestRecord],
    *,
    image_root: Path,
    weights: str = "IMAGENET1K_V2",
    batch_size: int = 32,
    device: str | None = None,
    limit: int | None = None,
    skip_missing: bool = False,
) -> FeatureExtractionResult:
    """Embed each manifest image with a pretrained ResNet-50 backbone.

    Thin wrapper over :func:`extract_backbone_features` kept for back-compat; the
    output is the 2048-dim pooled feature vector (IMAGENET1K_V2 weights).
    """

    return extract_backbone_features(
        records,
        backbone="resnet50",
        image_root=image_root,
        weights=weights,
        batch_size=batch_size,
        device=device,
        limit=limit,
        skip_missing=skip_missing,
    )

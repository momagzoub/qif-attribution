#!/usr/bin/env python
"""Extract frozen deep-backbone embeddings for the small set.

Produces a learned-feature *ceiling* bundle: each image is mapped to the
global-average-pooled penultimate activation of an ImageNet-pretrained backbone
(classifier head removed). ResNet-50 is the historical internship recipe;
stronger modern backbones (ConvNeXt-Large, ViT-L/16, Swin-V2-B) are available
via the registry in ``features/deep.py`` to test whether a better representation
raises the ceiling.

Requires the deep extra (torch + torchvision + pillow):

    pip install -e '.[deep]'

Run from the repo root (downloads pretrained weights on first run):

    PYTHONPATH=src python scripts/analysis/extract_resnet_small.py [backbone]

``backbone`` defaults to ``resnet50``; pass e.g. ``convnext_large``. Writes
data/processed/small/features_<backbone>/ (features.npy + index.csv +
features_meta.json), row-aligned with the other small-set bundles by image_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qif_attribution.data.manifest import load_manifest
from qif_attribution.features.deep import BACKBONES, extract_backbone_features, resolve_device
from qif_attribution.features.extract import write_feature_bundle

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "interim" / "scale" / "small" / "manifest.csv"
SMALL = ROOT / "data" / "processed" / "small"
BATCH_SIZE = 32


def main() -> int:
    backbone = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
    if backbone not in BACKBONES:
        raise SystemExit(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    spec = BACKBONES[backbone]
    out_dir = SMALL / f"features_{backbone}"

    records = load_manifest(MANIFEST)
    device = resolve_device()
    print(
        f"backbone={backbone} weights={spec.default_weights} dim={spec.dim}\n"
        f"images={len(records)} device={device} batch_size={BATCH_SIZE} "
        f"(downloads pretrained weights on first run)"
    )

    result = extract_backbone_features(
        records,
        backbone=backbone,
        image_root=ROOT,
        batch_size=BATCH_SIZE,
        device=device,
    )

    meta = write_feature_bundle(out_dir, result)
    print(
        f"\nextracted {meta['count']} features "
        f"(dim={meta['feature_dim']}, kind={meta['feature_kind']})"
    )
    print(f"generator_counts={meta['generator_counts']}")
    print(f"wrote bundle to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Extract frequency-domain feature arrays from manifest images."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qif_attribution.data.manifest import ManifestRecord
from qif_attribution.features.frequency import radial_fft_signature
from qif_attribution.features.patch_spectrum import density_matrix_signature, density_signature_dim

# Columns carried next to each feature row so downstream training can attach
# labels and join split assignments by image_id.
INDEX_FIELDS = (
    "image_id",
    "generator_id",
    "prompt_id",
    "prompt_source",
    "prompt_category",
    "sha256",
    "relative_path",
)


@dataclass(frozen=True)
class FeatureExtractionResult:
    features: np.ndarray
    index: list[dict[str, str]]
    bins: int
    include_phase: bool
    missing: list[str]
    feature_kind: str = "radial_fft"
    params: dict = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0


def load_image_array(path: Path) -> np.ndarray:
    """Load an image file as an (H, W, 3) uint8 RGB array."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - only hit without pillow installed
        raise RuntimeError(
            "Feature extraction requires pillow. Install with `pip install -e '.[analysis]'`."
        ) from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _walk_features(
    records: list[ManifestRecord],
    *,
    image_root: Path,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    limit: int | None,
    skip_missing: bool,
) -> tuple[list[np.ndarray], list[dict[str, str]], list[str]]:
    """Apply ``feature_fn`` to each manifest image, keeping rows aligned."""

    selected = records[:limit] if limit is not None else records
    vectors: list[np.ndarray] = []
    index: list[dict[str, str]] = []
    missing: list[str] = []
    for record in selected:
        image_path = image_root / record.relative_path
        if not image_path.exists():
            if skip_missing:
                missing.append(record.relative_path)
                continue
            raise FileNotFoundError(f"missing image for {record.image_id}: {image_path}")
        image = load_image_array(image_path)
        vectors.append(np.asarray(feature_fn(image), dtype=np.float64))
        index.append({name: getattr(record, name) for name in INDEX_FIELDS})
    return vectors, index, missing


def extract_manifest_features(
    records: list[ManifestRecord],
    *,
    image_root: Path,
    bins: int = 64,
    include_phase: bool = True,
    limit: int | None = None,
    skip_missing: bool = False,
) -> FeatureExtractionResult:
    """Compute a radial FFT signature for each image referenced by the manifest.

    Feature rows are aligned with ``index`` rows so labels and splits can be
    attached later by position or by image_id.
    """

    vectors, index, missing = _walk_features(
        records,
        image_root=image_root,
        feature_fn=lambda image: radial_fft_signature(
            image, bins=bins, include_phase=include_phase
        ),
        limit=limit,
        skip_missing=skip_missing,
    )
    feature_dim = bins * 2 if include_phase else bins
    features = np.vstack(vectors) if vectors else np.empty((0, feature_dim), dtype=np.float64)
    return FeatureExtractionResult(
        features=features,
        index=index,
        bins=bins,
        include_phase=include_phase,
        missing=missing,
        feature_kind="radial_fft",
        params={"bins": bins, "include_phase": include_phase},
    )


def extract_density_matrix_features(
    records: list[ManifestRecord],
    *,
    image_root: Path,
    patch_size: int = 8,
    top_eigenvalues: int = 16,
    color: bool = False,
    highpass: bool = False,
    highpass_radius: int = 2,
    limit: int | None = None,
    skip_missing: bool = False,
) -> FeatureExtractionResult:
    """Compute a density-matrix (quantum-state) signature for each manifest image.

    ``color`` stacks the RGB channels into each patch vector; ``highpass``
    replaces each channel plane with its residual after a box blur of radius
    ``highpass_radius`` (isolating the high-frequency content where generator
    fingerprints tend to live). The signature length is unchanged by either.
    """

    vectors, index, missing = _walk_features(
        records,
        image_root=image_root,
        feature_fn=lambda image: density_matrix_signature(
            image,
            patch_size=patch_size,
            top_eigenvalues=top_eigenvalues,
            color=color,
            highpass=highpass,
            highpass_radius=highpass_radius,
        ),
        limit=limit,
        skip_missing=skip_missing,
    )
    feature_dim = density_signature_dim(top_eigenvalues)
    features = np.vstack(vectors) if vectors else np.empty((0, feature_dim), dtype=np.float64)
    return FeatureExtractionResult(
        features=features,
        index=index,
        bins=0,
        include_phase=False,
        missing=missing,
        feature_kind="density_matrix",
        params={
            "patch_size": patch_size,
            "top_eigenvalues": top_eigenvalues,
            "color": color,
            "highpass": highpass,
            "highpass_radius": highpass_radius,
        },
    )


def write_feature_bundle(output_dir: Path, result: FeatureExtractionResult) -> dict:
    """Persist features.npy, index.csv, and features_meta.json under output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", result.features)

    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_FIELDS))
        writer.writeheader()
        writer.writerows(result.index)

    generator_counts: dict[str, int] = {}
    for row in result.index:
        generator_counts[row["generator_id"]] = generator_counts.get(row["generator_id"], 0) + 1

    meta = {
        "count": len(result.index),
        "feature_dim": result.feature_dim,
        "feature_kind": result.feature_kind,
        "params": result.params,
        "missing": len(result.missing),
        "generator_counts": generator_counts,
    }
    (output_dir / "features_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta

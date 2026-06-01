import numpy as np

from qif_attribution.data.manifest import ManifestRecord
from qif_attribution.features.extract import extract_manifest_features, write_feature_bundle
from qif_attribution.generation.png import write_solid_rgb_png


def _record(image_id: str, generator_id: str, relative_path: str) -> ManifestRecord:
    return ManifestRecord(
        image_id=image_id,
        sha256="a" * 64,
        prompt_id="p1",
        prompt_text="text",
        prompt_source="human",
        prompt_category="object",
        prompt_style="concise",
        generator_id=generator_id,
        generator_version="1",
        seed="0",
        sampler="euler",
        steps="1",
        cfg_scale="1.0",
        width="16",
        height="16",
        postprocess="none",
        license_notes="test",
        relative_path=relative_path,
    )


def test_extract_features_shape_and_alignment(tmp_path) -> None:
    write_solid_rgb_png(tmp_path / "a.png", width=16, height=16, rgb=(10, 20, 30))
    write_solid_rgb_png(tmp_path / "b.png", width=16, height=16, rgb=(200, 100, 50))
    records = [
        _record("img_a", "sd15", "a.png"),
        _record("img_b", "openjourney_v4", "b.png"),
    ]

    result = extract_manifest_features(records, image_root=tmp_path, bins=8, include_phase=True)

    assert result.features.shape == (2, 16)
    assert result.feature_dim == 16
    assert [row["image_id"] for row in result.index] == ["img_a", "img_b"]
    assert [row["generator_id"] for row in result.index] == ["sd15", "openjourney_v4"]
    assert np.all(np.isfinite(result.features))


def test_extract_features_skip_missing(tmp_path) -> None:
    write_solid_rgb_png(tmp_path / "a.png", width=16, height=16, rgb=(10, 20, 30))
    records = [
        _record("img_a", "sd15", "a.png"),
        _record("img_missing", "sd15", "missing.png"),
    ]

    result = extract_manifest_features(
        records, image_root=tmp_path, bins=4, include_phase=False, skip_missing=True
    )

    assert result.features.shape == (1, 4)
    assert result.missing == ["missing.png"]


def test_extract_features_limit(tmp_path) -> None:
    write_solid_rgb_png(tmp_path / "a.png", width=16, height=16, rgb=(10, 20, 30))
    write_solid_rgb_png(tmp_path / "b.png", width=16, height=16, rgb=(200, 100, 50))
    records = [_record("img_a", "sd15", "a.png"), _record("img_b", "sd15", "b.png")]

    result = extract_manifest_features(records, image_root=tmp_path, bins=8, limit=1)

    assert result.features.shape == (1, 16)
    assert [row["image_id"] for row in result.index] == ["img_a"]


def test_write_feature_bundle_writes_artifacts(tmp_path) -> None:
    write_solid_rgb_png(tmp_path / "a.png", width=16, height=16, rgb=(10, 20, 30))
    records = [_record("img_a", "sd15", "a.png")]
    result = extract_manifest_features(records, image_root=tmp_path, bins=8, include_phase=True)

    out = tmp_path / "features_out"
    meta = write_feature_bundle(out, result)

    assert (out / "features.npy").exists()
    assert (out / "index.csv").exists()
    assert (out / "features_meta.json").exists()
    assert meta["count"] == 1
    assert meta["feature_dim"] == 16
    assert meta["generator_counts"] == {"sd15": 1}
    assert np.load(out / "features.npy").shape == (1, 16)

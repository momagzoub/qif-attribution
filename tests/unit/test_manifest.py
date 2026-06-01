from qif_attribution.data.manifest import ManifestRecord, validate_manifest


def make_record(**overrides: str) -> ManifestRecord:
    values = {
        "image_id": "img_001",
        "sha256": "a" * 64,
        "prompt_id": "prompt_001",
        "prompt_text": "a red cube",
        "prompt_source": "human",
        "prompt_category": "object",
        "prompt_style": "concise",
        "generator_id": "sd15",
        "generator_version": "1.5",
        "seed": "101",
        "sampler": "euler",
        "steps": "30",
        "cfg_scale": "7.5",
        "width": "512",
        "height": "512",
        "postprocess": "none",
        "license_notes": "test",
        "relative_path": "raw/img_001.png",
    }
    values.update(overrides)
    return ManifestRecord(**values)


def test_valid_manifest_passes() -> None:
    records = [
        make_record(image_id="img_001", sha256="a" * 64, generator_id="sd15"),
        make_record(image_id="img_002", sha256="b" * 64, generator_id="sd15"),
    ]

    report = validate_manifest(records)

    assert report.ok


def test_duplicate_hash_fails() -> None:
    records = [
        make_record(image_id="img_001", sha256="a" * 64),
        make_record(image_id="img_002", sha256="a" * 64),
    ]

    report = validate_manifest(records)

    assert not report.ok
    assert any("duplicate sha256" in error for error in report.errors)


def test_bad_numeric_field_fails() -> None:
    report = validate_manifest([make_record(steps="thirty")])

    assert not report.ok
    assert any("steps must be an integer" in error for error in report.errors)

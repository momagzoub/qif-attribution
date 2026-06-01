from pathlib import Path

from qif_attribution.data.audit import audit_dataset, find_duplicate_prompt_text, format_audit
from tests.unit.test_manifest import make_record


def test_audit_reports_counts_and_ok_status() -> None:
    records = [
        make_record(image_id="a", sha256="a" * 64, generator_id="sd15"),
        make_record(image_id="b", sha256="b" * 64, generator_id="sd15"),
    ]

    audit = audit_dataset(records)

    assert audit.ok
    assert audit.counts["generator_id"] == {"sd15": 2}
    assert "status: ok" in format_audit(audit)


def test_audit_warns_about_missing_files() -> None:
    records = [make_record(relative_path="raw/not_here.png")]

    audit = audit_dataset(records, image_root=Path("/tmp/qif_missing"))

    assert any("image file(s) are missing" in warning for warning in audit.warnings)


def test_duplicate_prompt_text_normalizes_spacing_and_case() -> None:
    records = [
        make_record(image_id="a", sha256="a" * 64, prompt_text="A Red Cube"),
        make_record(image_id="b", sha256="b" * 64, prompt_text="a  red   cube"),
    ]

    duplicates = find_duplicate_prompt_text(records)

    assert duplicates == {"a red cube": ["a", "b"]}

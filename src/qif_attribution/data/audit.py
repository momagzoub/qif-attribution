"""Dataset audit helpers for attribution experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from qif_attribution.data.manifest import ManifestRecord, validate_manifest


@dataclass(frozen=True)
class DatasetAudit:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    counts: dict[str, dict[str, int]]

    @property
    def ok(self) -> bool:
        return not self.errors


def audit_dataset(
    records: list[ManifestRecord],
    *,
    label_fields: tuple[str, ...] = ("generator_id", "prompt_source", "prompt_category"),
    min_per_class: int = 2,
    image_root: Path | None = None,
) -> DatasetAudit:
    """Run lightweight checks before training attribution models."""

    validation = validate_manifest(records)
    errors = list(validation.errors)
    warnings = list(validation.warnings)
    counts: dict[str, dict[str, int]] = {}

    for field in label_fields:
        field_counts = Counter(str(getattr(record, field)) for record in records)
        counts[field] = dict(sorted(field_counts.items()))
        rare = sorted(label for label, count in field_counts.items() if count < min_per_class)
        if rare:
            warnings.append(
                f"{field} has {len(rare)} class(es) with fewer than {min_per_class} records: "
                f"{', '.join(rare[:5])}"
            )

    duplicate_prompts = find_duplicate_prompt_text(records)
    if duplicate_prompts:
        warnings.append(
            f"duplicate normalized prompt text appears in {len(duplicate_prompts)} group(s)"
        )

    if image_root is not None:
        missing = [
            record.image_id
            for record in records
            if not (image_root / record.relative_path).exists()
        ]
        if missing:
            warnings.append(f"{len(missing)} image file(s) are missing under {image_root}")

    return DatasetAudit(errors=tuple(errors), warnings=tuple(warnings), counts=counts)


def find_duplicate_prompt_text(records: list[ManifestRecord]) -> dict[str, list[str]]:
    """Return normalized prompt strings mapped to image IDs when they repeat."""

    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        normalized = " ".join(record.prompt_text.lower().split())
        groups[normalized].append(record.image_id)
    return {
        prompt: image_ids
        for prompt, image_ids in sorted(groups.items())
        if len(image_ids) > 1
    }


def format_audit(audit: DatasetAudit) -> str:
    """Format an audit report for CLI output."""

    lines: list[str] = ["dataset audit"]
    lines.append(f"status: {'ok' if audit.ok else 'invalid'}")
    for field, counts in audit.counts.items():
        rendered = ", ".join(f"{label}={count}" for label, count in counts.items())
        lines.append(f"{field}: {rendered}")
    if audit.errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in audit.errors)
    if audit.warnings:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in audit.warnings)
    return "\n".join(lines)

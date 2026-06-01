"""Manifest schema and validation for generated-image datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "image_id",
    "sha256",
    "prompt_id",
    "prompt_text",
    "prompt_source",
    "prompt_category",
    "prompt_style",
    "generator_id",
    "generator_version",
    "seed",
    "sampler",
    "steps",
    "cfg_scale",
    "width",
    "height",
    "postprocess",
    "license_notes",
    "relative_path",
)


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    sha256: str
    prompt_id: str
    prompt_text: str
    prompt_source: str
    prompt_category: str
    prompt_style: str
    generator_id: str
    generator_version: str
    seed: str
    sampler: str
    steps: str
    cfg_scale: str
    width: str
    height: str
    postprocess: str
    license_notes: str
    relative_path: str

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> ManifestRecord:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing manifest fields: {', '.join(missing)}")
        values = {field: str(row[field]).strip() for field in REQUIRED_FIELDS}
        return cls(**values)

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in REQUIRED_FIELDS}


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def load_manifest(path: Path) -> list[ManifestRecord]:
    """Load a CSV or JSONL manifest."""

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [ManifestRecord.from_mapping(row) for row in csv.DictReader(handle)]

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records: list[ManifestRecord] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(ManifestRecord.from_mapping(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
        return records

    raise ValueError(f"unsupported manifest format: {path.suffix}")


def validate_manifest(records: list[ManifestRecord]) -> ValidationReport:
    """Validate uniqueness, required numeric fields, and contamination-prone metadata."""

    errors: list[str] = []
    warnings: list[str] = []
    if not records:
        return ValidationReport(errors=("manifest has no records",))

    _require_unique(records, "image_id", errors)
    _require_unique(records, "sha256", errors)

    for index, record in enumerate(records):
        prefix = f"record {index} ({record.image_id})"
        has_bad_hash = len(record.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in record.sha256
        )
        if has_bad_hash:
            errors.append(f"{prefix}: sha256 must be 64 lowercase hex characters")
        for field in ("steps", "width", "height"):
            if not _is_int(getattr(record, field)):
                errors.append(f"{prefix}: {field} must be an integer")
        if not _is_float(record.cfg_scale):
            errors.append(f"{prefix}: cfg_scale must be numeric")
        if "/" in record.image_id or "\\" in record.image_id:
            errors.append(f"{prefix}: image_id must not encode paths")
        if not record.relative_path:
            errors.append(f"{prefix}: relative_path is empty")

    label_counts = _counts(records, "generator_id")
    if min(label_counts.values()) < 2:
        warnings.append("at least one generator_id has fewer than 2 records")

    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))


def _require_unique(records: list[ManifestRecord], field: str, errors: list[str]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        value = getattr(record, field)
        if value in seen:
            errors.append(
                f"duplicate {field}: {value} appears in {seen[value]} and {record.image_id}"
            )
        else:
            seen[value] = record.image_id


def _counts(records: list[ManifestRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = getattr(record, field)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _is_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True

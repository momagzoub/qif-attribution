"""Prompt-family schema for controlled image generation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMPT_FIELDS = (
    "prompt_id",
    "family_id",
    "semantic_id",
    "prompt_text",
    "prompt_source",
    "prompt_category",
    "prompt_style",
)


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    family_id: str
    semantic_id: str
    prompt_text: str
    prompt_source: str
    prompt_category: str
    prompt_style: str

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> PromptRecord:
        missing = [field for field in PROMPT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing prompt fields: {', '.join(missing)}")
        values = {field: str(row[field]).strip() for field in PROMPT_FIELDS}
        return cls(**values)


def load_prompt_records(path: Path) -> list[PromptRecord]:
    if path.suffix.lower() != ".csv":
        raise ValueError("prompt records currently require CSV input")
    with path.open(newline="", encoding="utf-8") as handle:
        return [PromptRecord.from_mapping(row) for row in csv.DictReader(handle)]


def validate_prompt_records(records: list[PromptRecord]) -> tuple[str, ...]:
    errors: list[str] = []
    if not records:
        return ("prompt file has no records",)

    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"prompt record {index} ({record.prompt_id})"
        if record.prompt_id in seen_ids:
            errors.append(f"duplicate prompt_id: {record.prompt_id}")
        seen_ids.add(record.prompt_id)
        for field in PROMPT_FIELDS:
            if not getattr(record, field):
                errors.append(f"{prefix}: {field} is empty")
    return tuple(errors)


def prompt_family_counts(records: list[PromptRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.family_id] = counts.get(record.family_id, 0) + 1
    return dict(sorted(counts.items()))

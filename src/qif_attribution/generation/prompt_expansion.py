"""Deterministic prompt expansion for controlled pilot datasets."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qif_attribution.data.prompts import PROMPT_FIELDS, PromptRecord

CONCEPT_FIELDS = ("concept_id", "category", "subject", "setting", "attributes")

DEFAULT_SOURCES = ("human", "llm_a", "llm_b")
DEFAULT_STYLES = ("concise", "photorealistic", "technical")


@dataclass(frozen=True)
class ConceptRecord:
    concept_id: str
    category: str
    subject: str
    setting: str
    attributes: str

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> ConceptRecord:
        missing = [field for field in CONCEPT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing concept fields: {', '.join(missing)}")
        values = {field: str(row[field]).strip() for field in CONCEPT_FIELDS}
        return cls(**values)


def load_concepts(path: Path) -> list[ConceptRecord]:
    if path.suffix.lower() != ".csv":
        raise ValueError("concept records currently require CSV input")
    with path.open(newline="", encoding="utf-8") as handle:
        return [ConceptRecord.from_mapping(row) for row in csv.DictReader(handle)]


def expand_concepts(
    concepts: list[ConceptRecord],
    *,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    styles: tuple[str, ...] = DEFAULT_STYLES,
) -> list[PromptRecord]:
    """Expand base concepts into prompt families.

    The generated `prompt_source` values are controlled source labels, not proof
    that a particular LLM wrote the prompt. Real LLM-generated prompts can use
    the same schema later.
    """

    prompts: list[PromptRecord] = []
    for concept in sorted(concepts, key=lambda item: item.concept_id):
        semantic_id = f"semantic_{concept.category}_{concept.subject.replace(' ', '_')}"
        for source in sources:
            for style in styles:
                prompt_text = render_prompt(concept, source=source, style=style)
                prompt_id = _prompt_id(concept.concept_id, source, style)
                prompts.append(
                    PromptRecord(
                        prompt_id=prompt_id,
                        family_id=f"family_{concept.concept_id}",
                        semantic_id=semantic_id,
                        prompt_text=prompt_text,
                        prompt_source=source,
                        prompt_category=concept.category,
                        prompt_style=style,
                    )
                )
    return prompts


def render_prompt(concept: ConceptRecord, *, source: str, style: str) -> str:
    attributes = _attributes(concept)
    attribute_text = ", ".join(attributes)
    if source == "human":
        base = f"{concept.subject} {concept.setting}"
    elif source == "llm_a":
        base = f"A detailed image of {concept.subject} {concept.setting}"
    elif source == "llm_b":
        base = f"Create a visually clear scene featuring {concept.subject} {concept.setting}"
    elif source == "llm_c":
        base = f"An analytical visual study of {concept.subject} {concept.setting}"
    else:
        base = f"{concept.subject} {concept.setting}"

    if style == "concise":
        return f"{base}, {attribute_text}".strip(", ")
    if style == "photorealistic":
        return (
            f"{base}, photorealistic, natural lighting, sharp detail, {attribute_text}"
        ).strip(", ")
    if style == "technical":
        return (
            f"{base}, controlled composition, neutral background, high-frequency detail, "
            f"{attribute_text}"
        ).strip(", ")
    if style == "cinematic":
        return f"{base}, cinematic lighting, rich contrast, {attribute_text}".strip(", ")
    if style == "illustration":
        return f"{base}, clean digital illustration, crisp edges, {attribute_text}".strip(", ")
    return f"{base}, {style}, {attribute_text}".strip(", ")


def write_prompt_records(path: Path, prompts: list[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROMPT_FIELDS)
        writer.writeheader()
        for prompt in prompts:
            writer.writerow({field: getattr(prompt, field) for field in PROMPT_FIELDS})


def write_concept_records(path: Path, concepts: list[ConceptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONCEPT_FIELDS)
        writer.writeheader()
        for concept in concepts:
            writer.writerow({field: getattr(concept, field) for field in CONCEPT_FIELDS})


def _attributes(concept: ConceptRecord) -> list[str]:
    return [item.strip() for item in concept.attributes.split(";") if item.strip()]


def _prompt_id(concept_id: str, source: str, style: str) -> str:
    digest = hashlib.sha1(f"{concept_id}:{source}:{style}".encode()).hexdigest()[:8]
    return f"prompt_{digest}"

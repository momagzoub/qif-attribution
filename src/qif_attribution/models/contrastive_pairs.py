"""SynCLR-style pair construction adapted for attribution experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from qif_attribution.data.manifest import ManifestRecord


@dataclass(frozen=True)
class ContrastivePair:
    anchor_id: str
    candidate_id: str
    relation: str


def build_attribution_pairs(
    records: list[ManifestRecord],
    *,
    content_key: str = "prompt_id",
    source_context_keys: tuple[str, ...] = ("prompt_category", "prompt_style"),
    max_pairs_per_anchor: int = 4,
    seed: int = 1729,
) -> list[ContrastivePair]:
    """Build deterministic content-positive and provenance-negative pairs.

    Relation meanings:

    - `content_positive`: same prompt/content group, different image.
    - `generator_hard_negative`: same content group but different generator.
    - `prompt_source_hard_negative`: same coarse context but different prompt source.

    The output is suitable for constructing multi-head contrastive losses later:
    one head can learn semantic invariance from positives, while attribution heads
    can use generator/source hard negatives.
    """

    if max_pairs_per_anchor <= 0:
        raise ValueError("max_pairs_per_anchor must be positive")

    pairs: list[ContrastivePair] = []
    ordered_records = sorted(records, key=lambda record: record.image_id)
    for anchor in ordered_records:
        candidates = [record for record in ordered_records if record.image_id != anchor.image_id]
        pairs.extend(
            _limited_pairs(
                anchor,
                candidates,
                relation="content_positive",
                predicate=lambda other, anchor=anchor: (
                    _value(other, content_key) == _value(anchor, content_key)
                ),
                max_pairs=max_pairs_per_anchor,
                seed=seed,
            )
        )
        pairs.extend(
            _limited_pairs(
                anchor,
                candidates,
                relation="generator_hard_negative",
                predicate=lambda other, anchor=anchor: (
                    _value(other, content_key) == _value(anchor, content_key)
                    and other.generator_id != anchor.generator_id
                ),
                max_pairs=max_pairs_per_anchor,
                seed=seed,
            )
        )
        pairs.extend(
            _limited_pairs(
                anchor,
                candidates,
                relation="prompt_source_hard_negative",
                predicate=lambda other, anchor=anchor: (
                    _same_values(anchor, other, source_context_keys)
                    and other.prompt_source != anchor.prompt_source
                ),
                max_pairs=max_pairs_per_anchor,
                seed=seed,
            )
        )
    return pairs


def relation_counts(pairs: list[ContrastivePair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.relation] = counts.get(pair.relation, 0) + 1
    return dict(sorted(counts.items()))


def _limited_pairs(
    anchor: ManifestRecord,
    candidates: list[ManifestRecord],
    *,
    relation: str,
    predicate,
    max_pairs: int,
    seed: int,
) -> list[ContrastivePair]:
    matches = [candidate for candidate in candidates if predicate(candidate)]
    matches = sorted(
        matches,
        key=lambda candidate: _stable_hash(
            f"{seed}:{relation}:{anchor.image_id}:{candidate.image_id}"
        ),
    )
    return [
        ContrastivePair(anchor.image_id, candidate.image_id, relation)
        for candidate in matches[:max_pairs]
    ]


def _same_values(
    anchor: ManifestRecord,
    other: ManifestRecord,
    keys: tuple[str, ...],
) -> bool:
    return all(_value(anchor, key) == _value(other, key) for key in keys)


def _value(record: ManifestRecord, key: str) -> str:
    return str(getattr(record, key))


def _stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

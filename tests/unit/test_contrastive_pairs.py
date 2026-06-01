from qif_attribution.models.contrastive_pairs import build_attribution_pairs, relation_counts
from tests.unit.test_manifest import make_record


def test_build_attribution_pairs_counts_core_relations() -> None:
    records = [
        make_record(
            image_id="a",
            sha256="a" * 64,
            prompt_id="p1",
            generator_id="g1",
            prompt_source="human",
        ),
        make_record(
            image_id="b",
            sha256="b" * 64,
            prompt_id="p1",
            generator_id="g2",
            prompt_source="human",
        ),
        make_record(
            image_id="c",
            sha256="c" * 64,
            prompt_id="p2",
            generator_id="g1",
            prompt_source="llm_a",
        ),
    ]

    pairs = build_attribution_pairs(records, max_pairs_per_anchor=2)
    counts = relation_counts(pairs)

    assert counts["content_positive"] == 2
    assert counts["generator_hard_negative"] == 2
    assert counts["prompt_source_hard_negative"] >= 2

from pathlib import Path

from qif_attribution.generation.prompt_expansion import (
    expand_concepts,
    load_concepts,
    render_prompt,
)


def test_expand_concepts_creates_source_style_grid() -> None:
    concepts = load_concepts(Path("data/manifests/pilot_concepts.csv"))

    prompts = expand_concepts(
        concepts[:1],
        sources=("human", "llm_a"),
        styles=("concise", "technical"),
    )

    assert len(prompts) == 4
    assert {prompt.prompt_source for prompt in prompts} == {"human", "llm_a"}
    assert {prompt.prompt_style for prompt in prompts} == {"concise", "technical"}
    assert len({prompt.prompt_id for prompt in prompts}) == 4


def test_render_prompt_preserves_concept_content() -> None:
    concept = load_concepts(Path("data/manifests/pilot_concepts.csv"))[0]

    prompt = render_prompt(concept, source="llm_b", style="photorealistic")

    assert "red cube" in prompt
    assert "on a table" in prompt
    assert "photorealistic" in prompt

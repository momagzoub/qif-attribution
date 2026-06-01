from pathlib import Path

from qif_attribution.generation.jobs import GeneratorSpec
from qif_attribution.generation.scale_plan import (
    SCALE_PROFILES,
    build_scaled_jobs,
    generate_scaled_concepts,
    write_scale_plan,
)


def make_generators() -> list[GeneratorSpec]:
    return [
        GeneratorSpec(
            generator_id=generator_id,
            generator_version="test",
            sampler="euler",
            steps="2",
            cfg_scale="1.0",
            width="64",
            height="64",
            model_id=f"test/{generator_id}",
            backend="placeholder",
        )
        for generator_id in SCALE_PROFILES["small"].generator_ids
    ]


def test_generate_scaled_concepts_balances_categories() -> None:
    concepts = generate_scaled_concepts(8)

    assert len(concepts) == 8
    assert {concept.category for concept in concepts} == {
        "architecture",
        "natural",
        "object",
        "scientific",
    }
    assert len({concept.concept_id for concept in concepts}) == 8


def test_small_profile_counts_are_factorial() -> None:
    profile = SCALE_PROFILES["small"]
    concepts, prompts, jobs = build_scaled_jobs(profile, make_generators())

    assert len(concepts) == 4
    assert len(prompts) == 64
    assert len(jobs) == 1024
    assert {job.generator_id for job in jobs} == set(profile.generator_ids)
    assert {job.seed for job in jobs} == {"0", "1", "2", "3"}


def test_write_scale_plan_writes_cluster_env(tmp_path: Path) -> None:
    profile = SCALE_PROFILES["small"]

    summary = write_scale_plan(
        tmp_path,
        profile=profile,
        generators=make_generators(),
        generator_spec_path="generators.csv",
    )

    assert summary["image_count"] == 1024
    assert (tmp_path / "concepts.csv").exists()
    assert (tmp_path / "prompts.csv").exists()
    assert (tmp_path / "jobs.csv").exists()
    assert (tmp_path / "summary.json").exists()
    env_text = (tmp_path / "slurm.env").read_text(encoding="utf-8")
    assert "export PROMPTS=" in env_text
    assert "export GENERATOR_IDS='sd15,sd21_base,openjourney_v4,dreamlike_photoreal_2'" in env_text

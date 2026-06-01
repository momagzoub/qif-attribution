from pathlib import Path

from qif_attribution.data.prompts import load_prompt_records
from qif_attribution.generation.jobs import (
    build_generation_jobs,
    load_generator_specs,
)


def test_build_generation_jobs_crosses_prompts_generators_and_seeds() -> None:
    prompts = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))[:2]
    generators = load_generator_specs(Path("data/manifests/example_generators.csv"))

    jobs = build_generation_jobs(prompts, generators, seeds=(7, 8))

    assert generators[0].model_id
    assert len(jobs) == 8
    assert all(job.relative_path.endswith(".png") for job in jobs)
    assert {job.seed for job in jobs} == {"7", "8"}
    assert {job.generator_id for job in jobs} == {"sd15", "sdxl"}

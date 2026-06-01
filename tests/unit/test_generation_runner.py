from pathlib import Path

from qif_attribution.data.prompts import load_prompt_records
from qif_attribution.generation.backends import PlaceholderImageBackend
from qif_attribution.generation.jobs import build_generation_jobs, load_generator_specs
from qif_attribution.generation.png import PNG_SIGNATURE
from qif_attribution.generation.runner import (
    filter_generation_jobs,
    run_generation_jobs,
    shard_generation_jobs,
)


def test_run_generation_jobs_with_placeholder_backend(tmp_path: Path) -> None:
    prompts = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))[:1]
    generators = load_generator_specs(Path("data/manifests/example_generators.csv"))[:1]
    jobs = build_generation_jobs(prompts, generators, seeds=(0, 1))

    summary = run_generation_jobs(
        prompts=prompts,
        jobs=jobs,
        generators=generators,
        image_root=tmp_path,
        backend=PlaceholderImageBackend(),
    )

    assert summary.ok
    assert summary.completed == 2
    assert summary.skipped == 0
    for job in jobs:
        assert (tmp_path / job.relative_path).read_bytes().startswith(PNG_SIGNATURE)


def test_run_generation_jobs_resumes_existing_outputs(tmp_path: Path) -> None:
    prompts = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))[:1]
    generators = load_generator_specs(Path("data/manifests/example_generators.csv"))[:1]
    jobs = build_generation_jobs(prompts, generators, seeds=(0,))
    existing = tmp_path / jobs[0].relative_path
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already here")

    summary = run_generation_jobs(
        prompts=prompts,
        jobs=jobs,
        generators=generators,
        image_root=tmp_path,
        backend=PlaceholderImageBackend(),
        resume=True,
    )

    assert summary.completed == 0
    assert summary.skipped == 1
    assert existing.read_bytes() == b"already here"


def test_filter_generation_jobs_by_generator_and_limit() -> None:
    prompts = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))[:2]
    generators = load_generator_specs(Path("data/manifests/example_generators.csv"))
    jobs = build_generation_jobs(prompts, generators, seeds=(0, 1))

    filtered = filter_generation_jobs(jobs, generator_ids=("sd15",), limit=2)

    assert len(filtered) == 2
    assert {job.generator_id for job in filtered} == {"sd15"}


def test_shard_generation_jobs_partitions_jobs() -> None:
    prompts = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))[:2]
    generators = load_generator_specs(Path("data/manifests/example_generators.csv"))
    jobs = build_generation_jobs(prompts, generators, seeds=(0, 1))

    shard_zero = shard_generation_jobs(jobs, shard_index=0, num_shards=2)
    shard_one = shard_generation_jobs(jobs, shard_index=1, num_shards=2)

    assert len(shard_zero) + len(shard_one) == len(jobs)
    assert {job.job_id for job in shard_zero}.isdisjoint({job.job_id for job in shard_one})


def test_filter_generation_jobs_applies_shard_before_limit() -> None:
    prompts = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))[:3]
    generators = load_generator_specs(Path("data/manifests/example_generators.csv"))
    jobs = build_generation_jobs(prompts, generators, seeds=(0, 1))

    filtered = filter_generation_jobs(jobs, shard_index=1, num_shards=2, limit=3)

    assert len(filtered) == 3
    assert filtered == shard_generation_jobs(jobs, shard_index=1, num_shards=2)[:3]

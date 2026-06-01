"""Resumable execution for text-to-image generation jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qif_attribution.data.prompts import PromptRecord
from qif_attribution.generation.backends import ImageGenerationBackend, ImageGenerationRequest
from qif_attribution.generation.jobs import GenerationJob, GeneratorSpec


@dataclass(frozen=True)
class JobFailure:
    job_id: str
    error: str


@dataclass(frozen=True)
class GenerationRunSummary:
    completed: int
    skipped: int
    failed: tuple[JobFailure, ...]

    @property
    def ok(self) -> bool:
        return not self.failed


def filter_generation_jobs(
    jobs: list[GenerationJob],
    *,
    generator_ids: tuple[str, ...] = (),
    prompt_ids: tuple[str, ...] = (),
    limit: int | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> list[GenerationJob]:
    filtered = jobs
    if generator_ids:
        allowed = set(generator_ids)
        filtered = [job for job in filtered if job.generator_id in allowed]
    if prompt_ids:
        allowed = set(prompt_ids)
        filtered = [job for job in filtered if job.prompt_id in allowed]
    if shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("shard_index and num_shards must be provided together")
        filtered = shard_generation_jobs(
            filtered,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        filtered = filtered[:limit]
    return filtered


def shard_generation_jobs(
    jobs: list[GenerationJob],
    *,
    shard_index: int,
    num_shards: int,
) -> list[GenerationJob]:
    """Return one deterministic shard from an ordered job list."""

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [job for index, job in enumerate(jobs) if index % num_shards == shard_index]


def run_generation_jobs(
    *,
    prompts: list[PromptRecord],
    jobs: list[GenerationJob],
    generators: list[GeneratorSpec],
    image_root: Path,
    backend: ImageGenerationBackend,
    resume: bool = True,
    stop_on_error: bool = False,
) -> GenerationRunSummary:
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    generator_by_id = {generator.generator_id: generator for generator in generators}
    completed = 0
    skipped = 0
    failures: list[JobFailure] = []

    for job in jobs:
        output_path = image_root / job.relative_path
        if resume and output_path.exists():
            skipped += 1
            continue

        try:
            prompt = prompt_by_id[job.prompt_id]
            generator = generator_by_id[job.generator_id]
            request = _request_from_job(prompt, job, generator, output_path)
            backend.generate(request)
            completed += 1
        except Exception as exc:
            failures.append(JobFailure(job.job_id, str(exc)))
            if stop_on_error:
                break

    return GenerationRunSummary(
        completed=completed,
        skipped=skipped,
        failed=tuple(failures),
    )


def _request_from_job(
    prompt: PromptRecord,
    job: GenerationJob,
    generator: GeneratorSpec,
    output_path: Path,
) -> ImageGenerationRequest:
    model_id = generator.model_id or generator.generator_id
    return ImageGenerationRequest(
        job_id=job.job_id,
        prompt=prompt.prompt_text,
        generator_id=job.generator_id,
        model_id=model_id,
        seed=int(job.seed),
        sampler=job.sampler,
        steps=int(job.steps),
        cfg_scale=float(job.cfg_scale),
        width=int(job.width),
        height=int(job.height),
        output_path=output_path,
    )

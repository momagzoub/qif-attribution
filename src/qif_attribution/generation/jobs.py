"""Generation job specifications for text-to-image experiments."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qif_attribution.data.prompts import PromptRecord

GENERATOR_FIELDS = (
    "generator_id",
    "generator_version",
    "sampler",
    "steps",
    "cfg_scale",
    "width",
    "height",
)

OPTIONAL_GENERATOR_FIELDS = ("model_id", "backend")

JOB_FIELDS = (
    "job_id",
    "prompt_id",
    "generator_id",
    "generator_version",
    "seed",
    "sampler",
    "steps",
    "cfg_scale",
    "width",
    "height",
    "relative_path",
)


@dataclass(frozen=True)
class GeneratorSpec:
    generator_id: str
    generator_version: str
    sampler: str
    steps: str
    cfg_scale: str
    width: str
    height: str
    model_id: str = ""
    backend: str = "diffusers"

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> GeneratorSpec:
        missing = [field for field in GENERATOR_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing generator fields: {', '.join(missing)}")
        values = {field: str(row[field]).strip() for field in GENERATOR_FIELDS}
        values.update(
            {
                field: str(row.get(field, "")).strip()
                for field in OPTIONAL_GENERATOR_FIELDS
            }
        )
        return cls(**values)


@dataclass(frozen=True)
class GenerationJob:
    job_id: str
    prompt_id: str
    generator_id: str
    generator_version: str
    seed: str
    sampler: str
    steps: str
    cfg_scale: str
    width: str
    height: str
    relative_path: str

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> GenerationJob:
        missing = [field for field in JOB_FIELDS if field not in row]
        if missing:
            raise ValueError(f"missing job fields: {', '.join(missing)}")
        values = {field: str(row[field]).strip() for field in JOB_FIELDS}
        return cls(**values)


def load_generator_specs(path: Path) -> list[GeneratorSpec]:
    if path.suffix.lower() != ".csv":
        raise ValueError("generator specs currently require CSV input")
    with path.open(newline="", encoding="utf-8") as handle:
        return [GeneratorSpec.from_mapping(row) for row in csv.DictReader(handle)]


def load_generation_jobs(path: Path) -> list[GenerationJob]:
    if path.suffix.lower() != ".csv":
        raise ValueError("generation jobs currently require CSV input")
    with path.open(newline="", encoding="utf-8") as handle:
        return [GenerationJob.from_mapping(row) for row in csv.DictReader(handle)]


def build_generation_jobs(
    prompts: list[PromptRecord],
    generators: list[GeneratorSpec],
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    output_prefix: str = "raw",
) -> list[GenerationJob]:
    jobs: list[GenerationJob] = []
    for prompt in sorted(prompts, key=lambda item: item.prompt_id):
        for generator in sorted(generators, key=lambda item: item.generator_id):
            for seed in seeds:
                job_id = _job_id(prompt.prompt_id, generator.generator_id, str(seed))
                relative_path = (
                    f"{output_prefix}/{generator.generator_id}/{prompt.prompt_id}/"
                    f"seed_{seed:06d}.png"
                )
                jobs.append(
                    GenerationJob(
                        job_id=job_id,
                        prompt_id=prompt.prompt_id,
                        generator_id=generator.generator_id,
                        generator_version=generator.generator_version,
                        seed=str(seed),
                        sampler=generator.sampler,
                        steps=generator.steps,
                        cfg_scale=generator.cfg_scale,
                        width=generator.width,
                        height=generator.height,
                        relative_path=relative_path,
                    )
                )
    return jobs


def write_generation_jobs(path: Path, jobs: list[GenerationJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_FIELDS)
        writer.writeheader()
        for job in jobs:
            writer.writerow({field: getattr(job, field) for field in JOB_FIELDS})


def _job_id(prompt_id: str, generator_id: str, seed: str) -> str:
    digest = hashlib.sha1(f"{prompt_id}:{generator_id}:{seed}".encode()).hexdigest()[:10]
    return f"job_{digest}"

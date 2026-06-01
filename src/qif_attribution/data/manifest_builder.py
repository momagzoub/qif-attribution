"""Build attribution manifests from completed generation jobs."""

from __future__ import annotations

import csv
from pathlib import Path

from qif_attribution.data.manifest import REQUIRED_FIELDS, ManifestRecord
from qif_attribution.data.prompts import PromptRecord
from qif_attribution.generation.jobs import GenerationJob
from qif_attribution.utils.hashing import sha256_file


def build_manifest_from_jobs(
    prompts: list[PromptRecord],
    jobs: list[GenerationJob],
    *,
    image_root: Path,
    postprocess: str = "none",
    license_notes: str = "generated for controlled research dataset",
    skip_missing: bool = False,
) -> list[ManifestRecord]:
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    records: list[ManifestRecord] = []
    for job in jobs:
        if job.prompt_id not in prompt_by_id:
            raise ValueError(f"job {job.job_id} references unknown prompt_id {job.prompt_id}")

        image_path = image_root / job.relative_path
        if not image_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"missing generated image for job {job.job_id}: {image_path}")

        prompt = prompt_by_id[job.prompt_id]
        records.append(
            ManifestRecord(
                image_id=job.job_id,
                sha256=sha256_file(image_path),
                prompt_id=prompt.prompt_id,
                prompt_text=prompt.prompt_text,
                prompt_source=prompt.prompt_source,
                prompt_category=prompt.prompt_category,
                prompt_style=prompt.prompt_style,
                generator_id=job.generator_id,
                generator_version=job.generator_version,
                seed=job.seed,
                sampler=job.sampler,
                steps=job.steps,
                cfg_scale=job.cfg_scale,
                width=job.width,
                height=job.height,
                postprocess=postprocess,
                license_notes=license_notes,
                relative_path=job.relative_path,
            )
        )
    return records


def write_manifest(path: Path, records: list[ManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_dict())

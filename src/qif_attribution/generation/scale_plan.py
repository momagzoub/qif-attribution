"""Small, medium, and large controlled dataset plans."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qif_attribution.generation.jobs import (
    GenerationJob,
    GeneratorSpec,
    build_generation_jobs,
    write_generation_jobs,
)
from qif_attribution.generation.prompt_expansion import (
    ConceptRecord,
    expand_concepts,
    write_concept_records,
    write_prompt_records,
)


@dataclass(frozen=True)
class ScaleProfile:
    name: str
    concept_count: int
    sources: tuple[str, ...]
    styles: tuple[str, ...]
    generator_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    jobs_per_shard_per_generator: int
    description: str

    @property
    def prompt_count(self) -> int:
        return self.concept_count * len(self.sources) * len(self.styles)

    @property
    def image_count(self) -> int:
        return self.prompt_count * len(self.generator_ids) * len(self.seeds)

    @property
    def images_per_generator(self) -> int:
        return self.prompt_count * len(self.seeds)

    @property
    def shards_per_generator(self) -> int:
        return max(1, math.ceil(self.images_per_generator / self.jobs_per_shard_per_generator))

    @property
    def total_slurm_tasks(self) -> int:
        return self.shards_per_generator * len(self.generator_ids)


PROMPT_SOURCES = ("human", "llm_a", "llm_b", "llm_c")
PROMPT_STYLES = ("concise", "photorealistic", "technical", "cinematic")
CONTROLLED_GENERATOR_IDS = (
    "sd15",
    "sd21_base",
    "openjourney_v4",
    "dreamlike_photoreal_2",
)

SCALE_PROFILES: dict[str, ScaleProfile] = {
    "small": ScaleProfile(
        name="small",
        concept_count=4,
        sources=PROMPT_SOURCES,
        styles=PROMPT_STYLES,
        generator_ids=CONTROLLED_GENERATOR_IDS,
        seeds=(0, 1, 2, 3),
        jobs_per_shard_per_generator=64,
        description="1K-image sanity dataset for end-to-end generation and baseline debugging.",
    ),
    "medium": ScaleProfile(
        name="medium",
        concept_count=40,
        sources=PROMPT_SOURCES,
        styles=PROMPT_STYLES,
        generator_ids=CONTROLLED_GENERATOR_IDS,
        seeds=(0, 1, 2, 3),
        jobs_per_shard_per_generator=128,
        description="10K-image research dataset for first source-attribution experiments.",
    ),
    "large": ScaleProfile(
        name="large",
        concept_count=200,
        sources=PROMPT_SOURCES,
        styles=PROMPT_STYLES,
        generator_ids=CONTROLLED_GENERATOR_IDS,
        seeds=(0, 1, 2, 3, 4, 5, 6, 7),
        jobs_per_shard_per_generator=256,
        description=(
            "100K-image dataset for robust attribution, open-set tests, and quantum analysis."
        ),
    ),
}

_CATEGORY_BANKS: dict[str, dict[str, tuple[str, ...]]] = {
    "object": {
        "subjects": (
            "red cube",
            "blue sphere",
            "brass key",
            "ceramic mug",
            "transparent bottle",
            "wooden chair",
            "silver watch",
            "folded paper crane",
        ),
        "settings": (
            "on a table",
            "inside a lightbox",
            "beside a window",
            "on a concrete floor",
            "against a neutral wall",
            "on a reflective tray",
        ),
        "attributes": (
            "matte surface",
            "glossy highlight",
            "simple composition",
            "visible texture",
            "centered framing",
            "soft shadow",
        ),
    },
    "architecture": {
        "subjects": (
            "glass house",
            "brick courtyard",
            "concrete library",
            "steel bridge",
            "small chapel",
            "modern apartment facade",
            "wooden pavilion",
            "train station canopy",
        ),
        "settings": (
            "in a forest",
            "at sunrise",
            "during overcast weather",
            "near a city street",
            "beside shallow water",
            "in winter light",
        ),
        "attributes": (
            "modern design",
            "clean geometry",
            "repeating windows",
            "strong perspective",
            "material contrast",
            "human-scale details",
        ),
    },
    "scientific": {
        "subjects": (
            "small lab robot",
            "microscope slide",
            "fluid simulation display",
            "quantum optics bench",
            "circuit board",
            "battery test rig",
            "medical sensor",
            "spectrometer module",
        ),
        "settings": (
            "on a workbench",
            "inside a laboratory",
            "under controlled lighting",
            "beside measurement equipment",
            "on an anti-static mat",
            "near a calibration chart",
        ),
        "attributes": (
            "visible sensors",
            "metal body",
            "precise alignment",
            "labeled components",
            "cable details",
            "clean background",
        ),
    },
    "natural": {
        "subjects": (
            "single maple leaf",
            "smooth river stone",
            "mushroom cluster",
            "pine cone",
            "desert succulent",
            "shell fragment",
            "snow-covered branch",
            "flower bud",
        ),
        "settings": (
            "on dark soil",
            "beside shallow water",
            "on a wooden surface",
            "in diffuse daylight",
            "against moss",
            "on pale sand",
        ),
        "attributes": (
            "fine texture",
            "natural color",
            "macro detail",
            "subtle shadow",
            "asymmetric shape",
            "organic pattern",
        ),
    },
}


def get_scale_profile(name: str) -> ScaleProfile:
    try:
        return SCALE_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCALE_PROFILES))
        message = f"unknown scale profile '{name}'. Available profiles: {available}"
        raise ValueError(message) from exc


def generate_scaled_concepts(count: int) -> list[ConceptRecord]:
    """Generate deterministic concept seeds balanced across content categories."""

    if count <= 0:
        raise ValueError("count must be positive")

    categories = tuple(sorted(_CATEGORY_BANKS))
    concepts: list[ConceptRecord] = []
    category_offsets = {category: 0 for category in categories}
    for index in range(count):
        category = categories[index % len(categories)]
        local_index = category_offsets[category]
        category_offsets[category] += 1
        bank = _CATEGORY_BANKS[category]
        subject = bank["subjects"][local_index % len(bank["subjects"])]
        setting = bank["settings"][(local_index // len(bank["subjects"])) % len(bank["settings"])]
        attr_a = bank["attributes"][local_index % len(bank["attributes"])]
        attr_b = bank["attributes"][(local_index + 2) % len(bank["attributes"])]
        concepts.append(
            ConceptRecord(
                concept_id=f"{category}_{local_index:04d}",
                category=category,
                subject=subject,
                setting=setting,
                attributes=f"{attr_a};{attr_b}",
            )
        )
    return concepts


def select_generators(
    generators: list[GeneratorSpec],
    generator_ids: tuple[str, ...],
) -> list[GeneratorSpec]:
    by_id = {generator.generator_id: generator for generator in generators}
    missing = [generator_id for generator_id in generator_ids if generator_id not in by_id]
    if missing:
        missing_ids = ", ".join(missing)
        raise ValueError(f"generator file is missing required generator_id(s): {missing_ids}")
    return [by_id[generator_id] for generator_id in generator_ids]


def build_scaled_jobs(
    profile: ScaleProfile,
    generators: list[GeneratorSpec],
    *,
    output_prefix: str = "raw",
) -> tuple[list[ConceptRecord], list[Any], list[GenerationJob]]:
    concepts = generate_scaled_concepts(profile.concept_count)
    prompts = expand_concepts(concepts, sources=profile.sources, styles=profile.styles)
    selected_generators = select_generators(generators, profile.generator_ids)
    jobs = build_generation_jobs(
        prompts,
        selected_generators,
        seeds=profile.seeds,
        output_prefix=output_prefix,
    )
    return concepts, prompts, jobs


def write_scale_plan(
    output_dir: Path,
    *,
    profile: ScaleProfile,
    generators: list[GeneratorSpec],
    output_prefix: str = "raw",
    generator_spec_path: str,
) -> dict[str, Any]:
    concepts, prompts, jobs = build_scaled_jobs(
        profile,
        generators,
        output_prefix=output_prefix,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    concept_path = output_dir / "concepts.csv"
    prompt_path = output_dir / "prompts.csv"
    job_path = output_dir / "jobs.csv"
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"
    env_path = output_dir / "slurm.env"

    write_concept_records(concept_path, concepts)
    write_prompt_records(prompt_path, prompts)
    write_generation_jobs(job_path, jobs)

    summary = scale_summary(
        profile,
        concept_path=concept_path,
        prompt_path=prompt_path,
        job_path=job_path,
        manifest_path=manifest_path,
        generator_spec_path=generator_spec_path,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env_path.write_text(format_slurm_env(summary), encoding="utf-8")
    return summary


def scale_summary(
    profile: ScaleProfile,
    *,
    concept_path: Path,
    prompt_path: Path,
    job_path: Path,
    manifest_path: Path,
    generator_spec_path: str,
) -> dict[str, Any]:
    estimated_storage_gb = round(profile.image_count * 1.0 / 1024, 1)
    return {
        "profile": profile.name,
        "description": profile.description,
        "concept_count": profile.concept_count,
        "prompt_count": profile.prompt_count,
        "image_count": profile.image_count,
        "prompt_sources": list(profile.sources),
        "prompt_styles": list(profile.styles),
        "generator_ids": list(profile.generator_ids),
        "seed_count": len(profile.seeds),
        "seeds": list(profile.seeds),
        "images_per_generator": profile.images_per_generator,
        "jobs_per_shard_per_generator": profile.jobs_per_shard_per_generator,
        "shards_per_generator": profile.shards_per_generator,
        "total_slurm_tasks": profile.total_slurm_tasks,
        "estimated_png_storage_gb": estimated_storage_gb,
        "concepts": str(concept_path),
        "prompts": str(prompt_path),
        "jobs": str(job_path),
        "manifest": str(manifest_path),
        "generators": generator_spec_path,
    }


def format_profile_table() -> str:
    lines = [
        "profile concepts prompts images generators seeds shards_per_generator total_tasks",
    ]
    for profile in SCALE_PROFILES.values():
        lines.append(
            " ".join(
                (
                    profile.name,
                    str(profile.concept_count),
                    str(profile.prompt_count),
                    str(profile.image_count),
                    str(len(profile.generator_ids)),
                    str(len(profile.seeds)),
                    str(profile.shards_per_generator),
                    str(profile.total_slurm_tasks),
                )
            )
        )
    return "\n".join(lines)


def format_slurm_env(summary: dict[str, Any]) -> str:
    generator_ids = ",".join(summary["generator_ids"])
    return "\n".join(
        (
            f"export PROMPTS={_shell_quote(summary['prompts'])}",
            f"export JOBS={_shell_quote(summary['jobs'])}",
            f"export GENERATORS={_shell_quote(summary['generators'])}",
            "export IMAGE_ROOT=.",
            f"export MANIFEST_OUT={_shell_quote(summary['manifest'])}",
            f"export NUM_SHARDS={summary['shards_per_generator']}",
            f"export GENERATOR_IDS={_shell_quote(generator_ids)}",
            "",
        )
    )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"

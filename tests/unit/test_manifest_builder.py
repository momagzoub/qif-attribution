from pathlib import Path

from qif_attribution.data.manifest import validate_manifest
from qif_attribution.data.manifest_builder import build_manifest_from_jobs
from qif_attribution.data.prompts import load_prompt_records
from qif_attribution.generation.jobs import build_generation_jobs, load_generator_specs


def test_build_manifest_from_completed_jobs(tmp_path: Path) -> None:
    prompts = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))[:1]
    generators = load_generator_specs(Path("data/manifests/example_generators.csv"))[:1]
    jobs = build_generation_jobs(prompts, generators, seeds=(3,))
    image_path = tmp_path / jobs[0].relative_path
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake png bytes")

    records = build_manifest_from_jobs(prompts, jobs, image_root=tmp_path)

    assert len(records) == 1
    assert records[0].prompt_id == prompts[0].prompt_id
    assert records[0].generator_id == generators[0].generator_id
    assert validate_manifest(records).ok


def test_build_manifest_can_skip_missing_jobs(tmp_path: Path) -> None:
    prompts = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))[:1]
    generators = load_generator_specs(Path("data/manifests/example_generators.csv"))[:1]
    jobs = build_generation_jobs(prompts, generators, seeds=(3,))

    records = build_manifest_from_jobs(prompts, jobs, image_root=tmp_path, skip_missing=True)

    assert records == []

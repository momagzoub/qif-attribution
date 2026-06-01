from pathlib import Path

from qif_attribution.data.prompts import (
    load_prompt_records,
    prompt_family_counts,
    validate_prompt_records,
)


def test_load_and_validate_prompt_records() -> None:
    records = load_prompt_records(Path("data/manifests/example_prompt_families.csv"))

    assert validate_prompt_records(records) == ()
    assert prompt_family_counts(records) == {"family_cube": 2, "family_sphere": 2}

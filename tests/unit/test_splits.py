import csv

from qif_attribution.data.splits import (
    audit_split_leakage,
    group_split,
    write_split_assignments,
)
from tests.unit.test_manifest import make_record


def test_group_split_keeps_prompt_ids_together() -> None:
    records = []
    for prompt_index in range(10):
        for replica in range(2):
            records.append(
                make_record(
                    image_id=f"img_{prompt_index}_{replica}",
                    sha256=f"{prompt_index:02x}{replica:02x}" + "a" * 60,
                    prompt_id=f"prompt_{prompt_index}",
                    seed=str(replica),
                )
            )

    assignments = group_split(records, group_key="prompt_id", seed=1)

    by_prompt: dict[str, set[str]] = {}
    for record in records:
        by_prompt.setdefault(record.prompt_id, set()).add(assignments[record.image_id])

    assert all(len(splits) == 1 for splits in by_prompt.values())


def test_audit_detects_seed_leakage() -> None:
    records = [
        make_record(image_id="a", sha256="a" * 64, prompt_id="p1", seed="123"),
        make_record(image_id="b", sha256="b" * 64, prompt_id="p2", seed="123"),
    ]
    assignments = {"a": "train", "b": "test"}

    leakage = audit_split_leakage(records, assignments, keys=("seed",))

    assert leakage == {"seed": ["123"]}


def test_audit_does_not_flag_shared_seed_values_by_default() -> None:
    records = [
        make_record(image_id="a", sha256="a" * 64, prompt_id="p1", seed="123"),
        make_record(image_id="b", sha256="b" * 64, prompt_id="p2", seed="123"),
    ]
    assignments = {"a": "train", "b": "test"}

    leakage = audit_split_leakage(records, assignments)

    assert leakage == {}


def test_write_split_assignments_writes_rows_with_group_key(tmp_path) -> None:
    records = [
        make_record(image_id="a", sha256="a" * 64, prompt_id="p1", seed="0"),
        make_record(image_id="b", sha256="b" * 64, prompt_id="p2", seed="0"),
    ]
    assignments = {"a": "train", "b": "test"}
    path = tmp_path / "nested" / "splits.csv"

    count = write_split_assignments(path, records, assignments, group_key="prompt_id")

    assert count == 2
    assert path.exists()
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"image_id": "a", "prompt_id": "p1", "split": "train"},
        {"image_id": "b", "prompt_id": "p2", "split": "test"},
    ]


def test_write_split_assignments_omits_group_key_when_image_id(tmp_path) -> None:
    records = [make_record(image_id="a", sha256="a" * 64, prompt_id="p1", seed="0")]
    assignments = {"a": "val"}
    path = tmp_path / "splits.csv"

    write_split_assignments(path, records, assignments, group_key="image_id")

    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["image_id", "split"]
        assert list(reader) == [{"image_id": "a", "split": "val"}]

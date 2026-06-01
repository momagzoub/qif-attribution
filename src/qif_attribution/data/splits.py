"""Deterministic group-aware splits and leakage audits."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from qif_attribution.data.manifest import ManifestRecord

SplitName = Literal["train", "val", "test"]


def group_split(
    records: list[ManifestRecord],
    *,
    group_key: str = "prompt_id",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 1729,
) -> dict[str, SplitName]:
    """Assign records to splits without separating members of the same group."""

    _validate_fractions(train_fraction, val_fraction, test_fraction)
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, group_key))].append(record)

    ordered_groups = sorted(groups, key=lambda value: _stable_hash(f"{seed}:{value}"))
    train_cut = int(round(len(ordered_groups) * train_fraction))
    val_cut = train_cut + int(round(len(ordered_groups) * val_fraction))

    assignments: dict[str, SplitName] = {}
    for index, group in enumerate(ordered_groups):
        if index < train_cut:
            split: SplitName = "train"
        elif index < val_cut:
            split = "val"
        else:
            split = "test"
        for record in groups[group]:
            assignments[record.image_id] = split
    return assignments


def audit_split_leakage(
    records: Iterable[ManifestRecord],
    assignments: dict[str, SplitName],
    *,
    keys: tuple[str, ...] = ("prompt_id", "sha256"),
) -> dict[str, list[str]]:
    """Return values that occur in more than one split for each audited key.

    Seeds are intentionally not part of the default audit because controlled
    generation commonly reuses the same seed grid for every prompt.
    """

    buckets: dict[str, dict[str, set[SplitName]]] = {
        key: defaultdict(set) for key in keys
    }
    for record in records:
        split = assignments[record.image_id]
        for key in keys:
            buckets[key][str(getattr(record, key))].add(split)

    leakage: dict[str, list[str]] = {}
    for key, value_splits in buckets.items():
        leaked = sorted(value for value, splits in value_splits.items() if len(splits) > 1)
        if leaked:
            leakage[key] = leaked
    return leakage


def write_split_assignments(
    path: Path,
    records: Iterable[ManifestRecord],
    assignments: dict[str, SplitName],
    *,
    group_key: str = "prompt_id",
) -> int:
    """Write per-image split assignments to CSV, returning the row count."""

    fieldnames = ["image_id"]
    if group_key != "image_id":
        fieldnames.append(group_key)
    fieldnames.append("split")

    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {"image_id": record.image_id, "split": assignments[record.image_id]}
            if group_key != "image_id":
                row[group_key] = str(getattr(record, group_key))
            writer.writerow(row)
            count += 1
    return count


def _stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _validate_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> None:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")
    for name, value in {
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
    }.items():
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1")

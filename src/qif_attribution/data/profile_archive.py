"""Archive profile manifests and their referenced images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from qif_attribution.data.manifest import load_manifest

PROFILE_FILE_NAMES = (
    "concepts.csv",
    "prompts.csv",
    "jobs.csv",
    "manifest.csv",
    "summary.json",
    "slurm.env",
)


@dataclass(frozen=True)
class ProfileArchiveSummary:
    output: Path
    records: int
    images: int
    metadata_files: int
    missing_images: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_images


def write_profile_archive(
    *,
    profile_dir: Path,
    image_root: Path,
    output: Path,
    allow_missing: bool = False,
) -> ProfileArchiveSummary:
    """Write a zip containing profile metadata and images referenced by its manifest."""

    manifest_path = profile_dir / "manifest.csv"
    records = load_manifest(manifest_path)
    metadata_paths = [
        profile_dir / name for name in PROFILE_FILE_NAMES if (profile_dir / name).exists()
    ]
    image_paths = [image_root / record.relative_path for record in records]
    missing = tuple(
        record.relative_path
        for record, image_path in zip(records, image_paths, strict=True)
        if not image_path.exists()
    )
    if missing and not allow_missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"profile archive is missing {len(missing)} image(s): {preview}")

    output.parent.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    image_count = 0
    metadata_count = 0
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in metadata_paths:
            archive_name = _archive_name(path, image_root)
            if _should_skip_archive_name(archive_name):
                continue
            if archive_name not in written:
                archive.write(path, archive_name)
                written.add(archive_name)
                metadata_count += 1
        for image_path in image_paths:
            if not image_path.exists():
                continue
            archive_name = _archive_name(image_path, image_root)
            if archive_name not in written:
                archive.write(image_path, archive_name)
                written.add(archive_name)
                image_count += 1

    return ProfileArchiveSummary(
        output=output,
        records=len(records),
        images=image_count,
        metadata_files=metadata_count,
        missing_images=missing,
    )


def _archive_name(path: Path, image_root: Path) -> str:
    resolved_root = image_root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(path)


def _should_skip_archive_name(name: str) -> bool:
    parts = Path(name).parts
    return any(part == ".DS_Store" or part.startswith("._") for part in parts)

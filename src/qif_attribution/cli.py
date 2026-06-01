"""Command-line entry points for dataset and experiment hygiene."""

from __future__ import annotations

import argparse
from pathlib import Path

from qif_attribution.data.audit import audit_dataset, format_audit
from qif_attribution.data.manifest import load_manifest, validate_manifest
from qif_attribution.data.manifest_builder import build_manifest_from_jobs, write_manifest
from qif_attribution.data.profile_archive import write_profile_archive
from qif_attribution.data.prompts import load_prompt_records, validate_prompt_records
from qif_attribution.data.splits import (
    audit_split_leakage,
    group_split,
    write_split_assignments,
)
from qif_attribution.eval.baseline import (
    format_baseline_report,
    run_baseline_from_paths,
    write_baseline_report,
)
from qif_attribution.features.extract import (
    extract_density_matrix_features,
    extract_manifest_features,
    write_feature_bundle,
)
from qif_attribution.generation.backends import make_backend
from qif_attribution.generation.jobs import (
    build_generation_jobs,
    load_generation_jobs,
    load_generator_specs,
    write_generation_jobs,
)
from qif_attribution.generation.prompt_expansion import (
    expand_concepts,
    load_concepts,
    write_prompt_records,
)
from qif_attribution.generation.runner import filter_generation_jobs, run_generation_jobs
from qif_attribution.generation.scale_plan import (
    format_profile_table,
    get_scale_profile,
    write_scale_plan,
)
from qif_attribution.models.contrastive_pairs import build_attribution_pairs, relation_counts


def _validate_manifest(args: argparse.Namespace) -> int:
    records = load_manifest(Path(args.manifest))
    report = validate_manifest(records)
    if report.ok:
        print(f"manifest ok: {len(records)} records")
        return 0

    print(f"manifest invalid: {len(report.errors)} error(s)")
    for error in report.errors:
        print(f"- {error}")
    return 1


def _make_splits(args: argparse.Namespace) -> int:
    records = load_manifest(Path(args.manifest))
    assignments = group_split(
        records,
        group_key=args.group_by,
        train_fraction=args.train,
        val_fraction=args.val,
        test_fraction=args.test,
        seed=args.seed,
    )
    leakage_keys = tuple(_split_csv_arg(args.leakage_keys))
    leakage = audit_split_leakage(records, assignments, keys=leakage_keys)
    if leakage:
        print("split leakage detected")
        for key, values in leakage.items():
            print(f"- {key}: {', '.join(values[:5])}")
        return 1

    counts = {split: list(assignments.values()).count(split) for split in ("train", "val", "test")}
    if args.output:
        written = write_split_assignments(
            Path(args.output), records, assignments, group_key=args.group_by
        )
        print(f"wrote {written} split assignments to {args.output}")
    print(f"splits ok: train={counts['train']} val={counts['val']} test={counts['test']}")
    return 0


def _audit_manifest(args: argparse.Namespace) -> int:
    records = load_manifest(Path(args.manifest))
    image_root = Path(args.image_root) if args.image_root else None
    audit = audit_dataset(records, image_root=image_root)
    print(format_audit(audit))
    return 0 if audit.ok else 1


def _pair_summary(args: argparse.Namespace) -> int:
    records = load_manifest(Path(args.manifest))
    pairs = build_attribution_pairs(
        records,
        content_key=args.content_key,
        max_pairs_per_anchor=args.max_pairs_per_anchor,
        seed=args.seed,
    )
    counts = relation_counts(pairs)
    print(f"pairs: {len(pairs)}")
    for relation, count in counts.items():
        print(f"{relation}: {count}")
    return 0


def _expand_prompts(args: argparse.Namespace) -> int:
    concepts = load_concepts(Path(args.concepts))
    sources = _split_csv_arg(args.sources)
    styles = _split_csv_arg(args.styles)
    prompts = expand_concepts(concepts, sources=sources, styles=styles)
    errors = validate_prompt_records(prompts)
    if errors:
        print(f"prompt expansion invalid: {len(errors)} error(s)")
        for error in errors:
            print(f"- {error}")
        return 1
    write_prompt_records(Path(args.output), prompts)
    print(f"wrote {len(prompts)} prompts to {args.output}")
    return 0


def _build_jobs(args: argparse.Namespace) -> int:
    prompts = load_prompt_records(Path(args.prompts))
    generators = load_generator_specs(Path(args.generators))
    seeds = tuple(int(value) for value in _split_csv_arg(args.seeds))
    jobs = build_generation_jobs(
        prompts,
        generators,
        seeds=seeds,
        output_prefix=args.output_prefix,
    )
    write_generation_jobs(Path(args.output), jobs)
    print(f"wrote {len(jobs)} jobs to {args.output}")
    return 0


def _build_manifest(args: argparse.Namespace) -> int:
    prompts = load_prompt_records(Path(args.prompts))
    jobs = load_generation_jobs(Path(args.jobs))
    records = build_manifest_from_jobs(
        prompts,
        jobs,
        image_root=Path(args.image_root),
        postprocess=args.postprocess,
        license_notes=args.license_notes,
        skip_missing=args.skip_missing,
    )
    write_manifest(Path(args.output), records)
    print(f"wrote {len(records)} manifest records to {args.output}")
    return 0


def _run_generation(args: argparse.Namespace) -> int:
    prompts = load_prompt_records(Path(args.prompts))
    jobs = load_generation_jobs(Path(args.jobs))
    generators = load_generator_specs(Path(args.generators))
    selected_jobs = filter_generation_jobs(
        jobs,
        generator_ids=tuple(args.generator_id or ()),
        prompt_ids=tuple(args.prompt_id or ()),
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    backend = make_backend(
        args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )
    summary = run_generation_jobs(
        prompts=prompts,
        jobs=selected_jobs,
        generators=generators,
        image_root=Path(args.image_root),
        backend=backend,
        resume=not args.no_resume,
        stop_on_error=args.stop_on_error,
    )
    print(
        f"generation complete: completed={summary.completed} "
        f"skipped={summary.skipped} failed={len(summary.failed)}"
    )
    for failure in summary.failed[:10]:
        print(f"- {failure.job_id}: {failure.error}")
    return 0 if summary.ok else 1


def _list_scale_profiles(_: argparse.Namespace) -> int:
    print(format_profile_table())
    return 0


def _build_scale_plan(args: argparse.Namespace) -> int:
    profile = get_scale_profile(args.profile)
    generators = load_generator_specs(Path(args.generators))
    summary = write_scale_plan(
        Path(args.output_dir),
        profile=profile,
        generators=generators,
        output_prefix=args.output_prefix,
        generator_spec_path=args.generators,
    )
    print(
        f"scale plan '{profile.name}' ok: "
        f"prompts={summary['prompt_count']} images={summary['image_count']} "
        f"shards_per_generator={summary['shards_per_generator']} "
        f"total_tasks={summary['total_slurm_tasks']}"
    )
    print(f"wrote plan files to {args.output_dir}")
    return 0


def _package_profile(args: argparse.Namespace) -> int:
    summary = write_profile_archive(
        profile_dir=Path(args.profile_dir),
        image_root=Path(args.image_root),
        output=Path(args.output),
        allow_missing=args.allow_missing,
    )
    print(
        f"wrote {summary.output}: records={summary.records} "
        f"images={summary.images} metadata_files={summary.metadata_files}"
    )
    if summary.missing_images:
        print(f"missing_images={len(summary.missing_images)}")
        for missing in summary.missing_images[:10]:
            print(f"- {missing}")
    return 0 if summary.ok or args.allow_missing else 1


def _extract_features(args: argparse.Namespace) -> int:
    records = load_manifest(Path(args.manifest))
    image_root = Path(args.image_root)
    if args.feature == "density-matrix":
        result = extract_density_matrix_features(
            records,
            image_root=image_root,
            patch_size=args.patch_size,
            top_eigenvalues=args.top_eigenvalues,
            color=args.color,
            highpass=args.highpass,
            highpass_radius=args.highpass_radius,
            limit=args.limit,
            skip_missing=args.skip_missing,
        )
    else:
        result = extract_manifest_features(
            records,
            image_root=image_root,
            bins=args.bins,
            include_phase=not args.no_phase,
            limit=args.limit,
            skip_missing=args.skip_missing,
        )
    meta = write_feature_bundle(Path(args.output_dir), result)
    print(
        f"extracted {meta['count']} features "
        f"(dim={meta['feature_dim']}, kind={result.feature_kind}) to {args.output_dir}"
    )
    if result.missing:
        print(f"missing_images={len(result.missing)}")
    return 0


def _train_eval(args: argparse.Namespace) -> int:
    report = run_baseline_from_paths(
        feature_dir=Path(args.feature_dir),
        splits_path=Path(args.splits),
        label_field=args.label_field,
        model=args.model,
        metric=args.metric,
        k=args.k,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    print(format_baseline_report(report))
    if args.output:
        write_baseline_report(Path(args.output), report)
        print(f"wrote metrics to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qif-attr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest", help="Validate a dataset manifest")
    validate.add_argument("manifest")
    validate.set_defaults(func=_validate_manifest)

    audit = subparsers.add_parser("audit-manifest", help="Audit a manifest before training")
    audit.add_argument("manifest")
    audit.add_argument("--image-root")
    audit.set_defaults(func=_audit_manifest)

    split = subparsers.add_parser("make-splits", help="Create and audit deterministic splits")
    split.add_argument("manifest")
    split.add_argument("--group-by", default="prompt_id")
    split.add_argument("--train", type=float, default=0.7)
    split.add_argument("--val", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=1729)
    split.add_argument(
        "--leakage-keys",
        default="prompt_id,sha256",
        help="Comma-separated manifest fields that must not cross splits.",
    )
    split.add_argument("--output", help="Write split assignments to this CSV path.")
    split.set_defaults(func=_make_splits)

    pairs = subparsers.add_parser("pair-summary", help="Summarize SynCLR-style attribution pairs")
    pairs.add_argument("manifest")
    pairs.add_argument("--content-key", default="prompt_id")
    pairs.add_argument("--max-pairs-per-anchor", type=int, default=4)
    pairs.add_argument("--seed", type=int, default=1729)
    pairs.set_defaults(func=_pair_summary)

    extract = subparsers.add_parser(
        "extract-features",
        help="Compute radial FFT feature arrays from manifest images",
    )
    extract.add_argument("manifest")
    extract.add_argument("output_dir")
    extract.add_argument("--image-root", default=".")
    extract.add_argument(
        "--feature",
        choices=("radial-fft", "density-matrix"),
        default="radial-fft",
        help="Feature family to compute.",
    )
    extract.add_argument("--bins", type=int, default=64, help="radial-fft: number of annuli.")
    extract.add_argument("--no-phase", action="store_true", help="radial-fft: drop phase channel.")
    extract.add_argument(
        "--patch-size", type=int, default=8, help="density-matrix: square patch edge in pixels."
    )
    extract.add_argument(
        "--top-eigenvalues",
        type=int,
        default=16,
        help="density-matrix: number of leading eigenvalues to keep.",
    )
    extract.add_argument(
        "--color",
        action="store_true",
        help="density-matrix: stack RGB channels into each patch vector.",
    )
    extract.add_argument(
        "--highpass",
        action="store_true",
        help="density-matrix: use high-pass residual planes (box-blur subtracted).",
    )
    extract.add_argument(
        "--highpass-radius",
        type=int,
        default=2,
        help="density-matrix: box-blur radius for the high-pass residual.",
    )
    extract.add_argument("--limit", type=int)
    extract.add_argument("--skip-missing", action="store_true")
    extract.set_defaults(func=_extract_features)

    traineval = subparsers.add_parser(
        "train-eval",
        help="Fit a nearest-centroid baseline on a feature bundle and report split metrics",
    )
    traineval.add_argument(
        "feature_dir", help="Directory containing features.npy and index.csv"
    )
    traineval.add_argument("splits", help="Split assignments CSV with image_id,split columns")
    traineval.add_argument("--label-field", default="generator_id")
    traineval.add_argument(
        "--model",
        choices=("nearest-centroid", "knn", "logreg"),
        default="nearest-centroid",
    )
    traineval.add_argument(
        "--metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="Distance for nearest-centroid and knn.",
    )
    traineval.add_argument("--k", type=int, default=5, help="knn: number of neighbors.")
    traineval.add_argument("--max-iter", type=int, default=500, help="logreg: GD iterations.")
    traineval.add_argument(
        "--learning-rate", type=float, default=0.5, help="logreg: GD step size."
    )
    traineval.add_argument("--l2", type=float, default=1e-3, help="logreg: L2 strength.")
    traineval.add_argument("--output", help="Write metrics JSON to this path.")
    traineval.set_defaults(func=_train_eval)

    expand = subparsers.add_parser("expand-prompts", help="Expand concept seeds into prompts")
    expand.add_argument("concepts")
    expand.add_argument("output")
    expand.add_argument("--sources", default="human,llm_a,llm_b")
    expand.add_argument("--styles", default="concise,photorealistic,technical")
    expand.set_defaults(func=_expand_prompts)

    jobs = subparsers.add_parser("build-jobs", help="Build generation job CSV")
    jobs.add_argument("prompts")
    jobs.add_argument("generators")
    jobs.add_argument("output")
    jobs.add_argument("--seeds", default="0,1,2")
    jobs.add_argument("--output-prefix", default="raw")
    jobs.set_defaults(func=_build_jobs)

    manifest = subparsers.add_parser(
        "build-manifest",
        help="Build a completed-image manifest from prompts and generation jobs",
    )
    manifest.add_argument("prompts")
    manifest.add_argument("jobs")
    manifest.add_argument("image_root")
    manifest.add_argument("output")
    manifest.add_argument("--postprocess", default="none")
    manifest.add_argument("--license-notes", default="generated for controlled research dataset")
    manifest.add_argument("--skip-missing", action="store_true")
    manifest.set_defaults(func=_build_manifest)

    run = subparsers.add_parser("run-generation", help="Run text-to-image generation jobs")
    run.add_argument("prompts")
    run.add_argument("jobs")
    run.add_argument("generators")
    run.add_argument("image_root")
    run.add_argument("--backend", choices=("diffusers", "placeholder"), default="diffusers")
    run.add_argument("--limit", type=int)
    run.add_argument("--generator-id", action="append")
    run.add_argument("--prompt-id", action="append")
    run.add_argument("--shard-index", type=int)
    run.add_argument("--num-shards", type=int)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--stop-on-error", action="store_true")
    run.add_argument("--device", default="auto")
    run.add_argument(
        "--torch-dtype",
        default="auto",
        choices=("auto", "float16", "float32", "bfloat16"),
    )
    run.add_argument("--local-files-only", action="store_true")
    run.set_defaults(func=_run_generation)

    subparsers.add_parser(
        "list-scale-profiles",
        help="List controlled small/medium/large dataset scale profiles",
    ).set_defaults(func=_list_scale_profiles)

    scale = subparsers.add_parser(
        "build-scale-plan",
        help="Build prompt/job CSVs for a controlled scale profile",
    )
    scale.add_argument("profile", choices=("small", "medium", "large"))
    scale.add_argument("output_dir")
    scale.add_argument(
        "--generators",
        default="data/manifests/controlled_generators.csv",
        help="Generator spec CSV containing the profile generator IDs.",
    )
    scale.add_argument("--output-prefix", default="raw")
    scale.set_defaults(func=_build_scale_plan)

    package = subparsers.add_parser(
        "package-profile",
        help="Zip a scale profile manifest with exactly its referenced images",
    )
    package.add_argument("profile_dir")
    package.add_argument("output")
    package.add_argument("--image-root", default=".")
    package.add_argument("--allow-missing", action="store_true")
    package.set_defaults(func=_package_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _split_csv_arg(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("comma-separated argument cannot be empty")
    return items


if __name__ == "__main__":
    raise SystemExit(main())

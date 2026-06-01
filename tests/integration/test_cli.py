import csv

import numpy as np

from qif_attribution.cli import main


def test_validate_manifest_cli_passes(capsys) -> None:
    exit_code = main(["validate-manifest", "data/manifests/example_manifest.csv"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "manifest ok" in captured.out


def test_audit_manifest_cli_passes(capsys) -> None:
    exit_code = main(["audit-manifest", "data/manifests/example_manifest.csv"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dataset audit" in captured.out


def test_pair_summary_cli_passes(capsys) -> None:
    exit_code = main(["pair-summary", "data/manifests/example_manifest.csv"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "pairs:" in captured.out


def test_list_scale_profiles_cli_passes(capsys) -> None:
    exit_code = main(["list-scale-profiles"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "small" in captured.out
    assert "medium" in captured.out
    assert "large" in captured.out


def test_build_scale_plan_cli_writes_files(tmp_path, capsys) -> None:
    exit_code = main(["build-scale-plan", "small", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "images=1024" in captured.out
    assert (tmp_path / "prompts.csv").exists()
    assert (tmp_path / "jobs.csv").exists()
    assert (tmp_path / "slurm.env").exists()


def test_package_profile_cli_writes_manifest_defined_images(tmp_path, capsys) -> None:
    image_path = tmp_path / "raw" / "gen" / "prompt" / "seed_000000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "manifest.csv").write_text(
        "\n".join(
            (
                "image_id,sha256,prompt_id,prompt_text,prompt_source,prompt_category,"
                "prompt_style,generator_id,generator_version,seed,sampler,steps,cfg_scale,"
                "width,height,postprocess,license_notes,relative_path",
                "img,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,prompt,text,"
                "human,object,concise,gen,1,0,euler,1,1.0,64,64,none,test,"
                "raw/gen/prompt/seed_000000.png",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "profile.zip"

    exit_code = main(
        ["package-profile", str(profile_dir), str(output), "--image-root", str(tmp_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert "records=1 images=1" in captured.out


def test_make_splits_cli_allows_reused_seed_grid(capsys) -> None:
    exit_code = main(["make-splits", "data/manifests/example_manifest.csv"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "splits ok" in captured.out


def test_make_splits_cli_writes_assignments(tmp_path, capsys) -> None:
    output = tmp_path / "splits.csv"

    exit_code = main(
        ["make-splits", "data/manifests/example_manifest.csv", "--output", str(output)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "split assignments to" in captured.out
    assert output.exists()
    assert "splits ok" in captured.out


def test_expand_prompts_cli_writes_file(tmp_path, capsys) -> None:
    output = tmp_path / "prompts.csv"

    exit_code = main(
        [
            "expand-prompts",
            "data/manifests/pilot_concepts.csv",
            str(output),
            "--sources",
            "human,llm_a",
            "--styles",
            "concise",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert "wrote 8 prompts" in captured.out


def test_build_jobs_cli_writes_file(tmp_path, capsys) -> None:
    output = tmp_path / "jobs.csv"

    exit_code = main(
        [
            "build-jobs",
            "data/manifests/example_prompt_families.csv",
            "data/manifests/example_generators.csv",
            str(output),
            "--seeds",
            "1,2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert "wrote 16 jobs" in captured.out


def test_build_manifest_cli_can_skip_missing(tmp_path, capsys) -> None:
    jobs = tmp_path / "jobs.csv"
    main(
        [
            "build-jobs",
            "data/manifests/example_prompt_families.csv",
            "data/manifests/example_generators.csv",
            str(jobs),
            "--seeds",
            "1",
        ]
    )
    output = tmp_path / "manifest.csv"

    exit_code = main(
        [
            "build-manifest",
            "data/manifests/example_prompt_families.csv",
            str(jobs),
            str(tmp_path),
            str(output),
            "--skip-missing",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.exists()
    assert "wrote 0 manifest records" in captured.out


def test_run_generation_cli_placeholder_writes_images(tmp_path, capsys) -> None:
    jobs = tmp_path / "jobs.csv"
    main(
        [
            "build-jobs",
            "data/manifests/example_prompt_families.csv",
            "data/manifests/example_generators.csv",
            str(jobs),
            "--seeds",
            "0",
        ]
    )

    exit_code = main(
        [
            "run-generation",
            "data/manifests/example_prompt_families.csv",
            str(jobs),
            "data/manifests/example_generators.csv",
            str(tmp_path),
            "--backend",
            "placeholder",
            "--limit",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "completed=2" in captured.out
    assert len(list(tmp_path.glob("raw/**/*.png"))) == 2


def test_extract_features_cli_writes_bundle(tmp_path, capsys) -> None:
    from qif_attribution.generation.png import write_solid_rgb_png

    image_root = tmp_path / "images"
    write_solid_rgb_png(
        image_root / "raw/sd15/p/seed_000000.png", width=16, height=16, rgb=(10, 20, 30)
    )
    write_solid_rgb_png(
        image_root / "raw/openjourney_v4/p/seed_000000.png", width=16, height=16, rgb=(200, 100, 50)
    )
    header = (
        "image_id,sha256,prompt_id,prompt_text,prompt_source,prompt_category,"
        "prompt_style,generator_id,generator_version,seed,sampler,steps,cfg_scale,"
        "width,height,postprocess,license_notes,relative_path"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            (
                header,
                "img_a," + "a" * 64 + ",p,text,human,object,concise,sd15,1,0,euler,1,1.0,16,16,"
                "none,test,raw/sd15/p/seed_000000.png",
                "img_b," + "b" * 64 + ",p,text,human,object,concise,openjourney_v4,1,0,euler,1,"
                "1.0,16,16,none,test,raw/openjourney_v4/p/seed_000000.png",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "features"

    exit_code = main(
        [
            "extract-features",
            str(manifest),
            str(output),
            "--image-root",
            str(image_root),
            "--bins",
            "8",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "extracted 2 features (dim=16, kind=radial_fft)" in captured.out
    assert (output / "features.npy").exists()
    assert (output / "index.csv").exists()
    assert (output / "features_meta.json").exists()


def test_extract_features_cli_density_matrix(tmp_path, capsys) -> None:
    from qif_attribution.generation.png import write_solid_rgb_png

    image_root = tmp_path / "images"
    write_solid_rgb_png(
        image_root / "raw/sd15/p/seed_000000.png", width=16, height=16, rgb=(10, 20, 30)
    )
    header = (
        "image_id,sha256,prompt_id,prompt_text,prompt_source,prompt_category,"
        "prompt_style,generator_id,generator_version,seed,sampler,steps,cfg_scale,"
        "width,height,postprocess,license_notes,relative_path"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        header
        + "\nimg_a," + "a" * 64 + ",p,text,human,object,concise,sd15,1,0,euler,1,1.0,16,16,"
        "none,test,raw/sd15/p/seed_000000.png\n",
        encoding="utf-8",
    )
    output = tmp_path / "features"

    exit_code = main(
        ["extract-features", str(manifest), str(output), "--image-root", str(image_root),
         "--feature", "density-matrix", "--patch-size", "4", "--top-eigenvalues", "6"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    # 5 scalar descriptors + 6 eigenvalues = 11.
    assert "extracted 1 features (dim=11, kind=density_matrix)" in captured.out
    import json

    meta = json.loads((output / "features_meta.json").read_text(encoding="utf-8"))
    assert meta["feature_kind"] == "density_matrix"
    assert meta["params"] == {
        "patch_size": 4,
        "top_eigenvalues": 6,
        "color": False,
        "highpass": False,
        "highpass_radius": 2,
    }


def test_extract_features_cli_density_color_highpass(tmp_path, capsys) -> None:
    from qif_attribution.generation.png import write_solid_rgb_png

    image_root = tmp_path / "images"
    # Two non-solid images so the high-pass residual is non-degenerate.
    write_solid_rgb_png(
        image_root / "raw/sd15/p/seed_000000.png", width=16, height=16, rgb=(10, 200, 30)
    )
    write_solid_rgb_png(
        image_root / "raw/sd15/p/seed_000001.png", width=16, height=16, rgb=(200, 10, 90)
    )
    header = (
        "image_id,sha256,prompt_id,prompt_text,prompt_source,prompt_category,"
        "prompt_style,generator_id,generator_version,seed,sampler,steps,cfg_scale,"
        "width,height,postprocess,license_notes,relative_path"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        header
        + "\nimg_a," + "a" * 64 + ",p,text,human,object,concise,sd15,1,0,euler,1,1.0,16,16,"
        "none,test,raw/sd15/p/seed_000000.png"
        + "\nimg_b," + "b" * 64 + ",p,text,human,object,concise,sd15,1,1,euler,1,1.0,16,16,"
        "none,test,raw/sd15/p/seed_000001.png\n",
        encoding="utf-8",
    )
    output = tmp_path / "features"

    exit_code = main(
        ["extract-features", str(manifest), str(output), "--image-root", str(image_root),
         "--feature", "density-matrix", "--patch-size", "4", "--top-eigenvalues", "6",
         "--color", "--highpass", "--highpass-radius", "1"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    # color/high-pass do not change the signature length: 5 scalars + 6 eigenvalues.
    assert "extracted 2 features (dim=11, kind=density_matrix)" in captured.out
    import json

    meta = json.loads((output / "features_meta.json").read_text(encoding="utf-8"))
    assert meta["params"] == {
        "patch_size": 4,
        "top_eigenvalues": 6,
        "color": True,
        "highpass": True,
        "highpass_radius": 1,
    }


def test_train_eval_cli_reports_and_writes_metrics(tmp_path, capsys) -> None:
    import json

    from qif_attribution.generation.png import write_solid_rgb_png

    image_root = tmp_path / "images"
    write_solid_rgb_png(
        image_root / "raw/sd15/p/seed_000000.png", width=16, height=16, rgb=(10, 20, 30)
    )
    write_solid_rgb_png(
        image_root / "raw/openjourney_v4/p/seed_000000.png", width=16, height=16, rgb=(200, 100, 50)
    )
    header = (
        "image_id,sha256,prompt_id,prompt_text,prompt_source,prompt_category,"
        "prompt_style,generator_id,generator_version,seed,sampler,steps,cfg_scale,"
        "width,height,postprocess,license_notes,relative_path"
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            (
                header,
                "img_a," + "a" * 64 + ",p,text,human,object,concise,sd15,1,0,euler,1,1.0,16,16,"
                "none,test,raw/sd15/p/seed_000000.png",
                "img_b," + "b" * 64 + ",p,text,human,object,concise,openjourney_v4,1,0,euler,1,"
                "1.0,16,16,none,test,raw/openjourney_v4/p/seed_000000.png",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    feature_dir = tmp_path / "features"
    main(
        ["extract-features", str(manifest), str(feature_dir), "--image-root", str(image_root),
         "--bins", "8"]
    )
    capsys.readouterr()
    splits = tmp_path / "splits.csv"
    splits.write_text("image_id,split\nimg_a,train\nimg_b,test\n", encoding="utf-8")
    metrics = tmp_path / "metrics.json"

    exit_code = main(["train-eval", str(feature_dir), str(splits), "--output", str(metrics)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "baseline (nearest-centroid" in captured.out
    assert "wrote metrics to" in captured.out
    assert metrics.exists()
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["model"] == "nearest-centroid"
    assert payload["label_field"] == "generator_id"
    assert set(payload["splits"]) == {"train", "test"}


def test_train_eval_cli_supports_knn_and_logreg(tmp_path, capsys) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    features = np.array(
        [[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0], [1.5, 0.1], [0.1, 1.5]]
    )
    np.save(feature_dir / "features.npy", features)
    with (feature_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "generator_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"image_id": "a1", "generator_id": "sd15"},
                {"image_id": "a2", "generator_id": "sd15"},
                {"image_id": "b1", "generator_id": "openjourney_v4"},
                {"image_id": "b2", "generator_id": "openjourney_v4"},
                {"image_id": "a3", "generator_id": "sd15"},
                {"image_id": "b3", "generator_id": "openjourney_v4"},
            ]
        )
    splits = tmp_path / "splits.csv"
    splits.write_text(
        "image_id,split\na1,train\na2,train\nb1,train\nb2,train\na3,val\nb3,test\n",
        encoding="utf-8",
    )

    for model in ("knn", "logreg"):
        exit_code = main(["train-eval", str(feature_dir), str(splits), "--model", model])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert f"baseline ({model}" in captured.out

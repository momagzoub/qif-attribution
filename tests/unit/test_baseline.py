import csv
import json

import numpy as np
import pytest

from qif_attribution.eval.baseline import (
    BaselineReport,
    format_baseline_report,
    load_split_map,
    run_baseline,
    run_baseline_from_paths,
    write_baseline_report,
)


def _index(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"image_id": image_id, "generator_id": label} for image_id, label in pairs]


# Two well-separated clusters: sd15 along the x-axis, openjourney_v4 along the y-axis.
FEATURES = np.array(
    [
        [2.0, 0.0],  # a1 train sd15
        [1.0, 0.0],  # a2 train sd15
        [0.0, 2.0],  # b1 train openjourney_v4
        [0.0, 1.0],  # b2 train openjourney_v4
        [1.5, 0.1],  # a3 val sd15
        [0.1, 1.5],  # b3 test openjourney_v4
    ]
)
INDEX = _index(
    ("a1", "sd15"),
    ("a2", "sd15"),
    ("b1", "openjourney_v4"),
    ("b2", "openjourney_v4"),
    ("a3", "sd15"),
    ("b3", "openjourney_v4"),
)
SPLITS = {
    "a1": "train",
    "a2": "train",
    "b1": "train",
    "b2": "train",
    "a3": "val",
    "b3": "test",
}


def test_run_baseline_separable_clusters_scores_perfectly() -> None:
    report = run_baseline(FEATURES, INDEX, SPLITS)

    assert isinstance(report, BaselineReport)
    assert report.feature_dim == 2
    assert report.train_count == 4
    assert report.labels == ["openjourney_v4", "sd15"]
    by_split = {ev.split: ev for ev in report.evaluations}
    assert [ev.split for ev in report.evaluations] == ["train", "val", "test"]
    for ev in report.evaluations:
        assert ev.accuracy == 1.0
        assert ev.macro_f1 == 1.0
        assert ev.balanced_accuracy == 1.0
    assert by_split["val"].count == 1
    assert by_split["test"].confusion_matrix["openjourney_v4"]["openjourney_v4"] == 1


def test_run_baseline_euclidean_metric_also_separates() -> None:
    report = run_baseline(FEATURES, INDEX, SPLITS, metric="euclidean")

    assert report.model == "nearest-centroid"
    assert report.params["metric"] == "euclidean"
    assert all(ev.accuracy == 1.0 for ev in report.evaluations)


def test_run_baseline_knn_separates_clusters() -> None:
    report = run_baseline(FEATURES, INDEX, SPLITS, model="knn", k=1)

    assert report.model == "knn"
    assert report.params == {"k": 1, "metric": "cosine"}
    assert all(ev.accuracy == 1.0 for ev in report.evaluations)


def test_run_baseline_logreg_separates_clusters() -> None:
    report = run_baseline(FEATURES, INDEX, SPLITS, model="logreg", max_iter=200)

    assert report.model == "logreg"
    assert report.params["max_iter"] == 200
    assert all(ev.accuracy == 1.0 for ev in report.evaluations)


def test_run_baseline_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        run_baseline(FEATURES, INDEX, SPLITS, model="svm")


def test_run_baseline_requires_train_rows() -> None:
    splits = dict(SPLITS)
    for key in ("a1", "a2", "b1", "b2"):
        splits[key] = "test"

    with pytest.raises(ValueError, match="no training rows"):
        run_baseline(FEATURES, INDEX, splits)


def test_run_baseline_rejects_missing_split_assignment() -> None:
    splits = {key: value for key, value in SPLITS.items() if key != "b3"}

    with pytest.raises(KeyError, match="no split assignment"):
        run_baseline(FEATURES, INDEX, splits)


def test_run_baseline_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="align"):
        run_baseline(FEATURES[:-1], INDEX, SPLITS)


def test_load_split_map_requires_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("image_id,group\na,b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image_id and split"):
        load_split_map(path)


def test_write_baseline_report_writes_json(tmp_path) -> None:
    report = run_baseline(FEATURES, INDEX, SPLITS)
    out = tmp_path / "metrics" / "metrics.json"

    payload = write_baseline_report(out, report)

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == payload
    assert loaded["model"] == "nearest-centroid"
    assert loaded["params"] == {"metric": "cosine"}
    assert loaded["label_field"] == "generator_id"
    assert set(loaded["splits"]) == {"train", "val", "test"}
    assert loaded["splits"]["val"]["accuracy"] == 1.0


def test_format_baseline_report_lists_each_split() -> None:
    report = run_baseline(FEATURES, INDEX, SPLITS)

    text = format_baseline_report(report)

    assert "baseline (nearest-centroid metric=cosine" in text
    assert "train:" in text
    assert "val:" in text
    assert "test:" in text


def test_run_baseline_from_paths_round_trip(tmp_path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "features.npy", FEATURES)
    with (feature_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "generator_id"])
        writer.writeheader()
        writer.writerows(INDEX)
    splits_path = tmp_path / "splits.csv"
    with splits_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split"])
        writer.writeheader()
        writer.writerows({"image_id": k, "split": v} for k, v in SPLITS.items())

    report = run_baseline_from_paths(feature_dir=feature_dir, splits_path=splits_path)

    assert report.train_count == 4
    assert all(ev.accuracy == 1.0 for ev in report.evaluations)

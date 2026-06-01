#!/usr/bin/env python
"""Benchmark density-matrix feature variants on the small set.

Sweeps a feature x model grid over the radial-FFT signature and the four
density-matrix variants {gray, gray+highpass, rgb, rgb+highpass}, plus their
concatenations with the radial signature. For k-NN the neighbor count and
metric are chosen on the validation split only; test accuracy is reported but
never used for selection. The best-by-val config's full metrics are written to
``data/processed/small/metrics_best_density.json``.

Run from the repo root with the analysis extra installed:

    PYTHONPATH=src python scripts/analysis/benchmark_density_variants.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qif_attribution.eval.baseline import (
    BaselineReport,
    load_feature_index,
    load_split_map,
    run_baseline,
    write_baseline_report,
)

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT / "data" / "processed" / "small"
SPLITS = SMALL / "splits.csv"

BUNDLES = {
    "radial": SMALL / "features",
    "d_gray": SMALL / "features_density",
    "d_hp": SMALL / "features_density_hp",
    "d_rgb": SMALL / "features_density_rgb",
    "d_rgb_hp": SMALL / "features_density_rgb_hp",
}

# Each feature set is an ordered list of bundle keys whose columns are stacked.
FEATURE_SETS = {
    "radial": ["radial"],
    "d_gray": ["d_gray"],
    "d_hp": ["d_hp"],
    "d_rgb": ["d_rgb"],
    "d_rgb_hp": ["d_rgb_hp"],
    "radial+d_gray": ["radial", "d_gray"],
    "radial+d_hp": ["radial", "d_hp"],
    "radial+d_rgb": ["radial", "d_rgb"],
    "radial+d_rgb_hp": ["radial", "d_rgb_hp"],
    "radial+d_rgb+d_hp": ["radial", "d_rgb", "d_hp"],
    "radial+all_density": ["radial", "d_gray", "d_hp", "d_rgb", "d_rgb_hp"],
}

KNN_KS = (3, 5, 7, 9, 11)
KNN_METRICS = ("euclidean", "cosine")


def _acc(report: BaselineReport) -> dict[str, float]:
    return {ev.split: ev.accuracy for ev in report.evaluations}


def main() -> int:
    split_map = load_split_map(SPLITS)

    # Load every bundle and align all rows to the radial bundle's image_id order
    # so concatenated columns describe the same image on every row.
    raw = {
        name: (np.load(path / "features.npy"), load_feature_index(path / "index.csv"))
        for name, path in BUNDLES.items()
    }
    canonical_index = raw["radial"][1]
    order_ids = [row["image_id"] for row in canonical_index]
    aligned: dict[str, np.ndarray] = {}
    for name, (features, index) in raw.items():
        position = {row["image_id"]: i for i, row in enumerate(index)}
        if set(position) != set(order_ids):
            raise SystemExit(f"image_id set mismatch between radial and {name}")
        aligned[name] = features[[position[image_id] for image_id in order_ids]]

    def nearest_centroid(features: np.ndarray) -> tuple[BaselineReport, str]:
        report = run_baseline(
            features, canonical_index, split_map, model="nearest-centroid", metric="cosine"
        )
        return report, "cosine"

    def knn(features: np.ndarray) -> tuple[BaselineReport, str]:
        best: tuple[BaselineReport, str] | None = None
        for metric in KNN_METRICS:
            for k in KNN_KS:
                report = run_baseline(
                    features, canonical_index, split_map, model="knn", metric=metric, k=k
                )
                if best is None or _acc(report)["val"] > _acc(best[0])["val"]:
                    best = (report, f"{metric},k={k}")
        assert best is not None
        return best

    def logreg(features: np.ndarray) -> tuple[BaselineReport, str]:
        report = run_baseline(features, canonical_index, split_map, model="logreg")
        return report, "default"

    estimators = {"nc_cos": nearest_centroid, "knn": knn, "logreg": logreg}

    rows: list[tuple[str, int, str, str, float, float, BaselineReport]] = []
    for set_name, keys in FEATURE_SETS.items():
        features = np.hstack([aligned[key] for key in keys])
        for model_name, fit in estimators.items():
            report, config = fit(features)
            acc = _acc(report)
            rows.append(
                (
                    set_name,
                    int(features.shape[1]),
                    model_name,
                    config,
                    acc.get("val", float("nan")),
                    acc.get("test", float("nan")),
                    report,
                )
            )

    rows.sort(key=lambda row: row[4], reverse=True)
    header = f"{'feature_set':<20}{'dim':>5}  {'model':<8}{'config':<16}{'val':>7}{'test':>7}"
    print(header)
    print("-" * len(header))
    for set_name, dim, model_name, config, val, test, _ in rows:
        print(f"{set_name:<20}{dim:>5}  {model_name:<8}{config:<16}{val:>7.3f}{test:>7.3f}")

    best = rows[0]
    print(
        f"\nBEST by val: {best[0]} + {best[2]} ({best[3]}) "
        f"-> val={best[4]:.3f} test={best[5]:.3f} (dim={best[1]})"
    )
    out_path = SMALL / "metrics_best_density.json"
    write_baseline_report(out_path, best[6])
    print(f"wrote winning metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

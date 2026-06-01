#!/usr/bin/env python
"""Cross-validated benchmark of density-feature variants on the small set.

Confirms the single-split ranking under group-aware K-fold CV (folds split by
prompt_id, never by image) and probes three numpy-only accuracy levers:
PCA-whitening before k-NN, a richer color-rho eigenvalue tail (top 48 vs 16),
and multi-scale patches (4/8/16). Every (feature set, estimator) pair is scored
with a 95% confidence interval so we can tell a real gain from 144-image noise.

Run from the repo root:

    PYTHONPATH=src python scripts/analysis/cross_validate_small.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qif_attribution.eval.baseline import load_feature_index
from qif_attribution.eval.crossval import CrossValResult, cross_validate, paired_difference_ci
from qif_attribution.models.knn import KNearestNeighborsClassifier
from qif_attribution.models.logreg import LogisticRegressionClassifier
from qif_attribution.models.nearest_centroid import NearestCentroidClassifier
from qif_attribution.models.scaling import Whitener

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT / "data" / "processed" / "small"

BUNDLES = {
    "radial": SMALL / "features",
    "gray": SMALL / "features_density",
    "hp": SMALL / "features_density_hp",
    "rgb": SMALL / "features_density_rgb",
    "rgb_hp": SMALL / "features_density_rgb_hp",
    "rgb_e48": SMALL / "features_density_rgb_e48",
    "rgb_hp_e48": SMALL / "features_density_rgb_hp_e48",
    "p4": SMALL / "features_density_p4",
    "rgb_p4": SMALL / "features_density_rgb_p4",
    "p16": SMALL / "features_density_p16",
    "rgb_p16": SMALL / "features_density_rgb_p16",
    "resnet": SMALL / "features_resnet50",
    "convnext": SMALL / "features_convnext_large",
}

FEATURE_SETS = {
    "radial": ["radial"],
    "radial+gray": ["radial", "gray"],
    "radial+all4(212)": ["radial", "gray", "hp", "rgb", "rgb_hp"],
    "radial+all4_e48color": ["radial", "gray", "hp", "rgb_e48", "rgb_hp_e48"],
    "radial+all4+multiscale": [
        "radial", "gray", "hp", "rgb", "rgb_hp", "p4", "rgb_p4", "p16", "rgb_p16",
    ],
    "radial+everything": [
        "radial", "gray", "hp", "rgb_e48", "rgb_hp_e48",
        "p4", "rgb_p4", "p16", "rgb_p16",
    ],
    "radial+rgb_e48+rgb_hp_e48": ["radial", "rgb_e48", "rgb_hp_e48"],
    "all4(no radial)": ["gray", "hp", "rgb", "rgb_hp"],
    # ResNet-50 ImageNet embeddings: the learned-feature ceiling, plus a hybrid
    # to test whether the numpy-only physics features add signal on top of it.
    "resnet(2048)": ["resnet"],
    "radial+all4+resnet": ["radial", "gray", "hp", "rgb", "rgb_hp", "resnet"],
    # ConvNeXt-Large: a stronger, more modern frozen backbone -- does a better
    # representation raise the deep-feature ceiling above ResNet-50's ~0.88?
    "convnext(1536)": ["convnext"],
    "radial+all4+convnext": ["radial", "gray", "hp", "rgb", "rgb_hp", "convnext"],
}

N_SPLITS = 5
SEED = 1729

# Pre-registered headline config and the comparison we actually care about. We
# DECLARE these before scoring so the reported result is not a max over the
# ~50-cell grid (that argmax is optimistically biased by multiple comparisons).
# The grid stays in the output, but only as an exploratory map -- any "X beats
# Y" claim must clear the paired across-fold test below, not just rank higher.
PRIMARY_CONFIG = ("resnet(2048)", "logreg")
HYBRID_CONFIG = ("radial+all4+resnet", "logreg")


class WhitenedKNN:
    """PCA-whiten on the training fold, then run k-NN with no extra scaling."""

    def __init__(self, k: int = 5, metric: str = "euclidean", eps: float = 1e-8) -> None:
        self.k = k
        self.metric = metric
        self.eps = eps

    def fit(self, features: np.ndarray, labels: list[str]) -> WhitenedKNN:
        self._whitener = Whitener(eps=self.eps).fit(features)
        self._knn = KNearestNeighborsClassifier(
            k=self.k, metric=self.metric, standardize=False
        ).fit(self._whitener.transform(features), labels)
        return self

    def predict(self, features: np.ndarray) -> list[str]:
        return self._knn.predict(self._whitener.transform(features))


ESTIMATORS = {
    "nc_cos": lambda: NearestCentroidClassifier(metric="cosine"),
    "knn5_std": lambda: KNearestNeighborsClassifier(k=5, metric="euclidean"),
    "knn5_whiten": lambda: WhitenedKNN(k=5),
    "knn3_whiten": lambda: WhitenedKNN(k=3),
    "logreg": lambda: LogisticRegressionClassifier(),
}


def main() -> int:
    raw = {
        name: (np.load(path / "features.npy"), load_feature_index(path / "index.csv"))
        for name, path in BUNDLES.items()
    }
    canonical = raw["radial"][1]
    order_ids = [row["image_id"] for row in canonical]
    labels = [row["generator_id"] for row in canonical]
    groups = [row["prompt_id"] for row in canonical]
    print(f"images={len(order_ids)} groups(prompt_id)={len(set(groups))} folds={N_SPLITS}\n")

    aligned: dict[str, np.ndarray] = {}
    for name, (features, index) in raw.items():
        position = {row["image_id"]: i for i, row in enumerate(index)}
        if set(position) != set(order_ids):
            raise SystemExit(f"image_id set mismatch between radial and {name}")
        aligned[name] = features[[position[image_id] for image_id in order_ids]]

    rows: list[tuple[str, int, str, dict[str, float], CrossValResult]] = []
    results: dict[tuple[str, str], CrossValResult] = {}
    for set_name, keys in FEATURE_SETS.items():
        features = np.hstack([aligned[key] for key in keys])
        for est_name, factory in ESTIMATORS.items():
            result = cross_validate(
                features, labels, groups, factory, n_splits=N_SPLITS, seed=SEED
            )
            results[(set_name, est_name)] = result
            rows.append((set_name, features.shape[1], est_name, result.summary("accuracy"), result))

    rows.sort(key=lambda row: row[3]["mean"], reverse=True)
    header = (
        f"{'feature_set':<26}{'dim':>5}  {'estimator':<12}"
        f"{'mean':>7}{'95% CI':>17}{'folds(min-max)':>18}"
    )
    print(header)
    print("-" * len(header))
    for set_name, dim, est_name, summ, _ in rows:
        ci = f"[{summ['ci_low']:.3f},{summ['ci_high']:.3f}]"
        span = f"{summ['min']:.3f}-{summ['max']:.3f}"
        print(f"{set_name:<26}{dim:>5}  {est_name:<12}{summ['mean']:>7.3f}{ci:>17}{span:>18}")

    # The pre-registered headline number (declared before scoring), not the
    # grid max. Report this as THE result; treat the grid as exploratory.
    primary = results[PRIMARY_CONFIG].summary("accuracy")
    print(
        f"\nPRE-REGISTERED primary: {PRIMARY_CONFIG[0]} + {PRIMARY_CONFIG[1]} -> "
        f"acc {primary['mean']:.3f} 95% CI [{primary['ci_low']:.3f}, {primary['ci_high']:.3f}]"
    )
    # Does adding the hand-crafted physics block beat ResNet alone? Paired across
    # the (identical) folds -- a CI straddling 0 means "no, it's fold noise".
    paired = paired_difference_ci(
        results[HYBRID_CONFIG].fold_accuracy, results[PRIMARY_CONFIG].fold_accuracy
    )
    verdict = "SIGNIFICANT" if paired["significant"] else "NOT significant (CI crosses 0)"
    print(
        f"hybrid - resnet (paired across folds): mean_diff {paired['mean_diff']:+.3f} "
        f"95% CI [{paired['ci_low']:+.3f}, {paired['ci_high']:+.3f}] -> {verdict}"
    )
    best = rows[0]
    summ = best[3]
    print(
        f"\nTOP point estimate on the {len(rows)}-cell exploratory grid (NOT a locked "
        f"result; pick on val, then lock the test set): {best[0]} ({best[1]}-dim) + "
        f"{best[2]} -> acc {summ['mean']:.3f} 95% CI [{summ['ci_low']:.3f}, {summ['ci_high']:.3f}]"
    )
    payload = {
        "n_splits": N_SPLITS,
        "seed": SEED,
        "primary_config": {
            "feature_set": PRIMARY_CONFIG[0],
            "estimator": PRIMARY_CONFIG[1],
            "accuracy": primary,
            "fold_accuracy": list(results[PRIMARY_CONFIG].fold_accuracy),
        },
        "hybrid_vs_resnet_paired": paired,
        "grid_top_is_exploratory_max": True,
        "grid_top": {
            "feature_set": best[0],
            "dim": best[1],
            "estimator": best[2],
            "accuracy": best[3],
            "fold_accuracy": list(best[4].fold_accuracy),
        },
        "grid": [
            {
                "feature_set": set_name,
                "dim": dim,
                "estimator": est_name,
                "accuracy_mean": summ["mean"],
                "accuracy_ci": [summ["ci_low"], summ["ci_high"]],
                "fold_accuracy": list(result.fold_accuracy),
            }
            for set_name, dim, est_name, summ, result in rows
        ],
    }
    out_path = SMALL / "metrics_cv_small.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote CV metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

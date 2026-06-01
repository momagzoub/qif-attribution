#!/usr/bin/env python
"""Class-level quantum (density-matrix) audit of real deep embeddings.

Wires up the previously-dead centerpiece: `quantum.analysis.summarize_embedding
_classes` was exercised only by unit tests, and `configs/quantum/default.yaml`
pointed at a non-existent `pilot_embeddings.npy`, so it produced zero findings.
Here we run it on the *real* ResNet-50 embeddings already extracted for the
small set and ask a concrete question that adjudicates whether the density-
matrix lens adds anything beyond ordinary statistics:

    Does the trace distance between two generators' class density matrices rank
    their confusability differently from the trivial distance between their
    class-mean embeddings?

If the two orderings agree (Spearman ~ 1), the "quantum" distance is recovering
the same between-class separation a centroid distance already gives -- i.e. the
density-matrix framing is a re-description of second-moment statistics, exactly
what a covariance/PCA analysis yields, with no quantum advantage. Any divergence
is the density matrix's sensitivity to per-class *covariance shape* (not just the
mean) -- still classical spectral statistics, but a genuinely richer summary.
Either way the result is reported honestly; this script exists to produce the
finding, not to manufacture a win.

Run from the repo root (needs deep features from extract_resnet_small.py):

    PYTHONPATH=src python scripts/analysis/quantum_embedding_audit.py

Writes data/processed/small/metrics_quantum_audit_small.json.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from qif_attribution.eval.baseline import load_feature_index
from qif_attribution.quantum.analysis import summarize_embedding_classes

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT / "data" / "processed" / "small"
BUNDLE = SMALL / "features_resnet50"
PROJECTION_DIM = 64
LABEL_FIELD = "generator_id"


def pca_project(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Project onto the top-`dim` principal directions (centered, numpy SVD)."""

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    components = np.linalg.svd(centered, full_matrices=False)[2][:dim]
    return centered @ components.T


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] < 2:
        return float("nan")
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def main() -> int:
    if not (BUNDLE / "features.npy").exists():
        raise SystemExit(f"missing {BUNDLE}/features.npy -- run extract_resnet_small.py first")
    embeddings = np.load(BUNDLE / "features.npy")
    index = load_feature_index(BUNDLE / "index.csv")
    labels = [row[LABEL_FIELD] for row in index]
    classes = sorted(set(labels))
    print(
        f"embeddings={embeddings.shape[0]}x{embeddings.shape[1]} "
        f"classes={len(classes)} projection_dim={PROJECTION_DIM}\n"
    )

    projected = pca_project(embeddings, PROJECTION_DIM)
    summary = summarize_embedding_classes(projected, labels)

    # Classical contrast: distance between class-mean embeddings in the same space.
    centroids = {
        cls: projected[[i for i, lab in enumerate(labels) if lab == cls]].mean(axis=0)
        for cls in classes
    }
    pairs = list(combinations(classes, 2))
    centroid_dist = {
        pair: float(np.linalg.norm(centroids[pair[0]] - centroids[pair[1]])) for pair in pairs
    }
    quantum_td = np.array([summary.pairwise_trace_distance[pair] for pair in pairs])
    centroid_d = np.array([centroid_dist[pair] for pair in pairs])
    rho = spearman(quantum_td, centroid_d)
    most_confusable = pairs[int(np.argmin(quantum_td))]

    print("per-class density-matrix descriptors (64-d):")
    for cls in classes:
        print(
            f"  {cls:<24} purity={summary.purity_by_class[cls]:.4f} "
            f"entropy(bits)={summary.entropy_by_class[cls]:.3f}"
        )
    print("\npairwise generator distinguishability:")
    print(f"  {'pair':<46}{'trace_dist':>11}{'centroid':>11}{'fidelity':>10}")
    for pair in pairs:
        label = f"{pair[0]} | {pair[1]}"
        print(
            f"  {label:<46}{summary.pairwise_trace_distance[pair]:>11.4f}"
            f"{centroid_dist[pair]:>11.4f}{summary.pairwise_fidelity[pair]:>10.4f}"
        )
    print(f"\nSpearman(trace_distance, centroid_distance) over {len(pairs)} pairs = {rho:.3f}")
    confusable_label = f"{most_confusable[0]} | {most_confusable[1]}"
    print(f"least-distinguishable (most-confusable) pair: {confusable_label}")
    verdict = (
        "tracks centroid separation -> density-matrix distance re-describes "
        "second-moment statistics (no signal beyond class means)"
        if rho >= 0.9
        else "diverges from centroid separation -> density matrix adds per-class "
        "covariance-shape information (still classical spectral statistics)"
    )
    print(f"verdict: {verdict}")

    payload = {
        "bundle": BUNDLE.name,
        "projection_dim": PROJECTION_DIM,
        "classes": classes,
        "purity_by_class": summary.purity_by_class,
        "entropy_bits_by_class": summary.entropy_by_class,
        "pairwise": [
            {
                "pair": list(pair),
                "trace_distance": summary.pairwise_trace_distance[pair],
                "fidelity": summary.pairwise_fidelity[pair],
                "centroid_distance": centroid_dist[pair],
            }
            for pair in pairs
        ],
        "spearman_trace_vs_centroid": rho,
        "most_confusable_pair": list(most_confusable),
        "verdict": verdict,
    }
    out_path = SMALL / "metrics_quantum_audit_small.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote quantum-audit metrics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

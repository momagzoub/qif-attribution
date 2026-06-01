"""Class-level quantum summaries for embedding spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qif_attribution.quantum.states import (
    class_density_matrix,
    fidelity_mixed,
    purity,
    trace_distance,
    von_neumann_entropy,
)


@dataclass(frozen=True)
class QuantumClassSummary:
    class_labels: tuple[str, ...]
    purity_by_class: dict[str, float]
    entropy_by_class: dict[str, float]
    pairwise_trace_distance: dict[tuple[str, str], float]
    pairwise_fidelity: dict[tuple[str, str], float]


def summarize_embedding_classes(
    embeddings: np.ndarray,
    labels: list[str],
) -> QuantumClassSummary:
    """Summarize class distinguishability after treating embeddings as states."""

    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if array.shape[0] != len(labels):
        raise ValueError("number of embeddings must match number of labels")
    if not labels:
        raise ValueError("labels cannot be empty")

    class_labels = tuple(sorted(set(labels)))
    densities = {
        label: class_density_matrix(
            array[[idx for idx, item in enumerate(labels) if item == label]]
        )
        for label in class_labels
    }
    purity_by_class = {label: purity(rho) for label, rho in densities.items()}
    entropy_by_class = {label: von_neumann_entropy(rho) for label, rho in densities.items()}

    pairwise_trace_distance: dict[tuple[str, str], float] = {}
    pairwise_fidelity: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(class_labels):
        for right in class_labels[left_index + 1 :]:
            key = (left, right)
            pairwise_trace_distance[key] = trace_distance(densities[left], densities[right])
            pairwise_fidelity[key] = fidelity_mixed(densities[left], densities[right])

    return QuantumClassSummary(
        class_labels=class_labels,
        purity_by_class=purity_by_class,
        entropy_by_class=entropy_by_class,
        pairwise_trace_distance=pairwise_trace_distance,
        pairwise_fidelity=pairwise_fidelity,
    )

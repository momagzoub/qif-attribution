"""Quantum-information utilities for embedding analysis."""

from qif_attribution.quantum.analysis import QuantumClassSummary, summarize_embedding_classes
from qif_attribution.quantum.states import (
    class_density_matrix,
    fidelity_mixed,
    fidelity_pure,
    normalize_state,
    pure_density_matrix,
    purity,
    trace_distance,
    von_neumann_entropy,
)

__all__ = [
    "QuantumClassSummary",
    "class_density_matrix",
    "fidelity_mixed",
    "fidelity_pure",
    "normalize_state",
    "pure_density_matrix",
    "purity",
    "summarize_embedding_classes",
    "trace_distance",
    "von_neumann_entropy",
]

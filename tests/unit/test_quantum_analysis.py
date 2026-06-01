import numpy as np

from qif_attribution.quantum.analysis import summarize_embedding_classes
from qif_attribution.quantum.states import fidelity_mixed, pure_density_matrix


def test_mixed_fidelity_matches_pure_fidelity_for_pure_states() -> None:
    rho = pure_density_matrix(np.array([1.0, 0.0]))
    sigma = pure_density_matrix(np.array([0.0, 1.0]))

    assert np.isclose(fidelity_mixed(rho, sigma), 0.0)
    assert np.isclose(fidelity_mixed(rho, rho), 1.0)


def test_summarize_embedding_classes_returns_pairwise_metrics() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    labels = ["a", "a", "b", "b"]

    summary = summarize_embedding_classes(embeddings, labels)

    assert summary.class_labels == ("a", "b")
    assert np.isclose(summary.purity_by_class["a"], 1.0)
    assert np.isclose(summary.pairwise_trace_distance[("a", "b")], 1.0)

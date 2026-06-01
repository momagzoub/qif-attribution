import numpy as np

from qif_attribution.models.nearest_centroid import NearestCentroidClassifier


def test_nearest_centroid_predicts_by_cosine_similarity() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = ["generator_a", "generator_a", "generator_b", "generator_b"]
    model = NearestCentroidClassifier(metric="cosine").fit(embeddings, labels)

    predictions = model.predict(np.array([[0.95, 0.05], [0.05, 0.95]]))

    assert predictions == ["generator_a", "generator_b"]

from qif_attribution.eval.metrics import accuracy, balanced_accuracy, confusion_matrix, macro_f1


def test_classification_metrics() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]

    assert accuracy(y_true, y_pred) == 0.75
    assert balanced_accuracy(y_true, y_pred) == 0.75
    assert round(macro_f1(y_true, y_pred), 6) == round((2 / 3 + 0.8) / 2, 6)
    assert confusion_matrix(y_true, y_pred)["a"]["b"] == 1

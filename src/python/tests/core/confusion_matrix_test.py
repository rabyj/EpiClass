"""Test module for the ConfusionMatrixWriter class."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from epi_ml.core.confusion_matrix import ConfusionMatrixWriter


@pytest.fixture(params=[2, 5, 15, 25])
def sklearn_confusion_matrix(request):
    """
    Generates a confusion matrix for a simulated classification problem with the given number of classes.

    Args:
        num_classes (int): The number of classes in the classification problem.
        num_samples (int): The number of samples to simulate.

    Returns:
        np.ndarray: The confusion matrix.
    """
    num_samples = 500
    num_classes = request.param
    true_labels = np.random.randint(0, num_classes, size=num_samples)
    predicted_labels = np.random.randint(0, num_classes, size=num_samples)
    return confusion_matrix(true_labels, predicted_labels)


def test_to_png(
    sklearn_confusion_matrix: np.ndarray,
):  # pylint: disable=redefined-outer-name
    """Tests the to_png method of the ConfusionMatrixWriter class."""
    labels = [
        f"reallylonglabelnameforrealaaaahhhh {i}"
        for i in range(sklearn_confusion_matrix.shape[0])
    ]
    cm = ConfusionMatrixWriter(labels, sklearn_confusion_matrix)
    cm.to_png(f"test_c{len(labels)}.png")

"""Test module for the ConfusionMatrixWriter class."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from epi_ml.core.confusion_matrix import ConfusionMatrixWriter

THIS_FILE = Path(__file__).resolve()


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


@pytest.mark.parametrize("name_base", ["class", "reallylonglabelnameforrealaaaahhhh"])
def test_to_png(
    sklearn_confusion_matrix: np.ndarray, name_base: str
):  # pylint: disable=redefined-outer-name
    """Tests the to_png method of the ConfusionMatrixWriter class."""
    labels = [f"{name_base} {i}" for i in range(sklearn_confusion_matrix.shape[0])]
    cm = ConfusionMatrixWriter(labels, sklearn_confusion_matrix)

    output_dir = THIS_FILE.parent / "test_matrix"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"test_c{len(labels)}_{name_base}.png"

    cm.to_png(output)

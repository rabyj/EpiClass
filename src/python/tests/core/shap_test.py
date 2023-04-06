"""Test SHAP related modules."""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore", message=".*IPython display.*")

import numpy as np
import pytest

from epi_ml.core.data import DataSet
from epi_ml.core.shap_values import SHAP_Handler


class Test_SHAP_Handler:
    """Class to test SHAP_Handler class."""

    @pytest.fixture
    def logdir(self, make_specific_logdir) -> Path:
        """Test logdir"""
        return make_specific_logdir("shap")

    @pytest.fixture(autouse=True)
    def clean_up(self, logdir: Path):
        """Clean up log directories after test."""
        yield
        for file in os.listdir(logdir):
            os.remove(os.path.join(logdir, file))
        os.rmdir(logdir)

    @pytest.fixture
    def handler(self, logdir: Path, test_NN_model) -> SHAP_Handler:
        """SHAP_Handler instance"""
        return SHAP_Handler(test_NN_model, logdir)

    @pytest.fixture
    def mock_shap_values(self, test_epiatlas_dataset: DataSet) -> List[np.ndarray]:
        """Mock shape values for evaluation on two examples."""
        shap_values = [
            np.zeros(test_epiatlas_dataset.validation.signals.shape)
            for _ in test_epiatlas_dataset.classes
        ]
        return shap_values

    @pytest.fixture
    def fake_ids(self, test_epiatlas_dataset: DataSet):
        """Fake signal ids"""
        num_signals = test_epiatlas_dataset.validation.num_examples
        return [f"id{i}" for i in range(num_signals)]

    def test_compute_NN(self, handler: SHAP_Handler, test_epiatlas_dataset: DataSet):
        """Test shapes of return of compute_NN method."""
        dset = test_epiatlas_dataset
        _, shap_values = handler.compute_NN(
            background_dset=dset.train, evaluation_dset=dset.validation, save=False  # type: ignore
        )
        print(f"len(shap_values) = {len(shap_values)}")
        print(f"shap_values[0].shape = {shap_values[0].shape }")

        n_signals, n_dims = dset.validation.signals.shape[:]
        assert shap_values[0].shape == (n_signals, n_dims)

    def test_save_load_pickle(
        self, handler: SHAP_Handler, mock_shap_values: List, fake_ids: List
    ):
        """Test pickle save/load methods."""
        path = handler.save_to_pickle(mock_shap_values, fake_ids)
        data = handler.load_from_pickle(path)
        assert data["ids"] == fake_ids
        assert np.array_equal(data["shap"], mock_shap_values)

        fake_ids = ["miaw" for _ in fake_ids]
        path = handler.save_to_pickle(mock_shap_values, ids=fake_ids, name="miaw")
        data = handler.load_from_pickle(path)
        assert data["ids"] == fake_ids
        assert np.array_equal(data["shap"], mock_shap_values)

    def test_save_load_csv(self, handler: SHAP_Handler, mock_shap_values, fake_ids):
        """Test pickle save/load methods."""
        shaps = mock_shap_values[0]
        path = handler.save_to_csv(shaps, fake_ids, name="test")

        data = handler.load_from_csv(path)
        assert list(data.index) == fake_ids
        assert np.array_equal(shaps, data.values)

    def test_save_to_csv_list_input(
        self, handler: SHAP_Handler, mock_shap_values, fake_ids
    ):
        """Test effect of list input."""
        shap_values_matrix = [mock_shap_values[0]]
        name = "test_csv"

        with pytest.raises(ValueError):
            handler.save_to_csv(shap_values_matrix, fake_ids, name)

    def test_create_filename(self, handler: SHAP_Handler):
        """Test filename creation method. Created by GPT4 lol."""
        ext = "pickle"
        name = "test_name"

        filename = handler._create_filename(ext, name)  # pylint: disable=protected-access
        assert filename.name.startswith(f"shap_{name}_")
        assert filename.name.endswith(f".{ext}")
        assert filename.parent == Path(handler.logdir)

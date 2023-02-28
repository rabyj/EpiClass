"""pytest setup/configuration"""
# pylint: disable=unused-argument
from __future__ import annotations

from pathlib import Path

import pytest

from src.python.core.data import DataSet
from src.python.core.epiatlas_treatment import EpiAtlasFoldFactory
from src.python.core.model_pytorch import LightningDenseClassifier
from src.python.tests.fixtures.epilap_test_data import EpiAtlasTreatmentTestData

# def pytest_collection_modifyitems(session, config, items):
#     """Ignore certain names from collection."""
#     items[:] = [item for item in items if item.name != "test_logdir"]


def general_test_logdir():
    """Return logdir for tests. (in /tmp)."""
    logdir = Path("/tmp/pytest")
    logdir.mkdir(exist_ok=True, parents=True)
    return logdir


@pytest.fixture(scope="session", name="tmp_logdir")
def fixture_logdir() -> Path:
    """General test logdir"""
    return general_test_logdir()


@pytest.fixture(scope="session", autouse=True)
def make_specific_logdir(tmp_logdir):
    """Return fct to create test subdirectory."""

    def _make_specific_logdir(name: str) -> Path:
        logdir = tmp_logdir / name
        logdir.mkdir(exist_ok=True, parents=True)
        return logdir

    return _make_specific_logdir


@pytest.fixture(scope="session", name="test_epiatlas_data_handler")
def fixture_epiatlas_data_handler() -> EpiAtlasFoldFactory:
    """Return logdir for tests. (in /tmp)."""
    return EpiAtlasTreatmentTestData.default_test_data()


@pytest.fixture(scope="session", name="test_epiatlas_dataset")
def fixture_epiatlas_dataset(test_epiatlas_data_handler: EpiAtlasFoldFactory) -> DataSet:
    """Return mock dataset."""
    return next(test_epiatlas_data_handler.yield_split())


@pytest.fixture(scope="session", name="test_NN_model")
def fixture_NN_model(
    test_epiatlas_dataset: DataSet, tmp_logdir: Path
) -> LightningDenseClassifier:
    """Return small test neural network"""
    test_mapping = tmp_logdir / "test_mapping.tsv"
    test_epiatlas_dataset.save_mapping(test_mapping)
    test_mapping = test_epiatlas_dataset.load_mapping(test_mapping)

    return LightningDenseClassifier(
        input_size=test_epiatlas_dataset.train.signals.shape[1],
        output_size=len(test_epiatlas_dataset.classes),
        mapping=test_mapping,
        hparams={},
        nb_layer=1,
        hl_units=100,
    )

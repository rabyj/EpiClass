"""Test module for epiatlas_training_no_valid.py."""
import os
import sys
from pathlib import Path

import pytest

from epi_ml.epiatlas_training_no_valid import main as main_module
from tests.epilap_test_data import FIXTURES_DIR, EpiAtlasTreatmentTestData


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("epiatlas_training_no_valid")


def test_training(test_dir: Path):
    """Test if basic training succeeds."""
    os.environ["MIN_CLASS_SIZE"] = "3"

    datasource = EpiAtlasTreatmentTestData.default_test_data().epiatlas_dataset.datasource

    hparams_file = FIXTURES_DIR / "test_human_hparams.json"

    sys.argv = [
        "epiatlas_training_no_valid.py",
        "biomaterial_type",
        f"{hparams_file}",
        f"{datasource.hdf5_file}",
        f"{datasource.chromsize_file}",
        f"{datasource.metadata_file}",
        str(test_dir),
        "--offline",
    ]

    main_module()

"""Test module for other_estimators.py content"""
import os
import sys
from pathlib import Path

import src.python.core.estimators
from src.python.other_estimators import main as main_module
from src.python.tests.fixtures.epilap_test_data import EpiAtlasTreatmentTestData


def test_hyperparams():
    """Test if hyperparameter file is handled properly."""
    os.environ["MIN_CLASS_SIZE"] = "3"
    src.python.core.estimators.NFOLD_PREDICT = 2

    datasource = EpiAtlasTreatmentTestData.default_test_data().epiatlas_dataset.datasource

    current_dir = Path(__file__).parent.resolve()

    hparams_file = current_dir.parent / "fixtures" / "other_estimators_hparams.json"

    sys.argv = [
        "other_estimators.py",
        "--predict",
        "biomaterial_type",
        f"{datasource.hdf5_file}",
        f"{datasource.chromsize_file}",
        f"{datasource.metadata_file}",
        f"{current_dir}",
        "--hyperparams",
        f"{hparams_file}",
    ]

    main_module()

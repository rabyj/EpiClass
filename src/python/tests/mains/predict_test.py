"""Test module for predict.py."""
import sys
from pathlib import Path

import pytest

from epi_ml.predict import main as main_module
from tests.epilap_test_data import FIXTURES_DIR


@pytest.fixture(name="test_dir")
def fixture_test_dir(mk_logdir) -> Path:
    """Make temp logdir for tests."""
    return mk_logdir("predict")


def test_training(test_dir: Path):
    """Test if basic training succeeds."""

    saccer3_fixtures_dir = FIXTURES_DIR / "saccer3"
    file_list = saccer3_fixtures_dir / "hdf5_10kb_all_none.list"
    hdf5_dir = saccer3_fixtures_dir / "hdf5"
    chroms = saccer3_fixtures_dir / "saccer3.can.chrom.sizes"

    sys.argv = [
        "predict.py",
        str(file_list),
        str(chroms),
        str(test_dir),  # logdir
        "--offline",
        "--model",
        str(saccer3_fixtures_dir),
        "--hdf5_dir",
        str(hdf5_dir),
    ]
    main_module()

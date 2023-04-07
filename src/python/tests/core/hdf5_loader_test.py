"""Test module for hdf5_loader file."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import pytest

from epi_ml.core.epiatlas_treatment import EpiAtlasDataset
from epi_ml.core.hdf5_loader import Hdf5Loader
from tests.fixtures.epilap_test_data import EpiAtlasTreatmentTestData


class Test_Hdf5Loader:
    """Test class Test_Hdf5Loader"""

    @pytest.fixture(scope="class", autouse=True)
    def test_folder(self, mk_logdir) -> Path:
        """Return temp hdf5 storage folder."""
        return mk_logdir("temp_hdf5s")

    @pytest.fixture(scope="function")
    def test_data(self) -> EpiAtlasDataset:
        """Mock test EpiAtlasFoldFactory."""
        return EpiAtlasTreatmentTestData.default_test_data().epiatlas_dataset

    def test_load_hdf5_wrong_name(self, test_data: EpiAtlasDataset):
        """Verify that files are loading
        even if the internal filename does not match the md5sum.
        """
        hdf5_list = Hdf5Loader.read_list(test_data.datasource.hdf5_file)
        chosen_file = list(hdf5_list.values())[0]

        # modify the header group of an hdf5 file
        with h5py.File(chosen_file, "r+") as f:  # type: ignore
            # print(list(f.items()))
            # print(f.name)
            # print(list(f.attrs.items()))
            dset = list(f.values())[0]
            # print(dset.name)
            dset.move(dset.name, "/miaw")
            new_id = list(f.keys())[0]
            assert new_id == "miaw"
            f.close()

        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)

        with pytest.warns(UserWarning, match="header is different"):
            hdf5_loader.load_hdf5s(test_data.datasource.hdf5_file)

    def test_load_hdf5_corrupted(self, test_data: EpiAtlasDataset):
        """Verify that file corruption errors are caught/raised."""
        hdf5_list = Hdf5Loader.read_list(test_data.datasource.hdf5_file)
        chosen_file = list(hdf5_list.values())[0]

        # corrupt a file
        os.system(
            f"dd if=/dev/urandom of={chosen_file} bs=1024 seek=$((RANDOM%10)) count=1 conv=notrunc"
        )

        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
        with pytest.raises(OSError, match="file signature not found"):
            hdf5_loader.load_hdf5s(test_data.datasource.hdf5_file)

    def test_adapt_to_environment(self, test_folder: Path, test_data: EpiAtlasDataset):
        """Test that the existence of $SLURM_TMPDIR/hdf5s affects hdf5 loading."""
        # setup
        os.environ["SLURM_TMPDIR"] = str(test_folder)
        shutil.rmtree(test_folder)
        os.makedirs(test_folder / "hdf5s", exist_ok=True)

        # test
        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
        files = hdf5_loader.read_list(test_data.datasource.hdf5_file)
        files = hdf5_loader.adapt_to_environment(files)

        a_path = list(files.values())[0]
        assert str(test_folder) in str(a_path)

        # tearup
        del os.environ["SLURM_TMPDIR"]

    def test_adapt_to_environment_2(self, test_folder: Path, test_data: EpiAtlasDataset):
        """Test that the existence of $SLURM_TMPDIR/$HDF5_PARENT affects hdf5 loading."""
        # setup
        new_parent = "test"
        hdf5_dir = test_folder / new_parent

        os.environ["SLURM_TMPDIR"] = str(test_folder)
        os.environ["HDF5_PARENT"] = str(new_parent)
        shutil.rmtree(test_folder)

        os.makedirs(hdf5_dir, exist_ok=True)

        # test
        hdf5_loader = Hdf5Loader(test_data.datasource.chromsize_file, True)
        files = hdf5_loader.read_list(test_data.datasource.hdf5_file)
        files = hdf5_loader.adapt_to_environment(files)

        a_path = list(files.values())[0]
        assert str(hdf5_dir) in str(a_path)

        # tearup
        del os.environ["SLURM_TMPDIR"]
        del os.environ["HDF5_PARENT"]

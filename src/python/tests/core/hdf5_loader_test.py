"""Test module for hdf5_loader file."""
from __future__ import annotations

import os

import h5py
import pytest

from src.python.core.epiatlas_treatment import EpiAtlasDataset
from src.python.core.hdf5_loader import Hdf5Loader
from src.python.tests.fixtures.epilap_test_data import EpiAtlasTreatmentTestData


class Test_Hdf5Loader:
    """Test class Test_Hdf5Loader"""

    @pytest.fixture(scope="class", autouse=True)
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
        with h5py.File(chosen_file, "r+") as f:
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

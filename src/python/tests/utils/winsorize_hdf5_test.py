"""Tests for the winsorize_hdf5 module.""" ""
import os
import sys

import h5py
import numpy as np
from scipy.stats import mstats

from epi_ml.utils.winsorize_hdf5 import main


def test_winsorization():
    """Integrative test: test winsorize_hdf5 main."""
    # Prepare a small test hdf5 file
    test_file_name = "test.h5"
    test_paths_list_name = "test_list.txt"

    expected_output_name = "test_winsorized-0-0.01.hdf5"

    upper_limit = 0.01

    # Prepare a small test hdf5 file
    # Input data is a fixed numpy array
    input_data = np.arange(0, 100)

    # Expected data is the winsorized version of the input_data
    expected_data = mstats.winsorize(input_data, limits=(0, upper_limit))

    test_file = "test.h5"
    with h5py.File(test_file, "w") as f:
        group = f.create_group("header")
        for i in range(1, 23):
            data = group.create_dataset(f"chr{i}", (100,), dtype="i")
            data[:] = expected_data
        # Add chrX
        data = group.create_dataset("chrX", (100,), dtype="i")
        data[:] = expected_data

    with open(test_paths_list_name, "w", encoding="utf-8") as f:
        f.write(test_file_name)

    # Run the winsorization function
    sys.argv = ["script.py", test_paths_list_name, ".", f"{upper_limit}"]
    main()

    # Check that the data in the new file is as expected
    with h5py.File(expected_output_name, "r") as f:
        header = list(f.keys())[0]
        hdf5_data = f[header]

        for dataset in hdf5_data.values():
            assert np.allclose(
                dataset[:],
                expected_data,
            )

    # Cleanup
    for file in [test_file_name, test_paths_list_name, expected_output_name]:
        os.remove(file)

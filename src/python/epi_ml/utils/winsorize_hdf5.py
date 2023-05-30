"""
Module: winsorize_hdf5

This module provides utility functions for winsorizing HDF5 files.
The main function processes a list of HDF5 files, concatenates all chromosome datasets into one array,
applies winsorization to the global array, and then de-concatenates the winsorized array
back into individual chromosome datasets. The modified HDF5 files are saved in the specified output directory.

Note:
This module requires the h5py and scipy libraries, and the epi_ml package for additional utilities.

Usage:
The main function is used for processing HDF5 files based on the provided command line arguments.
It expects the following arguments:
- hdf5_list (a file with HDF5 filenames)
- output_dir (the directory where the modified HDF5 files will be created).
By running the module as a script, it will execute the main function.

Example:
$ python winsorize_hdf5.py hdf5_list.txt output_directory

Author:
Joanny Raby
"""
# pylint: disable=unexpected-keyword-arg, R0801, line-too-long
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.stats.mstats import winsorize

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker


def winsorize_dataset(array: np.ndarray, limits=(0, 0.01)):
    """Winsorizes a dataset.

    Args:
        dataset: The vector to be winsorized.
        limits: Pair of (lower limit, upper limit) for the winsorization. Defaults to (0, 0.01).

    Returns:
        The winsorized dataset.
    """
    return winsorize(array, limits=limits)


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to create the new hdf5 files.",
    )
    arg_parser.add_argument(
        "ceiling",
        type=float,
        help="Upper limit for winsorization. Ex: 0.01 which signifies setting the top 1% of values to the 99th percentile.",
    )
    return arg_parser.parse_args()


def main() -> None:
    """Main function."""
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    output_dir = Path(cli.output_dir).resolve()

    upper_limit = cli.ceiling
    if upper_limit < 0 or upper_limit > 1:
        raise ValueError(
            f"Upper limit must be between 0 and 1. Provided value: {upper_limit}"
        )
    limits = (0, upper_limit)

    with open(hdf5_list_path, "r", encoding="utf8") as f:
        hdf5_files_path = [Path(line.strip()) for line in f if line.strip()]

    for og_hdf5_path in hdf5_files_path:
        hdf5_path = output_dir / (
            og_hdf5_path.stem + f"_winsorized-{limits[0]}-{limits[1]}.hdf5"
        )
        shutil.copy(og_hdf5_path, hdf5_path)

        with h5py.File(hdf5_path, "r+") as file:
            header = list(file.keys())[0]
            hdf5_data = file[header]

            # Step 1 - Concatenate all chromosomes into one array
            chrom_signals = [hdf5_data[chrom][...] for chrom in hdf5_data.keys()]
            concat_array = np.concatenate(chrom_signals, dtype=np.float32)

            # Step 2 - Winsorize the global array
            winsorized_array = winsorize(concat_array, limits=limits)

            # Step 3 - De-concatenate and update the hdf5 file with new chromosome datasets
            start = 0
            for chrom in hdf5_data.keys():
                end = start + hdf5_data[chrom].shape[0]
                hdf5_data[chrom][...] = winsorized_array[start:end]
                start = end

            file.close()


if __name__ == "__main__":
    main()

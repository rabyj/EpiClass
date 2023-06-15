"""
This module provides functionalities to copy HDF5 files to a new directory, convert their datasets to float32 data type, and repack them to reduce their sizes.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import traceback
import warnings
from pathlib import Path

import h5py
import numpy as np

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )

    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to create to output the new hdf5 files.",
    )
    # fmt: on
    return arg_parser.parse_args()


def copy_hdf5_file(file_path: Path, logdir: Path) -> Path | None:
    """
    Copies an HDF5 file to a new location, appending "_float32.hdf5" to the filename.

    Args:
        file_path (Path): The path to the original HDF5 file.
        logdir (Path): The directory to copy the HDF5 file to.

    Returns:
        Path or None: The path to the copied file if successful, or None if the file already exists at the new location.
    """
    new_hdf5_path = logdir / (file_path.stem + "_float32.hdf5")

    if new_hdf5_path.is_file():
        warnings.warn(f"{new_hdf5_path} already exists. Skipping.")
        return None

    shutil.copy(file_path, new_hdf5_path)

    return new_hdf5_path


def cast_datasets_to_float32(file_path: Path) -> bool:
    """
    Casts all the datasets in an HDF5 file to float32 data type.

    Args:
        file_path (Path): The path to the HDF5 file.

    Returns:
        bool: True if any dataset was casted, False otherwise.
    """
    modified = False
    with h5py.File(file_path, "r+") as f:
        for _, group in f.items():
            for dataset_name, dataset in list(group.items()):
                # Cast the dataset to float32, remove the old dataset and save the new one
                if dataset.dtype == np.float64:
                    modified = True
                    attrs = dict(dataset.attrs.items())

                    og_dataset = dataset[:]
                    casted_dataset = dataset.astype(np.float32)[:]

                    # Verify the difference between the original dataset and the casted dataset
                    diff = np.abs(casted_dataset - og_dataset)
                    max_diff = np.max(diff)
                    if max_diff > 1e-5:
                        raise ValueError(
                            f"Recasting failed, max(diff) > 1e-5: {file_path}"
                        )

                    del group[dataset_name]

                    group.create_dataset(
                        dataset_name, data=casted_dataset, dtype=np.float32
                    )
                    group[dataset_name].attrs.update(attrs)

    return modified


def repack_hdf5_file(file_path: Path) -> None:
    """
    Repacks an HDF5 file to reduce its size. Uses the h5repack command line tool.

    Args:
        file_path (Path): The path to the HDF5 file.
    """
    tmp_path = str(file_path)
    tmp_path = tmp_path + "_repacked.hdf5"
    subprocess.run(["h5repack", file_path, tmp_path], check=True)
    shutil.move(tmp_path, file_path)


def main():
    """
    Main function that parses command-line arguments and performs the operations to copy and cast HDF5 files.
    """
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    logdir = cli.output_dir.resolve()

    if shutil.which("h5repack") is None:
        raise RuntimeError(
            "The required 'h5repack' tool is not available in your system's PATH. Please install it before running this program."
        )

    with open(hdf5_list_path, "r", encoding="utf8") as f:
        hdf5_files = [Path(line.strip()) for line in f if line.strip()]

    for hdf5_file in hdf5_files:
        try:
            new_filepath = copy_hdf5_file(hdf5_file, logdir)
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error: {e}. Skipping file {hdf5_file}\n{traceback.format_exc()}")
            continue

        if new_filepath:
            try:
                modified = cast_datasets_to_float32(new_filepath)
            except ValueError as e:
                warnings.warn(f"Error: {e}.\n Skipping file {new_filepath}")
                new_filepath.unlink(missing_ok=True)
                continue
            if modified:
                print(
                    f"Casting and verification successful. Repacking file {new_filepath}"
                )
                repack_hdf5_file(new_filepath)
            else:
                print("File already existing or no casting needed. Skipping.")
                new_filepath.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

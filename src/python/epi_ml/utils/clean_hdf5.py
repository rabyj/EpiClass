"""
Module: clean_hdf5

This module provides functionality for working with HDF5 files, particularly for genomic data processing.
It offers a set of utility functions that allow loading of BED (Browser Extensible Data) files,
preprocessing of BED intervals, filtering positions based on a provided blacklist,
and modifying HDF5 datasets accordingly.

It primarily consists of the following functions:

- is_position_in_blacklist(start, end, chromosome_intervals, chromosome, verbose=False) -> bool:
    Checks if a given genomic position falls within a predefined blacklist.

- process_file(og_hdf5_path, blacklist_chromosome_intervals, output_dir):
    Processes an individual HDF5 file. Copies the file to an output directory, applies the blacklist filter,
    and writes the changes back into the file.

- main() -> None:
    The main driver function that organizes the entire process of reading, filtering and writing the HDF5 files.

It defines the following type alias:

- BEDIntervals: Alias for List[Tuple[str, int, int]]. Represents genomic intervals from a BED file.

Dependencies:
This module requires the h5py library for HDF5 file operations and
the epi_ml package for argument parsing and directory checking utilities.

Usage:
The main function is invoked to process a list of HDF5 files according to command line arguments.
It expects the following arguments:
- hdf5_list (a file containing a list of HDF5 filenames)
- bed_filter (a path to a BED file containing the blacklist positions)
- output_dir (the directory where the modified HDF5 files will be saved)
- n_jobs (optional; the number of parallel jobs to run)

Example:
$ python clean_hdf5.py hdf5_list.txt blacklist.bed output_directory -n 4

Author:
Joanny Raby
"""
from __future__ import annotations

import argparse
import bisect
import shutil
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import h5py

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker

BEDInterval = Tuple[str, int, int]
BEDIntervals = List[BEDInterval]


def print_h5py_structure(file: str) -> None:
    """Prints the structure of an h5py file.

    Args:
        file: The path to the h5py file.
    """

    def print_structure(name: str, obj: h5py.Group) -> None:
        """Recursive function to print the structure of the h5py file."""
        print(name, type(obj))
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                print_structure(name + "/" + key, obj[key])  # type: ignore

    with h5py.File(file, "r") as f:
        f.visititems(print_structure)


def load_bed(path: Path | str) -> BEDIntervals:
    """Loads the contents of a BED file.

    Args:
        path: The path to the BED file.

    Returns:
        A list of tuples representing the BED intervals (chromosome, start, end).
    """
    data = []
    with open(path, "r", encoding="utf8") as file:
        for line in file:
            parts = line.strip().split("\t")
            chromosome = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            data.append((chromosome, start, end))
    return data


def preprocess_bed(bed: BEDIntervals) -> Dict[str, BEDIntervals]:
    """Preprocesses the BED intervals.

    Args:
        bed: A list of tuples representing the BED intervals (chromosome, start, end).

    Returns:
        A dictionary mapping chromosomes to sorted intervals.
    """
    chromosome_intervals = {}

    for interval in bed:
        chromosome = interval[0]

        if chromosome not in chromosome_intervals:
            chromosome_intervals[chromosome] = []

        chromosome_intervals[chromosome].append(interval)

    for intervals in chromosome_intervals.values():
        intervals.sort(key=lambda interval: interval[1])

    return chromosome_intervals


def is_position_in_blacklist(
    start: int,
    end: int,
    chromosome_intervals: Dict[str, BEDIntervals],
    chromosome: str,
    verbose: bool = False,
) -> bool:
    """Checks if a position is in the blacklist.

    Args:
        start: The start position of the region.
        end: The end position of the region.
        chromosome_intervals: A dictionary mapping chromosomes to sorted intervals.
        chromosome: The chromosome to check.
        verbose: Flag indicating whether to print verbose output. Default is False.

    Returns:
        True if the position is in the blacklist, False otherwise.
    """
    if chromosome not in chromosome_intervals:
        return False

    intervals = chromosome_intervals[chromosome]
    start_positions = [interval[1] for interval in intervals]
    end_positions = [interval[2] - 1 for interval in intervals]  # 1 to 0 based

    start_index = bisect.bisect_right(start_positions, start)

    if start_index > 0:
        end_index = start_index - 1
        interval_start = start_positions[end_index]
        interval_end = end_positions[end_index]
        if interval_start <= end and start <= interval_end:
            if verbose:
                print(
                    f"{chromosome}:{start}-{end} is overlapping {interval_start}-{interval_end}"
                )
            return True

    if start_index < len(start_positions):
        interval_start = start_positions[start_index]
        interval_end = end_positions[start_index]
        if interval_start <= end and start <= interval_end:
            if verbose:
                print(
                    f"{chromosome}:{start}-{end} is overlapping {interval_start}-{interval_end}"
                )
            return True

    return False


def process_file(
    og_hdf5_path: Path,
    blacklist_chromosome_intervals: Dict[str, BEDIntervals],
    output_dir: Path,
):
    """Clean one hdf5 file. Save copy with regions that touch blacklisted regions to 0."""
    hdf5_path = output_dir / (og_hdf5_path.stem + "_0blklst.hdf5")

    if hdf5_path.is_file():
        warnings.warn(f"{hdf5_path} already exists. Skipping.")
        return

    shutil.copy(og_hdf5_path, hdf5_path)

    with h5py.File(hdf5_path, "r+") as file:
        bin_resolution = file.attrs["bin"][0]  # type: ignore # pylint: disable=unsubscriptable-object

        header = list(file.keys())[0]
        hdf5_data = file[header]
        for chromosome, dataset in hdf5_data.items():  # type: ignore
            if chromosome in blacklist_chromosome_intervals:
                for i, _ in enumerate(dataset):
                    position = i * bin_resolution
                    if is_position_in_blacklist(
                        position,
                        position + bin_resolution,
                        blacklist_chromosome_intervals,
                        chromosome,
                        verbose=False,
                    ):
                        dataset[i] = 0


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "bed_filter",
        type=Path,
        help="Path of the bed to of position to put to 0, like a blacklist.",
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to create the new hdf5 files.",
    )
    arg_parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel.",
    )
    return arg_parser.parse_args()


def main() -> None:
    """Main function."""
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    output_dir = cli.output_dir
    blacklist_path = cli.bed_filter
    n_cores = cli.n_jobs

    if n_cores < 1:
        raise ValueError("n_jobs must be >= 1")

    with open(hdf5_list_path, "r", encoding="utf8") as f:
        hdf5_files = [Path(line.strip()) for line in f if line.strip()]

    blacklist_bed = load_bed(blacklist_path)
    blacklist_chromosome_intervals = preprocess_bed(blacklist_bed)

    with Pool(n_cores) as pool:
        pool.starmap(
            process_file,
            [(file, blacklist_chromosome_intervals, output_dir) for file in hdf5_files],
        )


if __name__ == "__main__":
    main()

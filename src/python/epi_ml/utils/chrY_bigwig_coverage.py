"""
This module provides functionalities compute the chrY and chrX coverage from bigwig files.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import pandas as pd
import pyBigWig

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.hdf5_loader import Hdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to create to output the new hdf5 files.",
    )
    return arg_parser.parse_args()


def compute_coverage(file_path: Path) -> Tuple[str, int, int, int]:
    """Compute mean signal value in chrY and chrX

    Return
        Tuple[filename, chrY_coverage, chrX_coverage, chrY_coverage/chrX_coverage]
    """
    try:
        bw = pyBigWig.open(str(file_path), "r")
    except (RuntimeError, OSError) as err:
        print(f"{err}: Could not process {file_path}.", flush=True, file=sys.stderr)
        return (file_path.name, 0, 0, 0)
    chrY_coverage = bw.stats("chrY", exact=True)[0]
    chrX_coverage = bw.stats("chrX", exact=True)[0]
    bw.close()
    return (
        file_path.name,
        chrY_coverage,
        chrX_coverage,
        chrY_coverage / chrX_coverage,
    )


def main():
    """
    Main function that parses command-line arguments and performs the operations to copy and cast HDF5 files.
    """
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    logdir = cli.output_dir.resolve()
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))

    hdf5_files = list(Hdf5Loader.read_list(hdf5_list_path, adapt=False).values())

    with ThreadPoolExecutor(max_workers) as executor:
        result = list(executor.map(compute_coverage, hdf5_files))

    pd.DataFrame(result, columns=["filename", "chrY", "chrX", "chrY/chrX"]).to_csv(
        logdir / "coverage.csv", index=False
    )


if __name__ == "__main__":
    main()

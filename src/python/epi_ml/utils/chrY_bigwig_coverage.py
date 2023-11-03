"""
This module provides functionalities compute the chrY and chrX coverage from bigwig files.
"""
# pylint: disable=invalid-name
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generator, List, Tuple

import pandas as pd
import pyBigWig

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.hdf5_loader import Hdf5Loader
from epi_ml.utils.time import time_now_str


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


def chunks(lst: List, n: int) -> Generator[List, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def compute_coverage(file_path: Path) -> Tuple[str, int, int]:
    """Compute mean signal value in chrY and chrX

    Return
        Tuple[filename, chrY_coverage, chrX_coverage, chrY_coverage/chrX_coverage]
    """
    try:
        bw = pyBigWig.open(str(file_path), "r")
    except (RuntimeError, OSError) as err:
        print(f"{err}: Could not process {file_path}.", flush=True, file=sys.stderr)
        return (file_path.name, 0, 0)

    try:
        chrY_coverage = bw.stats("chrY", exact=True)[0]
        chrX_coverage = bw.stats("chrX", exact=True)[0]
    except RuntimeError as err:
        print(f"{err}: Could not process {file_path}.", flush=True, file=sys.stderr)
        chrY_coverage = 0
        chrX_coverage = 0
    finally:
        bw.close()

    return (
        file_path.name,
        chrY_coverage,
        chrX_coverage,
    )


def main():
    """
    Main function that parses command-line arguments and
    performs the operations to copy and cast HDF5 files.
    """
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    logdir = cli.output_dir.resolve()
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "4"))
    log_every_n_files = 1000  # Define how many bigwigs to process before logging

    hdf5_files = list(Hdf5Loader.read_list(hdf5_list_path, adapt=False).values())

    all_results = []

    # Divide hdf5_files into chunks and process them concurrently
    for idx, chunk in enumerate(chunks(hdf5_files, log_every_n_files)):
        with ThreadPoolExecutor(max_workers) as executor:
            chunk_result = list(executor.map(compute_coverage, chunk))

        all_results.extend(chunk_result)

        # Log the results of the current chunk
        chunk_name = logdir / f"coverage_chunk_{idx}.csv"
        if chunk_name.exists():
            chunk_name = logdir / f"coverage_chunk_{time_now_str()}_.csv"

        pd.DataFrame(chunk_result, columns=["filename", "chrY", "chrX"]).to_csv(
            chunk_name, index=False
        )

    # Combine all results and save if required
    final_name = logdir / "coverage_combined.csv"
    if final_name.exists():
        final_name = logdir / f"coverage_combined_{time_now_str()}.csv"
    pd.DataFrame(all_results, columns=["filename", "chrY", "chrX"]).to_csv(
        final_name, index=False
    )


if __name__ == "__main__":
    main()

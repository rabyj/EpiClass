"""
This module provides a command-line interface for computing the metric of signal bins
for a given set of genomic data.

The script reads genomic data from hdf5 files, which are provided as input through the
command line. Additional inputs include chromosome sizes.

The command line interface requires four arguments:
    1) A file containing hdf5 filenames.
    2) A file containing the sizes of chromosomes.
    3) A directory for log outputs.

The hdf5 files are used to create a dict of signal data, from which the metrics for
each signal bin is computed over all hdf5s. The metrics per bin are then written to a npz file in the log
directory.

Metrics:
    - Mean: The mean of each signal bin.
    - Standard deviation: The standard deviation of each signal bin.
    - Median: The median of each signal bin.
    - IQR: The interquartile range of each signal bin.

npz output: {metric: [list of values]}

Typical usage example:
    $ python compute_variance.py hdf5s.list chrom.sizes logs/
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.data import Hdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes."
        )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="A directory for the logs."
    )
    # fmt: on
    return arg_parser.parse_args()


def compute_metrics(hdf5s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Computes various metrics for signal bins from HDF5 signals.

    Args:
        hdf5s (Dict[str, np.ndarray]): Dictionary of signal data.

    Returns:
        Dict[str, np.ndarray]: Dictionary of computed metrics.
    """
    hdf5_list = list(hdf5s.values())
    mean = np.mean(hdf5_list, axis=0, dtype=np.float64)
    std = np.std(hdf5_list, axis=0, dtype=np.float64)
    median = np.median(hdf5_list, axis=0)
    iqr = np.percentile(hdf5_list, 75, axis=0) - np.percentile(hdf5_list, 25, axis=0)
    return {"mean": mean, "std": std, "median": median, "iqr": iqr}


def main():
    """main called from command line, edit to change behavior"""
    # parse params
    epiml_options = parse_arguments()
    hdf5_list_path = epiml_options.hdf5
    chromsize_path = epiml_options.chromsize
    logdir = epiml_options.logdir

    # md5:signal dict
    signals = (
        Hdf5Loader(chromsize_path, normalization=False)
        .load_hdf5s(hdf5_list_path, strict=True, verbose=False)
        .signals
    )

    metrics = compute_metrics(signals)

    # write to log
    log_file = logdir / "metrics_raw.npz"
    np.savez(log_file, **metrics)

    print(f"Metrics written to {log_file}")


if __name__ == "__main__":
    main()

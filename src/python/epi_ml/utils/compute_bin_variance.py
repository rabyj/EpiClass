"""
This module provides a command-line interface for computing the variance of signal bins
for a given set of genomic data.

The script reads genomic data from hdf5 files, which are provided as input through the
command line. Additional inputs include chromosome sizes and metadata files.

The command line interface requires four arguments:
    1) A file containing hdf5 filenames.
    2) A file containing the sizes of chromosomes.
    3) A metadata JSON file.
    4) A directory for log outputs.

The hdf5 files are used to create a dict of signal data, from which the variance for
each signal bin is computed. The variance is then written to a bedgraph file in the log
directory.

The script utilizes functions from the `argparseutils`, `core.data`, and `utils.bed_utils`
modules of the `epi_ml` package.

Functions:
- parse_arguments(): Parses command-line arguments.
- compute_variance(hdf5s: Dict[str, np.ndarray]): Computes and returns the variance of each signal bin.
- main(): Main script execution. It parses the arguments, loads the data, computes the variance,
  and writes the variance to a bedgraph file.

Typical usage example:
    $ python compute_variance.py hdf5s.txt chrom_sizes.txt metadata.json logs/
"""
import argparse
from pathlib import Path

import numpy as np

from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core import data
from epi_ml.utils import bed_utils


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument("chromsize", type=Path, help="A file with chrom sizes.")
    arg_parser.add_argument("metadata", type=Path, help="A metadata JSON file.")
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="A directory for the logs."
    )
    return arg_parser.parse_args()


def compute_variance(hdf5s):
    """Return array of variance per signal bin from hdf5 signals dict."""
    return np.var(list(hdf5s.values()), axis=0)


def main():
    """main called from command line, edit to change behavior"""
    # parse params
    epiml_options = parse_arguments()

    # load external files
    my_datasource = data.EpiDataSource(
        epiml_options.hdf5, epiml_options.chromsize, epiml_options.metadata
    )

    # load useful info
    hdf5_resolution = my_datasource.hdf5_resolution()
    chroms = my_datasource.load_chrom_sizes()

    # md5:signal dict
    signals = (
        data.Hdf5Loader(my_datasource.chromsize_file, normalization=True)
        .load_hdf5s(my_datasource.hdf5_file)
        .signals
    )

    variance = compute_variance(signals)
    bedgraph_path = epiml_options.logdir / "variance.bedgraph"
    bed_utils.values_to_bedgraph(variance, chroms, hdf5_resolution, bedgraph_path)


if __name__ == "__main__":
    main()

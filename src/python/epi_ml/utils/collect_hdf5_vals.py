"""Collect values from hdf5 files and save them as csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.hdf5_loader import Hdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
    # fmt: off
    arg_parser.add_argument(
        "hdf5_list", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument(
        "chromsize",
        type=Path,
        help="A file with chrom sizes.")
    arg_parser.add_argument(
        "feature_list",
        type=Path,
        help="A file with feature bin indexes in json format.",
    )
    arg_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the values in the hdf5 files as z-scores.",
    )
    arg_parser.add_argument(
        "output_dir",
        type=DirectoryChecker(),
        help="Directory where to save the hdf5 values, as csv.",
    )
    # fmt: on
    return arg_parser.parse_args()


def main() -> None:
    """Main function to collect values from hdf5 files and save them as csv."""
    cli = parse_arguments()

    hdf5_list_path = cli.hdf5_list
    chromsize_path = cli.chromsize
    feature_list_path = cli.feature_list
    normalize_hdf5 = cli.normalize
    output_dir = cli.output_dir

    with open(feature_list_path, "r", encoding="utf8") as f:
        feature_list = json.load(f)

    hdf5_loader = Hdf5Loader(chrom_file=chromsize_path, normalization=normalize_hdf5)
    hdf5_loader.load_hdf5s(data_file=hdf5_list_path, verbose=False, strict=True)

    selected_values = {}
    for md5sum, signal in hdf5_loader.signals.items():
        feature_values = signal[feature_list]
        selected_values[md5sum] = feature_values

    df = pd.DataFrame.from_dict(selected_values, orient="index", columns=feature_list)
    df.to_csv(
        output_dir / f"hdf5_values_{hdf5_list_path.stem}_{feature_list_path.stem}.csv"
    )


if __name__ == "__main__":
    main()

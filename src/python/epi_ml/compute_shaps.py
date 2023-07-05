"""Compute SHAP values of a model."""
# pylint: disable=import-error, line-too-long
from __future__ import annotations

import argparse
import os
from pathlib import Path

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.data import UnknownData
from epi_ml.core.estimators import EstimatorAnalyzer
from epi_ml.core.hdf5_loader import Hdf5Loader
from epi_ml.core.model_pytorch import LightningDenseClassifier
from epi_ml.core.shap_values import LGBM_SHAP_Handler, NN_SHAP_Handler


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Namespace object with parsed arguments.
    """
    arg_parser = ArgumentParser()

    # fmt: off
    gen_group = arg_parser.add_argument_group("General arguments")
    gen_group.add_argument(
        "-m", "--model", required=True, choices=["NN", "LGBM"], help="Model to explain. Neural network or LightGBM.",
    )
    gen_group.add_argument(
        "--background_hdf5", required=True, metavar="background-hdf5", type=Path, help="A file with hdf5 filenames for the explainer background. Use absolute path!"
    )
    gen_group.add_argument(
        "--explain_hdf5", required=True, metavar="explain-hdf5", type=Path, help="A file with hdf5 filenames on which to compute SHAP values. Use absolute path!",
    )
    gen_group.add_argument(
        "--chromsize", required=True, type=Path, help="A file with chrom sizes.",
    )
    gen_group.add_argument(
        "-l", "--logdir", type=DirectoryChecker(), help="Directory for the output logs.",
    )
    gen_group.add_argument(
        "-o", "--output_name", metavar="--output-name", default="", help="Name (not path) of outputted pickle file containing computed SHAP values",
    )
    depend_group = arg_parser.add_argument_group("Model dependant arguments")

    depend_group.add_argument(
        "--model_file", metavar="model_file", type=Path, help="Needed for LGBM. Specify the model file to load.",
    )
    depend_group.add_argument(
        "--model_dir", type=DirectoryChecker(), help="Needed for neural netowork. Directory with 'best_checkpoint.list' file.",
    )
    # fmt: on
    return arg_parser.parse_args()


def compute_shap(
    cli: argparse.Namespace,
    shap_computer: NN_SHAP_Handler | LGBM_SHAP_Handler,
    output_name: str,
):
    """
    Compute SHAP values for a given model.

    Args:
        cli (argparse.Namespace): Parsed command-line arguments.
        model (LightningDenseClassifier or EstimatorAnalyzer): Model to compute SHAP values for.
        shap_computer (NN_SHAP_Handler or LGBM_SHAP_Handler): SHAP computer instance.
        output_name (str): Output name for the SHAP values.
        background_required (bool, optional): Whether the background dataset is required. Defaults to False.
    """
    signals = {}
    hdf5_loader = Hdf5Loader(chrom_file=cli.chromsize, normalization=True)

    hdf5_loader.load_hdf5s(cli.background_hdf5, strict=True)
    signals["background"] = hdf5_loader.signals

    hdf5_loader.load_hdf5s(cli.explain_hdf5, strict=True)
    signals["explain"] = hdf5_loader.signals

    background_set = UnknownData(
        list(signals["background"].keys()),
        list(signals["background"].values()),
        None,
        None,
    )

    explain_set = UnknownData(
        list(signals["explain"].keys()),
        list(signals["explain"].values()),
        None,
        None,
    )

    shap_computer.compute_shaps(
        background_dset=background_set,
        evaluation_dset=explain_set,
        save=True,
        name=output_name,
        num_workers=int(os.getenv("SLURM_CPUS_PER_TASK", "1")),
    )


def main():
    """main"""
    cli = parse_arguments()

    name = cli.output_name

    logdir = cli.logdir
    if logdir is None:
        logdir = Path.cwd()

    model_dir = cli.model_dir

    model_name = cli.model
    if model_name == "NN":
        if not cli.model_dir:
            raise ValueError(
                "Must provide a model directory for neural network models. See help."
            )
        my_model = LightningDenseClassifier.restore_model(model_dir)

        shap_handler = NN_SHAP_Handler(model=my_model, logdir=logdir)
        compute_shap(cli, shap_handler, name)

    elif model_name == "LGBM":
        if not cli.model_file:
            raise ValueError("Must provide a model file for LGBM models.")
        model_analyzer = EstimatorAnalyzer.restore_model_from_path(cli.model_file)

        shap_handler = LGBM_SHAP_Handler(model_analyzer=model_analyzer, logdir=logdir)
        compute_shap(cli, shap_handler, name)


if __name__ == "__main__":
    main()

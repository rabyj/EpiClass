"""Compute SHAP values of a model."""
# pylint: disable=import-error
import argparse
from pathlib import Path

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.analysis import SHAP_Handler
from epi_ml.core.data import DataSetFactory, KnownData
from epi_ml.core.data_source import EpiDataSource
from epi_ml.core.hdf5_loader import Hdf5Loader
from epi_ml.core.metadata import Metadata
from epi_ml.core.model_pytorch import LightningDenseClassifier
from epi_ml.utils.time import time_now


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Namespace object with parsed arguments.
    """
    arg_parser = ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "background_hdf5", metavar="background-hdf5", type=Path, help="A file with hdf5 filenames for the explainer background. Use absolute path!", # pylint: disable=line-too-long
    )
    arg_parser.add_argument(
        "explain_hdf5", metavar="explain-hdf5", type=Path, help="A file with hdf5 filenames on which to compute SHAP values. Use absolute path!", # pylint: disable=line-too-long
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file.",
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs.",
    )
    arg_parser.add_argument(
        "model", type=DirectoryChecker(), help="Directory where to load the classifier to explain.",
    )
    arg_parser.add_argument(
        "-o", "--output_name", metavar="--output-name", type=str, help="Name (not path) of outputed pickle file containing computed SHAP values", # pylint: disable=line-too-long
    )
    # fmt: on
    return arg_parser.parse_args()


def benchmark(metadata: Metadata, datasource: EpiDataSource, model):
    """
    Benchmark the time taken for computing SHAP values based on the size of the background dataset.

    Args:
        metadata (Metadata): A Metadata object containing dataset metadata.
        datasource (EpiDataSource): An EpiDataSource object for accessing the data.
        model: The model used for computing SHAP values.
    """
    full_data = DataSetFactory.from_epidata(
        datasource=datasource,
        label_category="assay",
        metadata=metadata,
        min_class_size=1,
        test_ratio=0,
        validation_ratio=0,
        oversample=False,
    )

    for n in [250]:
        train_data = full_data.train.subsample(list(range(n)))
        shap_computer = SHAP_Handler(model=model, logdir=None)

        eval_size = 25
        evaluation_data = full_data.train.subsample(list(range(n, n + eval_size)))
        t_a = time_now()
        shap_computer.compute_NN(
            background_dset=train_data,
            evaluation_dset=evaluation_data,
            save=False,
        )
        print(f"Time taken with n={n}: {time_now() - t_a}")


def test_background_effect(
    my_metadata: Metadata, my_datasource: EpiDataSource, my_model, logdir: Path
):
    """
    Test the effect of different background datasets on SHAP value computation.

    Args:
        my_metadata (Metadata): A Metadata object containing dataset metadata.
        my_datasource (EpiDataSource): An EpiDataSource object for accessing the data.
        my_model: The model used for computing SHAP values.
        logdir (Path): The path to the output log directory.
    """
    # --- Prefilter metadata ---
    my_metadata.display_labels("assay")
    my_metadata.select_category_subsets("track_type", ["pval", "Unique_plusRaw"])

    assay_list = ["h3k9me3", "h3k36me3", "rna_seq"]
    my_metadata.select_category_subsets("assay", assay_list)

    md5_per_classes = my_metadata.md5_per_class("assay")
    background_1_md5s = md5_per_classes["h3k9me3"][0:10]
    background_2_md5s = md5_per_classes["rna_seq"][0:10]

    evaluation_md5s = (
        md5_per_classes["h3k9me3"][10:20]
        + md5_per_classes["rna_seq"][10:20]
        + md5_per_classes["h3k36me3"][0:10]
    )
    all_md5s = set(background_1_md5s + background_2_md5s + evaluation_md5s)

    for md5 in list(my_metadata.md5s):
        if md5 not in all_md5s:
            del my_metadata[md5]

    full_data = DataSetFactory.from_epidata(
        datasource=my_datasource,
        label_category="assay",
        metadata=my_metadata,
        min_class_size=1,
        test_ratio=0,
        validation_ratio=0,
        oversample=False,
    )

    background_1_idxs = [
        i
        for i, signal_id in enumerate(full_data.train.ids)
        if signal_id in set(background_1_md5s)
    ]
    background_2_idxs = [
        i
        for i, signal_id in enumerate(full_data.train.ids)
        if signal_id in set(background_2_md5s)
    ]
    evaluation_idxs = list(
        set(range(full_data.train.num_examples))
        - set(background_1_idxs + background_2_idxs)
    )

    assert isinstance(full_data.train, KnownData)
    background_1_data = full_data.train.subsample(background_1_idxs)
    background_2_data = full_data.train.subsample(background_2_idxs)

    evaluation_data = full_data.train.subsample(evaluation_idxs)

    for background_data in [background_1_data, background_2_data]:
        shap_computer = SHAP_Handler(model=my_model, logdir=logdir)
        shap_computer.compute_NN(
            background_dset=background_data,
            evaluation_dset=evaluation_data,
            save=True,
            name="background_effect_test",
        )


def main():
    """main"""
    cli = parse_arguments()

    name = cli.output_name
    if name is None:
        name = "shap_values"

    metadata = cli.metadata
    logdir = cli.logdir
    model_dir = logdir
    if cli.model is not None:
        model_dir = cli.model
    my_model = LightningDenseClassifier.restore_model(model_dir)

    signals = {}
    hdf5_loader = Hdf5Loader(chrom_file=cli.chromsize, normalization=True)

    hdf5_loader.load_hdf5s(cli.background_hdf5, strict=True)
    signals["background"] = hdf5_loader.signals

    hdf5_loader.load_hdf5s(cli.explain_hdf5, strict=True)
    signals["explain"] = hdf5_loader.signals

    background_set = KnownData(
        list(signals["background"].keys()),
        list(signals["background"].values()),
        None,
        None,
        metadata,
    )
    explain_set = KnownData(
        list(signals["explain"].keys()),
        list(signals["explain"].values()),
        None,
        None,
        metadata,
    )

    shap_computer = SHAP_Handler(model=my_model, logdir=logdir)
    shap_computer.compute_NN(
        background_dset=background_set,
        evaluation_dset=explain_set,
        save=True,
        name=name,
    )


if __name__ == "__main__":
    main()

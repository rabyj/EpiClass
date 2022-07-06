"""Main"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from epi_ml.python.argparseutils.directorychecker import DirectoryChecker

from epi_ml.python.core import metadata
from epi_ml.python.core.data_source import EpiDataSource
from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment
from epi_ml.python.core.estimators import Svm, Ensemble


class DatasetError(Exception):
    """Custom error"""
    def __init__(self, *args: object) -> None:
        print("\n--- ERROR : Verify source files, filters, and min_class_size. ---\n", file=sys.stderr)
        super().__init__(*args)


def time_now():
    """Return datetime of call without microseconds"""
    return datetime.utcnow().replace(microsecond=0)


def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("category", type=str, help="The metatada category to analyse.")
    arg_parser.add_argument("hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!")
    arg_parser.add_argument("chromsize", type=Path, help="A file with chrom sizes.")
    arg_parser.add_argument("metadata", type=Path, help="A metadata JSON file.")
    arg_parser.add_argument("logdir", type=DirectoryChecker(), help="Directory for the output logs.")
    return arg_parser.parse_args(args)


def main(args):
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments(args)

    my_datasource = EpiDataSource(
        cli.hdf5,
        cli.chromsize,
        cli.metadata
        )


    # --- LOAD DATA ---
    my_metadata = metadata.Metadata(my_datasource.metadata_file)
    my_metadata.remove_category_subsets(label_category="track_type", labels=["Unique.raw"])


    # --- DO THE STUFF ---
    if os.getenv("ASSAY_LIST") is not None:
        assay_list = json.loads(os.environ["ASSAY_LIST"])
        print(f"Going to only keep targets with {assay_list}")
    else:
        assay_list = my_metadata.unique_classes(cli.category)
        print("No assay list")


    loading_begin = time_now()
    ea_handler = EpiAtlasTreatment(my_datasource, cli.category, assay_list)
    loading_time = time_now() - loading_begin
    print(f"Initial hdf5 loading time: {loading_time}")

    time_before_split = time_now()
    for i, my_data in enumerate(ea_handler.yield_split()):
        iteration_time = time_now() - time_before_split
        print(f"Set loading/splitting time: {iteration_time}")

        begin_loop = time_now()

        logdir = Path(cli.logdir)
        print(f"\n\nSplit {i} training size: {my_data.train.num_examples}")

        if my_data.train.num_examples == 0 or my_data.validation.num_examples == 0:
            raise DatasetError("Trying to train without any training or validation data.")

        # Warning : output mapping of model created from training dataset
        if i == 0:
            mapping_file = logdir / "training_mapping.tsv"
            my_data.save_mapping(mapping_file)
            nb_files = len(set(my_data.train.ids.tolist() + my_data.validation.ids.tolist()))
            print(f"Number of files used globally {nb_files}")

        # --- Create and fit estimators ---
        for my_model in [Svm(my_data), Ensemble(my_data)]:
            print(f"Estimator is {my_model}")

            before_train = time_now()
            my_model.train()
            training_time = time_now() - before_train
            print(f"training time: {training_time}")

            my_model.metrics()

            end_loop = time_now()
            loop_time = end_loop - begin_loop
            print(f"Loop time (excludes split time): {loop_time}\n")
            time_before_split = time_now()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])

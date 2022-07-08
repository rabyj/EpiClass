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

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

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


def tune_svm(ea_handler: EpiAtlasTreatment):
    """Apply Bayesian optimization over hyper parameters"""
    my_svm = Svm(ea_handler.classes)

    params = {
        'C': Real(1e-6, 1e+6, prior='log-uniform'),
        'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'degree': Integer(1,8),
        'kernel': Categorical(['linear', 'poly', 'rbf'])
    }

    opt = BayesSearchCV(
        my_svm, search_spaces=params, cv=ea_handler.split(),
        n_iter=100, random_state=42, return_train_score=True, error_score=-1
        )

    total_data = ea_handler.create_total_data()
    print(f"Number of files used globally {len(total_data)}")

    opt.fit(X=total_data.signals, y=total_data.encoded_labels)

    print(f"val. score: {opt.best_score_}")
    print(f"best params: {opt.best_params_}")


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


    my_metadata = metadata.Metadata(my_datasource.metadata_file)
    my_metadata.remove_category_subsets(label_category="track_type", labels=["Unique.raw"])


    if os.getenv("ASSAY_LIST") is not None:
        assay_list = json.loads(os.environ["ASSAY_LIST"])
        print(f"Going to only keep targets with {assay_list}")
    else:
        assay_list = my_metadata.unique_classes(cli.category)
        print("No assay list")


    loading_begin = time_now()
    ea_handler = EpiAtlasTreatment(my_datasource, cli.category, assay_list, n_fold=2, test_ratio=0, min_class_size=1)
    loading_time = time_now() - loading_begin
    print(f"Initial hdf5 loading time: {loading_time}")

    tune_svm(ea_handler)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])

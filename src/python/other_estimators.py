"""Main"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import src.python.core.estimators as estimators
from src.python.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from src.python.argparseutils.directorychecker import DirectoryChecker
from src.python.core import data, metadata
from src.python.core.data_source import EpiDataSource
from src.python.core.epiatlas_treatment import EpiAtlasFoldFactory
from src.python.core.lgbm import tune_lgbm
from src.python.utils.modify_metadata import filter_cell_types_by_pairs
from src.python.utils.time import time_now

if os.getenv("CONCURRENT_CV") is not None:
    CONCURRENT_CV = int(os.environ["CONCURRENT_CV"])
else:
    CONCURRENT_CV = 1


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    # fmt: off
    parser = ArgumentParser()
    group1 = parser.add_argument_group("General")
    group1.add_argument(
        "category", type=str, help="The metatada category to analyse."
        )
    group1.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    group1.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes."
        )
    group1.add_argument(
        "metadata", type=Path, help="A metadata JSON file."
        )
    group1.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs."
    )
    group1.add_argument(
        "--models", nargs="+", type=str, help="Specify models to tune and/or predict.",
        choices=["all", "LinearSVC", "RF", "LR", "LGBM"], default=["all"]
        )

    mode = parser.add_argument_group("Mode")
    mode = mode.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--tune", action="store_true", help="Search best hyperparameters."
        )
    mode.add_argument(
        "--predict", action="store_true", help="Fit and predict using hyperparameters."
        )
    mode.add_argument(
        "--predict-new", action="store_true", help="Use saved models to predict labels of new samples."
        )
    mode.add_argument(
        "--full-run", action="store_true", help="Tune then predict"
        )

    tune = parser.add_argument_group("Tune")
    tune.add_argument(
        "-n",
        type=int,
        default=30,
        help="Number of BayesSearchCV hyperparameters iterations.",
    )
    # fmt: on
    return parser.parse_args()


def main():
    """Takes command line arguments."""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments()

    if cli.tune:
        mode_tune = True
        mode_predict = False
    elif cli.predict:
        mode_tune = False
        mode_predict = True
    elif cli.full_run:
        mode_tune = True
        mode_predict = True
    elif cli.predict_new:
        mode_tune = False
        mode_predict = False
    else:
        raise ValueError("Houston we have a problem.")

    if "all" in cli.models:
        models = estimators.model_mapping.keys()
    else:
        models = cli.models

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)

    # --- Prefilter metadata, must put in EpiAtlasDataset to actually use it ---
    my_metadata = metadata.Metadata(my_datasource.metadata_file)
    my_metadata.remove_category_subsets(
        label_category="track_type", labels=["Unique.raw"]
    )

    label_list = metadata.env_filtering(my_metadata, cli.category)

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = 10

    if cli.category == "harm_sample_ontology_intermediate":
        my_metadata = filter_cell_types_by_pairs(my_metadata)

    # Tuning mode
    loading_begin = time_now()
    if mode_tune is True:  # type: ignore
        print("Entering tuning mode")
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            my_datasource,
            cli.category,
            label_list,
            n_fold=estimators.NFOLD_TUNE,
            test_ratio=0.1,
            min_class_size=min_class_size,
            metadata=my_metadata,
        )
        loading_time = time_now() - loading_begin
        print(f"Initial hdf5 loading time: {loading_time}")

        n_iter = cli.n

        for name in models:
            if name == "LGBM":
                # optuna.logging.set_verbosity(optuna.logging.DEBUG)  # type: ignore
                tune_lgbm(ea_handler, cli.logdir)
            else:
                estimators.optimize_estimator(ea_handler, cli.logdir, n_iter, name)

    # Predict mode
    # TODO: Pre-check, with a separate init script for best_params.json existence
    if mode_predict is True:  # type: ignore
        print("Entering fit/prediction mode")

        pattern = f"{cli.logdir / estimators.best_params_file_format.format(name='*')}"
        hparam_files = glob.glob(pattern)
        if hparam_files:

            ea_handler = EpiAtlasFoldFactory.from_datasource(
                my_datasource,
                cli.category,
                label_list,
                n_fold=estimators.NFOLD_PREDICT,
                test_ratio=0,
                min_class_size=min_class_size,
                metadata=my_metadata,
            )
            loading_time = time_now() - loading_begin
            print(f"Initial hdf5 loading time: {loading_time}")

            # Intersect available hparam files with chose models.
            available = set([estimators.get_model_name(path) for path in hparam_files])
            chosen = available & set(models)
            hparam_files = [pattern.replace("*", name) for name in chosen]

            if not hparam_files:
                print("No parameters file found, finishing now.")
                sys.exit()

            for filepath in hparam_files:

                print(f"Using {Path(filepath).resolve()}.")
                with open(filepath, "r", encoding="utf-8") as file:
                    hparams = json.load(file)

                name = estimators.get_model_name(filepath)
                if name == "LGBM":
                    hparams = {
                        k: v
                        for k, v in hparams.items()
                        if k in estimators.lgbm_allowed_params
                    }

                estimator = estimators.model_mapping[name]
                estimator.set_params(**hparams)

                estimators.run_predictions(ea_handler, estimator, name, cli.logdir)

        else:
            print("No parameters file found, finishing now.")
            sys.exit()

    # Giving predictions with chosen models, for all files in hdf5 list.
    if cli.predict_new:

        pattern = "{log}/**{name}*.pickle"
        to_load = []
        for model in models:
            save_name = estimators.save_mapping[model]
            to_load += glob.glob(pattern.format(log=cli.logdir, name=save_name))

        if to_load:

            my_data = data.DataSetFactory.from_epidata(
                my_datasource,
                my_metadata,
                cli.category,
                min_class_size=min_class_size,
                validation_ratio=0,
                test_ratio=1,
                onehot=False,
                oversample=False,
            )

            for model_path in to_load:
                my_model = estimators.EstimatorAnalyzer.restore_model_from_path(
                    full_path=model_path
                )
                my_analyzer = estimators.EstimatorAnalyzer(
                    classes=my_metadata.unique_classes(cli.category), estimator=my_model
                )

                predict_path = (
                    cli.logdir / f"{Path(model_path).stem}_prediction_{cli.hdf5.stem}.csv"
                )

                print(f"Saving predictions to: {predict_path}")
                my_analyzer.predict_file(
                    ids=my_data.test.ids,
                    X=my_data.test.signals,
                    y=my_data.test.encoded_labels,
                    log=predict_path,
                )

        else:
            print("No saved model file found, finishing now.")
            sys.exit()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()

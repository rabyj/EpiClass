"""Main"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper
from skopt.space import Categorical, Integer, Real
from tabulate import tabulate

from epi_ml.python.argparseutils.directorychecker import DirectoryChecker
from epi_ml.python.core import metadata
from epi_ml.python.core.data_source import EpiDataSource
from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment
from epi_ml.python.core.estimators import EstimatorAnalyzer
from epi_ml.python.utils.time import time_now

NFOLD_TUNE = 9
NFOLD_PREDICT = 10
RNG = np.random.RandomState(42)
SCORES = {
    "acc": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "mcc": make_scorer(matthews_corrcoef),
}
SVM_RBF_SEARCH = {
    "model__C": Real(1e-6, 1e6, prior="log-uniform"),
    "model__gamma": Real(1e-6, 1e1, prior="log-uniform"),
}
SVM_LIN_SEARCH = {
    "model__C": Real(1e-6, 1e6, prior="log-uniform"),
    "model__loss": Categorical(["hinge", "squared_hinge"]),
    "model__intercept_scaling": Categorical([1, 5]),
}
RF_SEARCH = {
    "model__n_estimators": Categorical([500, 1000]),
    "model__criterion": Categorical(["gini", "entropy", "log_loss"]),
    "model__max_features": Categorical(["sqrt", "log2"]),
    "model__bootstrap": Categorical([True]),
    "model__random_state": Categorical([RNG]),
    "model__min_samples_leaf": Integer(1, 5),
    "model__min_samples_split": Categorical([0.01, 0.05, 0.1, 0.3]),
}

mapping = {
    "LinearSVC": Pipeline(steps=[("scaler", StandardScaler()), ("model", LinearSVC())]),
    "SVC": Pipeline(steps=[("scaler", StandardScaler()), ("model", SVC())]),
    "RF": Pipeline(steps=[("model", RandomForestClassifier())]),
}

if os.getenv("CONCURRENT_CV") is not None:
    CONCURRENT_CV = int(os.environ["CONCURRENT_CV"])
else:
    CONCURRENT_CV = 1

# TODO: make recurring path pattern a global variable that gets formatted with a specific name


def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    # fmt: off
    parser = argparse.ArgumentParser()
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

    mode = parser.add_argument_group("Mode")
    mode = mode.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--tune", action="store_true", help="Search best hyperparameters."
    )
    mode.add_argument(
        "--predict", action="store_true", help="Fit and predict using hyperparameters."
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
    tune.add_argument(
        "--only-svm", action="store_true", help="Only use SVM estimator. Runs both linear and RBF."
        )
    tune.add_argument(
        "--only-rf", action="store_true", help="Only use random forest estimator."
    )

    predict = parser.add_argument_group("Predict")
    predict.add_argument(
        "--hparams",
        type=Path,
        help="A file with chosen hyperparameters for each estimator. Needs a specific format. Will search for a file for each estimator by default.",
    )
    # fmt: on
    return parser.parse_args(args)


def best_params_cb(result):
    """BayesSearchCV callback"""
    print(f"Best params yet: {result.x}")


deadline_cb = DeadlineStopper(total_time=60 * 60 * 8)


def tune_estimator(
    model,
    ea_handler: EpiAtlasTreatment,
    params: dict,
    standard_scaling: bool,
    n_iter: int,
    concurrent_cv: int = 1,
    n_jobs: int | None = None,
):
    """Apply Bayesian optimization over hyperparameters search space.

    n_iter: Total number of parameter settings to sample.
    concurrent_cv: Number of full cross-validation process (X folds) to run in parallel
    n_jobs: Number of jobs to run in parallel. Max NFOLD_TUNE * concurrent_cv.
    """
    if standard_scaling:
        pipe = Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])
    else:
        pipe = Pipeline(steps=[("model", model)])

    if n_jobs is None:
        n_jobs = int(NFOLD_TUNE * concurrent_cv)

    if n_jobs > 48:
        raise AssertionError("More jobs than cores asked, max 48 jobs.")

    total_data = ea_handler.create_total_data()
    print(f"Number of files used globally {len(total_data)}")

    opt = BayesSearchCV(
        pipe,
        search_spaces=params,
        cv=ea_handler.split(total_data),
        random_state=RNG,
        return_train_score=True,
        error_score=-1,  # type: ignore
        verbose=3,
        scoring=SCORES,
        refit="acc",  # type: ignore
        n_jobs=n_jobs,
        n_points=concurrent_cv,
        n_iter=n_iter,
    )

    opt.fit(
        X=total_data.signals,
        y=total_data.encoded_labels,
        callback=[best_params_cb, deadline_cb],
    )

    print(f"best params: {opt.best_params_}")
    return opt


def optimize_svm(ea_handler: EpiAtlasTreatment, logdir: Path, n_iter: int):
    """Optimize an sklearn SVC over a hyperparameter space. n_iter divided in two for linear and rbf kernel."""
    print("Starting SVM optimization")
    n_iter = n_iter // 2

    start_train = time_now()
    opt = tune_estimator(
        model=LinearSVC(),
        ea_handler=ea_handler,
        params=SVM_LIN_SEARCH,
        standard_scaling=True,
        n_iter=n_iter,
        concurrent_cv=CONCURRENT_CV,
    )
    print(f"Total linear SVM optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers="keys", tablefmt="psql"))  # type: ignore
    df.to_csv(logdir / "LinearSVC_optim.csv", sep=",")

    with open(logdir / "LinearSVC_best_params.json", "w", encoding="utf-8") as f:
        json.dump(obj=opt.best_params_, fp=f)

    start_train = time_now()
    opt = tune_estimator(
        SVC(cache_size=3000, kernel="rbf"),
        ea_handler=ea_handler,
        params=SVM_RBF_SEARCH,
        standard_scaling=True,
        n_iter=n_iter,
        concurrent_cv=CONCURRENT_CV,
    )
    print(f"Total rbf SVM optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers="keys", tablefmt="psql"))  # type: ignore
    df.to_csv(logdir / "SVC_optim.csv", sep=",")

    with open(logdir / "SVC_best_params.json", "w", encoding="utf-8") as f:
        json.dump(obj=opt.best_params_, fp=f)


def optimize_rf(ea_handler: EpiAtlasTreatment, logdir: Path, n_iter: int):
    """Optimize an sklearn random forest over a hyperparameter space."""
    print("Starting RF optimization")
    start_train = time_now()
    opt = tune_estimator(
        model=RandomForestClassifier(),
        ea_handler=ea_handler,
        params=RF_SEARCH,
        standard_scaling=False,
        n_iter=n_iter,
        concurrent_cv=CONCURRENT_CV,
    )
    print(f"Total RF optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers="keys", tablefmt="psql"))  # type: ignore
    df.to_csv(logdir / "RF_optim.csv", sep=",")

    with open(logdir / "RF_best_params.json", "w", encoding="utf-8") as f:
        json.dump(obj=opt.best_params_, fp=f)


def run_predictions(ea_handler: EpiAtlasTreatment, estimator, name: str, logdir: Path):
    """Run predictions"""
    for i, my_data in enumerate(ea_handler.yield_split()):

        print(f"Split {i} training size: {my_data.train.num_examples}")
        nb_files = len(
            set(my_data.train.ids.tolist() + my_data.validation.ids.tolist())
        )
        print(f"Total nb of files: {nb_files}")

        estimator.fit(X=my_data.train.signals, y=my_data.train.encoded_labels)

        analyzer = EstimatorAnalyzer(my_data.classes, estimator)

        X, y = my_data.validation.signals, my_data.validation.encoded_labels
        analyzer.metrics(X, y)
        analyzer.predict_file(
            my_data.validation.ids,
            X,
            y,
            logdir / f"{name}_split{i}_validation_prediction.csv",
        )


def main(args):
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments(args)

    if cli.tune is True:
        mode_tune = True
        mode_predict = False
    elif cli.predict is True:
        mode_tune = False
        mode_predict = True
    elif cli.full_run is True:
        mode_tune = True
        mode_predict = True

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)

    my_metadata = metadata.Metadata(my_datasource.metadata_file)
    my_metadata.remove_category_subsets(
        label_category="track_type", labels=["Unique.raw"]
    )

    if os.getenv("EXCLUDE_LIST") is not None:
        exclude_list = json.loads(os.environ["EXCLUDE_LIST"])
        my_metadata.remove_category_subsets(
            label_category=cli.category, labels=exclude_list
        )

    if os.getenv("ASSAY_LIST") is not None:
        assay_list = json.loads(os.environ["ASSAY_LIST"])
        print(f"Going to only keep targets with {assay_list}")
    else:
        assay_list = my_metadata.unique_classes(cli.category)
        print("No assay list")

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = 10

    loading_begin = time_now()
    if mode_tune is True:  # type: ignore
        print("Entering tuning mode")
        ea_handler = EpiAtlasTreatment(
            my_datasource,
            cli.category,
            assay_list,
            n_fold=NFOLD_TUNE,
            test_ratio=0.1,
            min_class_size=min_class_size,
        )
        loading_time = time_now() - loading_begin
        print(f"Initial hdf5 loading time: {loading_time}")

        n_iter = cli.n
        if cli.only_svm:
            optimize_svm(ea_handler, cli.logdir, n_iter)
            sys.exit()
        elif cli.only_rf:
            optimize_rf(ea_handler, cli.logdir, n_iter)
            sys.exit()
        else:
            optimize_rf(ea_handler, cli.logdir, n_iter)
            optimize_svm(ea_handler, cli.logdir, n_iter)

    if mode_predict is True:  # type: ignore
        print("Entering fit/prediction mode")
        if cli.hparams is not None and not cli.hparams.exists():
            raise AssertionError("hparams file does not exist: {cli.hparams}")

        ea_handler = EpiAtlasTreatment(
            my_datasource,
            cli.category,
            assay_list,
            n_fold=NFOLD_PREDICT,
            test_ratio=0,
            min_class_size=min_class_size,
        )
        loading_time = time_now() - loading_begin
        print(f"Initial hdf5 loading time: {loading_time}")

        if cli.hparams is not None:
            with open(cli.hparams, "r", encoding="utf-8") as file:
                hparams = json.load(file)

            for name, hparams in hparams.items():
                estimator = mapping[name]
                estimator.set_params(**hparams)
                print(
                    f"Fitting and making predictions with {name} estimator. Fit with hparams {hparams}."
                )
                run_predictions(ea_handler, estimator, name, cli.logdir)

        else:
            pattern = f"{cli.logdir / '*_best_params.json'}"
            hparam_files = glob.glob(pattern)
            for filepath in hparam_files:

                with open(filepath, "r", encoding="utf-8") as file:
                    hparams = json.load(file)

                filepath = Path(filepath)
                name = filepath.stem.split(sep="_", maxsplit=1)[0]
                estimator = mapping[name]
                estimator.set_params(**hparams)

                run_predictions(ea_handler, estimator, name, cli.logdir)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])

"""Main"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from tabulate import tabulate
import pandas as pd

from epi_ml.python.argparseutils.directorychecker import DirectoryChecker

from epi_ml.python.utils.time import time_now
from epi_ml.python.core import metadata
from epi_ml.python.core.data_source import EpiDataSource
from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment
from epi_ml.python.core.estimators import EstimatorAnalyzer

NFOLD = 9
RNG = np.random.RandomState(42)
SCORES = {
    "acc":"accuracy",
    "precision":"precision_macro",
    "recall":"recall_macro",
    "f1":"f1_macro",
    "mcc":make_scorer(matthews_corrcoef)
}
SVM_RBF_SEARCH = {
    "model__C": Real(1e-6, 1e+6, prior="log-uniform"),
    "model__gamma": Real(1e-6, 1e+1, prior="log-uniform"),
}
SVM_LIN_SEARCH = {
    "model__C": Real(1e-6, 1e+6, prior="log-uniform"),
    "model__loss": Categorical(["hinge", "squared_hinge"]),
    "model__intercept_scaling": Categorical([1, 5]),
}
RF_SEARCH = {
    "model__n_estimators": Categorical([500, 1000]),
    "model__criterion": Categorical(["gini", "entropy", "log_loss"]),
    "model__max_features": Categorical(["sqrt", "log2"]),
    "model__bootstrap": Categorical([True]),
    "model__random_state": Categorical([RNG]),
    "model__min_samples_leaf": Integer(1,5),
    "model__min_samples_split": Categorical([0.01, 0.05, 0.1, 0.3])
}


def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--only-svm", action="store_true", help="Only test SVM estimator.")
    arg_parser.add_argument("--only-rf", action="store_true", help="Only test random forest estimator.")
    arg_parser.add_argument("category", type=str, help="The metatada category to analyse.")
    arg_parser.add_argument("hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!")
    arg_parser.add_argument("chromsize", type=Path, help="A file with chrom sizes.")
    arg_parser.add_argument("metadata", type=Path, help="A metadata JSON file.")
    arg_parser.add_argument("logdir", type=DirectoryChecker(), help="Directory for the output logs.")
    arg_parser.add_argument("-n", type=int, default=30, help="Number of BayesSearchCV hyperparameters iterations.")
    return arg_parser.parse_args(args)

def time_now():
    """Return datetime of call without microseconds"""
    return datetime.utcnow().replace(microsecond=0)


def on_step(result):
    """BayesSearchCV callback"""
    print(f"Best params yet: {result.x}")

def tune_estimator(my_model, ea_handler: EpiAtlasTreatment, params: dict, n_iter: int, concurrent_cv: int=2):
    """Apply Bayesian optimization over hyperparameters search space."""
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", my_model)
    ])

    opt = BayesSearchCV(
        pipe, search_spaces=params, cv=ea_handler.split(),
        random_state=RNG, return_train_score=True, error_score=-1, verbose=3,
        scoring=SCORES, refit="acc",
        n_jobs=NFOLD*concurrent_cv, n_iter=n_iter
        )

    total_data = ea_handler.create_total_data()
    print(f"Number of files used globally {len(total_data)}")

    opt.fit(X=total_data.signals, y=total_data.encoded_labels, callback=on_step)

    print(f"best params: {opt.best_params_}")
    return opt

def optimize_svm(ea_handler: EpiAtlasTreatment, logdir: Path, n_iter: int):
    """Optimize an sklearn SVC over a hyperparameter space."""
    n_iter = n_iter//2
    concurrent_cv = 3
    print("Starting SVM optimization")
    start_train = time_now()
    opt = tune_estimator(
        LinearSVC(), ea_handler, SVM_LIN_SEARCH,
        n_iter=n_iter, concurrent_cv=concurrent_cv
    )
    print(f"Total linear SVM optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers='keys', tablefmt='psql'))  # type: ignore
    df.to_csv(logdir / "SVM_lin_optim.csv", sep=",")

    start_train = time_now()
    opt = tune_estimator(
        SVC(cache_size=3000, kernel="rbf"), ea_handler, SVM_RBF_SEARCH,
        n_iter=n_iter, concurrent_cv=concurrent_cv
        )
    print(f"Total rbf SVM optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers='keys', tablefmt='psql'))  # type: ignore
    df.to_csv(logdir / "SVM_rbf_optim.csv", sep=",")

def optimize_rf(ea_handler: EpiAtlasTreatment, logdir: Path, n_iter: int):
    """Optimize an sklearn random forest over a hyperparameter space."""
    print("Starting RF optimization")
    start_train = time_now()
    opt = tune_estimator(RandomForestClassifier(), ea_handler, RF_SEARCH, n_iter, concurrent_cv=5)
    print(f"Total RF optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers='keys', tablefmt='psql'))  # type: ignore
    df.to_csv(logdir / "RF_optim.csv", sep=",")


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

    if os.getenv("EXCLUDE_LIST") is not None:
        exclude_list = json.loads(os.environ["EXCLUDE_LIST"])
        my_metadata.remove_category_subsets(label_category=cli.category, labels=exclude_list)

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
    ea_handler = EpiAtlasTreatment(my_datasource, cli.category, assay_list,
    n_fold=NFOLD, test_ratio=0.1, min_class_size=min_class_size
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


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])

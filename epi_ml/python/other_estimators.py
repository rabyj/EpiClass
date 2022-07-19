"""Main"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from tabulate import tabulate
import pandas as pd

from epi_ml.python.argparseutils.directorychecker import DirectoryChecker

from epi_ml.python.core import metadata
from epi_ml.python.core.data_source import EpiDataSource
from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment
from epi_ml.python.core.estimators import EstimatorAnalyzer


RNG = np.random.RandomState(42)
SCORES = {
    "acc":"accuracy",
    "precision":"precision_macro",
    "recall":"recall_macro",
    "f1":"f1_macro",
    "mcc":make_scorer(matthews_corrcoef)
}
SVM_SEARCH = {
    "model__C": Real(1e-6, 1e+6, prior="log-uniform"),
    "model__gamma": Real(1e-6, 1e+1, prior="log-uniform"),
    "model__degree": Integer(1,8),
    "model__kernel": Categorical(["linear", "poly", "rbf"])
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


def on_step(result):
    """BayesSearchCV callback"""
    print(f"Best params yet: {result.x}")


def tune_estimator(my_model, ea_handler: EpiAtlasTreatment, params:dict):
    """Apply Bayesian optimization over hyperparameters search space."""
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", my_model)
    ])

    opt = BayesSearchCV(
        pipe, search_spaces=params, cv=ea_handler.split(),
        n_iter=10, random_state=RNG, return_train_score=True, error_score=-1, verbose=3,
        scoring=SCORES, refit="acc",
        n_jobs=1
        )

    total_data = ea_handler.create_total_data()
    print(f"Number of files used globally {len(total_data)}")

    opt.fit(X=total_data.signals, y=total_data.encoded_labels, callback=on_step)

    print(f"best params: {opt.best_params_}")
    return opt


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
    ea_handler = EpiAtlasTreatment(my_datasource, cli.category, assay_list, n_fold=9, test_ratio=0.1, min_class_size=10)
    loading_time = time_now() - loading_begin
    print(f"Initial hdf5 loading time: {loading_time}")

    print("Starting RF optimization")
    start_train = time_now()
    opt = tune_estimator(RandomForestClassifier(), ea_handler, RF_SEARCH)
    print(f"Total RF optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv(cli.logdir / "RF_optim.csv", sep=",")


    print("Starting SVM optimization")
    start_train = time_now()
    opt = tune_estimator(SVC(), ea_handler, SVM_SEARCH)
    print(f"Total SVM optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    df.to_csv(cli.logdir / "SVM_optim.csv", sep=",")



if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])

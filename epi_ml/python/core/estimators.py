"""Module for wrappers around simple sklearn machine learning estimators."""
# pylint: disable=no-member
from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper
from skopt.space import Categorical, Integer, Real
from tabulate import tabulate

from .analysis import write_pred_table
from epi_ml.python.core.data import DataSet
from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment
from epi_ml.python.utils.check_dir import create_dirs
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
SVM_LIN_SEARCH = {
    "model__C": Real(1e-6, 1e6, prior="log-uniform"),
    "model__loss": Categorical(["hinge", "squared_hinge"]),
    "model__intercept_scaling": Categorical([1, 5]),
}
RF_SEARCH = {
    "model__n_estimators": Categorical([500, 1000]),
    "model__criterion": Categorical(["gini", "entropy", "log_loss"]),
    "model__max_features": Categorical(["sqrt", "log2"]),
    "model__min_samples_leaf": Integer(1, 5),
    "model__min_samples_split": Categorical([0.002, 0.01, 0.05, 0.1, 0.3]),
}
LR_SEARCH = {
    "model__C": Real(1e-6, 1e6, prior="log-uniform"),
}

# fmt: off
model_mapping = {
    "LinearSVC": Pipeline(steps=[("scaler", StandardScaler()), ("model", LinearSVC())]),
    "RF": Pipeline(steps=[("model", RandomForestClassifier(random_state=RNG, bootstrap=True))]),
    "LR": Pipeline(steps=[
        ("model", LogisticRegression(penalty="l2", multi_class="multinomial", solver="lbfgs", dual=False, fit_intercept=True, warm_start=True, max_iter=1000))
        ]),
    "LGBM": Pipeline(steps=[("model", LGBMClassifier())]),  # type: ignore
}
# fmt: on

search_mapping = {
    "LinearSVC": SVM_LIN_SEARCH,
    "RF": RF_SEARCH,
    "LR": LR_SEARCH,
}

save_mapping = {
    "LinearSVC": "LinearSVC",
    "RF": "RandomForestClassifier",
    "LR": "LogisticRegression",
    "LGBM":"LGBMClassifier"
}

tune_results_file_format = "{name}_optim.csv"
best_params_file_format = "{name}_best_params.json"


class EstimatorAnalyzer(object):
    """Generic class to analyze results given by an estimator."""

    def __init__(self, classes, estimator):
        self.classes = sorted(classes)
        self.mapping = dict(enumerate(self.classes))
        self.encoder = LabelBinarizer().fit(list(self.mapping.keys()))

        self._clf = estimator
        self._name = self._get_name(estimator)

    @staticmethod
    def _get_name(estimator) -> str:
        """Return estimator model name."""
        name = type(estimator).__name__
        if name == "Pipeline":
            name = type(estimator.named_steps["model"]).__name__
        return name

    @property
    def name(self) -> str:
        """Return classifier name."""
        return self._get_name(self._clf)

    def metrics(self, X, y, verbose=True):
        """Return a dict of metrics over given set"""
        y_pred = self._clf.predict(X)
        y_true = y

        val_acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        val_precision = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
        val_recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
        val_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
        val_mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)

        metrics_dict = {
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_mcc": val_mcc,
        }

        if verbose:
            EstimatorAnalyzer.print_metrics(metrics_dict)

        return metrics_dict

    @staticmethod
    def print_metrics(metrics_dict: dict):
        """Print metrics"""
        print(f"Validation Accuracy: {metrics_dict['val_acc']}")
        print(f"Validation Precision: {metrics_dict['val_precision']}")
        print(f"Validation Recall: {metrics_dict['val_recall']}")
        print(f"Validation F1_score: {metrics_dict['val_f1']}")
        print(f"Validation MCC: {metrics_dict['val_mcc']}")

    def predict_file(self, ids, X, y, log):
        """Write predictions table for validation set."""

        try:
            if self.name == "LGBMClassifier":
                pred_results, _, _  = self._clf.predict_proba(X)
            else:
                pred_results = self._clf.predict_proba(X)
        except AttributeError:
            int_results = self._clf.predict(X)
            pred_results = self.encoder.transform(int_results)
            if pred_results.shape[1] == 1:  # 2 classes
                pred_results = [[1, 0] if i == 0 else [0, 1] for i in int_results]

        str_preds = [
            self.mapping[encoded_label] for encoded_label in np.argmax(pred_results, axis=1)  # type: ignore
        ]

        str_y = [self.mapping[encoded_label] for encoded_label in y]

        write_pred_table(
            predictions=pred_results,
            str_preds=str_preds,
            str_targets=str_y,
            classes=self.classes,
            md5s=ids,
            path=log,
        )

    def save_model(self, logdir: Path, name=None):
        """Save model to pickle file. If a filename is given, it will be appended to model name."""
        save_name = f"{self._name}"
        if name is not None:
            save_name += f"_{name}"

        time = str(time_now()).replace(" ", "_")
        save_name = logdir / f"{save_name}_{time}.pickle"

        print(f"Saving model to {save_name}")
        with open(save_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def restore_model(
        cls, logdir: str, auto_name: str, full_path=None
    ) -> EstimatorAnalyzer:
        """Restore most recent EstimatorAnalyzer instance from a previous save.
        Automatic mode restores most recent model with cli name. Full path takes it literally
        """
        if full_path is not None:
            filepath = full_path
        else:
            name = save_mapping[auto_name]
            path = Path(logdir) / f"{name}*.pickle"
            list_of_files = glob.glob(str(path))
            try:
                filepath = max(list_of_files, key=os.path.getctime)
            except ValueError as err:
                print(
                    f"Did not find any model file following pattern {path}",
                    file=sys.stderr,
                )
                raise err

        print(f"Loading model {filepath}")
        with open(filepath, "rb") as f:
            model = pickle.load(f)

        return model


def best_params_cb(result):
    """BayesSearchCV callback"""
    print(f"Best params yet: {result.x}")


def tune_estimator(
    model: Pipeline,
    ea_handler: EpiAtlasTreatment,
    params: dict,
    n_iter: int,
    concurrent_cv: int,
    n_jobs: int | None = None,
) -> BayesSearchCV:
    """
    Apply Bayesian optimization on model, over hyperparameters search space.

    Args:
      model (Pipeline): The model to tune.
      ea_handler (EpiAtlasTreatment): Dataset
      params (dict): Hyperparameters search space.
      n_iter (int): Total number of parameter settings to sample.
      concurrent_cv (int): Number of full cross-validation process (X folds) to run
    in parallel.
      n_jobs (int | None): Number of jobs to run in parallel. Max NFOLD_TUNE *
    concurrent_cv.

    Returns:
      A BayesSearchCV object
    """
    deadline_cb = DeadlineStopper(total_time=60 * 60 * 8)
    if n_jobs is None:
        n_jobs = int(NFOLD_TUNE * concurrent_cv)

    if n_jobs > 48:
        raise AssertionError("More jobs than cores asked, max 48 jobs.")

    total_data = ea_handler.create_total_data()
    print(f"Number of files used globally {len(total_data)}")

    opt = BayesSearchCV(
        model,
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
    print(f"Current model params: {model.get_params()}")
    print(f"best params: {opt.best_params_}")
    return opt


def optimize_estimator(
    ea_handler: EpiAtlasTreatment,
    logdir: Path,
    n_iter: int,
    name: str,
    concurrent_cv: int = 1,
):
    """
    It takes a dataset and model name, and then it optimizes the model with the given name
    using the search space with the same name.

    Args:
      ea_handler (EpiAtlasTreatment): Dataset
      logdir (Path): The directory where the results will be saved.
      n_iter (int): Number of different search space sampling.
      name (str): The name of the model we're tuning.
      concurrent_cv (int): Number of full cross-validation process (X folds) to run
    in parallel. Defaults to 1.
    """

    print(f"Starting {name} optimization")
    start_train = time_now()
    opt = tune_estimator(
        model_mapping[name],
        ea_handler,
        search_mapping[name],
        n_iter=n_iter,
        concurrent_cv=concurrent_cv,
    )
    print(f"Total {name} optimisation time: {time_now()-start_train}")

    log_tune_results(logdir, name, opt)


def log_tune_results(logdir: Path, name: str, opt: BayesSearchCV):
    """
    It takes the results of a parameter optimization run and saves them to a CSV
    file.

    Args:
      logdir (Path): The directory where the results will be saved.
      name (str): The name of the model.
      opt (BayesSearchCV): Optimizer after tuning.
    """
    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers="keys", tablefmt="psql"))  # type: ignore

    file = tune_results_file_format.format(name=name)
    df.to_csv(logdir / file, sep=",")

    file = best_params_file_format.format(name=name)
    with open(logdir / file, "w", encoding="utf-8") as f:
        json.dump(obj=opt.best_params_, fp=f)


def run_predictions(
    ea_handler: EpiAtlasTreatment, estimator: Pipeline, name: str, logdir: Path
):
    """
    It will fit and run a prediction for each of the k-folds in the EpiAtlasTreatment
    object, using the estimator provided. Will use all available cpus.

    Args:
      ea_handler (EpiAtlasTreatment): Dataset
      estimator (Pipeline): The model to use.
      name (str): The name of the model.
      logdir (Path): The directory where the results will be saved.
    """
    nb_workers = ea_handler.k
    available_cpus = len(os.sched_getaffinity(0))
    if available_cpus < nb_workers:
        nb_workers = available_cpus

    func = partial(run_prediction, estimator=estimator, name=name, logdir=logdir)
    items = enumerate(ea_handler.yield_split())

    with mp.Pool(nb_workers) as pool:
        pool.starmap(func, items)


def run_prediction(
    i: int,
    my_data: DataSet,
    estimator: Pipeline,
    name: str,
    logdir: Path,
    verbose=True,
    save_model=True,
):
    """
    It takes a dataset, fits the model on the training data, and then predicts on
    the validation data

    Args:
      i (int): the index of the split
      my_data (DataSet): DataSet
      estimator (Pipeline): The model to use.
      name (str): The name of the model.
      logdir (Path): The directory where the results will be saved.
      verbose: Whether to print out the metrics. Defaults to True
    """
    if verbose:
        print(f"Split {i} training size: {my_data.train.num_examples}")

    if i == 0:
        nb_files = len(
            set(my_data.train.ids.tolist() + my_data.validation.ids.tolist())
        )
        print(f"Total nb of files: {nb_files}")

    estimator.fit(X=my_data.train.signals, y=my_data.train.encoded_labels)

    analyzer = EstimatorAnalyzer(my_data.classes, estimator)

    X, y = my_data.validation.signals, my_data.validation.encoded_labels

    if verbose:
        print(f"Split {i} metrics:")
        analyzer.metrics(X, y, verbose=True)

    try:
        logdir = logdir / f"{name}"
        create_dirs(logdir)
    except KeyboardInterrupt:
        print("Shutdown requested (KeyboardInterrupt)...exiting")
        sys.exit(1)
    except Exception as err:
        (print(err))
        print("Continuing with default logdir.")

    analyzer.predict_file(
        my_data.validation.ids,
        X,
        y,
        logdir / f"{name}_split{i}_validation_prediction.csv",
    )

    if save_model:
        analyzer.save_model(logdir, name=f"split{i}")

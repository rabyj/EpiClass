"""Module containing result analysis code."""
from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import torch
import torchmetrics
from torch import Tensor
from torch.utils.data import TensorDataset

from .confusion_matrix import ConfusionMatrixWriter
from .data import Data, DataSet
from .model_pytorch import LightningDenseClassifier
from .types import SomeData, TensorData
from epi_ml.utils.time import time_now_str


class Analysis(object):
    """Class containing main analysis methods desired."""

    def __init__(
        self,
        model: LightningDenseClassifier,
        datasets_info: DataSet,
        logger: pl.loggers.CometLogger,  # type: ignore
        train_dataset: Optional[TensorData] = None,
        val_dataset: Optional[TensorData] = None,
        test_dataset: Optional[TensorData] = None,
    ):
        self._model = model
        self._classes = sorted(list(self._model.mapping.values()))
        self._logger = logger

        # Original DataSet object (legacy)
        self.datasets = datasets_info
        self._set_dict = {
            "training": self.datasets.train,
            "validation": self.datasets.validation,
            "test": self.datasets.test,
        }

        # TensorDataset objects (pytorch)
        self._train = train_dataset
        self._val = val_dataset
        self._test = test_dataset

    def _log_metrics(self, metric_dict, prefix=""):
        """Log metrics from TorchMetrics metrics dict object. (key: tensor(val))"""
        for metric, val in metric_dict.items():
            name = f"{prefix[0:3]}_{metric}"
            self._logger.experiment.log_metric(name, val.item())

    @staticmethod
    def print_metrics(metric_dict, name):
        """Print metrics from TorchMetrics dict."""
        print(f"--- {name} METRICS ---")
        vals = []
        for metric, val in metric_dict.items():
            str_val = f"{val.item():.3f}"
            print(metric, str_val)
            vals.append(str_val)
        print(*vals)

    def _generic_metrics(self, dataset, name, verbose):
        """General treatment to compute and print metrics"""
        if dataset is None:
            print(f"Cannot compute {name} metrics : No {name} dataset given")
            metrics_dict = None
        else:
            metrics_dict = self._model.compute_metrics(dataset)
            if self._logger is not None:
                self._log_metrics(metrics_dict, prefix=name)
            if verbose:
                Analysis.print_metrics(metrics_dict, name=f"{name} set")
        return metrics_dict

    def get_training_metrics(self, verbose=True):
        """Compute and print training set metrics."""
        return self._generic_metrics(self._train, "training", verbose)

    def get_validation_metrics(self, verbose=True):
        """Compute and print validation set metrics."""
        return self._generic_metrics(self._val, "validation", verbose)

    def get_test_metrics(self, verbose=True):
        """Compute and print test set metrics."""
        return self._generic_metrics(self._test, "test", verbose)

    def _generic_write_prediction(
        self, to_predict: TensorData | None, name, path, verbose=True
    ):
        """General treatment to write predictions
        Name can be {training, validation, test}.

        to_predict: Object that contains samples to predict.
        """
        if path is None:
            path = self._logger.save_dir / f"{name}_prediction.csv"

        if to_predict is None:
            print(f"Cannot compute {name} predictions : No {name} dataset given")
            return

        if isinstance(to_predict, TensorDataset):
            preds, targets = self._model.compute_predictions_from_dataset(to_predict)
            str_targets = [self._model.mapping[int(val.item())] for val in targets]
        elif isinstance(to_predict, Tensor):
            preds = self._model.compute_predictions_from_features(to_predict)
            str_targets = ["Unknown" for _ in range(to_predict.size(dim=1))]

        write_pred_table(
            predictions=preds,
            str_preds=[
                self._model.mapping[int(val.item())]
                for val in torch.argmax(preds, dim=-1)
            ],
            str_targets=str_targets,
            md5s=self._set_dict[name].ids,
            classes=self._classes,
            path=path,
        )
        self._logger.experiment.log_asset(file_data=path, file_name=f"{name}_prediction")

        if verbose:
            print(f"'{path.name}' written to '{path.parent}'")

    def write_training_prediction(self, path=None):
        """Compute and write training predictions to file."""
        self._generic_write_prediction(self._train, name="training", path=path)

    def write_validation_prediction(self, path=None):
        """Compute and write validation predictions to file."""
        self._generic_write_prediction(self._val, name="validation", path=path)

    def write_test_prediction(self, path=None):
        """Compute and write test predictions to file."""
        self._generic_write_prediction(self._test, name="test", path=path)

    def _generic_confusion_matrix(self, dataset: TensorData | None, name) -> np.ndarray:
        """General treatment to write confusion matrices."""
        if dataset is None:
            raise ValueError(
                f"Cannot compute {name} confusion matrix : No {name} dataset given"
            )
        elif isinstance(dataset, Tensor):
            raise ValueError(
                f"Cannot compute {name} confusion matrix : No targets in given dataset."
            )

        preds, targets = self._model.compute_predictions_from_dataset(dataset)

        final_pred = torch.argmax(preds, dim=-1)

        mat = torchmetrics.functional.confusion_matrix(
            final_pred, targets, num_classes=len(self._classes), normalize=None
        )
        return mat.detach().cpu().numpy()

    def _save_matrix(self, mat: ConfusionMatrixWriter, set_name, path: Path | None):
        """Save matrix to files"""
        if path is None:
            parent = Path(self._logger.save_dir)
            name = f"{set_name}_confusion_matrix"
        else:
            parent = path.parent
            name = path.with_suffix("").name
        csv, csv_rel, png = mat.to_all_formats(logdir=parent, name=name)
        self._logger.experiment.log_asset(file_data=csv, file_name=f"{csv.name}")
        self._logger.experiment.log_asset(file_data=csv_rel, file_name=f"{csv_rel.name}")  # fmt: skip
        self._logger.experiment.log_asset(file_data=png, file_name=f"{png.name}")

    def train_confusion_matrix(self, path=None):
        """Compute and write train confusion matrix to file."""
        set_name = "train"
        mat = self._generic_confusion_matrix(self._train, name=set_name)
        mat = ConfusionMatrixWriter(labels=self._classes, confusion_matrix=mat)
        self._save_matrix(mat, set_name, path)

    def validation_confusion_matrix(self, path=None):
        """Compute and write validation confusion matrix to file."""
        set_name = "validation"
        mat = self._generic_confusion_matrix(self._val, name=set_name)
        mat = ConfusionMatrixWriter(labels=self._classes, confusion_matrix=mat)
        self._save_matrix(mat, set_name, path)

    def test_confusion_matrix(self, path=None):
        """Compute and write test confusion matrix to file."""
        set_name = "test"
        mat = self._generic_confusion_matrix(self._test, name=set_name)
        mat = ConfusionMatrixWriter(labels=self._classes, confusion_matrix=mat)
        self._save_matrix(mat, set_name, path)


class SHAP_Handler:
    """Handle shap computations and data saving/loading."""

    def __init__(self, model, logdir):
        self.model = model
        self.logdir = logdir
        self.filename_template = "shap_values{name}_{time}.{ext}"

    def _create_filename(self, ext: str, name="") -> Path:
        if name:
            name = "_" + name
        filename = self.filename_template.format(name=name, ext=ext, time=time_now_str())

        return Path(self.logdir) / filename

    def compute_NN(
        self, background_dset: SomeData, evaluation_dset: SomeData, save=True, name=""
    ) -> Tuple[shap.DeepExplainer, List[np.ndarray]]:
        """Compute shap values of deep learning model on evaluation dataset
        by creating an explainer with background dataset.

        Returns explainer and shap values (as a list of matrix per class)
        """
        explainer = shap.DeepExplainer(
            model=self.model, data=Tensor(background_dset.signals)
        )

        shap_values = explainer.shap_values(Tensor(evaluation_dset.signals))

        if save:
            self.save_to_pickle(shap_values, evaluation_dset.ids, name)

        return explainer, shap_values  # type: ignore

    def save_to_pickle(self, shap_values, ids, name="") -> Path:
        """Save shap values with assigned signals ids to a pickle file.

        Returns path of saved file."""

        filename = self._create_filename(name=name, ext="pickle")

        print(f"Saving SHAP values to: {filename}")
        with open(filename, "wb") as f:
            pickle.dump({"shap": shap_values, "ids": ids}, f)

        return filename

    @staticmethod
    def load_from_pickle(path) -> Dict[str, Any]:
        """Load pickle file with shap values and ids.

        Returns {"shap": shap_values, "ids": ids} dict.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    def save_to_csv(self, shap_values_matrix: np.ndarray, ids, name) -> Path:
        """Save a single shap value matrix (shape (n_samples, #features)) to csv.
        Giving a name is mandatory contrary to pickle.

        Returns path of saved file.
        """
        if isinstance(shap_values_matrix, list):
            raise ValueError(
                "{shap_values_matrix} is a list, not an array. Need one matrix of shape (n_samples, #features)"
            )

        filename = self._create_filename(name=name, ext="csv")

        n_dims = shap_values_matrix.shape[1]
        df = pd.DataFrame(data=shap_values_matrix, index=ids, columns=range(n_dims))

        print(f"Saving SHAP values to: {filename}")
        df.to_csv(filename)

        return filename

    @staticmethod
    def load_from_csv(path) -> pd.DataFrame:
        """Return pandas dataframe of shap values for loaded file."""
        return pd.read_csv(path, index_col=0)


# TODO: Insert "ID" in header, and make sure subsequent script use that (e.g. the bash one liner, for sorting)
def write_pred_table(predictions, str_preds, str_targets, md5s, classes, path):
    """Write to "path" a csv containing class probability predictions.

    pred : Prediction vectors
    str_preds : List of predictions, but in string form
    str_targets : List of corresponding targets, but in string form
    md5s : List of corresponding md5s
    classes : Ordered list of the output classes
    path : Where to write the file
    """
    df = pd.DataFrame(data=predictions, index=md5s, columns=classes)

    df.insert(loc=0, column="True class", value=str_targets)
    df.insert(loc=1, column="Predicted class", value=str_preds)

    df.to_csv(path, encoding="utf8")


def predict_concat_size(chroms, resolution):
    """Compute the size of a concatenated genome from the resolution of each chromosome."""
    concat_size = 0
    for _, size in chroms:
        size_of_mean = size // resolution
        if size_of_mean % resolution == 0:
            concat_size += size_of_mean
        else:
            concat_size += size_of_mean + 1

    return concat_size


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def assert_correct_resolution(chroms, resolution, signal_length):
    """Raise AssertionError if the given resolution is not coherent with
    the input size of the network.
    """
    if predict_concat_size(chroms, resolution) != signal_length:
        raise AssertionError(
            f"Signal_length not coherent with given resolution of {resolution}."
        )


def values_to_bedgraph(values, chroms, resolution, bedgraph_path):
    """Write a bedgraph from a full genome values iterable (e.g. importance).
    The chromosome coordinates are zero-based, half-open (from 0 to N-1).
    """
    i = 0
    with open(bedgraph_path, "w", encoding="utf-8") as my_bedgraph:
        for name, size in chroms:

            positions = itertools.chain(range(0, size, resolution), [size - 1])

            for pos1, pos2 in pairwise(positions):

                line = [name, pos1, pos2, values[i]]
                my_bedgraph.write("{}\t{}\t{}\t{}\n".format(*line))
                i += 1

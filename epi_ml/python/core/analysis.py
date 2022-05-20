"""Module containing result analysis code."""
import itertools
from pathlib import Path
from typing import Union, Optional

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset
import torchmetrics

from core.data import DataSet
from core.model_pytorch import LightningDenseClassifier
from core.confusion_matrix import ConfusionMatrixWriter


class Analysis(object):
    """Class containing main analysis methods desired."""
    def __init__(self,
    model: Union[pl.LightningModule, LightningDenseClassifier],
    datasets_info: DataSet,
    logger: pl.loggers.CometLogger,
    train_dataset: Optional[TensorDataset] = None,
    val_dataset: Optional[TensorDataset] = None,
    test_dataset: Optional[TensorDataset] = None
    ):
        self._model = model
        self._classes = sorted(list(self._model.mapping.values()))
        self._logger = logger

        # Original DataSet object (legacy)
        self.datasets = datasets_info
        self._set_dict = {
            "training": self.datasets.train,
            "validation": self.datasets.validation,
            "test": self.datasets.test
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

    def _generic_write_prediction(self, dataset, name, path, verbose=True):
        """General treatment to write predictions
        Name can be {training, validation, test}.
        """
        if path is None:
            path = self._logger.save_dir / f"{name}_prediction.csv"

        if dataset is None:
            print(f"Cannot compute {name} predictions : No {name} dataset given")
            return

        preds, targets = self._model.compute_predictions(dataset)

        write_pred_table(
            predictions=preds,
            str_preds=[self._model.mapping[int(val.item())] for val in torch.argmax(preds, dim=-1)],
            str_targets=[self._model.mapping[int(val.item())] for val in targets],
            md5s=self._set_dict[name].ids,
            classes=self._classes,
            path=path
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

    def _generic_confusion_matrix(self, dataset, name) -> np.ndarray:
        """General treatment to write confusion matrices."""
        if dataset is None:
            print(f"Cannot compute {name} confusion matrix : No {name} dataset given")
            return

        preds, targets = self._model.compute_predictions(dataset)

        final_pred = torch.argmax(preds, dim=-1)

        mat = torchmetrics.functional.confusion_matrix(
            final_pred,
            targets,
            num_classes=len(self._classes),
            normalize=None
            )
        return mat.detach().cpu().numpy()

    def _save_matrix(self, mat: ConfusionMatrixWriter, set_name, path: Path):
        """Save matrix to files"""
        if path is None:
            parent = self._logger.save_dir
            name = f"{set_name}_confusion_matrix"
        else:
            parent = path.parent
            name = path.with_suffix("").name
        csv, csv_rel, png = mat.to_all_formats(logdir=parent, name=name)
        self._logger.experiment.log_asset(file_data=csv, file_name=f"{csv.name}")
        self._logger.experiment.log_asset(file_data=csv_rel, file_name=f"{csv_rel.name}")
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
        size_of_mean = size//resolution
        if size_of_mean%resolution == 0:
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
        raise AssertionError(f"Signal_length not coherent with given resolution of {resolution}.")

def values_to_bedgraph(values, chroms, resolution, bedgraph_path):
    """Write a bedgraph from a full genome values iterable (e.g. importance).
    The chromosome coordinates are zero-based, half-open (from 0 to N-1).
    """
    i = 0
    with open(bedgraph_path, 'w', encoding="utf-8") as my_bedgraph:
        for name, size in chroms:

            positions = itertools.chain(range(0, size, resolution), [size-1])

            for pos1, pos2 in pairwise(positions):

                line = [name, pos1, pos2, values[i]]
                my_bedgraph.write("{}\t{}\t{}\t{}\n".format(*line))
                i += 1

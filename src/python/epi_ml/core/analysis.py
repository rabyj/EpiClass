"""Module containing result analysis code."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import Tensor
from torch.utils.data import TensorDataset

from epi_ml.core.confusion_matrix import ConfusionMatrixWriter
from epi_ml.core.data import DataSet
from epi_ml.core.model_pytorch import LightningDenseClassifier
from epi_ml.core.types import TensorData


class Analysis:
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
        if isinstance(dataset, Tensor):
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


def write_to_bed(
    bed_ranges: List[Tuple[str, int, int]], bed_path: str | Path, verbose: bool = False
) -> None:
    """Writes the given bed ranges to a .bed file.

    Args:
        bed_ranges (List[Tuple[str, int, int]]): List of tuples, each containing
            (chromosome name, start position, end position).
        bed_path (str): The path where the .bed file should be written.

    Note:
        The function doesn't return anything. It writes directly to a file.
    """
    with open(bed_path, "w", encoding="utf8") as file:
        for bed_range in bed_ranges:
            file.write(f"{bed_range[0]}\t{bed_range[1]}\t{bed_range[2]}\n")
    if verbose:
        print(f"Bed file written to {bed_path}")


def bins_to_bed_ranges(
    bin_indexes: List[int], chroms: List[Tuple[str, int]], resolution: int
) -> List[Tuple[str, int, int]]:
    """Convert multiple global genome bins to chromosome ranges.

    Args:
        bin_indexes (List[int]): List of bin indexes in the genome.
        chroms (List[Tuple[str, int]]): List of tuples (ordered by chromosome order),
            where each tuple contains a chromosome name and its length in base pairs.
        resolution (int): The size of each bin.

    Returns:
        List[Tuple[str, int, int]]: List of tuples, each containing (chromosome name, start position, end position).

    Raises:
        IndexError: If any bin index is not in any chromosome,
        i.e., it's greater than the total number of bins in the genome.

    Note:
        The function assumes that chromosomes in `chroms` are ordered as they appear in the genome.
        The functions assumes that the binning was done per chromosome and then joined.
        The bin indexes are zero-based and span the entire genome considering the resolution.
        The returned ranges are half-open intervals [start, end).
    """
    bin_indexes = sorted(bin_indexes)
    bin_ranges = []

    # Calculate the cumulative bin positions at the start of each chromosome
    cumulative_bins = [0]
    for _, chrom_size in chroms:
        bins_in_chrom = chrom_size // resolution + (chrom_size % resolution > 0)
        cumulative_bins.append(cumulative_bins[-1] + bins_in_chrom)

    for bin_index in bin_indexes:
        # Find the chromosome that contains this bin
        for chrom_index, (chrom_start_bin, chrom_end_bin) in enumerate(
            zip(cumulative_bins[:-1], cumulative_bins[1:])
        ):
            if chrom_start_bin <= bin_index < chrom_end_bin:
                # The bin is in this chromosome
                bin_in_chrom = bin_index - chrom_start_bin
                start = bin_in_chrom * resolution
                end = min((bin_in_chrom + 1) * resolution, chroms[chrom_index][1])
                bin_ranges.append((chroms[chrom_index][0], start, end))
                break
        else:
            # The bin index is out of range
            raise IndexError("bin_index out of range")

    return bin_ranges

"""Module containing result analysis code."""
import itertools
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from core.model_pytorch import LightningDenseClassifier

class Analysis(object):
    """Class containing main analysis methods desired."""
    def __init__(self,
    model : Union[pl.LightningModule, LightningDenseClassifier],
    train_dataset=None, val_dataset=None, test_dataset=None,
    logger=None
    ):
        self._model = model
        self._train = train_dataset
        self._val = val_dataset
        self._test = test_dataset
        self._logger = logger

    def _log_metrics(self, metric_dict, prefix=""):
        """Log metrics from TorchMetrics metrics dict object. (key : tensor(val))"""
        for metric, val in metric_dict.items():
            self._logger.experiment.log_metric(f"{prefix[0:3]}_{metric}", val.item())

    def _generic_metrics(self, dataset, name, verbose):
        """General treatment to compute and print metrics"""
        if dataset is None:
            print(f"Cannot compute {name} metrics : No {name} dataset given")
            metrics_dict = None
        else:
            metrics_dict = self._model.compute_metrics(dataset)
            if self._logger is not None :
                self._log_metrics(metrics_dict, prefix=name)
            if verbose:
                print_metrics(metrics_dict, name=f"{name} set")
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

    # def write_training_prediction(self, path):
    #     write_pred_table(self._trainer.training_pred(), self._data.classes, self._data.train, path)

    # def write_validation_prediction(self, path):
    #     write_pred_table(self._trainer.validation_pred(), self._data.classes, self._data.validation, path)

    # def write_test_prediction(self, path):
    #     write_pred_table(self._trainer.test_pred(), self._data.classes, self._data.test, path)

    # def training_confusion_matrix(self, logdir, name="training_confusion_matrix"):
    #     mat = ConfusionMatrix(self._data.classes, self._trainer.training_mat())
    #     mat.to_all_formats(logdir, name)

    # def validation_confusion_matrix(self, logdir, name="validation_confusion_matrix"):
    #     mat = ConfusionMatrix(self._data.classes, self._trainer.validation_mat())
    #     mat.to_all_formats(logdir, name)

    # def test_confusion_matrix(self, logdir, name="test_confusion_matrix"):
    #     mat = ConfusionMatrix(self._data.classes, self._trainer.test_mat())
    #     mat.to_all_formats(logdir, name)

    # def importance(self):
    #     return importance(self._trainer.weights())


class ConfusionMatrix(object):
    """Class to create/handle confusion matrices"""
    def __init__(self, labels, tf_confusion_mat):
        self._labels = labels
        self._confusion_matrix = self._create_confusion_matrix(tf_confusion_mat) #pd dataframe

    @classmethod
    def from_csv(cls, csv_path):
        obj = cls.__new__(cls)  # Does not call __init__
        obj._confusion_matrix = pd.read_csv(csv_path, sep=',', index_col=0)
        return obj

    def _create_confusion_matrix(self, tf_confusion_mat):
        labels_count = tf_confusion_mat.sum(axis=0)
        labels_w_count = [f"{label}({label_count})" for label, label_count in zip(self._labels, labels_count)]

        confusion_mat = self._to_relative_confusion_matrix(labels_count, tf_confusion_mat)

        return pd.DataFrame(data=confusion_mat, index=labels_w_count, columns=self._labels)

    def _to_relative_confusion_matrix(self, labels_count, tf_confusion_mat):
        confusion_mat = tf_confusion_mat/labels_count
        confusion_mat = np.nan_to_num(confusion_mat)
        confusion_mat = confusion_mat.T #one label per row instead of column
        return confusion_mat

    def to_png(self, path):
        plt.figure()

        data_mask = np.ma.masked_where(self._confusion_matrix == 0, self._confusion_matrix)

        cdict = {'red': ((0.0, 0.1, 0.1),
                         (1.0, 1.0, 1.0)),

                 'green': ((0.0, 0.1, 0.1),
                           (1.0, 0.1, 0.1)),

                 'blue': ((0.0, 1.0, 1.0),
                          (1.0, 0.1, 0.1))}

        blue_red = matplotlib.colors.LinearSegmentedColormap('BlueRed', cdict, N=1000)

        nb_labels = len(self._confusion_matrix.columns)
        grid_width = 0.5 - nb_labels/400.0
        label_size = 15*np.exp(-0.02*nb_labels)

        fig, ax = plt.subplots()
        cm = ax.pcolormesh(data_mask, cmap=blue_red, alpha=0.8, edgecolors='k', linewidths=grid_width)
        ax.set_frame_on(False)
        ax.set_xticks(np.arange(nb_labels) + 0.5, minor=False)
        ax.set_yticks(np.arange(nb_labels) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(self._confusion_matrix.columns, fontsize=label_size)
        ax.set_yticklabels(self._confusion_matrix.index, fontsize=label_size)
        plt.xticks(rotation=90)

        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        cbar = fig.colorbar(cm, ax=ax, shrink=0.75)
        cbar.ax.tick_params(labelsize=4)

        plt.tight_layout()
        plt.savefig(path, format='png', dpi=500)

        # buf = io.BytesIO()
        # plt.savefig(buf, format='png', dpi=400)
        # buf.seek(0)
        # image = tf.image.decode_png(buf.getvalue(), channels=4)
        # image = tf.expand_dims(image, 0)
        # summary = tf.summary.image("Confusion Matrix", image, max_outputs=1)
        # self._writer.add_summary(summary.eval(session=self._sess))

    def to_csv(self, path):
        self._confusion_matrix.to_csv(path, encoding="utf8", float_format='%.4f')

    def to_all_formats(self, logdir, name):
        outpath = Path(logdir) / name
        self.to_csv(outpath.with_suffix(".csv"))
        self.to_png(outpath.with_suffix(".png"))


def write_pred_table(pred, classes, data_subset, path):
    """Write to "path" a csv containing class probability predictions of data_subset."""
    labels = data_subset.labels
    md5s = data_subset.ids

    string_labels = [classes[np.argmax(label)] for label in labels]
    df = pd.DataFrame(data=pred, index=md5s, columns=classes)

    df.insert(loc=0, column="class", value=string_labels)

    df.to_csv(path, encoding="utf8")

def convert_matrix_csv_to_png(in_path, out_path):
    """Convert csv of confusion matrix to a png, and write it to out_path."""
    mat = ConfusionMatrix.from_csv(in_path)
    mat.to_png(out_path)

def print_metrics(metric_dict, name):
    """Print metrics from torchmetrics dict."""
    print(f"--- {name} METRICS ---")
    vals = []
    for metric, val in metric_dict.items():
        str_val = f"{val.item():.3f}"
        print(metric, str_val)
        vals.append(str_val)
    print(*vals)

def importance(w):
    """garson algorithm, w for weights."""
    #TODO: generalise, put in model
    total_w = w[0]
    for i in range(2, len(w), 2):
        total_w = np.dot(total_w, w[i])
    total_w = np.absolute(total_w)
    sum_w = np.sum(total_w, axis=None)
    total_w = np.sum(total_w/sum_w, axis=1)
    # print((total_w > 1e-04).sum())
    # return ','.join([str(x) for x in total_w])
    return [x for x in total_w]

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

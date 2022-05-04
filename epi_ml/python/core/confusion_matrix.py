"""ConfusionMatrixWriter class"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class ConfusionMatrixWriter(object):
    """Class to create/handle confusion matrices"""
    def __init__(self, labels, confusion_mat):
        """Confusion mat is a matrix that counts each final prediction (int matrix)
        Expects a confusion matrix input with prediction rows and target colomns."""
        self._labels = labels
        self.confusion_mat = confusion_mat
        self.pd_confusion_matrix = self.create_confusion_matrix(relative=True) #pd dataframe

    @classmethod
    def from_csv(cls, csv_path):
        """Create instance from file."""
        obj = cls.__new__(cls)  # Does not call __init__
        obj.pd_confusion_matrix = pd.read_csv(csv_path, sep=',', index_col=0)
        return obj

    def create_confusion_matrix(self, relative):
        """Returns confusion matrix with labels (pandas df) from int matrix.
        Expects prediction rows and target columns.
        Can be normalized on rows (relative) or not.
        """
        labels_count = self.confusion_mat.sum(axis=1) # total nb examples of each label
        labels_w_count = [f"{label}({label_count})" for label, label_count in zip(self._labels, labels_count)]

        confusion_mat = self.confusion_mat
        if relative:
            confusion_mat = ConfusionMatrixWriter.to_relative_confusion_matrix(labels_count, self.confusion_mat)

        return pd.DataFrame(data=confusion_mat, index=labels_w_count, columns=self._labels)

    @staticmethod
    def to_relative_confusion_matrix(labels_count, confusion_mat : np.array):
        """Normalize confusion matrix per row.
        Expects prediction rows and target columns.
        """
        confusion_mat1 = torch.div(confusion_mat.T, labels_count)
        confusion_mat2 = np.nan_to_num(confusion_mat1)
        return confusion_mat2.T

    def to_png(self, path):
        """Write to path an image of the confusion matrix."""
        plt.figure()

        data_mask = np.ma.masked_where(self.pd_confusion_matrix == 0, self.pd_confusion_matrix)

        cdict = {'red': ((0.0, 0.1, 0.1),
                         (1.0, 1.0, 1.0)),

                 'green': ((0.0, 0.1, 0.1),
                           (1.0, 0.1, 0.1)),

                 'blue': ((0.0, 1.0, 1.0),
                          (1.0, 0.1, 0.1))}

        blue_red = matplotlib.colors.LinearSegmentedColormap('BlueRed', cdict, N=1000)

        nb_labels = len(self.pd_confusion_matrix.columns)
        grid_width = 0.5 - nb_labels/400.0
        label_size = 15*np.exp(-0.02*nb_labels)

        fig, ax = plt.subplots()
        cm = ax.pcolormesh(data_mask, cmap=blue_red, alpha=0.8, edgecolors='k', linewidths=grid_width)
        ax.set_frame_on(False)
        ax.set_xticks(np.arange(nb_labels) + 0.5, minor=False)
        ax.set_yticks(np.arange(nb_labels) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(self.pd_confusion_matrix.columns, fontsize=label_size)
        ax.set_yticklabels(self.pd_confusion_matrix.index, fontsize=label_size)
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
        """Write to path a csv file of the confusion matrix."""
        self.pd_confusion_matrix.to_csv(path, encoding="utf8", float_format='%.4f')

    def to_all_formats(self, logdir, name):
        """Write to path both a png and csv file of the confusion matrix."""
        outpath = Path(logdir) / name
        self.to_csv(outpath.with_suffix(".csv"))
        self.to_png(outpath.with_suffix(".png"))

    @staticmethod
    def convert_matrix_csv_to_png(in_path, out_path):
        """Convert csv of confusion matrix to a png, and write it to out_path."""
        mat = ConfusionMatrixWriter.from_csv(in_path)
        mat.to_png(out_path)

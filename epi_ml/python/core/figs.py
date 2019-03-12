import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class ConfusionMatrix(object):
    def __init__(self, labels, confusion_matrix):
        self._labels = labels
        self._confusion_matrix = self._create_confusion_matrix(confusion_matrix)
        
    @classmethod
    def from_csv(cls, csv_path):
        obj = cls.__new__(cls)  # Does not call __init__
        obj._confusion_matrix = pd.read_csv(csv_path, sep=',', index_col=0)
        return obj

    def _create_confusion_matrix(self, confusion_matrix):

        labels_count = confusion_matrix.sum(axis=0)
        labels_w_count = ["{}({})".format(label, label_count) for label, label_count in zip(self._labels, labels_count)]

        confusion_matrix = self._to_relative_confusion_matrix(labels_count, confusion_matrix)

        return pd.DataFrame(data=confusion_matrix, index=labels_w_count, columns=self._labels)

    def _to_relative_confusion_matrix(self, labels_count, confusion_matrix):
        confusion_matrix = confusion_matrix/labels_count 
        confusion_matrix = np.nan_to_num(confusion_matrix)
        confusion_matrix = confusion_matrix.T #one label per row instead of column
        return confusion_matrix

    def to_png(self, path):
        plt.figure()
        
        data_mask = np.ma.masked_where(self._confusion_matrix == 0, self._confusion_matrix)

        cdict = {'red':   ((0.0,  0.1, 0.1),
                           (1.0,  1.0, 1.0)),

                'green': ((0.0,  0.1, 0.1),
                          (1.0,  0.1, 0.1)),

                'blue': ((0.0,  1.0, 1.0),
                         (1.0,  0.1, 0.1))}

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

        cbar = fig.colorbar(cm, ax=ax, shrink=0.5)
        cbar.ax.tick_params(labelsize=4)

        plt.tight_layout()
        plt.savefig(path, format='png', dpi=400)

        # buf = io.BytesIO()
        # plt.savefig(buf, format='png', dpi=400)
        # buf.seek(0)
        # image = tf.image.decode_png(buf.getvalue(), channels=4)
        # image = tf.expand_dims(image, 0)
        # summary = tf.summary.image("Confusion Matrix", image, max_outputs=1)
        # self._writer.add_summary(summary.eval(session=self._sess))

    def to_csv(self, path):
        self._confusion_matrix.to_csv(path, encoding="utf8", float_format='%.4f')

def convert_matrix_csv_to_png(in_path, out_path):
    confusion_matrix = ConfusionMatrix.from_csv(in_path)
    confusion_matrix.to_png(out_path)

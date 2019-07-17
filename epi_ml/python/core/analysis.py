import io
import itertools
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf


class Analysis(object):
    def __init__(self, trainer):
        self._trainer = trainer
        self._model = trainer._model
        self._data = trainer._data

    def training_metrics(self):
        print("Training set metrics")
        metrics(self._trainer.training_acc(), self._trainer.training_pred(), self._data.train)

    def validation_metrics(self):
        print("Validation set metrics")
        metrics(self._trainer.validation_acc(), self._trainer.validation_pred(), self._data.validation)

    def test_metrics(self):
        print("Test set metrics")
        metrics(self._trainer.test_acc(), self._trainer.test_pred(), self._data.test)

    def training_prediction(self, path):
        write_pred_table(self._trainer.training_pred(), self._data.classes, self._data.train, path)

    def validation_prediction(self, path):
        write_pred_table(self._trainer.validation_pred(), self._data.classes, self._data.validation, path)

    def test_prediction(self, path):
        write_pred_table(self._trainer.test_pred(), self._data.classes, self._data.test, path)

    def training_confusion_matrix(self, logdir, name="training_confusion_matrix"):
        mat = ConfusionMatrix(self._data.classes, self._trainer.training_mat())
        mat.to_all_formats(logdir, name)

    def validation_confusion_matrix(self, logdir, name="validation_confusion_matrix"):
        mat = ConfusionMatrix(self._data.classes, self._trainer.validation_mat())
        mat.to_all_formats(logdir, name)

    def test_confusion_matrix(self, logdir, name="test_confusion_matrix"):
        mat = ConfusionMatrix(self._data.classes, self._trainer.test_mat())
        mat.to_all_formats(logdir, name)

    def importance(self):
        return importance(self._trainer.weights())


class ConfusionMatrix(object):
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
        labels_w_count = ["{}({})".format(label, label_count) for label, label_count in zip(self._labels, labels_count)]

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
        outpath = os.path.join(logdir, name)
        self.to_csv(outpath + ".csv")
        self.to_png(outpath + ".png")


def write_pred_table(pred, classes, data_subset, path):
    labels = data_subset.labels
    md5s = data_subset.ids

    string_labels = [classes[np.argmax(label)] for label in labels]
    df = pd.DataFrame(data=pred, index=md5s, columns=classes)

    df.insert(loc=0 , column="class", value=string_labels)
    
    df.to_csv(path, encoding="utf8")

def convert_matrix_csv_to_png(in_path, out_path):
    mat = ConfusionMatrix.from_csv(in_path)
    mat.to_png(out_path)

def metrics(acc, pred, data_subset):
    #TODO: separate metrics
    print("Accuracy: %s" % (acc))
    y_true = np.argmax(data_subset.labels, 1)
    y_pred = np.argmax(pred, 1)
    print ("Precision: %s" % sklearn.metrics.precision_score(y_true, y_pred, average="macro"))
    print ("Recall: %s" % sklearn.metrics.recall_score(y_true, y_pred, average="macro"))
    print ("f1_score: %s" % sklearn.metrics.f1_score(y_true, y_pred, average="macro"))
    print ("MCC: %s" % sklearn.metrics.matthews_corrcoef(y_true, y_pred))

def importance(w):
    #garson algorithm, w for weights 
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
        raise AssertionError("Signal_length not coherent with given resolution of {}.".format(resolution))

def bedgraph_from_importance(importance, chroms, resolution, bedgraph_path):
    """Write a bedgraph from the computed importance of features.
    The chromosome coordinates are zero-based, half-open (from 0 to N-1).
    """
    importance_index = 0
    with open(bedgraph_path, 'w') as my_bedgraph:
        for name, size in chroms:

            positions = itertools.chain(range(0, size, resolution), [size-1])

            for pos1, pos2 in pairwise(positions):

                line = [name, pos1, pos2, importance[importance_index]]
                my_bedgraph.write("{}\t{}\t{}\t{:.6f}\n".format(*line))
                importance_index += 1

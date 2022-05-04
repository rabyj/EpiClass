"""ConfusionMatrixWriter class"""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
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
        """Write to path an image of the confusion matrix.
        Colors : https://i.stack.imgur.com/cmk1J.png
        Ref code : https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        https://stackoverflow.com/questions/35710931/remove-a-section-of-a-colormap
        """
        plt.figure()

        # Check matrix type https://stackoverflow.com/questions/1342601/pythonic-way-of-checking-if-a-condition-holds-for-any-element-of-a-list
        # compute color bar ranges
        relative = True
        if np.any(self.pd_confusion_matrix > 1):
            relative = False

        if relative:
            vmax = 0.999 #this is so exactly 1.0 is a different color from the rest
            vmin = 0.0
        else:
            vmax = np.max(self.pd_confusion_matrix)
            vmin = 1.0

        # mask empty values, so they are white in the image
        data_mask = np.ma.masked_where(self.pd_confusion_matrix == 0, self.pd_confusion_matrix)

        # prep colormap
        nb_colors = 20
        gnuplot = cm.get_cmap("gnuplot", nb_colors) #20 colors
        newcolors = gnuplot(np.linspace(0., 1., nb_colors))
        new_cmap = ListedColormap(newcolors)
        new_cmap.set_over(matplotlib.colors.to_rgba("GreenYellow")) # color for max values, do it LAST

        # prep labels / ticks
        nb_labels = len(self.pd_confusion_matrix.columns)
        grid_width = 0.5 - nb_labels/400.0
        label_size = 15*np.exp(-0.02*nb_labels)

        # create color mesh and arrange ticks
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(data_mask, cmap=new_cmap, vmin=vmin, vmax=vmax, edgecolors='k', linewidths=grid_width)

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

        # arrange color bar
        bounds = np.linspace(vmin, math.ceil(vmax), nb_colors+1) #just to have the max tick appear properly
        ticks = np.linspace(vmin, math.ceil(vmax), 11)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, boundaries=bounds, ticks=ticks)
        cbar.ax.tick_params(labelsize=4)

        plt.tight_layout()
        plt.savefig(path, format='png', dpi=500)


    def to_csv(self, path):
        """Write to path a csv file of the confusion matrix."""
        self.pd_confusion_matrix.to_csv(path, encoding="utf8", float_format='%.4f')
        return path

    def to_all_formats(self, logdir, name):
        """Write to path both a png and csv file of the confusion matrix."""
        outpath = Path(logdir) / name
        out1 = outpath.with_suffix(".csv")
        out2 = outpath.with_suffix(".png")
        self.to_csv(out1)
        self.to_png(out2)
        return out1, out2

    @staticmethod
    def convert_matrix_csv_to_png(in_path, out_path):
        """Convert csv of confusion matrix to a png, and write it to out_path."""
        mat = ConfusionMatrixWriter.from_csv(in_path)
        mat.to_png(out_path)

"""ConfusionMatrixWriter class"""
from __future__ import annotations
import math
from pathlib import Path
import re
from typing import Tuple
import warnings


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

class InputMatrixError(Exception):
    pass


class ConfusionMatrixWriter(object):
    """Class to create/handle confusion matrices.

    labels : list of classes string representation
    confusion_matrix : A confusion matrix that counts each final prediction (int matrix)
    Expects a confusion matrix input with prediction rows (row value: pred1 pred2 pred3 ...) and target columns.
    """
    def __init__(self, labels, confusion_matrix: np.ndarray):
        self._labels = sorted(labels)
        self._og_confusion_mat = np.array(confusion_matrix)
        self._pd_matrix, self._pd_rel_matrix = self.init_confusion_matrices(confusion_matrix) #pd dataframe

    def __add__(self, other: ConfusionMatrixWriter) -> ConfusionMatrixWriter:
        if set(self._labels) != set(other._labels):
            warnings.warn("Cannot add matrices with different labels")
            return None

        new_mat = self._og_confusion_mat + other._og_confusion_mat
        new_mat = ConfusionMatrixWriter(self._labels, new_mat)
        return new_mat

    @staticmethod
    def _extract_class(label: str):
        """Extract class for a label with count, e.g. input(42)."""
        return str(label).split("(", 1)[0]

    @classmethod
    def from_csv(cls, csv_path, relative: bool):
        """Create instance from already written file.
        The state of the matrix (relative or not) needs to be specified.
        """
        obj = cls.__new__(cls)  # Does not call __init__
        content = pd.DataFrame(pd.read_csv(csv_path, sep=',', index_col=0))
        labels_w_count = content.index.tolist()
        values = content.values
        if relative:
            if np.any(a=values>=2, axis=None):
                raise InputMatrixError("Inputed file seems to contain count values, but relative=True was given.")

            obj._pd_rel_matrix = content
            obj._labels = sorted([ConfusionMatrixWriter._extract_class(val) for val in labels_w_count])

            labels_count = re.findall(r"\(([0-9]+)\)", "".join(labels_w_count))
            labels_count = np.array(labels_count, dtype=int)

            mat = np.array((content.values.T * labels_count).T, dtype=float)
            obj._og_confusion_mat = np.around(mat).astype(int)

            obj._pd_matrix = pd.DataFrame(data=obj._og_confusion_mat, index=content.index, columns=content.columns) # pylint: disable=no-member
        else:
            if np.any(a=(values<=0.99)&(values>=0.01), axis=None):
                raise InputMatrixError("Inputed file seems to contain relative values, but relative=False was given.")

            obj._pd_matrix = content
            obj._labels = sorted([ConfusionMatrixWriter._extract_class(val) for val in labels_w_count]) # pylint: disable=no-member

            obj._og_confusion_mat = content.to_numpy() # pylint: disable=no-member

            rel_mat = ConfusionMatrixWriter.to_relative_confusion_matrix(
                labels_count=obj._og_confusion_mat.sum(axis=1),
                confusion_matrix=obj._og_confusion_mat
            )
            obj._pd_rel_matrix = pd.DataFrame(data=rel_mat, index=content.index, columns=content.columns) # pylint: disable=no-member
        return obj

    def init_confusion_matrices(self, confusion_matrix: np.ndarray):
        """Returns confusion matrices with labels (pandas df) from int matrix.
        Expects prediction rows (target pred1 pred2 pred3 ....) and target columns.
        Returns original and normalized on rows matrices.
        """
        labels_count = confusion_matrix.sum(axis=1) # total nb examples of each label
        labels_w_count = [f"{label}({label_count})" for label, label_count in zip(self._labels, labels_count)]

        count_matrix = pd.DataFrame(data=confusion_matrix, index=labels_w_count, columns=self._labels, dtype=int)

        rel_confusion_mat = ConfusionMatrixWriter.to_relative_confusion_matrix(labels_count, confusion_matrix)
        rel_matrix = pd.DataFrame(data=rel_confusion_mat, index=labels_w_count, columns=self._labels)

        return count_matrix, rel_matrix

    @staticmethod
    def to_relative_confusion_matrix(labels_count: np.ndarray, confusion_matrix: np.ndarray):
        """Normalize confusion matrix per row.
        Expects prediction rows (target pred1 pred2 pred3 ....) and target columns.
        """
        confusion_mat1 = np.divide(confusion_matrix.T, labels_count)
        confusion_mat2 = np.nan_to_num(confusion_mat1)
        return confusion_mat2.T

    def to_png(self, path):
        """Write to path an image of the confusion matrix.
        Colors : https://i.stack.imgur.com/cmk1J.png
        Ref code : https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        https://stackoverflow.com/questions/35710931/remove-a-section-of-a-colormap
        """
        plt.figure()

        vmax = 0.9999 #this is so exactly 1.0 is a different color from the rest
        vmin = 0.0

        # mask empty values, so they are white in the image
        data_mask = np.ma.masked_where(self._pd_rel_matrix == 0, self._pd_rel_matrix)

        # prep colormap
        nb_colors = 20
        gnuplot = cm.get_cmap("gnuplot", nb_colors) #20 colors
        newcolors = gnuplot(np.linspace(0., 1., nb_colors))
        new_cmap = ListedColormap(newcolors)
        new_cmap.set_over(matplotlib.colors.to_rgba("GreenYellow")) # color for max values, do it LAST


        # create color mesh and arrange ticks
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(data_mask, cmap=new_cmap, vmin=vmin, vmax=vmax, edgecolors='k')

        nb_labels = self._pd_rel_matrix.columns.shape[0]
        ax.set_frame_on(False)
        ax.set_xticks(np.arange(nb_labels), minor=False)
        ax.set_yticks(np.arange(nb_labels), minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(self._pd_rel_matrix.columns)
        ax.set_yticklabels(self._pd_rel_matrix.index)
        plt.xticks(rotation=70, ha="left")
        plt.yticks(va="top")

        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # annotate each matrix value position
        for i in np.arange(nb_labels):
            for j in np.arange(nb_labels):
                count = self._pd_matrix.iat[i, j]
                if count != 0:
                    text = f"{count}\n{self._pd_rel_matrix.iat[i, j]*100:.1f}%"
                    text_obj = plt.text(
                        x=j+0.5, y=i+0.5, s=text,
                        horizontalalignment='center',
                        verticalalignment='center',
                        color="w",
                        size=5
                        )
                    text_obj.set_path_effects(
                        [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()]
                        )

        # Color bar
        bounds = np.linspace(vmin, math.ceil(vmax), nb_colors+1) #just to have the max tick appear properly
        ticks = np.linspace(vmin, math.ceil(vmax), 11)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, boundaries=bounds, ticks=ticks, extend="max")
        cbar.ax.tick_params(labelsize=7)


        plt.tight_layout()
        plt.savefig(path, format='png', dpi=500)

    def to_csv(self, path, relative):
        """Write to path a csv file of the confusion matrix.

        The type of matrix (relative by row, or not) needs to be specified.
        """
        if relative:
            self._pd_rel_matrix.to_csv(path, encoding="utf8", float_format='%.4f')
        else:
            self._pd_matrix.to_csv(path, encoding="utf8")

    def to_all_formats(self, logdir: str, name: str) -> Tuple[Path, Path, Path]:
        """Write to logdir files of the confusion matrix.
        out 1 : Path of csv of non-normalized matrix
        out 2 : Path of csv of normalized matrix
        out 3 : Path of png of matrix
        """
        outpath = Path(logdir) / name

        out1 = outpath.with_suffix(".csv")
        out1_rel = outpath.with_name(f"{name}_relative.csv")
        out2 = outpath.with_suffix(".png")

        self.to_csv(out1, relative=False)
        self.to_csv(out1_rel, relative=True)
        self.to_png(out2)

        return out1, out1_rel, out2

    @staticmethod
    def convert_matrix_csv_to_png(in_path, out_path, relative):
        """Convert csv of confusion matrix to a png, and write it to out_path.

        The state of the read matrix (relative by row or not) needs to be specified.
        """
        writer = ConfusionMatrixWriter.from_csv(in_path, relative)
        writer.to_png(out_path)

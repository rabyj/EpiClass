"""
This Python script is intended to merge or create confusion matrices.
It uses the `epi_ml` library's `ConfusionMatrixWriter` for tasks related to confusion matrices.

Command line arguments.
    It expects '--from_existing' to combine CSV matrices or '--from_prediction'
    to convert a classification prediction file into a confusion matrix.

Note: The prediction file used by the "from_prediction" option must
at least have 'True class' and 'Predicted class' headers to generate the confusion matrix.
Other columns, if present, will be ignored.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sklearn.metrics

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.confusion_matrix import ConfusionMatrixWriter


def parse_arguments() -> argparse.Namespace:
    """Return argument line parser."""
    parser = ArgumentParser()
    # fmt: off
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from_existing", metavar="logdir", type=DirectoryChecker(), help="Location of csv matrix logdir to combine.",
    )
    group.add_argument(
        "--from_prediction", metavar="pred-file", type=Path, help="Prediction file to convert to a confusion matrix.",
    )
    # fmt: on
    return parser.parse_args()


def add_matrices(logdir: str):
    """Add several matrices together from folds."""
    gen = logdir + "/split{i}/validation_confusion_matrix.csv"

    mat = ConfusionMatrixWriter.from_csv(csv_path=gen.format(i=0), relative=False)
    for i in range(1, 10):
        csv_path = Path(gen.format(i=i))
        if csv_path.exists():
            mat2 = ConfusionMatrixWriter.from_csv(csv_path=csv_path, relative=False)
            mat = mat + mat2
        else:
            print(f"File does not exist: {csv_path}")

    mat.to_all_formats(logdir=logdir, name="full-10fold-validation")


def main():
    """Augment a label prediction file with new metadata categories.

    File header format important. Expects [md5sum, true class, predicted class, labels] lines.
    """
    args = parse_arguments()

    if args.from_existing:
        add_matrices(logdir=args.from_logdir.parent)
    else:
        pred_file = args.from_prediction
        logdir = pred_file.parent

        df = pd.read_csv(
            pred_file,
            sep=",",
            usecols=["True class", "Predicted class"],
        )

        true, pred = df.iloc[:, 0], df.iloc[:, 1]
        labels = sorted(set(true.unique().tolist() + pred.unique().tolist()))
        confusion_mat = sklearn.metrics.confusion_matrix(true, pred, labels=labels)

        writer = ConfusionMatrixWriter(labels=labels, confusion_matrix=confusion_mat)
        writer.to_all_formats(
            logdir=logdir,
            name=str(pred_file.stem) + "-confusion-matrix",
        )


if __name__ == "__main__":
    main()

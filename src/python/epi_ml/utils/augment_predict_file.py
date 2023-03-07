"""Augment a label prediction file with new metadata categories.

File header format important. Expects [md5sum, true class, predicted class, labels] lines.
"""
from __future__ import annotations

import argparse
import csv
import decimal
import os
import os.path
from pathlib import Path

import numpy as np
import pandas as pd

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.core.confusion_matrix import ConfusionMatrixWriter as ConfusionMatrix
from epi_ml.core.metadata import Metadata

TARGET = "assay_epiclass"


def parse_arguments() -> argparse.Namespace:
    """Return argument line parser."""
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument(
        "predict_file", metavar="predict-file", type=Path, help="Predict file to augment with metadata.",
    )
    parser.add_argument(
        "metadata", type=Path, help="Metadata file to use.",
    )
    parser.add_argument(
        "--correct-true",
        metavar="LABEL_CATEGORY",
        type=str,
        help="Replace 'True class' field labels with metadata values of given LABEL_CATEGORY.",
        default=None
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--categories", nargs="+", type=str, help="Specific metadata categories to add.",
    )
    group.add_argument(
        "--all-categories",
        action="store_true",
        help="Add all available metadata categories.",
    )
    # fmt: on
    return parser.parse_args()


def augment_header(header, categories):
    """Augment the file header with new metadata categories"""
    targets_headers = header[1:3]
    pred_labels = header[3:]
    second_pred_info = [
        "Same?",
        "Max pred",
        "2nd pred class",
        "1rst/2nd prob diff",
        "1rst/2nd prob ratio",
    ]

    new_header = (
        ["md5sum"] + categories + targets_headers + second_pred_info + pred_labels
    )
    return new_header


def augment_line(line, metadata: Metadata, categories, classes):
    """Augment a non-header line with new metadata labels and additional info on 2nd highest prob."""
    md5 = line[0]
    targets = line[1:3]
    is_same = targets[0] == targets[1]

    prec4 = decimal.Decimal(".0001")
    prec2 = decimal.Decimal(".01")
    decimal.setcontext(decimal.ExtendedContext)

    preds = [decimal.Decimal(val).quantize(prec4) for val in line[3:]]

    order = np.argsort(preds)  # type: ignore
    i_1 = order[-1]
    i_2 = order[-2]
    diff = preds[i_1] - preds[i_2]
    ratio = (preds[i_1] / preds[i_2]).quantize(prec2)

    class_2 = classes[i_2]

    # get all labels for given categories
    new_labels = [metadata[md5].get(category, "--empty--") for category in categories]

    new_line = (
        [md5] + new_labels + targets + [is_same, preds[i_1], class_2, diff, ratio] + preds
    )
    return new_line


def augment_predict(
    metadata: Metadata, predict_path: Path, categories, append_name: str | None = None
):
    """Read -> augment -> write, row by row.

    Expects [md5sum, true class, predicted class, labels] lines.
    """
    root, ext = os.path.splitext(predict_path)

    new_root = root + "_augmented"
    if append_name is not None:
        new_root = new_root + f"-{append_name}"

    new_path = new_root + ext

    with open(predict_path, "r", encoding="utf-8") as infile, open(
        new_path, "w", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile, delimiter=",")
        writer = csv.writer(outfile, delimiter=",")

        header = next(reader)
        classes = header[3:]

        new_header = augment_header(header, categories)
        writer.writerow(new_header)

        for line in reader:
            new_line = augment_line(line, metadata, categories, classes)
            writer.writerow(new_line)

    return new_path


def add_matrices(logdir: str):
    """Add several matrices from 10fold together."""
    gen = logdir + "/split{i}/validation_confusion_matrix.csv"

    mat = ConfusionMatrix.from_csv(csv_path=gen.format(i=0), relative=False)
    for i in range(1, 10):

        csv_path = Path(gen.format(i=i))
        if csv_path.exists():
            mat2 = ConfusionMatrix.from_csv(csv_path=csv_path, relative=False)
            mat = mat + mat2
        else:
            print(f"File does not exist: {csv_path}")

    mat.to_all_formats(logdir=logdir, name="full-10fold-validation")


def write_coherence(path, category: str):
    """Read file, add coherence for given category, write it updated to same path."""
    df = pd.read_csv(path, sep=",")
    add_coherence(df, category)
    add_track_type_coherence(df)
    df.to_csv(path, sep=",", index=False)


def add_coherence(df: pd.DataFrame, category: str):
    """Add another metric based on multiple lines. Needs a file with EpiRR column.

    https://stackoverflow.com/questions/17995024/how-to-assign-a-name-to-the-size-column
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transform.html
    """
    df.set_index("md5sum")
    groups = df.groupby("EpiRR")
    if "files/epiRR" not in df.columns:
        df["files/epiRR"] = groups["md5sum"].transform("size")

    # compute coherence count/ratio, make category column countable beforehand
    col_copy = df[category].copy(deep=True)
    df[category] = df[category].astype(str)

    groups = df.groupby(["EpiRR", category])
    df[f"{category} Coherence count"] = groups["md5sum"].transform("size")

    df[f"{category} Coherence ratio"] = df[f"{category} Coherence count"] / (
        1.0 * df["files/epiRR"]
    )

    # return category to initial form
    df[category] = col_copy

def add_track_type_coherence(df):
    """Add a more complex coherence metric.
    Tells us how much tracks types agree on predicted class for a unique experiment
    """
    df.set_index("md5sum")

    cat1 = "files/experiment"
    cat2 = "Track type coherence count"
    cat3 = "Track type coherence ratio"

    groups = df.groupby(["EpiRR", TARGET])
    if cat1 not in df.columns:
        df[cat1] = groups["md5sum"].transform("size")

    groups = df.groupby(["EpiRR", TARGET, "Predicted class"])
    df[cat2] = groups["md5sum"].transform("size")

    df[cat3] = df[cat2] / (1.0 * df[cat1])


def correct_true(path: Path, category: str, metadata: Metadata):
    """Read file and replace 'True class' labels with metadata values for given category."""
    df = pd.read_csv(path, sep=",", header=0, index_col=0)
    for row in df.itertuples():
        md5 = row.Index
        df.at[md5, "True class"] = metadata[md5][category]
    df.to_csv(path, sep=",")


def main():
    """Augment a label prediction file with new metadata categories.

    File header format important. Expects [md5sum, true class, predicted class, labels] lines."""
    if os.getenv("LOG") is not None:
        logdir = os.environ["LOG"]
        add_matrices(logdir)

    args = parse_arguments()

    metadata = Metadata(args.metadata)
    pred_file = args.predict_file

    if args.correct_true:
        correct_true(path=pred_file, category=args.correct_true, metadata=metadata)

    categories = args.categories
    if args.all_categories:
        categories = metadata.get_categories()
        new_path = augment_predict(metadata, pred_file, categories, append_name="all")
        write_coherence(new_path, "Predicted class")
    else:
        augment_predict(metadata, pred_file, categories)


if __name__ == "__main__":
    main()

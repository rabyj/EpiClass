import argparse
import csv
import os.path
from pathlib import Path
import sys

import numpy as np

from epi_ml.python.core.metadata import Metadata

def parse_args(argv):
    """Return argument line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("predict", type=Path, help="Predict file to augment with metadata.")
    parser.add_argument("metadata", type=Path, help="Metadata file to use.")
    parser.add_argument("categories", nargs='+', type=str, help="Metadata categories to add.")
    return parser.parse_args(argv)


def augment_header(header, categories):
    """Augment the file header with new metadata categories"""
    targets_headers = header[1:3]
    pred_labels = header[3:]
    second_pred_info = ["2nd pred class", "1rst/2nd prob diff", "1rst/2nd prob ratio"]

    new_header = ["md5sum"] + categories + targets_headers + second_pred_info + pred_labels
    return new_header

def augment_line(line, metadata: Metadata, categories, classes):
    """Augment a non-header line with new metadata labels and additional info on 2nd highest prob."""
    md5 = line[0]
    targets = line[1:3]
    preds = [float(val) for val in line[3:]]

    order = np.argsort(preds)
    i_1 = order[-1]
    i_2 = order[-2]
    diff = preds[i_1] - preds[i_2]
    ratio = preds[i_1]/preds[i_2]
    class_2 = classes[i_2]

    new_labels = [
        metadata.get(md5).get(category, "--empty--")
        for category in categories
        ]

    new_line = [md5] + new_labels + targets + [class_2] + [diff] + [ratio] + preds
    return new_line

def augment_predict(metadata: Metadata, predict_path: Path, categories):
    """Read -> augment -> write, row by row.

    Expects [md5sum, true class, predicted class, labels] lines.
    """
    root, ext = os.path.splitext(predict_path)
    new_root = root + "_augmented"
    new_path = new_root + ext

    with open(predict_path, 'r', encoding="utf-8") as infile, open(new_path, 'w', encoding="utf-8") as outfile:
        reader = csv.reader(infile, delimiter=',')
        writer = csv.writer(outfile, delimiter=',')

        header = next(reader)
        classes = header[3:]

        new_header = augment_header(header, categories)
        writer.writerow(new_header)

        for line in reader:
            new_line = augment_line(line, metadata, categories, classes)
            writer.writerow(new_line)


def main(argv):
    """Augment a label prediction file with new metadata categories.

    File header format important. Expects [md5sum, true class, predicted class, labels] lines."""
    args = parse_args(argv)
    metadata = Metadata(args.metadata)
    augment_predict(metadata, args.predict, args.categories)

def cli():
    """Ignore program path."""
    main(sys.argv[1:])

if __name__ == "__main__":
    cli()

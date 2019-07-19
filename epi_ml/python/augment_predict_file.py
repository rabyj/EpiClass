import argparse
import csv
import os.path
import sys

from core.metadata import Metadata

def parse_args(argv):
    """Return argument line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("predict", type=str, help="Predict file to augment with metadata.")
    parser.add_argument("metadata", type=str, help="Metadata file to use.")
    parser.add_argument("categories", nargs='+', type=str, help="Metadata categories to add.")
    return parser.parse_args(argv)

def augment_header(header, categories):
    """Augment the file header with new metadata categories."""
    return header[:1] + categories + header[1:]

def augment_line(line, metadata, categories):
    """Augment a non-header line with new metadata labels."""
    md5 = line[0]
    labels = [
        metadata.get(md5).get(category, "--empty--")
        for category in categories
        ]
    return line[:1] + labels + line[1:]

def augment_predict(metadata, predict_path, categories):
    """Read -> augment -> write, row by row."""
    root, ext = os.path.splitext(predict_path)
    new_root = root + "_augmented"
    new_path = new_root + ext

    with open(predict_path, 'r') as infile, open(new_path, 'w') as outfile:
        reader = csv.reader(infile, delimiter=',')
        writer = csv.writer(outfile, delimiter=',')

        header = next(reader)
        new_header = augment_header(header, categories)
        writer.writerow(new_header)

        for line in reader:
            new_line = augment_line(line, metadata, categories)
            writer.writerow(new_line)

def main(argv):
    """Augment a label prediction file with new metadata categories."""
    args = parse_args(argv)
    metadata = Metadata.from_path(args.metadata)
    augment_predict(metadata, args.predict, args.categories)

def cli():
    main(sys.argv[1:])

if __name__ == "__main__":
    cli()

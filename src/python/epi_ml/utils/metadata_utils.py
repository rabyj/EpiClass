"""Metadata analysis basic utility functions. Also defines some constants."""
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from epi_ml.core.metadata import Metadata


def make_table(my_metadata: Metadata, cat1: str, cat2: str, filepath: str):
    """Write metadata content tsv table for given metadata categories"""
    counter = count_pairs(my_metadata, cat1, cat2)
    triplets = [(pair[0], pair[1], int(count)) for pair, count in sorted(counter.items())]

    df = pd.DataFrame(triplets, columns=[cat1, cat2, "count"])
    table = df.pivot_table(
        values="count",
        index=cat1,
        columns=cat2,
        aggfunc=np.sum,
        fill_value=0,
        margins=True,
        margins_name="Total",
    )
    table = table.T.sort_values(by="Total", ascending=False)
    table.to_csv(Path(filepath).with_suffix(".csv"), sep=",")


def count_pairs(my_metadata: Metadata, cat1: str, cat2: str):
    """Return label pairs (label_cat1, label_cat2) counter from the given metadata categories."""
    counter = collections.Counter(
        (dset.get(cat1, "--empty--"), dset.get(cat2, "--empty--"))
        for dset in my_metadata.datasets
    )
    return counter


def print_pairs(my_metadata: Metadata, cat1: str, cat2: str):
    """Print label pairs from the given metadata categories."""
    counter = count_pairs(my_metadata, cat1, cat2)
    for pair, count in sorted(counter.items()):
        print(pair, count)


EPIATLAS_CATS = set(
    [
        "uuid",
        "inputs",
        "inputs_ctl",
        "original_read_length",
        "original_read_length_ctl",
        "trimmed_read_length",
        "trimmed_read_length_ctl",
        "software_version",
        "chastity_passed",
        "paired_end_mode",
        "antibody_nan",
        "gembs_config",
        "rna_seq_type",
        "analyzed_as_stranded",
        "paired",
        "antibody",
        "upload_date",
    ]
)

EPIATLAS_ASSAYS = [
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k4me1",
    "h3k4me3",
    "h3k9me3",
    "input",
    "rna_seq",
    "mrna_seq",
    "wgbs",
    "wgbs-standard",
    "wgbs-pbat",
]

DP_ASSAYS = [
    "chromatin_acc",
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k4me1",
    "h3k4me3",
    "h3k9me3",
    "input",
    "mrna_seq",
    "rna_seq",
    "wgb_seq",
]

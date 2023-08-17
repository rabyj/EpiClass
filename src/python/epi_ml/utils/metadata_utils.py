"""Metadata analysis basic utility functions. Also defines some constants."""
import collections
import typing
from pathlib import Path
from typing import Tuple

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


def count_pairs(
    my_metadata: Metadata, cat1: str, cat2: str, use_uuid: bool = False
) -> typing.Counter[Tuple[str, str]]:
    """
    Return label pairs (label_cat1, label_cat2) counter from the given metadata categories.

    Args:
        my_metadata (Metadata): experiment metadata to analyze.
        cat1 (str): The first category to count.
        cat2 (str): The second category to count.
        use_uuid (bool): If True, count pairs by uuid; otherwise, by md5sum.

    Returns:
        Counter (Tuple[str, str], int): A counter object that counts label pairs.
    """
    id_label = "md5sum"
    if use_uuid:
        id_label = "uuid"

    unique_examples = collections.defaultdict(set)
    for dset in my_metadata.datasets:
        dset_id: str = dset[id_label]
        cat1_label: str = dset.get(cat1, "--empty--")
        cat2_label: str = dset.get(cat2, "--empty--")
        unique_examples[(cat1_label, cat2_label)].add(dset_id)

    counter = collections.Counter(
        {label_pair: len(ids) for label_pair, ids in unique_examples.items()}
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

import collections
import copy
from pathlib import Path
import os

import pandas as pd

from epi_ml.python.core.metadata import Metadata

def two_step_long_analysis(my_metadata: Metadata, category1, category2):

    print(f"First we remove small classes from {category1} and only keep datasets which have a {category2}.")
    my_metadata.remove_small_classes(10, category1)
    my_metadata.remove_missing_labels(category2)
    my_metadata.display_labels(category1)

    print(f"Then we examine the content of {category2} for each {category1}")
    nb_classes_kept = 0
    labels = my_metadata.label_counter(category1).keys()
    for label in sorted(labels):
        print(f"--{label}--")
        temp_metadata = copy.deepcopy(my_metadata)
        temp_metadata.select_category_subset(label, category1)
        temp_metadata.remove_small_classes(10, category2)
        if len(temp_metadata.label_counter(category2).most_common()) > 1:
            nb_classes_kept += 1
        temp_metadata.display_labels(category2)

    print(f"Number of classes with >1 cell_type label, from {category1}: {nb_classes_kept}/{len(labels)}")


def two_step_table_analysis(my_metadata: Metadata, category1, category2):

    my_metadata.remove_small_classes(10, category1)
    my_metadata.remove_small_classes(10, category2)
    my_metadata.remove_missing_labels(category2)

    counter_cat1 = my_metadata.label_counter(category1)

    info = {}
    for label in sorted(counter_cat1.keys()):
        info[label] = {}
        info[label]["total_examples_before_step2_filter"] = counter_cat1[label]

        temp_metadata = copy.deepcopy(my_metadata)
        temp_metadata.select_category_subset(label, category1)

        info[label]["total_classes_before_step2_filter"] = len(temp_metadata.label_counter(category2))

        temp_metadata.remove_small_classes(10, category2)

        counter_cat2 = temp_metadata.label_counter(category2)
        info[label]["total_examples_after_step2_filter"] = sum(counter_cat2.values())
        info[label]["total_classes_after_step2_filter"] = len(counter_cat2)

    line = "\t{total_classes_before_step2_filter}\t{total_classes_after_step2_filter}\t{total_examples_before_step2_filter}\t{total_examples_after_step2_filter}\t"
    print("class_step1\t#labels_before\t#labels_after\t#dsets_before\t#dsets_after\tratio #dsets/#labels")
    for label, infos in info.items():
        try:
            ratio = float(infos["total_examples_after_step2_filter"])/infos["total_classes_after_step2_filter"]
        except ZeroDivisionError:
            ratio = "--"
        print(label + line.format(**infos) + str(ratio))


def test(my_metadata: Metadata, category1):

    my_metadata.remove_small_classes(10, category1)
    counter_cat1 = my_metadata.label_counter(category1)
    for label in sorted(counter_cat1.keys()):
        temp_metadata = copy.deepcopy(my_metadata)
        temp_metadata.select_category_subset(label, category1)


def count_pairs(my_metadata: Metadata, cat1, cat2):
    """Return label pairs counter from the given metadata categories."""
    counter = collections.Counter(
        (dset.get(cat1, "--empty--"), dset.get(cat2, "--empty--"))
        for dset in my_metadata.datasets
    )
    return counter


def print_pairs(my_metadata: Metadata, cat1, cat2):
    """Print label pairs from the given metadata categories."""
    counter = count_pairs(my_metadata, cat1, cat2)
    for pair, count in sorted(counter.items()):
        print(pair, count)


def make_table(my_metadata: Metadata, cat1, cat2, filepath):
    """Write metadata content tsv table for given metadata categories"""
    counter = count_pairs(my_metadata, cat1, cat2)
    triplets = [(pair[0], pair[1], count) for pair, count in sorted(counter.items())]

    df = pd.DataFrame(triplets, columns=["assay", "cell_type", "count"])
    table = df.pivot_table(values="count", index="assay", columns="cell_type", fill_value=0)
    table.to_csv(filepath, sep='\t')


def analyze_chromatin_acc(my_metadata: Metadata):
    counter = collections.Counter(
        my_metadata[md5]["cell_type"] for md5 in my_metadata.md5s
        if my_metadata[md5]["assay"] == "chromatin_acc"
        )
    print(counter)
    counter = collections.Counter(
        my_metadata[md5]["assay"] for md5 in my_metadata.md5s
        if my_metadata[md5]["cell_type"] == "fetal_muscle_arm"
        )
    print(counter)


def main():

    base = Path("/home/local/USHERBROOKE/rabj2301/Projects/ihec/2018-10/")
    path = base / "hg19_2018-10_final.json"
    my_metadata = Metadata(path)

    cat1 = "assay"
    cat2 = "cell_type"

    # my_metadata = metadata.special_case(my_metadata)
    # my_metadata = metadata.special_case_2(my_metadata)

    # my_metadata = metadata.five_cell_types_selection(my_metadata)
    # assays_to_remove = [os.getenv(var, "") for var in ["REMOVE_ASSAY1", "REMOVE_ASSAY2", "REMOVE_ASSAY3"]]
    # my_metadata.remove_category_subsets(assays_to_remove, "assay")
    # my_metadata.select_category_subsets(["h3k4me1"], "assay")
    # my_metadata.remove_small_classes(10, "cell_type")

    # ---
    # two_step_long_analysis(my_metadata, cat1, cat2)
    # two_step_table_analysis(my_metadata, cat1, cat2)
    # test(my_metadata, cat1)
    # my_metadata = metadata.keep_major_cell_types(my_metadata)
    # my_metadata = metadata.keep_major_cell_types_alt(my_metadata)
    # ---

    # my_metadata.display_labels("assay")
    # my_metadata.display_labels("cell_type")

    # my_metadata.display_labels("publishing_group")
    # my_metadata.display_labels("releasing_group")

    # print_pairs(my_metadata, cat1, cat2)
    # make_table(my_metadata, cat1, cat2, "5_cell_types.tsv")

    # for md5 in sorted(my_metadata.md5s):
    #     print(md5)

if __name__ == "__main__":
    main()

import collections
import copy
from pathlib import Path

import pandas as pd

from epi_ml.python.core.metadata import Metadata


def keep_major_cell_types_alt(my_metadata: Metadata):
    """Return a filtered metadata with certain assays. Datasets which are
    not part of a cell_type which has at least 10 signals are removed.
    """
    my_meta = copy.deepcopy(my_metadata)

    # remove useless assays and cell_types
    assays = [
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
    my_meta.select_category_subsets("assay", assays)
    my_meta.remove_small_classes(10, "cell_type")
    my_meta.merge_classes("tissue_type", merge_fetal_tissues)

    new_meta = copy.deepcopy(my_meta)
    new_meta.empty()
    for assay in assays:
        temp_meta = copy.deepcopy(my_meta)
        temp_meta.select_category_subsets("assay", [assay])
        temp_meta.remove_small_classes(10, "cell_type")
        for md5 in temp_meta.md5s:
            new_meta[md5] = temp_meta[md5]

    return new_meta


def five_cell_types_selection(my_metadata: Metadata):
    """Return a filtered metadata with 5 major cell_types and certain assays."""
    cell_types = [
        "monocyte",
        "cd4_positive_helper_t_cell",
        "macrophage",
        "skeletal_muscle_tissue",
        "thyroid",
    ]
    assays = [
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
    my_metadata.select_category_subsets("cell_type", cell_types)
    my_metadata.select_category_subsets("assay", assays)
    return my_metadata


def special_case(my_metadata):
    """Return a filtered metadata with only rna_seq examples,
    but also add 3 thyroid (for model construction).

    Made to evaluate an already trained model,
    works with min_class_size=3 and oversample=False.
    """
    my_metadata = five_cell_types_selection(my_metadata)

    # get some thyroid examples md5s, there are none in rna_seq
    temp_meta = copy.deepcopy(my_metadata)
    temp_meta.select_category_subsets("assay", ["h3k9me3"])
    md5s = temp_meta.md5_per_class("cell_type")["thyroid"][0:3]

    # select only rna_seq examples + 3 thyroid examples for model making
    my_metadata.select_category_subsets("assay", ["rna_seq"])
    for md5 in md5s:
        my_metadata[md5] = temp_meta[md5]

    return my_metadata


def special_case_2(my_metadata):
    """Return a filtered metadata without 2 examples
    from all assay/cell_type pairs, and all mrna_seq.
    """
    my_metadata = five_cell_types_selection(my_metadata)
    my_metadata.remove_category_subsets(
        "assay",
        ["mrna_seq"],
    )

    cell_types = my_metadata.md5_per_class("cell_type").keys()
    to_del = []
    for cell_type in cell_types:
        temp_meta = copy.deepcopy(my_metadata)
        temp_meta.select_category_subsets("cell_type", [cell_type])
        for md5s in temp_meta.md5_per_class("assay").values():
            to_del.extend(md5s[0:2])

    for md5 in to_del:
        del my_metadata[md5]

    return my_metadata


def keep_major_assays_2019(my_metadata):
    """Combine rna_seq and polr2a classes pairs in the assay category.
    Written for the 2019-11 release.
    """
    print("Filtering, removing smrna_seq, and then merging rna/polr2a similar labels")
    my_metadata.remove_small_classes(10, "assay")
    my_metadata.remove_category_subsets(
        "assay",
        ["smrna_seq"],
    )
    for dataset in my_metadata.datasets:
        assay = dataset.get("assay", None)
        if assay == "mrna_seq":
            dataset["assay"] = "rna_seq"
        elif assay == "polr2aphosphos5":
            dataset["assay"] = "polr2a"

    return my_metadata


def keep_major_cell_types_2019(my_metadata):
    """Select 20 cell types in the major assays signal subset.
    A cell type needs to have at least 10 signals in one assay.
    Selection choices made out of the code.
    Written for the 2019-11 release.
    """
    my_metadata = keep_major_assays_2019(my_metadata)

    brain_labels = set(
        [
            "brain_occipetal_lobe_right",
            "brain_right_temporal",
            "brain_temporal_lobe_left",
            "brain",
            "brain_frontal_lobe_left",
        ]
    )
    hepatocyte_labels = set(["hepatocytes", "hepatocyte"])
    large_intestin_colon_labels = set(
        [
            "large_intestine_colon_ascending_(right)",
            "large_intestine_colon",
            "large_intestine_colon_rectosigmoid",
        ]
    )

    for dataset in my_metadata.datasets:
        cell_type = dataset.get("cell_type", None)
        if cell_type in brain_labels:
            dataset["cell_type"] = "brain"
        elif cell_type in hepatocyte_labels:
            dataset["cell_type"] = "hepatocytes"
        elif cell_type in large_intestin_colon_labels:
            dataset["cell_type"] = "large_intestine_colon"

    selected_cell_types = set(
        [
            "myeloid_cell",
            "venous_blood",
            "monocyte",
            "thyroid",
            "mature_neutrophil",
            "macrophage",
            "b_cells",
            "cd14_positive,_cd16_negative_classical_monocyte",
            "normal_human_colon_absorptive_epithelial_cells",
            "cd4_positive,_alpha_beta_t_cell",
            "precursor_b_cell",
            "naive_b_cell",
            "stomach",
            "lymph_node",
            "muscle_of_leg",
            "neoplastic_plasma_cell",
            "plasma_cell",
            "brain",
            "hepatocytes",
            "large_intestine_colon",
        ]
    )
    my_metadata.select_category_subsets(
        "cell_type",
        selected_cell_types,
    )

    return my_metadata


merge_fetal_tissues = {
    "fetal_intestine_large": "fetal_intestine",
    "fetal_intestine_small": "fetal_intestine",
    "fetal_lung_left": "fetal_lung",
    "fetal_lung_right": "fetal_lung",
    "fetal_muscle_arm": "fetal_muscle",
    "fetal_muscle_back": "fetal_muscle",
    "fetal_muscle_leg": "fetal_muscle",
    "fetal_renal_cortex": "fetal_kidney",
    "fetal_renal_pelvis": "fetal_kidney",
}

merge_molecule = {"rna": "total_rna", "polyadenylated_mrna": "polya_rna"}


def two_step_long_analysis(my_metadata: Metadata, category1, category2):

    print(
        f"First we remove small classes from {category1} and only keep datasets which have a {category2}."
    )
    my_metadata.remove_small_classes(10, category1)
    my_metadata.remove_missing_labels(category2)
    my_metadata.display_labels(category1)

    print(f"Then we examine the content of {category2} for each {category1}")
    nb_classes_kept = 0
    labels = my_metadata.label_counter(category1).keys()
    for label in sorted(labels):
        print(f"--{label}--")
        temp_metadata = copy.deepcopy(my_metadata)
        temp_metadata.select_category_subsets(label, [category1])
        temp_metadata.remove_small_classes(10, category2)
        if len(temp_metadata.label_counter(category2).most_common()) > 1:
            nb_classes_kept += 1
        temp_metadata.display_labels(category2)

    print(
        f"Number of classes with >1 cell_type label, from {category1}: {nb_classes_kept}/{len(labels)}"
    )


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
        temp_metadata.select_category_subsets(label, [category1])

        info[label]["total_classes_before_step2_filter"] = len(
            temp_metadata.label_counter(category2)
        )

        temp_metadata.remove_small_classes(10, category2)

        counter_cat2 = temp_metadata.label_counter(category2)
        info[label]["total_examples_after_step2_filter"] = sum(counter_cat2.values())
        info[label]["total_classes_after_step2_filter"] = len(counter_cat2)

    line = "\t{total_classes_before_step2_filter}\t{total_classes_after_step2_filter}\t{total_examples_before_step2_filter}\t{total_examples_after_step2_filter}\t"
    print(
        "class_step1\t#labels_before\t#labels_after\t#dsets_before\t#dsets_after\tratio #dsets/#labels"
    )
    for label, infos in info.items():
        try:
            ratio = (
                float(infos["total_examples_after_step2_filter"])
                / infos["total_classes_after_step2_filter"]
            )
        except ZeroDivisionError:
            ratio = "--"
        print(label + line.format(**infos) + str(ratio))


def test(my_metadata: Metadata, category1):

    my_metadata.remove_small_classes(10, category1)
    counter_cat1 = my_metadata.label_counter(category1)
    for label in sorted(counter_cat1.keys()):
        temp_metadata = copy.deepcopy(my_metadata)
        temp_metadata.select_category_subsets(label, [category1])


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
    table = df.pivot_table(
        values="count", index="assay", columns="cell_type", fill_value=0
    )
    table.to_csv(filepath, sep="\t")


def analyze_chromatin_acc(my_metadata: Metadata):
    counter = collections.Counter(
        my_metadata[md5]["cell_type"]
        for md5 in my_metadata.md5s
        if my_metadata[md5]["assay"] == "chromatin_acc"
    )
    print(counter)
    counter = collections.Counter(
        my_metadata[md5]["assay"]
        for md5 in my_metadata.md5s
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
    # my_metadata.remove_category_subsets("assay", assays_to_remove)
    # my_metadata.select_category_subsets("assay", ["h3k4me1"])
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

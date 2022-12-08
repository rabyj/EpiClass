import collections
import copy
from pathlib import Path

import pandas as pd

from .modify_metadata import count_pairs, epiatlas_cats
from src.python.core.epiatlas_treatment import TRACKS_MAPPING
from src.python.core.metadata import Metadata
from src.python.utils.augment_predict_file import add_coherence


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


def print_pairs(my_metadata: Metadata, cat1, cat2):
    """Print label pairs from the given metadata categories."""
    counter = count_pairs(my_metadata, cat1, cat2)
    for pair, count in sorted(counter.items()):
        print(pair, count)


def make_table(my_metadata: Metadata, cat1: str, cat2: str, filepath: str):
    """Write metadata content tsv table for given metadata categories"""
    counter = count_pairs(my_metadata, cat1, cat2)
    triplets = [(pair[0], pair[1], count) for pair, count in sorted(counter.items())]

    df = pd.DataFrame(triplets, columns=["assay", "cell_type", "count"])
    table = df.pivot_table(
        values="count", index="assay", columns="cell_type", fill_value=0
    )
    table.to_csv(Path(filepath).with_suffix(".tsv"), sep="\t")


def compute_coherence_on_all(meta: Metadata):
    """Compute coherence on all metadata categories."""
    for dset in list(meta.datasets):
        if dset["track_type"] in set({"pval", "fc"}):
            del meta[dset["md5sum"]]

    df = pd.DataFrame(list(meta.datasets))  # type: ignore
    df = df.replace(r"^\s*$", "--", regex=True)

    for category in list(df.columns):
        if category not in epiatlas_cats | set(
            [
                "md5sum",
                "EpiRR",
                "epirr_id",
                "track_type",
                "assay",
            ]
        ):
            add_coherence(df, category)

    df.drop(df.filter(regex="Coherence count").columns, axis=1, inplace=True)  # type: ignore

    df.to_csv("test.csv", index=False)


def check_epitatlas_uuid_premise(metadata: Metadata):
    """Check that there is only one file per track type, for a given uuid."""
    uuid_to_md5s = collections.defaultdict(collections.Counter)
    for dset in metadata.datasets:
        uuid = dset["uuid"]
        uuid_to_md5s[uuid].update([dset["track_type"]])

    for uuid, counter in uuid_to_md5s.items():
        for nb in counter.values():
            if nb != 1:
                print(uuid, counter)


def main():

    base = Path("/home/local/USHERBROOKE/rabj2301/Projects/epilap/input/metadata")
    path = base / "merge_EpiAtlas_allmetadata-v10.json"
    my_metadata = Metadata(path)

    md5_list = "/home/local/USHERBROOKE/rabj2301/Projects/sources/epi_ml/src/python/tests/core/test-epilap-empty-biotype-n40.md5"
    with open(md5_list, "r", encoding="utf8") as f:
        md5_set = set([md5.strip() for md5 in f.readlines()])

    print(md5_set)

    for md5 in list(my_metadata.md5s):
        if md5 not in md5_set:
            del my_metadata[md5]

    my_metadata.save(
        "/home/local/USHERBROOKE/rabj2301/Projects/sources/epi_ml/src/python/tests/core/test-epilap-empty-biotype-n40-metadata.json"
    )

    # cats = my_metadata.get_categories()
    # for cat in cats:
    #     if " " in cat:
    #         print(cat)
    # fix_roadmap(my_metadata)
    # my_metadata.display_labels("data_generating_centre")

    # merge_pair_end_info(my_metadata)
    # my_metadata.select_category_subsets("paired_end_mode", ["single_end", "paired_end"])
    # import numpy as np
    # import sklearn
    # from statsmodels.multivariate.manova import MANOVA

    # df = pd.DataFrame(my_metadata.datasets)
    # y = df["paired_end_mode"]
    # y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
    # classifier = sklearn.linear_model.LogisticRegression(penalty="none")
    # classifier = sklearn.ensemble.RandomForestClassifier(
    #     class_weight="balanced_subsample"
    # )
    # correlations = []
    # for category in my_metadata.get_categories():
    #     X = df[category]
    #     X = sklearn.preprocessing.LabelEncoder().fit_transform(X).reshape(-1, 1)
    #     acc = []
    #     classifier = classifier.fit(X, y)
    #     for i in range(10):
    #         X, y = sklearn.utils.shuffle(X, y)
    #         y_pred = classifier.predict(X)
    #         acc_1 = sklearn.metrics.accuracy_score(y, y_pred)
    #         adj_acc = sklearn.metrics.balanced_accuracy_score(y, y_pred, adjusted=True)
    #         acc.append((acc_1, adj_acc))
    #     acc = np.array(acc)
    #     mean_acc = acc.mean(axis=0)
    #     std = acc.std(axis=0)[0]
    #     correlations.append((category, str(mean_acc[0]), str(mean_acc[1]), str(std)))
    # for metrics in correlations:
    #     print(" ".join(metrics))
    # pivot = pd.crosstab(
    #     df["sample_ontology_term_high_order_unique"], df["paired_end_mode"]
    # )
    # print(pivot)
    # chi2_stat, p, dof, expected = scipy.stats.chi2_contingency(pivot)
    # print(chi2_stat, expected)
    # import statsmodels.api as sm
    # table = sm.stats.Table.from_data(
    #     df[["sample_ontology_term_high_order_unique", "paired_end_mode"]]
    # )
    # print(table.table_orig, "\n")
    # print(table.fittedvalues, "\n")
    # print(table.resid_pearson, "\n")
    # print(table.chi2_contribs, "\n")
    # encode labels
    # df = df.apply(lambda x: pd.factorize(x)[0])
    # maov = MANOVA.from_formula(
    #     "sample_ontology_term_high_order_unique ~ paired_end_mode", data=df
    # )
    # print(maov.mv_test())
    # compute_coherence_on_all(my_metadata)
    # cat1 = "assay"
    # cat2 = "cell_type"
    # my_metadata.display_labels("harm_donor_sex")
    # for cat in my_metadata.get_categories():
    #     print(cat)
    # filter_cell_types_by_pairs(my_metadata)
    # make_table(my_metadata, cat1, cat2, "test")
    # for cat in my_metadata.get_categories():
    #     print(cat)
    # my_metadata.display_labels("paired_end_mode")
    # for dset in list(my_metadata.datasets):
    #     if "paired_end_mode" in dset:
    #         del my_metadata[dset["md5sum"]]
    # my_metadata.display_labels("assay")
    # my_metadata.display_labels("paired")
    # merge_pair_end_info(my_metadata)
    # my_metadata.display_labels("paired_end_mode")


if __name__ == "__main__":
    main()

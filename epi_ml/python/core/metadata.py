import copy
import collections
import json
import io
import os.path

import tensorflow as tf
import numpy as np

from .data_source import EpiDataSource

class Metadata(object):
    """Wrapper around metadata md5:dataset dict."""
    def __init__(self, meta_file: io.IOBase):
        self._metadata = self._load_metadata(meta_file)

    @classmethod
    def from_path(cls, path):
        """Initialize from metadata filepath."""
        with open(path, 'r') as meta_file:
            return cls(meta_file)

    @classmethod
    def from_epidatasource(cls, datasource: EpiDataSource):
        """Initialize from EpiDataSource"""
        return cls(datasource.metadata_file)

    def empty(self):
        """Remove all entries."""
        self._metadata = {}

    def __setitem__(self, md5, value):
        self._metadata[md5] = value

    def __getitem__(self, md5):
        return self._metadata[md5]

    def __delitem__(self, md5):
        del self._metadata[md5]

    def __contains__(self, md5):
        return md5 in self._metadata

    def __len__(self):
        return len(self._metadata)

    def get(self, md5, default=None):
        """Dict .get"""
        return self._metadata.get(md5, default)

    @property
    def md5s(self):
        """Return keys."""
        return self._metadata.keys()

    @property
    def datasets(self):
        """Return values."""
        return self._metadata.values()

    def _load_metadata(self, meta_file: io.IOBase):
        """Return md5:dataset dict."""
        meta_file.seek(0)
        meta_raw = json.load(meta_file)
        return {dset["md5sum"]:dset for dset in meta_raw["datasets"]}

    def apply_filter(self, meta_filter=lambda item: True):
        """Apply a filter on items (md5:dataset)."""
        self._metadata = dict(filter(meta_filter, self._metadata.items()))

    def remove_missing_labels(self, label_category):
        """Remove datasets where the metadata category is missing."""
        filt = lambda item: label_category in item[1]
        self.apply_filter(filt)

    def md5_per_class(self, label_category):
        """Return {label/class:md5 list} dict for a given metadata category.

        Can fail if remove_missing_labels has not been ran before.
        """
        sorted_md5 = sorted(self._metadata.keys())
        data = collections.defaultdict(list)
        for md5 in sorted_md5:
            data[self._metadata[md5][label_category]].append(md5)
        return data

    def remove_small_classes(self, min_class_size, label_category):
        """Remove classes with less than min_class_size examples
        for a given metatada category.
        """
        data = self.md5_per_class(label_category)
        nb_class = len(data)

        nb_removed_class = 0
        for label, size in self.label_counter(label_category).most_common():
            if size < min_class_size:
                nb_removed_class += 1
                for md5 in data[label]:
                    del self._metadata[md5]

        print("{}/{} labels left from \"{}\" after removing classes with less than {} signals.".format(
            nb_class - nb_removed_class, nb_class, label_category, min_class_size))

    def select_category_subset(self, label, label_category):
        """Select only datasets which possess the given label
        for the given label category.
        """
        filt = lambda item: item[1].get(label_category) == label
        self.apply_filter(filt)

    def select_category_subsets(self, labels, label_category):
        """Select only datasets which possess the given labels
        for the given label category.
        """
        filt = lambda item: item[1].get(label_category) in set(labels)
        self.apply_filter(filt)

    def remove_category_subset(self, label, label_category):
        """Remove datasets which possess the given label
        for the given label category.
        """
        filt = lambda item: item[1].get(label_category) != label
        self.apply_filter(filt)

    def remove_category_subsets(self, labels, label_category):
        """Remove datasets which possess the given label
        for the given label category.
        """
        filt = lambda item: item[1].get(label_category) not in set(labels)
        self.apply_filter(filt)

    def label_counter(self, label_category):
        """Return a Counter() with label count from the given category."""
        counter = collections.Counter()
        for labels in self._metadata.values():
            label = labels[label_category]
            counter.update([label])
        return counter

    def display_labels(self, label_category):
        """Print number of examples for each label in given category."""
        print('\nExamples')
        i = 0
        for label, count in self.label_counter(label_category).most_common():
            print('{}: {}'.format(label, count))
            i += count
        print('For a total of {} examples\n'.format(i))

    def category_class_weights(self, label_category):
        """Return class weights for the given category, ordered
        by alphabetical order of labels.
        """
        counter = self.label_counter(label_category)
        weights = np.array([class_size for label, class_size in sorted(counter.most_common())])
        weights = 1. / (weights / np.amax(weights))
        return tf.constant(weights, shape=[1, weights.size], dtype=tf.float32)

    def create_healthy_category(self):
        """Combine "disease" and "donor_health_status" to create a "healthy" category.

        When a dataset has pairs with unknow correspondance, it does not add
        the category, and so these datasets are ignored through remove_missing_labels().
        """
        healthy_category = HealthyCategory()
        for dataset in self.datasets:
            healthy = healthy_category.get_healthy_status(dataset)
            if healthy == "?":
                continue
            dataset["healthy"] = healthy

    def merge_molecule_classes(self):
        """Combine similar classes pairs in the molecule category."""
        for dataset in self.datasets:
            molecule = dataset.get("molecule", None)
            if molecule == "rna":
                dataset["molecule"] = "total_rna"
            elif molecule == "polyadenylated_mrna":
                dataset["molecule"] = "polya_rna"

    def merge_fetal_tissues(self):
        """Combine similar fetal tissues in the cell_type category."""
        conversion = {
            "fetal_intestine_large":"fetal_intestine",
            "fetal_intestine_small":"fetal_intestine",
            "fetal_lung_left":"fetal_lung",
            "fetal_lung_right":"fetal_lung",
            "fetal_muscle_arm":"fetal_muscle",
            "fetal_muscle_back":"fetal_muscle",
            "fetal_muscle_leg":"fetal_muscle",
            "fetal_renal_cortex":"fetal_kidney",
            "fetal_renal_pelvis":"fetal_kidney"
        }
        for dataset in self.datasets:
            cell_type = dataset.get("cell_type", None)
            if cell_type in conversion:
                dataset["cell_type"] = conversion[cell_type]


class HealthyCategory(object):
    """Create/Represent/manipulate the "healthy" metadata category"""
    def __init__(self):
        self.pairs_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "healthy_category.tsv"
            )
        self.healthy_dict = self.read_healthy_pairs()

    @staticmethod
    def get_healthy_pairs(datasets):
        """Return set of (disease, donor_health_status) pairs."""
        pairs = set([])
        for dataset in datasets:
            disease = dataset.get("disease", "--empty--")
            donor_health_status = dataset.get("donor_health_status", "--empty--")
            pairs.add((disease, donor_health_status))
        return pairs

    def list_healthy_pairs(self, datasets):
        """List unique (disease, donor_health_status) pairs."""
        for pair in sorted(self.get_healthy_pairs(datasets)):
            print("{}\t{}".format(*pair))

    def read_healthy_pairs(self):
        """Return a (disease, donor_health_status):healthy dict defined in
        a tsv file with disease|donor_health_status|healthy columns.
        """
        healthy_dict = {}
        with open(self.pairs_file, "r") as tsv_file:
            next(tsv_file) # skip header
            for line in tsv_file:
                disease, donor_health_status, healthy = line.rstrip('\n').split('\t')
                healthy_dict[(disease, donor_health_status)] = healthy
        return healthy_dict

    def get_healthy_status(self, dataset):
        """Return "y", "n" or "?" depending of the healthy status of the dataset."""
        disease = dataset.get("disease", "--empty--")
        donor_health_status = dataset.get("donor_health_status", "--empty--")
        return self.healthy_dict[(disease, donor_health_status)]


def keep_major_cell_types(metadata):
    """Remove datasets which are not part of a cell_type which has
    at least 10 signals in two assays. Those assays must also have
    at least two cell_type.
    """
    # First pass to remove useless classes
    for category in ["assay", "cell_type"]:
        metadata.remove_small_classes(10, category)

    # Find big enough cell types in each assay
    md5s_per_assay = metadata.md5_per_class("assay")
    cell_types_in_assay = {}
    for assay, md5s in md5s_per_assay.items():

        # count cell_type occurence in assay
        cell_types_count = collections.Counter(
            metadata[md5]["cell_type"] for md5 in md5s
            )

        # remove small classes from counter
        cell_types_count = {
            cell_type:size for cell_type, size in cell_types_count.items()
            if size >= 10
            }

        # bad assay if only one class left with more than 10 examples
        if len(cell_types_count) == 1:
            cell_types_count = {}

        # delete small classes from metadata
        for md5 in md5s:
            if metadata[md5]["cell_type"] not in cell_types_count:
                del metadata[md5]

        # keep track of big enough classes
        if cell_types_count:
            cell_types_in_assay[assay] = set(cell_types_count.keys())

    # Count in how many assay each passing cell_type is
    cell_type_counter = collections.Counter()
    for cell_type_set in cell_types_in_assay.values():
        for cell_type in cell_type_set:
            cell_type_counter[cell_type] += 1

    # Remove signals which are not part of common+big cell_types
    for md5 in list(metadata.md5s):
        dset = metadata[md5]
        good_cell_type = cell_type_counter.get(dset["cell_type"], 0) > 1
        if not good_cell_type:
            del metadata[md5]

    return metadata


def keep_major_cell_types_alt(metadata):
    """Return a filtered metadata with certain assays. Datasets which are
    not part of a cell_type which has at least 10 signals are removed.
    """
    my_meta = copy.deepcopy(metadata)

    # remove useless assays and cell_types
    assays = [
        "chromatin_acc", "h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1",
        "h3k4me3", "h3k9me3", "input", "mrna_seq", "rna_seq", "wgb_seq"
        ]
    my_meta.select_category_subsets(assays, "assay")
    my_meta.remove_small_classes(10, "cell_type")
    my_meta.merge_fetal_tissues()

    new_meta = copy.deepcopy(my_meta)
    new_meta.empty()
    for assay in assays:
        temp_meta = copy.deepcopy(my_meta)
        temp_meta.select_category_subsets([assay], "assay")
        temp_meta.remove_small_classes(10, "cell_type")
        for md5 in temp_meta.md5s:
            new_meta[md5] = temp_meta[md5]

    return new_meta

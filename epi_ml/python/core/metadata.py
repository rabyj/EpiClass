"""Module from Metadata class and HealthyCategory."""
# pylint: disable=unnecessary-lambda-assignment
from __future__ import annotations

import collections
import copy
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import List


class Metadata(object):
    """
    Wrapper around metadata md5:dataset dict.

    path (Path): Path to json file containing metadata for some datasets.
    """

    def __init__(self, path: Path):
        self._metadata = self._load_metadata(path)
        self._rest = {}

    @classmethod
    def from_dict(cls, metadata: dict) -> Metadata:
        """Creates an object from a dict conforming to {md5sum:dset} format."""
        first_key = list(metadata.keys())[0]
        if len(first_key) != 32:
            raise Exception(
                f"Incorrect format of metadata. Key need to be md5sum (len=32). Is: {first_key}"
            )

        obj = cls.__new__(cls)
        obj._metadata = copy.deepcopy(metadata)
        return obj

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

    def update(self, info: Metadata) -> None:
        """Dict .update equivalent. Info needs to respect {md5sum:dset} format."""
        self._metadata.update(info.items)

    def save(self, path):
        """Save the metadata to path, in original epigeec_json format."""
        self._save_metadata(path)

    @property
    def md5s(self):
        """Return md5s (iterator). dict.keys() equivalent."""
        return self._metadata.keys()

    @property
    def datasets(self):
        """Return datasets (iterator). dict.values() equivalent."""
        return self._metadata.values()

    @property
    def items(self):
        """Return pairs (iterator). dict.items() equivalent"""
        return self._metadata.items()

    def _load_metadata(self, path):
        """Return md5:dataset dict."""
        with open(path, "r", encoding="utf-8") as file:
            meta_raw = json.load(file)

        self._rest = {k: v for k, v in meta_raw.items() if k != "datasets"}
        return {dset["md5sum"]: dset for dset in meta_raw["datasets"]}

    def _save_metadata(self, path):
        """Save the metadata to path, in original epigeec_json format.

        Only saves dataset information.
        """
        to_save = {"datasets": list(self.datasets)}
        to_save.update(self._rest)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(to_save, file)

    def apply_filter(self, meta_filter=lambda item: True):
        """Apply a filter on items (md5:dataset)."""
        self._metadata = dict(filter(meta_filter, self.items))

    def remove_missing_labels(self, label_category: str):
        """Remove datasets where the metadata category is missing."""
        filt = lambda item: label_category in item[1]
        self.apply_filter(filt)  # type: ignore

    def md5_per_class(self, label_category: str):
        """Return {label/class:md5 list} dict for a given metadata category.

        Can fail if remove_missing_labels has not been ran before.
        """
        sorted_md5 = sorted(self.md5s)
        data = collections.defaultdict(list)
        for md5 in sorted_md5:
            label = self[md5][label_category]
            data[label].append(md5)
        return data

    def remove_small_classes(self, min_class_size, label_category: str, verbose=True):
        """Remove classes with less than min_class_size examples
        for a given metatada category.

        Returns string of class ratio left if verbose.
        """
        data = self.md5_per_class(label_category)
        nb_class = len(data)

        nb_removed_class = 0
        for label, size in self.label_counter(label_category).most_common():
            if size < min_class_size:
                nb_removed_class += 1
                for md5 in data[label]:
                    del self[md5]

        if verbose:
            remaining = nb_class - nb_removed_class
            ratio = f"{remaining}/{nb_class}"
            print(
                f"{ratio} labels left from {label_category} "
                f"after removing classes with less than {min_class_size} signals."
            )

    def select_category_subsets(self, label_category: str, labels: Iterable[str]):
        """Select only datasets which possess the given labels
        for the given label category.
        """
        filt = lambda item: item[1].get(label_category) in set(labels)
        self.apply_filter(filt)  # type: ignore

    def remove_category_subsets(self, label_category: str, labels: Iterable[str]):
        """Remove datasets which possess the given labels
        for the given label category.
        """
        filt = lambda item: item[1].get(label_category) not in set(labels)
        self.apply_filter(filt)  # type: ignore

    def label_counter(self, label_category: str) -> collections.Counter[str]:
        """Return a Counter() with label count from the given category."""
        counter = collections.Counter()
        for dset in self.datasets:
            label = dset[label_category]
            counter.update([label])
        return counter

    def unique_classes(self, label_category: str) -> List[str]:
        """Return sorted list of unique classes currently existing for the given category."""
        sorted_md5 = sorted(self.md5s)
        uniq = set()
        for md5 in sorted_md5:
            uniq.add(self[md5][label_category])
        return sorted(list(uniq))

    def display_labels(self, label_category: str):
        """Print number of examples for each label in given category."""
        print("\nExamples")
        i = 0
        for label, count in self.label_counter(label_category).most_common():
            print(f"{label}: {count}")
            i += count
        print(f"For a total of {i} examples\n")

    def get_categories(self) -> list[str]:
        """Return a sorted list of all metadata categories."""
        categories = set()
        for dset in self.datasets:
            categories.update(dset.keys())
        return sorted(categories)

    def merge_classes(self, category: str, converter: dict):
        """Combine classes labels in the given category using the converter mapping."""
        for dataset in self.datasets:
            label = dataset.get(category, None)
            if label in converter:
                dataset[category] = converter[label]


def env_filtering(metadata: Metadata, category: str) -> List[str]:
    """Filter metadata using environment variables.

    Currently supports:
    EXCLUDE_LIST
    ASSAY_LIST
    LABEL_LIST
    """
    if os.getenv("EXCLUDE_LIST") is not None:
        exclude_list = json.loads(os.environ["EXCLUDE_LIST"])
        metadata.remove_category_subsets(label_category=category, labels=exclude_list)

    if os.getenv("ASSAY_LIST") is not None:
        assay_list = json.loads(os.environ["ASSAY_LIST"])
        print(f"Going to only keep examples with targets: {assay_list}")
        metadata.select_category_subsets("assay", assay_list)

    if os.getenv("LABEL_LIST") is not None:
        label_list = json.loads(os.environ["LABEL_LIST"])
        print(f"Going to only keep examples with labels: {label_list}")
        metadata.select_category_subsets(category, label_list)
    else:
        label_list = metadata.unique_classes(category)
        print("No label list")

    return label_list


class HealthyCategory(object):
    """Create/Represent/manipulate the "healthy" metadata category"""

    def __init__(self):
        self.pairs_file = Path(__file__).parent / "healthy_category.tsv"
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
        for x1, x2 in sorted(self.get_healthy_pairs(datasets)):
            print(f"{x1}\t{x2}")

    def read_healthy_pairs(self):
        """Return a (disease, donor_health_status):healthy dict defined in
        a tsv file with disease|donor_health_status|healthy columns.
        """
        healthy_dict = {}
        with open(self.pairs_file, "r", encoding="utf-8") as tsv_file:
            next(tsv_file)  # skip header
            for line in tsv_file:
                disease, donor_health_status, healthy = line.rstrip("\n").split("\t")
                healthy_dict[(disease, donor_health_status)] = healthy
        return healthy_dict

    def get_healthy_status(self, dataset):
        """Return "y", "n" or "?" depending of the healthy status of the dataset."""
        disease = dataset.get("disease", "--empty--")
        donor_health_status = dataset.get("donor_health_status", "--empty--")
        return self.healthy_dict[(disease, donor_health_status)]

    @staticmethod
    def create_healthy_category(metadata: Metadata):
        """Combine "disease" and "donor_health_status" to create a "healthy" category.

        When a dataset has pairs with unknow correspondance, it does not add
        the category, and so these datasets are ignored through remove_missing_labels().
        """
        healthy_category = HealthyCategory()
        for dataset in metadata.datasets:
            healthy = healthy_category.get_healthy_status(dataset)
            if healthy == "?":
                continue
            dataset["healthy"] = healthy

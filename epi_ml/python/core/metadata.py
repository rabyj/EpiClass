"""Module from Metadata class and HealthyCategory."""
from __future__ import annotations
import copy
import collections
from itertools import chain
import json
from pathlib import Path
from typing import List

class Metadata(object):
    """Wrapper around metadata md5:dataset dict."""
    def __init__(self, meta_file: Path):
        self._metadata = self._load_metadata(meta_file)

    @classmethod
    def from_dict(cls, metadata: dict) -> Metadata:
        """Creates an object from a dict conforming to {md5sum:dset} format."""
        first_key = list(metadata.keys())[0]
        if len(first_key) != 32:
            raise Exception(f"Incorrect format of metadata. Key need to be md5sum (len=32). Is: {first_key}")

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

    def _load_metadata(self, meta_file):
        """Return md5:dataset dict."""
        with open(meta_file, 'r', encoding="utf-8") as file:
            meta_raw = json.load(file)
        return {dset["md5sum"]:dset for dset in meta_raw["datasets"]}

    def apply_filter(self, meta_filter=lambda item: True):
        """Apply a filter on items (md5:dataset)."""
        self._metadata = dict(filter(meta_filter, self._metadata.items()))

    def remove_missing_labels(self, label_category: str):
        """Remove datasets where the metadata category is missing."""
        filt = lambda item: label_category in item[1]
        self.apply_filter(filt)

    def md5_per_class(self, label_category: str):
        """Return {label/class:md5 list} dict for a given metadata category.

        Can fail if remove_missing_labels has not been ran before.
        """
        sorted_md5 = sorted(self._metadata.keys())
        data = collections.defaultdict(list)
        for md5 in sorted_md5:
            data[self._metadata[md5][label_category]].append(md5)
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
                    del self._metadata[md5]

        if verbose:
            remaining = nb_class - nb_removed_class
            ratio = f"{remaining}/{nb_class}"
            print(
                f"{ratio} labels left from {label_category} "
                f"after removing classes with less than {min_class_size} signals."
            )

    def select_category_subsets(self, label_category: str, labels):
        """Select only datasets which possess the given labels
        for the given label category.
        """
        filt = lambda item: item[1].get(label_category) in set(labels)
        self.apply_filter(filt)

    def remove_category_subsets(self, label_category: str, labels):
        """Remove datasets which possess the given labels
        for the given label category.
        """
        filt = lambda item: item[1].get(label_category) not in set(labels)
        self.apply_filter(filt)

    def label_counter(self, label_category: str):
        """Return a Counter() with label count from the given category."""
        counter = collections.Counter()
        for labels in self._metadata.values():
            label = labels[label_category]
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
        print('\nExamples')
        i = 0
        for label, count in self.label_counter(label_category).most_common():
            print(f'{label}: {count}')
            i += count
        print(f"For a total of {i} examples\n")

    def get_categories(self):
        """Return a sorted list of all categories."""
        categories = set()
        for dset in self._metadata.values():
            categories.update(dset.keys())
        return sorted(categories)

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
        #TODO : No more specific merges, use a generic method w converter
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


def keep_major_cell_types(my_metadata: Metadata):
    """Remove datasets which are not part of a cell_type which has
    at least 10 signals in two assays. Those assays must also have
    at least two cell_type.
    """
    # First pass to remove useless classes
    for category in ["assay", "cell_type"]:
        my_metadata.remove_small_classes(10, category)

    # Find big enough cell types in each assay
    md5s_per_assay = my_metadata.md5_per_class("assay")
    cell_types_in_assay = {}
    for assay, md5s in md5s_per_assay.items():

        # count cell_type occurence in assay
        cell_types_count = collections.Counter(
            my_metadata[md5]["cell_type"] for md5 in md5s
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
            if my_metadata[md5]["cell_type"] not in cell_types_count:
                del my_metadata[md5]

        # keep track of big enough classes
        if cell_types_count:
            cell_types_in_assay[assay] = set(cell_types_count.keys())

    # Count in how many assay each passing cell_type is
    cell_type_counter = collections.Counter()
    for cell_type_set in cell_types_in_assay.values():
        for cell_type in cell_type_set:
            cell_type_counter[cell_type] += 1

    # Remove signals which are not part of common+big cell_types
    for md5 in list(my_metadata.md5s):
        dset = my_metadata[md5]
        good_cell_type = cell_type_counter.get(dset["cell_type"], 0) > 1
        if not good_cell_type:
            del my_metadata[md5]

    return my_metadata


def keep_major_cell_types_alt(my_metadata: Metadata):
    """Return a filtered metadata with certain assays. Datasets which are
    not part of a cell_type which has at least 10 signals are removed.
    """
    my_meta = copy.deepcopy(my_metadata)

    # remove useless assays and cell_types
    assays = [
        "chromatin_acc", "h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1",
        "h3k4me3", "h3k9me3", "input", "mrna_seq", "rna_seq", "wgb_seq"
        ]
    my_meta.select_category_subsets("assay", assays)
    my_meta.remove_small_classes(10, "cell_type")
    my_meta.merge_fetal_tissues()

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
        "monocyte", "cd4_positive_helper_t_cell", "macrophage",
        "skeletal_muscle_tissue", "thyroid"
        ]
    assays = [
        "chromatin_acc", "h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1",
        "h3k4me3", "h3k9me3", "input", "mrna_seq", "rna_seq", "wgb_seq"
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
    my_metadata.remove_category_subsets("assay", ["mrna_seq"],)

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
    my_metadata.remove_category_subsets("assay", ["smrna_seq"],)
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

    brain_labels = set(["brain_occipetal_lobe_right", "brain_right_temporal", "brain_temporal_lobe_left", "brain", "brain_frontal_lobe_left"])
    hepatocyte_labels = set(["hepatocytes", "hepatocyte"])
    large_intestin_colon_labels = set(["large_intestine_colon_ascending_(right)", "large_intestine_colon", "large_intestine_colon_rectosigmoid"])

    for dataset in my_metadata.datasets:
        cell_type = dataset.get("cell_type", None)
        if cell_type in brain_labels:
            dataset["cell_type"] = "brain"
        elif cell_type in hepatocyte_labels:
            dataset["cell_type"] = "hepatocytes"
        elif cell_type in large_intestin_colon_labels:
            dataset["cell_type"] = "large_intestine_colon"

    selected_cell_types = set(["myeloid_cell", "venous_blood", "monocyte", "thyroid", "mature_neutrophil", "macrophage", "b_cells", "cd14_positive,_cd16_negative_classical_monocyte", "normal_human_colon_absorptive_epithelial_cells", "cd4_positive,_alpha_beta_t_cell", "precursor_b_cell", "naive_b_cell", "stomach", "lymph_node", "muscle_of_leg", "neoplastic_plasma_cell", "plasma_cell", "brain", "hepatocytes", "large_intestine_colon"])
    my_metadata.select_category_subsets("cell_type", selected_cell_types,)

    return my_metadata

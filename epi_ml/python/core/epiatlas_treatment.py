"""Functions to split epiatlas datasets properly, keeping track types together in the different sets."""
import copy
import itertools

import numpy as np
from sklearn.model_selection import StratifiedKFold
from epi_ml.python.core import metadata
from epi_ml.python.core import data
from epi_ml.python.core.hdf5_loader import Hdf5Loader
from epi_ml.python.core.data_source import EpiDataSource


ASSAY_LIST = ["h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1", "h3k4me3", "h3k9me3", "input"]
LABEL_CATEGORY = "assay"
TEST_MD5S = set()

def get_complete_metadata(datasource: EpiDataSource, assay_list, label_category, verbose) -> metadata.Metadata:
    """Return metadata filtered for assay list and label_category."""
    meta = metadata.Metadata(datasource.metadata_file)
    meta.select_category_subsets(assay_list, label_category)
    meta.remove_small_classes(10, label_category, verbose)
    return meta


def _create_raw_dataset(datasource: EpiDataSource) -> data.DataSet:
    """Create a dataset with raw+ctl_raw signals, all in the training set."""
    print("Creating epiatlas raw signal training dataset")
    meta = get_complete_metadata(datasource, ASSAY_LIST, LABEL_CATEGORY, verbose=True)
    print("Theoretical maximum with complete dataset:")
    meta.display_labels(LABEL_CATEGORY)

    meta.select_category_subsets(["raw", "ctl_raw"], "track_type")
    print("Theoretical maximum with complete dataset:")
    meta.display_labels("track_type")

    print("Selected signals:")
    meta.display_labels("track_type")

    my_data = data.DataSetFactory.from_epidata(
        datasource, meta, LABEL_CATEGORY, min_class_size=10,
        validation_ratio=0, test_ratio=0,
        onehot=False, oversample=True
        )
    return my_data


def _epiatlas_prepare_split(datasource: EpiDataSource):
    """Return { raw_md5sum : {"pval":md5sum, "fc":md5sum} } dict assuming the datasource is complete."""
    complete_metatada = get_complete_metadata(datasource, ASSAY_LIST, LABEL_CATEGORY, verbose=False)

    uuid_to_md5s = {} #{ uuid : {track_type1:md5sum, track_type2:md5sum, ...} }
    for dset in complete_metatada.datasets:
        uuid = dset["uuid"]
        if uuid in uuid_to_md5s:
            uuid_to_md5s[uuid].update({dset["track_type"]:dset["md5sum"]})
        else:
            uuid_to_md5s[uuid] = {dset["track_type"]:dset["md5sum"]}

    raw_to_others = {} # { raw_md5sum : (pval_md5sum, fc_md5sum) }
    for val in uuid_to_md5s.values():
        raw_to_others[val["raw"]] = {"pval":val["pval"], "fc":val["fc"]}

    return raw_to_others, complete_metatada


def _load_other_tracks(datasource: EpiDataSource, raw_to_others: dict) -> dict:
    """Return Hdf5Loader.signals for md5s of other (fc, pval) signals"""
    hdf5_loader = Hdf5Loader(datasource.chromsize_file, normalization=True)
    md5s=itertools.chain.from_iterable([
            other_dict.values() for _, other_dict in raw_to_others.items()
            ])

    hdf5_loader.load_hdf5s(
        datasource.hdf5_file,
        md5s=md5s
        )
    return hdf5_loader.signals


def _add_other_tracks(selected_positions, other_signals: dict, raw_to_others: dict, dset: data.Data) -> data.Data:
    """Return a modified dset object with added tracks (pval + fc) for selected signals.

    The hdf5loader needs to contain at least the fc and pval tracks.
    """
    new_signals, new_str_labels, new_encoded_labels, new_md5s = [], [], [], []

    for selected_index in selected_positions:
        og_dset_metadata = dset.get_metadata(selected_index)
        md5 = dset.get_id(selected_index)
        label = dset.get_original_label(selected_index)
        encoded_label = dset.get_encoded_label(selected_index)
        signal = dset.get_signal(selected_index)

        if md5 != og_dset_metadata["md5sum"]:
            raise Exception("You dun fucked up")

        if og_dset_metadata["track_type"] == "raw":

            pval_md5 = raw_to_others[md5]["pval"]
            fc_md5 = raw_to_others[md5]["fc"]

            pval_signal = other_signals[pval_md5]
            fc_signal = other_signals[fc_md5]

            new_md5s.extend([md5, fc_md5, pval_md5])
            new_signals.extend([signal, fc_signal, pval_signal])
            new_encoded_labels.extend([encoded_label for _ in range(3)])
            new_str_labels.extend([label for _ in range(3)])

        elif og_dset_metadata["track_type"] == "ctl_raw":
            new_md5s.append(md5)
            new_signals.append(signal)
            new_encoded_labels.append(encoded_label)
            new_str_labels.append(label)
        else:
            raise Exception("You dun fucked up")

    new_dset = data.Data(new_md5s, new_signals, new_encoded_labels, new_str_labels, dset.metadata)

    return new_dset


def epiatlas_yield_split(datasource: EpiDataSource) -> data.DataSet:
    """Train dataset needs to be 100% of input+raw.
    Will not work with onehot encoding.

    Yield train and valid tensor datasets for one split
    """
    partial_data = _create_raw_dataset(datasource)

    raw_to_others, complete_metadata = _epiatlas_prepare_split(datasource)
    partial_data.train.metadata.update(complete_metadata)

    other_signals = _load_other_tracks(datasource, raw_to_others)

    new_datasets = data.DataSet.empty_collection()

    skf = StratifiedKFold(n_splits=10, shuffle=False)
    for train_idxs, valid_idxs in skf.split(
        np.zeros((partial_data.train.num_examples, len(partial_data.classes))),
        list(partial_data.train.encoded_labels)
        ):

        new_train = copy.deepcopy(partial_data.train)
        new_train = _add_other_tracks(train_idxs, other_signals, raw_to_others, new_train)

        new_valid = copy.deepcopy(partial_data.train)
        new_valid = _add_other_tracks(valid_idxs, other_signals, raw_to_others, new_valid)

        new_datasets.set_train(new_train)
        new_datasets.set_validation(new_valid)

        yield new_datasets

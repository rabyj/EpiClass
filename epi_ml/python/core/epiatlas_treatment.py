"""Functions to split epiatlas datasets properly, keeping track types together in the different sets."""
from __future__ import annotations
import copy
import itertools
from typing import List


import numpy as np
from sklearn.model_selection import StratifiedKFold
from epi_ml.python.core import metadata
from epi_ml.python.core import data
from epi_ml.python.core.hdf5_loader import Hdf5Loader
from epi_ml.python.core.data_source import EpiDataSource


class EpiAtlasTreatment(object):
    """Class that handles how epiatlas data is processed into datasets.

    datasource : where everything is read from
    label_category : The target category
    label_list : List of labels/classes to include from given category
    """
    def __init__(self, datasource: EpiDataSource, label_category: str, label_list: List[str]) -> None:
        self._datasource = datasource
        self._label_category = label_category
        self._label_list = label_list

        self._complete_metadata = self.get_complete_metadata(verbose=True)
        self._raw_dset = self._create_raw_dataset()
        self._raw_to_others = self._epiatlas_prepare_split()
        self._other_tracks = self._load_other_tracks()

    @property
    def datasource(self):
        """Return given datasource."""
        return self._datasource

    @property
    def target_category(self):
        """Return given label category (e.g. assay)"""
        return self._label_category

    @property
    def label_list(self):
        """Return given target labels inclusion list."""
        return self._label_list

    def get_complete_metadata(self, verbose: bool) -> metadata.Metadata:
        """Return metadata filtered for assay list and label_category."""
        meta = metadata.Metadata(self.datasource.metadata_file)
        meta.select_category_subsets(self.target_category, self.label_list)
        meta.remove_small_classes(10, self.target_category, verbose)
        return meta

    def _create_raw_dataset(self) -> data.DataSet:
        """Create a dataset with raw+ctl_raw signals, all in the training set."""
        print("Creating epiatlas raw signal training dataset")
        meta = copy.deepcopy(self._complete_metadata)
        print("Theoretical maximum with complete dataset:")
        meta.display_labels(self.target_category)

        meta.select_category_subsets("track_type", ["raw", "ctl_raw"])
        print("Theoretical maximum with complete dataset:")
        meta.display_labels("track_type")

        print("Selected signals:")
        meta.display_labels("track_type")

        my_data = data.DataSetFactory.from_epidata(
            self.datasource, meta, self.target_category, min_class_size=10,
            validation_ratio=0, test_ratio=0,
            onehot=False, oversample=False
            )
        return my_data

    def _epiatlas_prepare_split(self):
        """Return { raw_md5sum : {"pval":md5sum, "fc":md5sum} } dict assuming the datasource is complete."""
        meta = copy.deepcopy(self._complete_metadata)

        uuid_to_md5s = {} #{ uuid : {track_type1:md5sum, track_type2:md5sum, ...} }
        for dset in meta.datasets:
            uuid = dset["uuid"]
            if uuid in uuid_to_md5s:
                uuid_to_md5s[uuid].update({dset["track_type"]:dset["md5sum"]})
            else:
                uuid_to_md5s[uuid] = {dset["track_type"]:dset["md5sum"]}

        raw_to_others = {} # { raw_md5sum : (pval_md5sum, fc_md5sum) }
        for val in uuid_to_md5s.values():
            raw_to_others[val["raw"]] = {"pval":val["pval"], "fc":val["fc"]}

        return raw_to_others

    def _load_other_tracks(self) -> dict:
        """Return Hdf5Loader.signals for md5s of other (fc, pval) signals"""
        hdf5_loader = Hdf5Loader(self.datasource.chromsize_file, normalization=True)

        md5s=itertools.chain.from_iterable([
                other_dict.values() for _, other_dict in self._raw_to_others.items()
                ])

        hdf5_loader.load_hdf5s(
            self.datasource.hdf5_file,
            md5s=md5s
            )
        return hdf5_loader.signals

    def _add_other_tracks(self, selected_positions, dset: data.Data, resample: bool) -> data.Data:
        """Return a modified dset object with added tracks (pval + fc) for selected signals. """
        new_signals, new_str_labels, new_encoded_labels, new_md5s = [], [], [], []

        raw_dset = dset
        if resample:
            resampled_X, resampled_y, idxs = data.EpiData.oversample_data(dset.signals, dset.encoded_labels)
            raw_dset = data.Data(
                ids=np.take(dset.ids, idxs),
                x=resampled_X,
                y=resampled_y,
                y_str=np.take(dset.original_labels, idxs),
                metadata=dset.metadata
            )

        for selected_index in selected_positions:
            og_dset_metadata = raw_dset.get_metadata(selected_index)
            md5 = raw_dset.get_id(selected_index)
            label = raw_dset.get_original_label(selected_index)
            encoded_label = raw_dset.get_encoded_label(selected_index)
            signal = raw_dset.get_signal(selected_index)

            if md5 != og_dset_metadata["md5sum"]:
                raise Exception("You dun fucked up")

            if og_dset_metadata["track_type"] == "raw":

                pval_md5 = self._raw_to_others[md5]["pval"]
                fc_md5 = self._raw_to_others[md5]["fc"]

                pval_signal = self._other_tracks[pval_md5]
                fc_signal = self._other_tracks[fc_md5]

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


    def yield_split(self) -> data.DataSet:
        """Yield train and valid tensor datasets for one split.

        Depends on given init parameters.
        """
        new_datasets = data.DataSet.empty_collection()

        skf = StratifiedKFold(n_splits=10, shuffle=False)
        for train_idxs, valid_idxs in skf.split(
            np.zeros((self._raw_dset.train.num_examples, len(self._raw_dset.classes))),
            list(self._raw_dset.train.encoded_labels)
            ):

            new_train = copy.deepcopy(self._raw_dset.train)
            new_train = self._add_other_tracks(train_idxs, new_train, resample=True)

            new_valid = copy.deepcopy(self._raw_dset.train)
            new_valid = self._add_other_tracks(valid_idxs, new_valid, resample=False)

            new_datasets.set_train(new_train)
            new_datasets.set_validation(new_valid)

            yield new_datasets

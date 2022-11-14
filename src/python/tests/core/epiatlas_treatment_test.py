"""Functions to split epiatlas datasets properly, keeping track types together in the different sets."""
from __future__ import annotations

import collections
import copy
import itertools
from typing import Generator, List

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.python.core import data, metadata
from src.python.core.data_source import EpiDataSource
from src.python.core.epiatlas_treatment import LEADER_TRACKS, TRACKS_MAPPING
from src.python.core.hdf5_loader import Hdf5Loader


class EpiAtlasTreatment(object):
    """Class that handles how epiatlas data is processed into datasets.

    datasource : where everything is read from
    label_category : The target category
    label_list : List of labels/classes to include from given category
    """

    def __init__(
        self,
        datasource: EpiDataSource,
        label_category: str,
        label_list: List[str],
        n_fold: int = 10,
        test_ratio: float = 0,
        min_class_size: int = 10,
        meta=None,
    ) -> None:
        self._datasource = datasource
        self._label_category = label_category
        self._label_list = label_list
        self.k = n_fold

        if n_fold < 2:
            Exception(f"Need at least two folds for cross-validation. Got {n_fold}.")

        if meta is not None:
            self._complete_metadata = meta
        else:
            self._complete_metadata = self.get_complete_metadata(verbose=True)

        self._raw_to_others = self._epiatlas_prepare_split()
        # self._other_md5s = itertools.chain.from_iterable([
        #         other_dict.values() for _, other_dict in self._raw_to_others.items()
        #         ])

        # -- Load files --
        self._raw_dset = self._create_raw_dataset(test_ratio, min_class_size)
        self._other_tracks = self._load_other_tracks()

        self.classes = self._raw_dset.classes

    @property
    def datasource(self) -> EpiDataSource:
        """Return given datasource."""
        return self._datasource

    @property
    def target_category(self) -> str:
        """Return given label category (e.g. assay)"""
        return self._label_category

    @property
    def label_list(self) -> list:
        """Return given target labels inclusion list."""
        return self._label_list

    @property
    def raw_dataset(self) -> data.DataSet:
        """Return dataset of unmatched signals created during init."""
        return self._raw_dset

    @property
    def group_mapper(self) -> dict:
        """Return md5sum track_type mapping dict.

        e.g. 1 entry { raw_md5sum : {"pval":md5sum, "fc":md5sum} }
        """
        return self._raw_to_others

    def get_complete_metadata(self, verbose: bool) -> metadata.Metadata:
        """Return metadata filtered for assay list and label_category."""
        meta = metadata.Metadata(self.datasource.metadata_file)
        meta.select_category_subsets(self.target_category, self.label_list)
        meta.remove_small_classes(10, self.target_category, verbose)
        return meta

    def _create_raw_dataset(
        self, test_ratio: float, min_class_size: int
    ) -> data.DataSet:
        """Create a dataset with raw+ctl_raw signals, all in the training set."""
        print("Creating epiatlas 'raw' signal training dataset")
        meta = copy.deepcopy(self._complete_metadata)

        print("Theoretical maximum with complete dataset:")
        meta.display_labels(self.target_category)
        meta.display_labels("track_type")

        print("Selected signals in accordance with metadata:")
        meta.select_category_subsets("track_type", list(TRACKS_MAPPING.keys()))
        meta.display_labels("track_type")

        # important to not oversample now, because the train would bleed into valid during kfold.
        print("'Raw' dataset before oversampling and adding associated signals:")
        my_data = data.DataSetFactory.from_epidata(
            self.datasource,
            meta,
            self.target_category,
            min_class_size=min_class_size,
            validation_ratio=0,
            test_ratio=test_ratio,
            onehot=False,
            oversample=False,
        )
        meta.display_labels(self.target_category)

        return my_data

    def _epiatlas_prepare_split(self) -> dict:
        """Return track_type mapping dict assuming the datasource is complete.

        e.g. { raw_md5sum : {"pval":md5sum, "fc":md5sum} }
        """
        meta = copy.deepcopy(self._complete_metadata)

        uuid_to_md5s = {}  # { uuid : {track_type1:md5sum, track_type2:md5sum, ...} }
        for dset in meta.datasets:
            uuid = dset["uuid"]
            if uuid in uuid_to_md5s:
                uuid_to_md5s[uuid].update({dset["track_type"]: dset["md5sum"]})
            else:
                uuid_to_md5s[uuid] = {dset["track_type"]: dset["md5sum"]}

        raw_to_others = {}
        for val in uuid_to_md5s.values():
            for init in LEADER_TRACKS:
                if init in val:
                    others = TRACKS_MAPPING[init]
                    raw_to_others[val[init]] = {track: val[track] for track in others}

        return raw_to_others

    def _load_other_tracks(self) -> dict:
        """Return Hdf5Loader.signals for md5s of other (e.g. fc and pval) signals"""
        hdf5_loader = Hdf5Loader(self.datasource.chromsize_file, normalization=True)

        md5s = itertools.chain.from_iterable(
            [other_dict.values() for _, other_dict in self._raw_to_others.items()]
        )

        hdf5_loader.load_hdf5s(self.datasource.hdf5_file, md5s=md5s, verbose=False)
        return hdf5_loader.signals

    def _add_other_tracks(
        self, selected_positions, dset: data.Data, resample: bool
    ) -> data.Data:
        """Return a modified dset object with added tracks (pval + fc) for selected signals."""
        new_signals, new_str_labels, new_encoded_labels, new_md5s = [], [], [], []

        raw_dset = dset
        idxs = collections.Counter(i for i in np.arange(raw_dset.num_examples))
        if resample:
            resampled_X, resampled_y, idxs = data.EpiData.oversample_data(
                dset.signals, dset.encoded_labels
            )
            raw_dset = data.Data(
                ids=np.take(dset.ids, idxs),
                x=resampled_X,
                y=resampled_y,
                y_str=np.take(dset.original_labels, idxs),
                metadata=dset.metadata,
            )
            idxs = collections.Counter(i for i in idxs)

        for selected_index in selected_positions:
            og_dset_metadata = raw_dset.get_metadata(selected_index)
            chosen_md5 = raw_dset.get_id(selected_index)
            label = raw_dset.get_original_label(selected_index)
            encoded_label = raw_dset.get_encoded_label(selected_index)
            signal = raw_dset.get_signal(selected_index)

            track_type = og_dset_metadata["track_type"]

            if chosen_md5 != og_dset_metadata["md5sum"]:
                raise Exception("You dun fucked up")

            # oversampling specific to each "leader" signal
            for _ in range(idxs[selected_index]):

                if track_type in LEADER_TRACKS:

                    other_md5s = list(self._raw_to_others[chosen_md5].values())

                    other_signals = [self._other_tracks[md5] for md5 in other_md5s]

                    new_md5s.extend([chosen_md5] + other_md5s)
                    new_signals.extend([signal] + other_signals)
                    new_encoded_labels.extend(
                        [encoded_label for _ in range(len(other_md5s) + 1)]
                    )
                    new_str_labels.extend([label for _ in range(len(other_md5s) + 1)])

                elif track_type == "ctl_raw":
                    new_md5s.append(chosen_md5)
                    new_signals.append(signal)
                    new_encoded_labels.append(encoded_label)
                    new_str_labels.append(label)
                else:
                    raise Exception("You dun fucked up")

        new_dset = data.Data(
            new_md5s, new_signals, new_encoded_labels, new_str_labels, dset.metadata
        )

        return new_dset

    def yield_split(self) -> Generator[data.DataSet, None, None]:
        """Yield train and valid tensor datasets for one split.

        Depends on given init parameters.
        """
        new_datasets = data.DataSet.empty_collection()

        skf = StratifiedKFold(n_splits=self.k, shuffle=False)
        for train_idxs, valid_idxs in skf.split(
            np.zeros((self._raw_dset.train.num_examples, len(self.classes))),
            list(self._raw_dset.train.encoded_labels),
        ):

            new_train = copy.deepcopy(self._raw_dset.train)
            new_train = self._add_other_tracks(train_idxs, new_train, resample=True)

            new_valid = copy.deepcopy(self._raw_dset.train)
            new_valid = self._add_other_tracks(valid_idxs, new_valid, resample=False)

            new_datasets.set_train(new_train)
            new_datasets.set_validation(new_valid)

            yield new_datasets

    def create_total_data(self) -> data.Data:
        """Return a data set with all signals (no oversampling)"""
        return self._add_other_tracks(
            range(self._raw_dset.train.num_examples),
            self._raw_dset.train,
            resample=False,
        )

    def _find_other_tracks(
        self, selected_positions, dset: data.Data, resample: bool
    ) -> list:
        """Return indexes that sample from complete data, i.e. all signals with their match next to them."""
        raw_dset = dset
        idxs = collections.Counter(i for i in np.arange(raw_dset.num_examples))

        if resample:
            _, _, idxs = data.EpiData.oversample_data(
                np.zeros(shape=dset.signals.shape), dset.encoded_labels
            )
            repetitions = collections.Counter(i for i in idxs)
        else:
            repetitions = idxs

        new_selected_positions = []
        for selected_index in selected_positions:
            og_dset_metadata = raw_dset.get_metadata(selected_index)
            chosen_md5 = raw_dset.get_id(selected_index)
            track_type = og_dset_metadata["track_type"]

            if chosen_md5 != og_dset_metadata["md5sum"]:
                raise Exception("You dun fucked up")

            # oversampling specific to each "leader" signal
            rep = repetitions[selected_index]

            # number of matching signals (is it alone (ctl_raw), a pair, or a "fc,pval,raw" trio)
            other_nb = len(TRACKS_MAPPING[track_type])

            # add each group of indexes the required number of times (oversampling)
            if track_type in TRACKS_MAPPING:

                all_match_indexes = list(
                    range(selected_index, selected_index + other_nb + 1)
                )
                new_selected_positions.extend(all_match_indexes * rep)

            else:
                raise Exception("You dun fucked up")

        return new_selected_positions

    def split(self, X=None, y=None, groups=None):  # pylint: disable=unused-argument
        """Generate indices to split total data into training and validation set.

        Indexes match positions in output of create_total_data()
        X, y and groups :
            Always ignored, exist for compatibility.
        """
        skf = StratifiedKFold(n_splits=self.k, shuffle=False)
        for train_idxs, valid_idxs in skf.split(
            np.zeros((self._raw_dset.train.num_examples, len(self.classes))),
            list(self._raw_dset.train.encoded_labels),
        ):

            # The "complete" refers to the fact that the indexes are sampling over total data.
            complete_train_idxs = self._find_other_tracks(
                train_idxs, self._raw_dset.train, resample=True
            )

            complete_valid_idxs = self._find_other_tracks(
                valid_idxs, self._raw_dset.train, resample=False
            )

            yield complete_train_idxs, complete_valid_idxs

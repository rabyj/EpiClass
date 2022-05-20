import collections
import math
from pathlib import Path
import random
from typing import List

import h5py
import numpy as np

from .data_source import EpiDataSource
from .metadata import Metadata


class DataSetFactory(object):
    """Creation of DataSet from different sources."""
    @classmethod
    def from_epidata(cls, datasource: EpiDataSource, metadata: Metadata, label_category: str, oversample=False,
                     normalization=True, min_class_size=3, validation_ratio=0.1, test_ratio=0.1):
        """TODO : Write docstring"""

        return EpiData(
            datasource, metadata, label_category, oversample, normalization, min_class_size, validation_ratio, test_ratio
            ).dataset


class EpiData(object):
    """Used to load and preprocess epigenomic data. Data factory.

    Test ratio computed from validation ratio and test ratio. Be sure to set both correctly.
    """
    def __init__(self, datasource: EpiDataSource, metadata: Metadata, label_category: str, oversample=False,
                 normalization=True, min_class_size=3, validation_ratio=0.1, test_ratio=0.1):
        EpiData._assert_ratios(validation_ratio, test_ratio, verbose=True)
        self._label_category = label_category
        self._oversample = oversample

        #load
        self._hdf5s = Hdf5Loader(datasource.chromsize_file, datasource.hdf5_file, normalization).hdf5s
        self._metadata = self._load_metadata(metadata)

        #preprocess
        self._keep_meta_overlap()
        self._metadata.remove_small_classes(min_class_size, self._label_category)

        self._sorted_classes = self._metadata.unique_classes(label_category)
        self._onehot_dict = EpiData._create_onehot_dict(self._sorted_classes)

        self._split_data(validation_ratio, test_ratio)

    @property
    def dataset(self):
        """TODO : Write docstring"""
        return DataSet(self._train, self._validation, self._test, self._sorted_classes)

    @staticmethod
    def _assert_ratios(val_ratio, test_ratio, verbose):
        """Verify that splitting ratios make sense."""
        if val_ratio + test_ratio > 1:
            raise ValueError(
                f"Validation and test ratios are bigger than 100%: {val_ratio} and {test_ratio}"
                )
        elif verbose:
            print(
            f"training/validation/test split: {(1-val_ratio-test_ratio)*100}%/{val_ratio*100}%/{test_ratio*100}%"
            )

    def _load_metadata(self, metadata: Metadata) -> Metadata:
        metadata.remove_missing_labels(self._label_category)
        return metadata

    def _keep_meta_overlap(self):
        self._remove_md5_without_hdf5()
        self._remove_hdf5_without_md5()

    def _remove_md5_without_hdf5(self):
        self._metadata.apply_filter(lambda item: item[0] in self._hdf5s)

    def _remove_hdf5_without_md5(self):
        self._hdf5s = {md5:self._hdf5s[md5] for md5 in self._metadata.md5s}

    def _oversample_rates(self):
        """TODO : Write docstring"""
        sorted_md5 = sorted(self._metadata.md5s)
        label_count = {}
        for md5 in sorted_md5:
            label = self._metadata[md5][self._label_category]
            label_count[label] = label_count.get(label, 0) + 1

        max_count = 0
        max_label = ""
        for label, count in label_count.items():
            if count > max_count:
                max_count = count
                max_label = label

        oversample_rates = {}
        for label, count in label_count.items():
            oversample_rates[label] = count/max_count
        return oversample_rates

    def _split_data(self, validation_ratio, test_ratio):
        """TODO : Write docstring"""
        size_all_dict = self._metadata.label_counter(self._label_category)

        # A minimum of 3 examples are needed for each label (1 for each set), when splitting into three sets
        for label, size in size_all_dict.items():
            if size < 3:
                print(f"The label `{label}` countains only {size} datasets.")

        size_validation_dict = collections.Counter({label:math.ceil(size*validation_ratio) for label, size in size_all_dict.items()})
        size_test_dict = collections.Counter({label:math.ceil(size*test_ratio) for label, size in size_all_dict.items()})
        split_index_dict = size_validation_dict + size_test_dict

        data = self._metadata.md5_per_class(self._label_category)

        slice_data = lambda begin={}, end={}: sum([
            data[label][begin.get(label, 0):end.get(label, None)]
            for label in size_all_dict.keys()
        ], [])

        validation_md5s = slice_data(end=size_validation_dict)
        test_md5s = slice_data(begin=size_validation_dict, end=split_index_dict)
        train_md5s = slice_data(begin=split_index_dict)

        assert len(self._metadata.md5s) == len(set(sum([validation_md5s, test_md5s, train_md5s], [])))

        # separate hdf5 files
        validation_signals = [self._hdf5s[md5] for md5 in validation_md5s]
        test_signals = [self._hdf5s[md5] for md5 in test_md5s]
        train_signals = [self._hdf5s[md5] for md5 in train_md5s]

        # separate label values
        validation_labels = [self._metadata[md5][self._label_category] for md5 in validation_md5s]
        test_labels = [self._metadata[md5][self._label_category] for md5 in test_md5s]
        train_labels = [self._metadata[md5][self._label_category] for md5 in train_md5s]

        if self._oversample:
            train_signals, train_labels, train_md5s = self._oversample_data(train_signals, train_labels, train_md5s)

        self._to_onehot(validation_labels)
        self._to_onehot(test_labels)
        self._to_onehot(train_labels)

        self._validation = Data(validation_md5s, validation_signals, validation_labels, self._metadata)
        self._test = Data(test_md5s, test_signals, test_labels, self._metadata)
        self._train = Data(train_md5s, train_signals, train_labels, self._metadata)

        print(f"validation size {len(validation_labels)}")
        print(f"test size {len(test_labels)}")
        print(f"training size {len(train_labels)}")

    def _oversample_data(self, signals, labels, md5s):
        oversample_rates = self._oversample_rates()
        new_signals = []
        new_labels = []
        new_md5s = []
        for i, (signal, label) in enumerate(zip(signals, labels)):
            rate = oversample_rates[label]
            sample_rate = int(1/rate)
            if random.random() < (1/rate) % 1:
                sample_rate += 1
            for _ in range(sample_rate):
                new_signals.append(signal)
                new_labels.append(label)
                new_md5s.append(md5s[i])
        return new_signals, new_labels, new_md5s

    @staticmethod
    def _create_onehot_dict(classes):
        """Returns {label:onehot vector} dict corresponding given classes.

        Onehot vectors defined with given classes, no sorting done.
        """
        onehot_dict = {}
        for i, label in enumerate(classes):
            onehot = np.zeros(len(classes))
            onehot[i] = 1
            onehot_dict[label] = onehot
        return onehot_dict

    def _to_onehot(self, labels):
        """Transform labels into onehot vectors list."""
        for i, val in enumerate(labels):
            labels[i] = self._onehot_dict[val]


class Data(object): #class DataSet?
    """Generalised object to deal with data."""
    def __init__(self, ids, x, y, metadata: Metadata):
        self._ids = ids
        self._num_examples = len(x)
        self._signals = np.array(x)
        self._labels = np.array(y)
        self._shuffle_order = np.arange(self._num_examples)
        self._index = 0
        self._metadata = metadata

    def preprocess(self, f):
        """TODO : Write docstring"""
        self._signals = np.apply_along_axis(f, 1, self._signals)

    def next_batch(self, batch_size, shuffle=True):
        """TODO : Write docstring"""
        #if index exceeded num examples, start over
        if self._index >= self._num_examples:
            self._index = 0
        if self._index == 0:
            if shuffle:
                self._shuffle()
        start = self._index
        self._index += batch_size
        end = self._index
        return self._signals[start:end], self._labels[start:end]

    def _shuffle(self):
        """TODO : Write docstring"""
        rng_state = np.random.get_state()
        for array in [self._shuffle_order, self._signals, self._labels]:
            np.random.shuffle(array)
            np.random.set_state(rng_state)

    def get_metadata(self, index):
        """Get the metadata from the signal at the given position in the set."""
        return self._metadata.get(self._ids[index])

    @property
    def ids(self):
        """TODO : Write docstring"""
        return self._ids

    @property
    def signals(self):
        """TODO : Write docstring"""
        return self._signals

    @property
    def labels(self):
        """TODO : Write docstring"""
        return self._labels

    @property
    def num_examples(self):
        """TODO : Write docstring"""
        return self._num_examples


class DataSet(object): #class Data?
    """Contains training/valid/test Data objects."""
    def __init__(self, training: Data, validation: Data, test: Data, sorted_classes):
        self._train = training
        self._validation = validation
        self._test = test
        self._sorted_classes = sorted_classes

    @property
    def train(self) -> Data:
        """Training set"""
        return self._train

    @property
    def validation(self) -> Data:
        """Validation set"""
        return self._validation

    @property
    def test(self) -> Data:
        """Test set"""
        return self._test

    @property
    def classes(self) -> List[str]:
        """Return sorted classes present through datasets"""
        return self._sorted_classes

    def preprocess(self, f):
        """TODO : Write docstring"""
        if self._train.num_examples:
            self._train.preprocess(f)
        if self._validation.num_examples:
            self._validation.preprocess(f)
        if self._test.num_examples:
            self._test.preprocess(f)

    def save_mapping(self, path):
        """Write the 'output position --> label' mapping to path."""
        with open(path, 'w', encoding="utf-8") as map_file:
            for i, label in enumerate(self._sorted_classes):
                map_file.write(f"{i}\t{label}\n")

    def load_mapping(self, path):
        """Return dict object representation 'output position --> label' mapping from path."""
        with open(path, 'r', encoding="utf-8") as map_file:
            mapping = {}
            for line in map_file:
                i, label = line.rstrip().split('\t')
                mapping[int(i)] = label
        return mapping


class Hdf5Loader(object):
    """TODO : Write docstring"""
    def __init__(self, chrom_file, data_file, normalization: bool):
        self._normalization = normalization
        self._chroms = self._load_chroms(chrom_file)
        self._hdf5s = self._load_hdf5s(data_file)

    @property
    def hdf5s(self):
        """Return a md5:norm_concat_chroms dict."""
        return self._hdf5s

    def _load_chroms(self, chrom_file):
        """Return sorted chromosome names list."""
        with open(chrom_file, 'r', encoding="utf-8") as file:
            chroms = []
            for line in file:
                line = line.rstrip()
                if line:
                    chroms.append(line.split()[0])
            chroms.sort()
            return chroms

    def _load_hdf5s(self, data_file):
        """TODO : Write docstring"""
        with open(data_file, 'r', encoding="utf-8") as file_of_paths:
            hdf5s = {}
            for path in file_of_paths:
                path = Path(path.rstrip())
                md5 = self._extract_md5(path)
                datasets = []
                for chrom in self._chroms:
                    f = h5py.File(path)
                    array = f[md5][chrom][...]
                    datasets.append(array)
                hdf5s[md5] = self._normalize(np.concatenate(datasets))
        return hdf5s

    def _normalize(self, array):
        if self._normalization:
            return (array - array.mean()) / array.std()
        else:
            return array

    def _extract_md5(self, file_name: Path):
        return file_name.name.split("_")[0]

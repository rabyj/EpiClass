import collections
import math
from pathlib import Path
import random
from typing import List, Dict

import h5py
import numpy as np
from sklearn import preprocessing

from .data_source import EpiDataSource
from .metadata import Metadata


class DataSetFactory(object):
    """Creation of DataSet from different sources."""
    @classmethod
    def from_epidata(cls, datasource: EpiDataSource, metadata: Metadata, label_category: str, onehot=False, oversample=False,
                     normalization=True, min_class_size=3, validation_ratio=0.1, test_ratio=0.1):
        """TODO : Write docstring"""

        return EpiData(
            datasource, metadata, label_category, onehot, oversample, normalization, min_class_size,
            validation_ratio, test_ratio
            ).dataset


class EpiData(object):
    """Used to load and preprocess epigenomic data. Data factory.

    Test ratio computed from validation ratio and test ratio. Be sure to set both correctly.
    """
    def __init__(self, datasource: EpiDataSource, metadata: Metadata, label_category: str, onehot=False, oversample=False,
                 normalization=True, min_class_size=3, validation_ratio=0.1, test_ratio=0.1):
        EpiData._assert_ratios(validation_ratio, test_ratio, verbose=True)
        self._label_category = label_category
        self._oversample = oversample

        #load
        self._metadata = self._load_metadata(metadata)
        self._files = Hdf5Loader.read_list(datasource.hdf5_file)

        #preprocess
        self._keep_meta_overlap()
        self._metadata.remove_small_classes(min_class_size, self._label_category)

        self._hdf5s = Hdf5Loader(
            datasource.chromsize_file, normalization
            ).load_hdf5s(datasource.hdf5_file, md5s=self._files.keys()).signals

        self._sorted_classes = self._metadata.unique_classes(label_category)

        # TODO : Create encoder class separate from EpiData
        encoder = EpiData._make_encoder(self._sorted_classes, onehot=onehot)

        self._split_data(validation_ratio, test_ratio, encoder)

    @property
    def dataset(self):
        """Return DataSet object of processed data/metadata."""
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
        self._metadata.apply_filter(lambda item: item[0] in self._files)

    def _remove_hdf5_without_md5(self):
        self._files = {md5:self._files[md5] for md5 in self._metadata.md5s}

    @staticmethod
    def _create_onehot_dict(classes):
        """Returns {label:onehot vector} dict corresponding given classes.
        TODO : put into an encoder class
        Onehot vectors defined with given classes, no sorting done.
        """
        onehot_dict = {}
        for i, label in enumerate(classes):
            onehot = np.zeros(len(classes))
            onehot[i] = 1
            onehot_dict[label] = onehot
        return onehot_dict

    @staticmethod
    def _make_encoder(classes, onehot=False):
        """Return an int (default) or onehot vector encoder that takes label sets as entry.
        TODO : put into an encoder class
        Classes are sorted beforehand.
        """
        labels = sorted(classes)
        if onehot:
            encoding = EpiData._create_onehot_dict(labels)
            def to_onehot(labels):
                return [encoding[label] for label in labels]
            return to_onehot
        else:
            encoding = preprocessing.LabelEncoder().fit(labels) #int mapping
            def to_int(labels):
                if labels:
                    return encoding.transform(labels)
                else:
                    return []
            return to_int

    def _oversample_rates(self):
        """Return a {label:oversampling_rate} dict.

        The oversampling rates are label_count/max(label_counts).
        """
        sorted_md5 = sorted(self._metadata.md5s)
        label_count = {}
        for md5 in sorted_md5:
            label = self._metadata[md5][self._label_category]
            label_count[label] = label_count.get(label, 0) + 1

        max_count = 0
        for label, count in label_count.items():
            if count > max_count:
                max_count = count

        oversample_rates = {}
        for label, count in label_count.items():
            oversample_rates[label] = count/max_count
        return oversample_rates

    def _split_data(self, validation_ratio, test_ratio, encoder):
        """Split loaded data into three sets : Training/Validation/Test.

        The encoder/encoding function for a label list needs to be provided.
        """
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

        train_md5s = slice_data(begin=split_index_dict)
        validation_md5s = slice_data(end=size_validation_dict)
        test_md5s = slice_data(begin=size_validation_dict, end=split_index_dict)

        assert len(self._metadata.md5s) == len(set(sum([validation_md5s, test_md5s, train_md5s], [])))

        # separate hdf5 files
        train_signals = [self._hdf5s[md5] for md5 in train_md5s]
        validation_signals = [self._hdf5s[md5] for md5 in validation_md5s]
        test_signals = [self._hdf5s[md5] for md5 in test_md5s]

        # separate label values
        train_labels = [self._metadata[md5][self._label_category] for md5 in train_md5s]
        validation_labels = [self._metadata[md5][self._label_category] for md5 in validation_md5s]
        test_labels = [self._metadata[md5][self._label_category] for md5 in test_md5s]

        if self._oversample:
            train_signals, train_labels, train_md5s = self._oversample_data(train_signals, train_labels, train_md5s)

        encoded_labels = [encoder(labels) for labels in [train_labels, validation_labels, test_labels]]

        self._train = Data(train_md5s, train_signals, encoded_labels[0], train_labels, self._metadata)
        self._validation = Data(validation_md5s, validation_signals, encoded_labels[1], validation_labels, self._metadata)
        self._test = Data(test_md5s, test_signals, encoded_labels[2], test_labels, self._metadata)

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


class Data(object): #class DataSet?
    """Generalised object to deal with data."""
    def __init__(self, ids, x, y, y_str, metadata: Metadata):
        self._ids = ids
        self._num_examples = len(x)
        self._signals = np.array(x)
        self._labels = np.array(y)
        self._labels_str = y_str
        self._shuffle_order = np.arange(self._num_examples) # To be able to find back ids correctly
        self._index = 0
        self._metadata = metadata

    def preprocess(self, f):
        """Apply a preprocessing function on signals."""
        self._signals = np.apply_along_axis(f, 1, self._signals)

    def next_batch(self, batch_size, shuffle=True):
        """Return next (signals, targets) batch"""
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
        """Shuffle signals and labels together"""
        rng_state = np.random.get_state()
        for array in [self._shuffle_order, self._signals, self._labels]:
            np.random.shuffle(array)
            np.random.set_state(rng_state)

    def get_metadata(self, index):
        """Get the metadata from the signal at the given position in the set."""
        return self._metadata.get(self._ids[index])

    @property
    def ids(self):
        """Return md5s in current signals order."""
        return np.take(self._ids, list(self._shuffle_order))

    @property
    def signals(self):
        """Return signals in current order."""
        return self._signals

    @property
    def encoded_labels(self):
        """Return encoded labels of examples in current signal order."""
        return self._labels

    @property
    def original_labels(self):
        """Return string labels of examples in current signal order."""
        return np.take(self._labels_str, list(self._shuffle_order))

    @property
    def num_examples(self):
        """Return the number of examples contained in the set."""
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

    def get_encoder(self, mapping, using_file=False) -> preprocessing.LabelEncoder:
        """Load and return int label encoder.

        Requires the model mapping file itself, or its path (with using_file=True)
        """
        if using_file:
            mapping = self.load_mapping(mapping)

        labels = sorted(list(mapping.values()))
        return preprocessing.LabelEncoder().fit(labels)


class Hdf5Loader(object):
    """Handles loading/creating signals from hdf5 files"""
    def __init__(self, chrom_file, normalization: bool):
        self._normalization = normalization
        self._chroms = self._load_chroms(chrom_file)
        self._files = None
        self._signals = None

    @property
    def loaded_files(self) -> Dict[str, Path]:
        """Return a {md5:path} dict with last loaded files."""
        return self._files

    @property
    def signals(self) -> Dict[str, np.ndarray]:
        """Return a {md5:signal dict} with the last loaded signals,
        where the signal has concanenated chromosomes, and is normalized if set so.
        """
        return self._signals

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

    @staticmethod
    def read_list(data_file:Path) -> Dict[str, Path]:
        """Return {md5:file} dict from file of paths list."""
        with open(data_file, 'r', encoding="utf-8") as file_of_paths:
            files = {}
            for path in file_of_paths:
                path = Path(path.rstrip())
                files[Hdf5Loader.extract_md5(path)] = path
        return files

    def load_hdf5s(self, data_file: Path, md5s=None, verbose=True):
        """Load hdf5s from path list file.

        If a list of md5s is given, load only the corresponding files.
        Normalize if internal flag set so."""
        files = self.read_list(data_file)

        #Remove undesired files
        if md5s is not None:
            md5s = set(md5s)
            files = {
                md5:path for md5,path in files.items()
                if md5 in md5s
            }
        self._files = files

        #Load hdf5s and concatenate chroms into signals
        signals = {}
        for md5, file in files.items():
            f = h5py.File(file)
            chrom_signals = []
            for chrom in self._chroms:
                array = f[md5][chrom][...]
                chrom_signals.append(array)
            signals[md5] = self._normalize(np.concatenate(chrom_signals))

        self._signals = signals

        absent_md5s = md5s - set(files.keys())
        if absent_md5s and verbose:
            print("Following given md5s are absent of hdf5 list")
            for md5 in absent_md5s:
                print(md5)

        return self


    def _normalize(self, array):
        if self._normalization:
            return (array - array.mean()) / array.std()
        else:
            return array

    @staticmethod
    def extract_md5(file_name: Path):
        """Extract the md5 string from file path with specific naming convention."""
        return file_name.name.split("_")[0]

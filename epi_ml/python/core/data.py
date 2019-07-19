import h5py
import os.path
import numpy as np
from scipy import signal
import random
import collections
import math
import io

from .data_source import EpiDataSource
from .metadata import Metadata


class DataSetFactory(object):
    """Creation of DataSet from different sources."""
    @classmethod
    def from_epidata(cls, datasource: EpiDataSource, metadata: Metadata, label_category: str, oversample=False,
                 normalization=True, min_class_size=3):
        
        return EpiData(datasource, metadata, label_category, oversample, normalization, min_class_size).dataset


class EpiData(object):
    """Used to load and preprocess epigenomic data. Data factory."""
    def __init__(self, datasource: EpiDataSource, metadata: Metadata, label_category: str, oversample=False,
                 normalization=True, min_class_size=3):
        self._label_category = label_category
        self._oversample = oversample
        self._normalization = normalization

        #load
        self._load_chrom_sizes(datasource.chromsize_file)
        self._hdf5s = self._load_hdf5(datasource.hdf5_file)
        self._metadata = self._load_metadata(metadata)

        #preprocess
        self._keep_meta_overlap()
        self._metadata.remove_small_classes(min_class_size, self._label_category)
        self._split_data()

    @property
    def dataset(self):
        return DataSet(self._train, self._validation, self._test, self._sorted_classes)

    def _load_metadata(self, metadata):
        metadata.remove_missing_labels(self._label_category)
        return metadata

    def _keep_meta_overlap(self):
        self._remove_md5_without_hdf5()
        self._remove_hdf5_without_md5()

    def _remove_md5_without_hdf5(self):
       self._metadata.apply_filter(lambda item: item[0] in self._hdf5s)

    def _remove_hdf5_without_md5(self):
        self._hdf5s = {md5:self._hdf5s[md5] for md5 in self._metadata.md5s}

    def _load_chrom_sizes(self, chrom_file: io.IOBase):
        chrom_file.seek(0)
        self._chroms = []

        for line in chrom_file:
            line = line.strip()
            if line:
                line = line.split()
                self._chroms.append(line[0])
        self._chroms.sort()

    def _load_hdf5(self, data_file: io.IOBase):
        data_file.seek(0)
        hdf5s = {}
        for file_path in [line.strip() for line in data_file]:
            md5 = self._extract_md5(file_path)
            datasets = []
            for chrom in self._chroms:
                f = h5py.File(file_path)
                array = f[md5][chrom][...]
                datasets.append(array)
            hdf5s[md5] = self._normalize(np.concatenate(datasets))
        return hdf5s
    
    def _normalize(self, array):
        if self._normalization:
            return (array - array.mean()) / array.std()
        else:
            return array

    def _extract_md5(self, file_name):
        return os.path.basename(file_name).split("_")[0]

    def _oversample_rates(self):
        """"""
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
        
    def _split_data(self, validation_ratio=0.1, test_ratio=0.1):
        """"""
        size_all_dict = self._metadata.label_counter(self._label_category)

        # A minimum of 3 examples are needed for each label (1 for each set)
        for label, size in size_all_dict.items():
            if size < 3:
                print('The label `{}` countains only {} datasets.'.format(label, size))

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
            train_signals, train_labels = self._oversample_data(train_signals, train_labels)

        self._to_onehot(validation_labels)
        self._to_onehot(test_labels)
        self._to_onehot(train_labels)

        self._validation = Data(validation_md5s, validation_signals, validation_labels, self._metadata)
        self._test = Data(test_md5s, test_signals, test_labels, self._metadata)
        self._train = Data(train_md5s, train_signals, train_labels, self._metadata)

        print('validation size {}'.format(len(validation_labels)))
        print('test size {}'.format(len(test_labels)))
        print('training size {}'.format(len(train_labels)))

    def _oversample_data(self, signals, labels):
        oversample_rates = self._oversample_rates()
        new_signals = []
        new_labels = []
        for signal, label in zip(signals, labels):
            rate = oversample_rates[label]
            sample_rate = int(1/rate)
            if random.random() < (1/rate) % 1:
                sample_rate += 1
            for i in range(sample_rate):
                new_signals.append(signal)
                new_labels.append(label)
        return new_signals, new_labels

    def _to_onehot(self, labels):
        sorted_md5 = sorted(self._metadata.md5s)
        uniq = set()
        for md5 in sorted_md5:
            uniq.add(self._metadata[md5][self._label_category])
        self._sorted_classes = sorted(list(uniq))
        onehot_dict = {}
        for i in range(len(self._sorted_classes)):
            onehot = np.zeros(len(self._sorted_classes))
            onehot[i] = 1
            onehot_dict[self._sorted_classes[i]] = onehot
        for i in range(len(labels)):
            labels[i] = onehot_dict[labels[i]]


class Data(object): #class DataSet?
    """Generalised object to deal with data."""
    def __init__(self, ids, x, y , metadata: Metadata):
        self._ids = ids
        self._num_examples = len(x)
        self._signals = np.array(x)
        self._labels = np.array(y)
        self._shuffled_signals = None
        self._shuffled_labels = None
        self._index = 0
        self._metadata = metadata
        
    def preprocess(self, f):
        self._signals = np.apply_along_axis(f, 1, self._signals)
    
    def next_batch(self, batch_size, shuffle=True):
        #if index exceeded num examples, start over
        if self._index >= self._num_examples:
            self._index = 0
        if self._index == 0:
            if shuffle:
                self._shuffle()
        start = self._index
        self._index += batch_size
        end = self._index
        return self._shuffled_signals[start:end], self._shuffled_labels[start:end]

    def _shuffle(self):
        shuffle = np.arange(self._num_examples)
        np.random.shuffle(shuffle)
        self._shuffled_signals = self._signals[shuffle]
        self._shuffled_labels = self._labels[shuffle]

    def get_metadata(self, index):
        return self._metadata.get(self._ids[index])

    @property
    def ids(self):
        return self._ids

    @property
    def signals(self):
        return self._signals

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


class DataSet(object): #class Data?
    """Contains training/valid/test Data objects."""
    def __init__(self, training: Data, validation: Data, test: Data, sorted_classes):
        self._train = training
        self._validation = validation
        self._test = test
        self._sorted_classes = sorted_classes

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test

    @property
    def classes(self):
        return self._sorted_classes

    def preprocess(self, f):
        self._train.preprocess(f)
        self._validation.preprocess(f)
        self._test.preprocess(f)


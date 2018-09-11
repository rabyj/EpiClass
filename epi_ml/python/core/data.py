import glob
import h5py
import json
import os.path
import numpy as np
from scipy import signal
import random
import collections
import math

import io

class EpiDataSource(object):
    def __init__(self, hdf5: io.IOBase, chromsize: io.IOBase, metadata: io.IOBase):
        self._hdf5 = hdf5
        self._chromsize = chromsize
        self._metadata = metadata

    @property
    def hdf5_file(self) -> io.IOBase:
        return self._hdf5

    @property
    def chromsize_file(self) -> io.IOBase:
        return self._chromsize

    @property
    def metadata_file(self) -> io.IOBase:
        return self._metadata

class EpiData(object):
    def __init__(self, datasource: EpiDataSource, label_category: str, oversample=False, normalization=True, onehot=True):
        self._label_category = label_category
        self._oversample = oversample
        self._normalization = normalization
        self._onehot = onehot
        self._load_chrom_sizes(datasource.chromsize_file)
        self._load_hdf5(datasource.hdf5_file)
        self._load_metadata(datasource.metadata_file)
        self._build_data()

    def _load_metadata(self, meta_file: io.IOBase):
        meta_file.seek(0)
        meta_raw = json.load(meta_file)
        self._metadata = {}
        for dataset in meta_raw["datasets"]:
            if dataset["md5sum"] in self._hdf5s:
                self._metadata[dataset["md5sum"]] = dataset

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
        self._hdf5s = {}
        for file_path in [line.strip() for line in data_file]:
            md5 = self._extract_md5(file_path)
            datasets = []
            for chrom in self._chroms:
                f = h5py.File(file_path)
                array = f[md5][chrom][...]
                datasets.append(array)
            self._hdf5s[md5] = self._normalize(np.concatenate(datasets))
    
    def _normalize(self, array):
        if self._normalization:
            return (array - array.mean()) / array.std()
        else:
            return array

    def _extract_md5(self, file_name):
        return os.path.basename(file_name).split("_")[0]

    def _oversample_rates(self):
        sorted_md5 = sorted(self._metadata.keys())
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

    def _build_data(self, validation_ratio=0.1, test_ratio=0.1):
        sorted_md5 = sorted(self._metadata.keys())
        data = collections.defaultdict(list)
        for md5 in sorted_md5:
            data[self._metadata[md5][self._label_category]].append(md5)

        size_all_dict = collections.Counter({label:len(data[label]) for label in data.keys()})
        for label, size in size_all_dict.items():
            if size < 3:
                print('The label `{}` countains only {} datasets.'.format(label, size))

        size_validation_dict = collections.Counter({label:math.ceil(size*validation_ratio) for label, size in size_all_dict.items()})
        size_test_dict = collections.Counter({label:math.ceil(size*test_ratio) for label, size in size_all_dict.items()})
        split_index_dict = size_validation_dict + size_test_dict

        slice_data = lambda begin={}, end={}: sum([
            data[label][begin.get(label, 0):end.get(label, None)]
            for label in size_all_dict.keys()
        ], [])

        validation_md5s = slice_data(end=size_validation_dict)
        test_md5s = slice_data(begin=size_validation_dict, end=split_index_dict)
        train_md5s = slice_data(begin=split_index_dict)

        assert len(sorted_md5) == len(set(sum([validation_md5s, test_md5s, train_md5s], [])))

        validation_signals = [self._hdf5s[md5] for md5 in validation_md5s]
        test_signals = [self._hdf5s[md5] for md5 in test_md5s]
        train_signals = [self._hdf5s[md5] for md5 in train_md5s]

        validation_labels = [self._metadata[md5][self._label_category] for md5 in validation_md5s]
        test_labels = [self._metadata[md5][self._label_category] for md5 in test_md5s]
        train_labels = [self._metadata[md5][self._label_category] for md5 in train_md5s]

        if self._oversample:
            train_signals, train_labels = self._oversample_data(train_signals, train_labels)

        if self._onehot:
            self._to_onehot(validation_labels)
            self._to_onehot(test_labels)
            self._to_onehot(train_labels)

        self._validation = Data(validation_signals, validation_labels)
        self._test = Data(test_signals, test_labels)
        self._train = Data(train_signals, train_labels)

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
        sorted_md5 = sorted(self._metadata.keys())
        uniq = set()
        for md5 in sorted_md5:
            uniq.add(self._metadata[md5][self._label_category])
        self._sorted_choices = sorted(list(uniq))
        onehot_dict = {}
        for i in range(len(self._sorted_choices)):
            onehot = np.zeros(len(self._sorted_choices))
            onehot[i] = 1
            onehot_dict[self._sorted_choices[i]] = onehot
        for i in range(len(labels)):
            labels[i] = onehot_dict[labels[i]]

    def label_counter(self, n=None):
        counter = collections.Counter()
        for labels in self._metadata.values():
            label = labels[self._label_category]
            counter.update([label])

        return counter

    def display_labels(self):
        for label, count in self.label_counter().most_common():
            print('{}: {}'.format(label, count))

    def preprocess(self, f):
        self._train.preprocess(f)
        self._validation.preprocess(f)
        self._test.preprocess(f)

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
    def labels(self):
        return self._sorted_choices


class Data(object):
    def __init__(self, x, y):
        self._num_examples = len(x)
        self._signals = np.array(x)
        self._labels = np.array(y)
        self._index = 0

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
        return self._signals[start:end], self._labels[start:end]

    def _shuffle(self):
        shuffle = np.arange(self._num_examples)
        np.random.shuffle(shuffle)
        self._signals = self._signals[shuffle]
        self._labels = self._labels[shuffle]

    @property
    def signals(self):
        return self._signals

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

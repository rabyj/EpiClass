import glob
import h5py
import json
import os.path
import numpy as np
from scipy import signal
import random

import config

class EpiData(object):
    def __init__(self, label_category, oversample=False, normalization=True, onehot=True):
        self._label_category = label_category
        self._oversample = oversample
        self._normalization = normalization
        self._onehot = onehot
        self._load_chrom_sizes(config.CHROM_PATH)
        self._load_hdf5(config.DATA_PATH)
        self._load_metadata(config.META_PATH)
        self._build_data()

    def _load_metadata(self, meta_path):
        meta_raw = json.load(open(meta_path))
        self._metadata = {}
        for dataset in meta_raw["datasets"]:
            if dataset["md5sum"] in self._hdf5s:
                self._metadata[dataset["md5sum"]] = dataset

    def _load_chrom_sizes(self, chrom_path):
        self._chroms = []
        with open(chrom_path) as chrom_file:
            for line in chrom_file:
                line = line.strip()
                if line:
                    line = line.split()
                    self._chroms.append(line[0])
        self._chroms.sort()

    def _load_hdf5(self, data_path):
        self._hdf5s = {}
        for file_path in glob.glob(os.path.join(data_path, "*.hdf5")):
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

    def _build_data(self):
        data = [[], []]
        sorted_md5 = sorted(self._metadata.keys())
        for md5 in sorted_md5:
            data[0].append(self._hdf5s[md5])
            data[1].append(self._metadata[md5][self._label_category])

        size_all = len(data[0])
        size_validation = int(size_all*0.1)
        size_test = int(size_all*0.1)
        split_index = size_test+size_validation

        validation_signals = data[0][:size_validation]
        validation_labels = data[1][:size_validation]
        test_signals = data[0][size_validation:split_index]
        test_labels = data[1][size_validation:split_index]
        train_signals = data[0][split_index:]
        train_labels = data[1][split_index:]
        if self._oversample:
            train_signals, train_labels = self._oversample_data(train_signals, train_labels)

        if self._onehot:
            self._to_onehot(validation_labels)
            self._to_onehot(test_labels)
            self._to_onehot(train_labels)

        self._validation = Data(validation_signals, validation_labels)
        self._test = Data(test_signals, test_labels)
        self._train = Data(train_signals, train_labels)

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

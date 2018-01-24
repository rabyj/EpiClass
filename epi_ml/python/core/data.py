import glob
import h5py
import json
import os.path
import numpy as np

import config


class EpiData(object):
    def __init__(self, label_category):
        self._label_category = label_category
        self._load_metadata(config.META_PATH)
        self._load_chrom_sizes(config.CHROM_PATH)
        self._load_hdf5(config.DATA_PATH)
        self._build_data()

    def _load_metadata(self, meta_path):
        meta_raw = json.load(open(meta_path))
        self._metadata = {}
        for dataset in meta_raw["datasets"]:
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
        return (array - array.mean()) / array.std()

    def _extract_md5(self, file_name):
        return os.path.basename(file_name).split("_")[0]

    def _build_data(self):
        data = [[], []]
        sorted_md5 = sorted(self._metadata.keys())
        for md5 in sorted_md5:
            data[0].append(self._hdf5s[md5])
            data[1].append(self._metadata[md5][self._label_category])
        self._to_onehot(data[1])
        self._train = Data(data[0][:800], data[1][:800])
        self._validation = Data(data[0][800:900], data[1][800:900])
        self._test = Data(data[0][900:], data[1][900:])

    def _to_onehot(self, labels):
        uniq = set()
        for label in labels:
            uniq.add(label)
        self._sorted_choices = sorted(list(uniq))
        onehot_dict = {}
        for i in range(len(self._sorted_choices)):
            onehot = np.zeros(len(self._sorted_choices))
            onehot[i] = 1
            onehot_dict[self._sorted_choices[i]] = onehot
        for i in range(len(labels)):
            labels[i] = onehot_dict[labels[i]]

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

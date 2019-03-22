import json
import collections
import io

from dataSource import EpiDataSource

class Metadata(object):
    def __init__(self, datasource: EpiDataSource):
        self._metadata = self._load_metadata(datasource.metadata_file)

    def __getitem__(self, md5):
        return self._metadata[md5]

    def __delitem__(self, md5):
        del self._metadata[md5]

    def __contains__(self, md5):
        return md5 in self._metadata

    @property
    def md5s(self):
        return self._metadata.keys()

    @property
    def datasets(self):
        return self._metadata.values()

    def _load_metadata(self, meta_file: io.IOBase):
        meta_file.seek(0)
        meta_raw = json.load(meta_file)
        metadata = {}
        for dataset in meta_raw["datasets"]:
            metadata[dataset["md5sum"]] = dataset
        return metadata

    def apply_filter(self, meta_filter=lambda item:True):
        #item is md5:dataset
        self._metadata = dict(filter(meta_filter, self._metadata.items()))

    def remove_missing_labels(self, label_category):
        filt = lambda item:label_category in item[1] 
        self.apply_filter(filt)

    def md5_per_class(self, label_category):
        sorted_md5 = sorted(self._metadata.keys())
        data = collections.defaultdict(list)
        for md5 in sorted_md5:
            data[self._metadata[md5][label_category]].append(md5)
        return data

    def remove_small_classes(self, min_class_size, label_category):
        """"""
        # self._data  # label/class: md5 list
        # self._metadata #md5 : dataset_dict
        data = self.md5_per_class(label_category)
        nb_label_i =  len(data)
        for label, size in self.label_counter(label_category).most_common():
            if size < min_class_size:
                for md5 in data[label]:
                    del self._metadata[md5]

        print("{}/{} labels left after filtering.".format(len(data), nb_label_i))

    def label_counter(self, label_category):
        counter = collections.Counter()
        for labels in self._metadata.values():
            label = labels[label_category]
            counter.update([label])
        return counter

    def display_labels(self, label_category):
        print('\nExamples')
        i = 0
        for label, count in self.label_counter(label_category).most_common():
            print('{}: {}'.format(label, count))
            i += count
        print('For a total of {} examples\n'.format(i))


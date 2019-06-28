import json
import collections
import io

from data_source import EpiDataSource

class Metadata(object):
    def __init__(self, datasource: EpiDataSource):
        self._metadata = self._load_metadata(datasource.metadata_file)

    def __getitem__(self, md5):
        return self._metadata[md5]

    def __delitem__(self, md5):
        del self._metadata[md5]

    def __contains__(self, md5):
        return md5 in self._metadata

    def get(self, md5):
        return self._metadata.get(md5)

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

    def apply_filter(self, meta_filter=lambda item: True):
        #item is md5:dataset
        self._metadata = dict(filter(meta_filter, self._metadata.items()))

    def remove_missing_labels(self, label_category):
        filt = lambda item: label_category in item[1]
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
        nb_label_i = len(data)
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

    def create_healthy_category(self):
        """Combine "disease" and "donor_health_status" to create a "healthy" category."""

        healthy_pairs = {
            ("apparently_healthy", "apparently_healthy"),
            ("healthy", "na"),
            ("healthy", "healthy"),
            ("healthy", "healty"),
            ("healthy", "_"),
            ("healthy", "."),
            ("healthy", None),
            ("healthy", ""),
            ("no_ad_evident_at_demise", "no_ad_evident_at_demise"),
            ("no_ad_present", "no_ad_present"),
            ("none", "na"),
            ("none", None),
            ("none", ""),
            ("none", "normal"),
            ("none", "disease_free"),
            ("none", "healthy,_no_prior_medical_history_(no_diabetes,_hypertension,_coronary_artery_disease,_cancer)"),
            ("none", "normal,_renal_cell_ca_(clear_cell_type)"),
            ("none", "normal_smooth_muscle"),
            ("none", "non_localized_disease_free"),
            ("none", "normal_mucosa"),
            ("none", "polysubstance_abuse_(no_diabetes,_hypertension,_coronary_artery_disease,_cancer)"),
            ("none", "none"),
            ("none", "healthy"),
            ("normal", "normal"),
            ("negative_for_bac/fung/myc,_negative_for_hiv_1/hbv/hcv", "negative_for_bac/fung/myc,_negative_for_hiv_1/hbv/hcv"),
            ("obesity", "healthy"),
            ("presumed_normal", "presumed_normal"),
            ("presumed_normal", None),
            (None, "healthy"),
            ("", "healthy"),
            (None, "presumed_healthy"),
            ("", "presumed_healthy")
        }

        ignore_pairs = {
            ("na", "na"),
            ("none", "unknown"),
            ("unknown", "unknown"),
            ("unknown", "na"),
            ("autism_spectrum_disorder", "na"),
            ("disease_type_and/or_category_unknown", "na"),
            (None, "probable_tonsillitis"),
            ("", "probable_tonsillitis"),
            (None, "na"),
            ("", "na"),
            (None, None),
            ("", "")
        }

        # self._test_healthy_hg38()

        for dataset in self.datasets:
            disease = dataset.get("disease", None)
            donor_health_status = dataset.get("donor_health_status", None)

            if (disease, donor_health_status) in healthy_pairs:
                dataset["healthy"] = "y"
            elif (disease, donor_health_status) in ignore_pairs:
                continue
            else:
                dataset["healthy"] = "n"

        # self._test_healthy_hg38()

    def _test_healthy_hg38(self):

        test_md5s = ["379bed415f8e3fb17456115e68e5e773", "4749dbc09f7af9731f22cb1818dcefbd", "64b754ec8bc10b1f45b9990c387dda1f"]
        """ disease:donor_health_status
        "":""
        "":"na"
        "none":""
        """

        for md5 in test_md5s:
            dataset = self._metadata[md5]
            disease = dataset.get("disease", "absent")
            donor_health_status = dataset.get("donor_health_status", "absent")
            healthy = dataset.get("healthy", "absent")
            print("md5:{}\ndisease: {}\ndonor_health_status: {}\nhealthy:{}\n".format(md5, disease, donor_health_status, healthy))

    def merge_molecule_classes(self):
        """Combine similar classes pairs in the molecule category."""
        for dataset in self.datasets:
            molecule = dataset.get("molecule", None)
            if molecule == "rna":
                dataset["molecule"] = "total_rna"
            elif molecule == "polyadenylated_mrna":
                dataset["molecule"] = "polya_rna"

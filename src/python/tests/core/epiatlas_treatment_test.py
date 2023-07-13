"""EpiAtlas data treatment testing module."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from epi_ml.core.data_source import EpiDataSource
from epi_ml.core.epiatlas_treatment import (
    ACCEPTED_TRACKS,
    LEADER_TRACKS,
    OTHER_TRACKS,
    EpiAtlasFoldFactory,
)
from epi_ml.core.metadata import Metadata
from epi_ml.utils.general_utility import write_md5s_to_file
from tests.fixtures.epilap_test_data import EpiAtlasTreatmentTestData


class TestEpiAtlasFoldFactory:
    """Test class EpiAtlasFoldFactory.

    Preconditions: Exact same input labels list. (raw_dset.train.encoded_labels)
    """

    @pytest.fixture(scope="class", name="test_data")
    def test_data(self) -> EpiAtlasFoldFactory:
        """Mock test EpiAtlasFoldFactory."""
        return EpiAtlasTreatmentTestData.default_test_data()

    @pytest.fixture(name="big_test_data")
    def big_test_data(self, tmp_path) -> EpiAtlasFoldFactory:
        """Mock big EpiAtlasFoldFactory."""
        tmp_path = Path(tmp_path) / "big_test"
        tmp_path.mkdir(exist_ok=True, parents=True)

        target_category = "harmonized_donor_sex"

        meta_path = (
            Path.home()
            / "projects/epilap/input/metadata/hg38_2023_epiatlas_dfreeze_limited_categories.json"
        )
        # Reduce total nb of files
        meta = Metadata(meta_path)
        meta.select_category_subsets(target_category, ["female", "male"])
        md5_per_class = meta.md5_per_class(target_category)
        md5s = [md5 for md5_list in md5_per_class.values() for md5 in md5_list[:2000]]

        md5_file = write_md5s_to_file(md5s=md5s, logdir=str(tmp_path), name="big_test")
        return EpiAtlasTreatmentTestData(
            metadata_path=meta_path,
            logdir=tmp_path,
            md5_list_path=md5_file,
        ).get_ea_handler(label_category=target_category)

    @pytest.fixture(scope="class")
    def test_metadata(self, test_data) -> Metadata:
        """Basic test metadata, using real data."""
        meta_path = test_data.epiatlas_dataset.datasource.metadata_file
        return Metadata(meta_path)

    @pytest.fixture(scope="class")
    def test_datasource(self, test_data) -> EpiDataSource:
        """Basic test datasource, using real data."""
        return test_data.epiatlas_dataset.datasource

    @staticmethod
    def test_data_from_datasource(
        datasource: EpiDataSource, metadata: Metadata, target_category: str
    ) -> EpiAtlasFoldFactory:
        """Create EpiAtlasFoldFactory from datasource."""
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            datasource,
            label_category=target_category,
            min_class_size=1,
            n_fold=2,
            md5_list=list(metadata.md5s),
            force_filter=True,
            test_ratio=0,
        )
        return ea_handler

    @staticmethod
    def modified_metadata(test_metadata, del_tracks: List[str]) -> Metadata:
        """Remove tracks from metadata."""
        meta = copy.deepcopy(test_metadata)
        for del_track in del_tracks:
            for md5, dset in list(meta.items):
                if dset["track_type"] == del_track:
                    del meta[md5]
        return meta

    @staticmethod
    def assert_splits(ea_handler: EpiAtlasFoldFactory):
        """Assert that splits are correct."""
        total_data = ea_handler.train_val_dset
        assert total_data.num_examples == len(set(total_data.ids))

        for dset in ea_handler.yield_split():
            trains_ids = set(dset.train.ids)
            valid_ids = set(dset.validation.ids)
            assert len(trains_ids & valid_ids) == 0, "No overlap between train and valid"

            assert len(valid_ids) == len(dset.validation.ids), "No dups in validation"

            assert dset.test.num_examples == 0, "No test set"

            assert total_data.num_examples == len(trains_ids) + len(valid_ids)

    # @pytest.mark.parametrize("splitter", ["test_data", "big_test_data"])
    def test_yield_correct_split_1(self, test_data: EpiAtlasFoldFactory):
        """Test that splits contain the correct number of training and validation samples."""
        ea_handler = test_data

        datasource = test_data.epiatlas_dataset.datasource
        metadata = test_data.epiatlas_dataset.metadata
        ea_handler2 = TestEpiAtlasFoldFactory.test_data_from_datasource(
            datasource,
            metadata,
            target_category=test_data.epiatlas_dataset.target_category,
        )

        for handler in [ea_handler, ea_handler2]:
            TestEpiAtlasFoldFactory.assert_splits(handler)

    @pytest.mark.filterwarnings("ignore:.*Cannot read file directly.*")
    @pytest.mark.skip(reason="Takes too long.")
    def test_yield_correct_split_2(self, big_test_data: EpiAtlasFoldFactory):
        """Test that splits contain the correct number of training and validation samples, with real metadata."""
        TestEpiAtlasFoldFactory.assert_splits(big_test_data)

    # def test_yield_subsample_validation_1(self, test_data: EpiAtlasFoldFactory):
    #     """Test correct subsampling. Subsplit should partition initial validation split."""
    #     ea_handler = test_data

    #     # Is it coherent with initial split?
    #     # initial usual train/valid split
    #     for split_n in range(ea_handler.k):
    #         total_dataset = list(ea_handler.yield_split())[split_n]
    #         total_ids = set(total_dataset.validation.ids)

    #         # focus down on further splits of validation test
    #         for _ in range(2):
    #             for sub_dataset in ea_handler.yield_subsample_validation(
    #                 chosen_split=split_n, nb_split=2
    #             ):
    #                 train = sub_dataset.train.ids
    #                 valid = sub_dataset.validation.ids

    #                 ids = set(list(train) + list(valid))
    #                 assert ids == total_ids

    # def test_yield_subsample_validation_2(self, test_data: EpiAtlasFoldFactory):
    #     """Test correct subsampling. Repeated calls should lead to same outcome"""
    #     dset1 = next(test_data.yield_subsample_validation(chosen_split=0, nb_split=2))
    #     dset2 = next(test_data.yield_subsample_validation(chosen_split=0, nb_split=2))
    #     assert list(dset1.validation.ids) == list(dset2.validation.ids)

    # def test_yield_subsample_validation_outofrange(self, test_data: EpiAtlasFoldFactory):
    #     """Test correct subsampling range."""
    #     chosen_split = test_data.k  # one off error

    #     err_msg = f"{chosen_split}.*{test_data.k}"
    #     with pytest.raises(IndexError, match=err_msg):
    #         next(
    #             test_data.yield_subsample_validation(
    #                 chosen_split=chosen_split, nb_split=2
    #             )
    #         )

    # def test_yield_subsample_validation_toomanysplits(
    #     self, test_data: EpiAtlasFoldFactory
    # ):
    #     """Test that you cannot ask for too many splits."""
    #     nb_split = 10
    #     with pytest.raises(ValueError):
    #         next(test_data.yield_subsample_validation(chosen_split=0, nb_split=nb_split))

    @pytest.mark.parametrize("del_track,", ["raw", "pval", "fc", "Unique_minusRaw"])
    def test_yield_missing_tracks(self, test_datasource, test_metadata, del_track: str):
        """Make sure splitter can handle missing tracks."""
        meta = TestEpiAtlasFoldFactory.modified_metadata(test_metadata, [del_track])
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            test_datasource,
            label_category="biomaterial_type",
            min_class_size=2,
            n_fold=3,
            md5_list=list(meta.md5s),
            force_filter=True,
        )
        for _ in ea_handler.yield_split():
            pass

    def test_yield_only_lead(self, test_datasource, test_metadata):
        """Make sure splitter can handle multiple missing tracks."""
        meta = TestEpiAtlasFoldFactory.modified_metadata(test_metadata, ["fc", "pval"])
        labels_count = meta.label_counter("track_type")

        total_size = sum(labels_count[track_type] for track_type in ACCEPTED_TRACKS)
        total_raw_count = labels_count["raw"]

        ea_handler = EpiAtlasFoldFactory.from_datasource(
            test_datasource,
            label_category="biomaterial_type",
            min_class_size=2,
            n_fold=3,
            md5_list=list(meta.md5s),
            force_filter=True,
            test_ratio=0,
        )
        ref_dset = ea_handler.train_val_dset
        lead_tracks = len(
            [
                sample_id
                for sample_id in ref_dset.ids
                if meta[sample_id]["track_type"] in LEADER_TRACKS
            ]
        )
        other_tracks = len(
            [
                sample_id
                for sample_id in ref_dset.ids
                if meta[sample_id]["track_type"] in OTHER_TRACKS
            ]
        )

        assert total_size == lead_tracks + other_tracks

        valid_raw_sum = 0
        for dset in ea_handler.yield_split():
            trains_ids = set(dset.train.ids)
            valid_ids = set(dset.validation.ids)

            train_unique_size = len(trains_ids)
            valid_unique_size = len(valid_ids)

            assert total_size == train_unique_size + valid_unique_size

            track_type_counter = Counter(
                [meta[md5]["track_type"] for md5 in dset.validation.ids]
            )
            valid_raw_sum += track_type_counter["raw"]

        assert valid_raw_sum == total_raw_count

    def test_split_by_track_type(self):
        """Test that track types are distributed correctly but same uuid stay together."""
        raise NotImplementedError


@pytest.mark.skip(reason="One time. For sklearn code sanity check.")
def test_StratifiedKFold_sanity():
    """Test that StratifiedKFold yields same datasets every time.

    Preconditions: Exact same input labels list. (raw_dset.train.encoded_labels)
    """
    skf1 = StratifiedKFold(n_splits=5, shuffle=False)
    skf2 = StratifiedKFold(n_splits=5, shuffle=False)
    n_classes = 10
    num_examples = 150
    labels = np.random.choice(n_classes, size=num_examples)

    run_1 = list(
        skf1.split(
            np.zeros((num_examples, n_classes)),
            list(labels),
        )
    )
    run_2 = list(
        skf2.split(
            np.zeros((num_examples, n_classes)),
            list(labels),
        )
    )

    for elem1, elem2 in zip(run_1, run_2):
        train1, valid1 = elem1
        train2, valid2 = elem2
        assert np.array_equal(train1, train2)
        assert np.array_equal(valid1, valid2)

"""EpiAtlas data treatment testing module."""
from __future__ import annotations

import copy
from collections import Counter
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
from tests.fixtures.epilap_test_data import EpiAtlasTreatmentTestData


class TestEpiAtlasFoldFactory:
    """Test class EpiAtlasFoldFactory.

    Preconditions: Exact same input labels list. (raw_dset.train.encoded_labels)
    """

    @pytest.fixture(scope="class", autouse=True)
    def test_data(self) -> EpiAtlasFoldFactory:
        """Mock test EpiAtlasFoldFactory."""
        return EpiAtlasTreatmentTestData.default_test_data()

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
    def modified_metadata(test_metadata, del_tracks: List[str]) -> Metadata:
        """Remove tracks from metadata."""
        meta = copy.deepcopy(test_metadata)
        for del_track in del_tracks:
            for md5, dset in list(meta.items):
                if dset["track_type"] == del_track:
                    del meta[md5]
        return meta

    def test_yield_split_size(self, test_data: EpiAtlasFoldFactory):
        """Test that splits contain the correct number of training and validation samples."""
        total_data = test_data.train_val_dset
        assert total_data.num_examples == len(set(total_data.ids))

        for dset in test_data.yield_split():
            train_unique_size = len(set(dset.train.ids))
            valid_unique_size = len(set(dset.validation.ids))
            assert total_data.num_examples == train_unique_size + valid_unique_size

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

    @pytest.mark.parametrize("del_track,", ["pval", "fc", "Unique_minusRaw"])
    def test_yield_missing_tracks(self, test_datasource, test_metadata, del_track: str):
        """Make sure splitter can handle missing non-leading tracks."""
        meta = TestEpiAtlasFoldFactory.modified_metadata(test_metadata, [del_track])
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            test_datasource,
            label_category="biomaterial_type",
            min_class_size=2,
            n_fold=3,
            md5_list=list(meta.md5s),
        )
        for _ in ea_handler.yield_split():
            pass

    def test_yield_only_lead(self, test_datasource, test_metadata):
        """Make sure splitter can handle missing non-leading tracks."""
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
            train_unique_size = len(set(dset.train.ids))
            valid_unique_size = len(set(dset.validation.ids))
            assert dset.test.num_examples == 0

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

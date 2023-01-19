"""EpiAtlas data treatment testing module."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from src.python.core.epiatlas_treatment import EpiAtlasDataset, EpiAtlasFoldFactory
from src.python.core.metadata import Metadata
from src.python.tests.fixtures.epilap_test_data import EpiAtlasTreatmentTestData


class TestEpiAtlasFoldFactory:
    """Test class EpiAtlasFoldFactory.

    Preconditions: Exact same input labels list. (raw_dset.train.encoded_labels)
    """

    @pytest.fixture(scope="class", autouse=True)
    def test_data(self) -> EpiAtlasFoldFactory:
        """Mock test EpiAtlasFoldFactory."""
        return EpiAtlasTreatmentTestData.default_test_data()

    def test_yield_subsample_validation_1(self, test_data: EpiAtlasFoldFactory):
        """Test correct subsampling. Subsplit should partition initial validation split."""
        ea_handler = test_data

        # Is it coherent with initial split?
        # initial usual train/valid split
        for split_n in range(ea_handler.k):
            total_dataset = list(ea_handler.yield_split())[split_n]
            total_ids = set(total_dataset.validation.ids)

            # focus down on further splits of validation test
            for _ in range(2):
                for sub_dataset in ea_handler.yield_subsample_validation(
                    chosen_split=split_n, nb_split=2
                ):
                    train = sub_dataset.train.ids
                    valid = sub_dataset.validation.ids

                    ids = set(list(train) + list(valid))
                    assert ids == total_ids

    def test_yield_subsample_validation_2(self, test_data: EpiAtlasFoldFactory):
        """Test correct subsampling. Repeated calls should lead to same outcome"""
        dset1 = next(test_data.yield_subsample_validation(chosen_split=0, nb_split=2))
        dset2 = next(test_data.yield_subsample_validation(chosen_split=0, nb_split=2))
        assert list(dset1.validation.ids) == list(dset2.validation.ids)

    def test_yield_subsample_validation_outofrange(self, test_data: EpiAtlasFoldFactory):
        """Test correct subsampling range."""
        chosen_split = test_data.k  # one off error

        err_msg = f"{chosen_split}.*{test_data.k}"
        with pytest.raises(IndexError, match=err_msg):
            next(
                test_data.yield_subsample_validation(
                    chosen_split=chosen_split, nb_split=2
                )
            )

    def test_yield_subsample_validation_toomanysplits(
        self, test_data: EpiAtlasFoldFactory
    ):
        """Test that you cannot ask for too many splits."""
        nb_split = 10
        with pytest.raises(ValueError):
            next(test_data.yield_subsample_validation(chosen_split=0, nb_split=nb_split))


class TestEpiAtlasDataset:
    """Test class EpiAtlasDataset"""

    @pytest.fixture(scope="class")
    def test_metadata(self) -> Metadata:
        """Mock test EpiAtlasFoldFactory."""
        meta_path = (
            EpiAtlasTreatmentTestData.default_test_data().epiatlas_dataset.datasource.metadata_file
        )
        return Metadata(meta_path)

    @pytest.fixture(scope="class")
    def test_datasource(self):
        return EpiAtlasTreatmentTestData.default_test_data().epiatlas_dataset.datasource

    def modified_metadata(self, test_metadata, del_track: str):
        meta = copy.deepcopy(test_metadata)
        for md5, dset in list(meta.items):
            if dset["track_type"] == del_track:
                del meta[md5]
        return meta

    @pytest.mark.parametrize("del_track,", ["pval", "fc", "Unique_minusRaw"])
    def test_epiatlas_prepare_split(self, test_metadata: Metadata, del_track):
        """Verify that having missing non-leading tracks does not cause an error."""
        meta = self.modified_metadata(test_metadata, del_track)
        EpiAtlasDataset.epiatlas_prepare_split(meta)

    @pytest.mark.parametrize("del_track,", ["pval", "fc", "Unique_minusRaw"])
    def test_yield_missing_tracks(self, test_datasource, test_metadata, del_track: str):
        """Make sure splitter can handle missing non-leading tracks."""
        meta = self.modified_metadata(test_metadata, del_track)
        ea_handler = EpiAtlasFoldFactory.from_datasource(
            test_datasource,
            label_category="biomaterial_type",
            min_class_size=2,
            n_fold=3,
            metadata=meta,
        )
        for _ in ea_handler.yield_split():
            pass


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

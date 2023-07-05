"""Test SHAP related modules."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from lightgbm import LGBMClassifier
from shap import TreeExplainer
from sklearn.datasets import make_blobs

from epi_ml.core.data import DataSet, UnknownData
from epi_ml.core.estimators import EstimatorAnalyzer
from epi_ml.core.hdf5_loader import Hdf5Loader
from epi_ml.core.model_pytorch import LightningDenseClassifier
from epi_ml.core.shap_values import LGBM_SHAP_Handler, NN_SHAP_Handler, SHAP_Analyzer


class Test_NN_SHAP_Handler:
    """Class to test NN_SHAP_Handler class."""

    @pytest.fixture
    def logdir(self, mk_logdir) -> Path:
        """Test logdir"""
        return mk_logdir("shap")

    @pytest.fixture
    def handler(self, logdir: Path, test_NN_model) -> NN_SHAP_Handler:
        """NN_SHAP_Handler instance"""
        return NN_SHAP_Handler(test_NN_model, logdir)

    @pytest.fixture
    def mock_shap_values(self, test_epiatlas_dataset: DataSet) -> List[np.ndarray]:
        """Mock shape values for evaluation on two examples."""
        shap_values = [
            np.zeros(test_epiatlas_dataset.validation.signals.shape)
            for _ in test_epiatlas_dataset.classes
        ]
        return shap_values

    @pytest.fixture
    def fake_ids(self, test_epiatlas_dataset: DataSet):
        """Fake signal ids"""
        num_signals = test_epiatlas_dataset.validation.num_examples
        return [f"id{i}" for i in range(num_signals)]

    def test_compute_shaps(
        self, handler: NN_SHAP_Handler, test_epiatlas_dataset: DataSet
    ):
        """Test shapes of return of compute_shaps method."""
        dset = test_epiatlas_dataset
        _, shap_values = handler.compute_shaps(
            background_dset=dset.train, evaluation_dset=dset.validation, save=False  # type: ignore
        )
        print(f"len(shap_values) = {len(shap_values)}")
        print(f"shap_values[0].shape = {shap_values[0].shape }")

        n_signals, n_dims = dset.validation.signals.shape[:]
        assert shap_values[0].shape == (n_signals, n_dims)

    def test_save_load_csv(self, handler: NN_SHAP_Handler, mock_shap_values, fake_ids):
        """Test pickle save/load methods."""
        shaps = mock_shap_values[0]
        path = handler.saver.save_to_csv(shaps, fake_ids, name="test")

        data = handler.saver.load_from_csv(path)
        assert list(data.index) == fake_ids
        assert np.array_equal(shaps, data.values)

    def test_save_to_csv_list_input(
        self, handler: NN_SHAP_Handler, mock_shap_values, fake_ids
    ):
        """Test effect of list input."""
        shap_values_matrix = [mock_shap_values[0]]
        name = "test_csv"

        with pytest.raises(ValueError):
            handler.saver.save_to_csv(shap_values_matrix, fake_ids, name)  # type: ignore

    def test_create_filename(self, handler: NN_SHAP_Handler):
        """Test filename creation method. Created by GPT4 lol."""
        ext = "pickle"
        name = "test_name"

        filename = handler.saver._create_filename(  # pylint: disable=protected-access
            ext, name
        )
        assert filename.name.startswith(f"shap_{name}_")
        assert filename.name.endswith(f".{ext}")
        assert filename.parent == Path(handler.logdir)


@pytest.mark.skip(reason="One time thing")
def test_tree_explainer():
    """Minimal test to check if TreeExplainer works with LGBMClassifier."""
    X, y = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)  # type: ignore # pylint: disable=unbalanced-tuple-unpacking

    for boosting_method, model_output in itertools.product(
        ["gbdt", "dart"],
        ["raw", "probability", "log_loss", "predict", "predict_proba"],
    ):
        test_model = LGBMClassifier(boosting_type=boosting_method, objective="multiclass")
        test_model.fit(X, y)
        try:
            explainer = TreeExplainer(
                model=test_model,
                data=X,
                model_output=model_output,
                feature_perturbation="interventional",
            )
        except AttributeError:
            print(
                f"({boosting_method})(err2) TreeExplainer does not support multiclass + model_output={model_output}"
            )
            continue
        except Exception as e:
            if "Model does not have a known objective or output type" in e.args[0]:
                print(
                    f"({boosting_method})(err1) TreeExplainer does not support multiclass + model_output={model_output}"
                )
                continue
            raise e
        shap_values = explainer.shap_values(X)
        print(
            f"({boosting_method}) TreeExplainer supports multiclass + model_output={model_output}"
        )
        print(np.array(shap_values).shape)


class Test_LGBM_SHAP_Handler:
    """Class to test LGBM_SHAP_Handler class."""

    N = 100

    @staticmethod
    def create_test_model(
        nb_class: int, nb_features: int, nb_samples: int
    ) -> Tuple[EstimatorAnalyzer, UnknownData]:
        """Create a test LGBMClassifier model for testing."""
        X, y = make_blobs(n_samples=nb_samples, centers=nb_class, n_features=nb_features, random_state=42)  # type: ignore # pylint: disable=unbalanced-tuple-unpacking
        test_model = LGBMClassifier(
            boosting_type="dart",
        )
        test_model.fit(X, y)

        dataset = UnknownData(range(nb_samples), X, y, [str(val) for val in y])

        model_analyzer = EstimatorAnalyzer(
            classes=[str(i) for i in range(nb_class)],
            estimator=test_model,
        )

        return model_analyzer, dataset

    @pytest.fixture(name="model2c")
    def test_model_2classes(self) -> Tuple[EstimatorAnalyzer, UnknownData]:
        """Test model with 2 classes."""
        model_analyzer, dataset = Test_LGBM_SHAP_Handler.create_test_model(
            2, 4, Test_LGBM_SHAP_Handler.N
        )

        return model_analyzer, dataset

    @pytest.fixture(name="model3c")
    def test_model_3classes(self) -> Tuple[EstimatorAnalyzer, UnknownData]:
        """Test model with 3 classes."""
        model_analyzer, dataset = Test_LGBM_SHAP_Handler.create_test_model(
            3, 4, Test_LGBM_SHAP_Handler.N
        )

        return model_analyzer, dataset

    @pytest.mark.parametrize(
        "test_data,num_workers",
        [("model2c", 1), ("model3c", 1), ("model2c", 2), ("model3c", 2)],
    )
    def test_compute_shaps(self, test_data, num_workers, tmp_path, request):
        """
        Tests the compute_shaps method of the LGBM_SHAP_Handler class. It checks if the SHAP values and expected values
        are computed correctly and if they are saved correctly with the correct parameters.
        """
        model_analyzer, evaluation_dset = request.getfixturevalue(test_data)
        handler = LGBM_SHAP_Handler(model_analyzer, tmp_path)

        # Test compute_shaps
        shap_values, explainer = handler.compute_shaps(
            background_dset=evaluation_dset,
            evaluation_dset=evaluation_dset,
            save=True,
            name="test",
            num_workers=num_workers,
        )

        # Test output types
        expected_value = explainer.expected_value
        assert isinstance(shap_values, list)
        assert isinstance(shap_values[0], np.ndarray)
        assert isinstance(expected_value, (float, np.ndarray))

        # # Test output shapes
        nb_samples = Test_LGBM_SHAP_Handler.N
        nb_classes = len(model_analyzer.classes)
        nb_features = evaluation_dset.signals.shape[1]
        if isinstance(expected_value, np.ndarray):  # multiclass case
            assert expected_value.shape == (nb_classes,)
            assert shap_values[0].shape == (nb_samples, nb_features)
        else:  # binary case
            assert len(shap_values) == nb_samples
            assert shap_values[0].shape == (nb_features,)


class Test_SHAP_Analyzer:
    """Class to test SHAP_Analyzer class."""

    @pytest.fixture
    def test_folder(self, mk_logdir) -> Path:
        """Return temp shap test folder."""
        return mk_logdir("shap_test")

    @pytest.fixture
    def saccer3_dir(self) -> Path:
        """saccer3 params dir"""
        return Path(__file__).parent.parent / "fixtures" / "saccer3"

    @pytest.fixture
    def saccer3_model(self, saccer3_dir: Path) -> LightningDenseClassifier:
        """saccer3 test model"""
        saccer3_model_dir = saccer3_dir / "model"
        return LightningDenseClassifier.restore_model(saccer3_model_dir)

    @pytest.fixture
    def saccer3_signals(self, saccer3_dir: Path) -> Dict:
        """saccer3 epigenetic signals"""
        chrom_file = saccer3_dir / "saccer3.can.chrom.sizes"
        hdf5_filelist = saccer3_dir / "hdf5_10kb_all_none.list"
        hdf5_loader = Hdf5Loader(chrom_file=chrom_file, normalization=True)
        hdf5_loader.load_hdf5s(hdf5_filelist, strict=True)
        return hdf5_loader.signals

    @pytest.fixture
    def test_dsets(self, saccer3_signals: Dict) -> Tuple[UnknownData, UnknownData]:
        """Return background and evaluation datasets."""
        background_signals = list(saccer3_signals.values())[0:12]
        eval_signals = background_signals[10:12]

        background_dset = UnknownData(
            ids=range(len(background_signals)),
            x=background_signals,
            y=np.zeros(len(background_signals)),
            y_str=["NA" for _ in range(len(background_signals))],
        )

        eval_dset = UnknownData(
            ids=[-i for i in range(len(eval_signals))],
            x=eval_signals,
            y=np.zeros(len(eval_signals)),
            y_str=["NA" for _ in range(len(eval_signals))],
        )

        return background_dset, eval_dset

    def test_verify_shap_values_coherence(
        self,
        saccer3_model: LightningDenseClassifier,
        test_folder: Path,
        test_dsets: Tuple[UnknownData, UnknownData],
    ):
        """Test compute_shap_values method."""
        NN_shap_handler = NN_SHAP_Handler(model=saccer3_model, logdir=test_folder)

        explainer, shap_values = NN_shap_handler.compute_shaps(
            background_dset=test_dsets[0],
            evaluation_dset=test_dsets[1],
            save=True,
            name="test",
        )

        shap_analyzer = SHAP_Analyzer(saccer3_model, explainer)
        shap_analyzer.verify_shap_values_coherence(shap_values, test_dsets[1])
        assert True

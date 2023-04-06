"""Module containing shap values related code (e.g. handling computation, analysing results)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
from numpy.typing import ArrayLike
from torch import Tensor

from epi_ml.core.model_pytorch import LightningDenseClassifier
from epi_ml.core.types import SomeData
from epi_ml.utils.time import time_now_str


class SHAP_Handler:
    """Handle shap computations and data saving/loading."""

    def __init__(self, model: LightningDenseClassifier, logdir: Path | str):
        self.model = model
        self.logdir = logdir
        self.filename_template = "shap_{name}_{time}.{ext}"

    def _create_filename(self, ext: str, name="") -> Path:
        filename = self.filename_template.format(name=name, ext=ext, time=time_now_str())

        return Path(self.logdir) / filename

    def compute_NN(
        self,
        background_dset: SomeData,
        evaluation_dset: SomeData,
        save=True,
        name="",
        load_filepath: Path | str | None = None,
    ) -> Tuple[shap.DeepExplainer, List[np.ndarray]]:
        """Compute shap values of deep learning model on evaluation dataset
        by creating an explainer with background dataset.

        Loads explainer from given filepath if given.

        Returns explainer and shap values (as a list of matrix per class)
        """
        if load_filepath is not None:
            explainer = self.load_from_pickle(load_filepath)
        else:
            explainer = shap.DeepExplainer(
                model=self.model, data=Tensor(background_dset.signals)
            )

        if save and load_filepath is None:
            self.save_explainer(explainer, "explainer_" + name)

        shap_values = explainer.shap_values(Tensor(evaluation_dset.signals))

        if save:
            self.save_to_pickle(shap_values, evaluation_dset.ids, name, explainer)

        return explainer, shap_values  # type: ignore

    def save_explainer(self, explainer: shap.Explainer, name: str = "") -> Path:
        """Save explainer to pickle file."""
        filename = self._create_filename(name=name, ext="pickle")

        print(f"Saving explainer to: {filename}")
        with open(filename, "wb") as f:
            pickle.dump(explainer, f)

        return filename

    def save_to_pickle(
        self,
        shap_values: ArrayLike,
        ids: List[str],
        name: str = "",
        explainer: shap.Explainer | None = None,
    ) -> Path:
        """Save shap values with assigned signals ids to a pickle file.
        Also save the explainer if given.

        Returns path of saved file."""

        filename = self._create_filename(name=name, ext="pickle")

        print(f"Saving SHAP values and {explainer} to: {filename}")
        with open(filename, "wb") as f:
            pickle.dump({"shap": shap_values, "ids": ids, "explainer": explainer}, f)

        return filename

    @staticmethod
    def load_from_pickle(path: str | Path) -> Any:
        """Load pickle file.

        Returns {"shap": shap_values, "ids": ids, "explainer": shap_explainer} dict
        if given filepath of a saved shap values file.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    def save_to_csv(
        self, shap_values_matrix: np.ndarray, ids: List[str], name: str
    ) -> Path:
        """Save a single shap value matrix (shape (n_samples, #features)) to csv.
        Giving a name is mandatory contrary to pickle.

        Returns path of saved file.
        """
        if isinstance(shap_values_matrix, list):
            raise ValueError(
                "{shap_values_matrix} is a list, not an array. Need one matrix of shape (n_samples, #features)"
            )

        filename = self._create_filename(name=name, ext="csv")

        n_dims = shap_values_matrix.shape[1]
        df = pd.DataFrame(data=shap_values_matrix, index=ids, columns=range(n_dims))

        print(f"Saving SHAP values to: {filename}")
        df.to_csv(filename)

        return filename

    @staticmethod
    def load_from_csv(path: Path | str) -> pd.DataFrame:
        """Return pandas dataframe of shap values for loaded file."""
        return pd.read_csv(path, index_col=0)


class SHAP_Analyzer:
    """SHAP_Analyzer class for analyzing SHAP values of a model.

    Attributes:
        model: The trained model.
        explainer: The SHAP explainer object used to compute SHAP values.
    """

    def __init__(self, model, explainer):
        self.model = model
        self.explainer = explainer

    def verify_shap_values_coherence(
        self, shap_values: List, dset: SomeData, tolerance=1e-6
    ):
        """Verify the coherence of SHAP values with the model's output probabilities.

        Checks if the sum of SHAP values for each sample (across all classes) and the
        base values is approximately equal to the model's output probabilities.

        Args:
            shap_values: List of SHAP values for each class.
            dset: The dataset used to compute the SHAP values.
                  The samples need to be in the same order as in list of shap values.
            tolerance: The allowed tolerance for the difference between the sum of SHAP
                values and the model's output probabilities (default is 1e-6).

        Returns:
            bool: True if the SHAP values are coherent, False otherwise.
        """
        num_classes = len(shap_values)
        num_samples = shap_values[0].shape[0]

        # Calculate the sum of SHAP values for each sample (across all classes) and add base values
        shap_sum = np.zeros((num_samples, num_classes))
        for i, shap_values_class in enumerate(shap_values):
            shap_sum[:, i] = (
                shap_values_class.sum(axis=1) + self.explainer.expected_value[i]
            )

        # Compute the model's output probabilities for the samples
        model_output_probabilities = self.model(dset.signals).detach().numpy()

        # Compare the sum of SHAP values with the model's output probabilities
        diff = np.abs(shap_sum - model_output_probabilities)
        coherent = np.all(diff <= tolerance)

        if not coherent:
            problematic_samples = np.argwhere(diff > tolerance)
            print(f"SHAP values are not coherent for samples: {problematic_samples}")

        return coherent

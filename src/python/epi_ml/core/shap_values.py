"""Module containing shap values related code (e.g. handling computation, analysing results)."""
from __future__ import annotations

import concurrent.futures
import copy
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline

from epi_ml.core.estimators import EstimatorAnalyzer
from epi_ml.core.model_pytorch import LightningDenseClassifier
from epi_ml.core.types import SomeData
from epi_ml.utils.time import time_now_str


class SHAP_Saver:
    """Handle shap data saving/loading."""

    def __init__(self, logdir: Path | str):
        self.logdir = logdir
        self.filename_template = "shap_{name}_{time}.{ext}"

    def _create_filename(self, ext: str, name="") -> Path:
        """Create a filename with the given extension and name, and a timestamp."""
        filename = self.filename_template.format(name=name, ext=ext, time=time_now_str())
        filename = Path(self.logdir) / filename
        return filename

    def save_to_csv(
        self, shap_values_matrix: np.ndarray, ids: List[str], name: str
    ) -> Path:
        """Save a single shap value matrix (shape (n_samples, #features)) to csv.
        Giving a name is mandatory.

        Returns path of saved file.
        """
        if isinstance(shap_values_matrix, list):
            raise ValueError(
                f"Expected 'shap_values_matrix' to be a numpy array of shape (n_samples, #features), but got a list instead: {shap_values_matrix}"  # pylint: disable=line-too-long
            )
        filename = self._create_filename(name=name, ext="csv")

        n_dims = shap_values_matrix.shape[1]
        df = pd.DataFrame(data=shap_values_matrix, index=ids, columns=range(n_dims))

        print(f"Saving SHAP values to: {filename}")
        df.to_csv(filename)

        return filename

    def save_to_npz(self, name: str, verbose=True, **kwargs):
        """Save kwargs to numpy compressed npz file. Transforms everything into numpy arrays."""
        filename = self._create_filename(name=name, ext="npz")
        if verbose:
            print(f"Saving SHAP values to: {filename}")
        np.savez_compressed(
            file=filename,
            **kwargs,  # type: ignore
        )

    @staticmethod
    def load_from_csv(path: Path | str) -> pd.DataFrame:
        """Return pandas dataframe of shap values for loaded file."""
        return pd.read_csv(path, index_col=0)


class NN_SHAP_Handler:
    """Handle shap computations and data saving/loading."""

    def __init__(self, model: LightningDenseClassifier, logdir: Path | str):
        self.model = model
        self.model.eval()
        self.model_classes = list(self.model.mapping.items())
        self.logdir = logdir
        self.saver = SHAP_Saver(logdir=logdir)

    def compute_shaps(
        self,
        background_dset: SomeData,
        evaluation_dset: SomeData,
        save=True,
        name="",
        num_workers: int = 4,
    ) -> Tuple[shap.DeepExplainer, List[np.ndarray]]:
        """Compute shap values of deep learning model on evaluation dataset
        by creating an explainer with background dataset.

        Returns explainer and shap values (as a list of matrix per class)
        """
        explainer = shap.DeepExplainer(
            model=self.model, data=torch.from_numpy(background_dset.signals).float()
        )
        if save:
            self.saver.save_to_npz(
                name=name + "_explainer_background",
                background_md5s=background_dset.ids,
                background_expectation=explainer.expected_value,  # type: ignore
                classes=self.model_classes,
            )

        signals = torch.from_numpy(evaluation_dset.signals).float()
        shap_values = NN_SHAP_Handler._compute_shap_values_parallel(
            explainer, signals, num_workers
        )

        if save:
            self.saver.save_to_npz(
                name=name + "_evaluation",
                evaluation_md5s=evaluation_dset.ids,
                shap_values=shap_values,
                classes=self.model_classes,
            )

        return explainer, shap_values  # type: ignore

    @staticmethod
    def _compute_shap_values_parallel(
        explainer: shap.DeepExplainer,
        signals: torch.Tensor,
        num_workers: int,
    ) -> List[np.ndarray]:
        """Compute SHAP values in parallel using a ThreadPoolExecutor.

        Args:
            explainer (shap.DeepExplainer): The SHAP explainer object used for computing SHAP values.
            signals (torch.Tensor): The evaluation dataset samples as a torch Tensor of shape (#samples, #features).
            num_workers (int): The number of parallel threads to use for computation.

        Returns:
            List[np.ndarray]: A list of SHAP values matrices (one per output class) of shape (#samples, #features).
        """
        signal_chunks = torch.tensor_split(signals, num_workers)

        def worker(chunk: torch.Tensor) -> np.ndarray:
            explainer_copy = copy.deepcopy(explainer)
            return explainer_copy.shap_values(chunk)  # type: ignore

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            shap_values_chunks = list(executor.map(worker, signal_chunks))

        shap_values = [
            np.concatenate([chunk[i] for chunk in shap_values_chunks], axis=0)
            for i in range(len(shap_values_chunks[0]))
        ]

        return shap_values


class LGBM_SHAP_Handler:
    """Handle shap computations and data saving/loading."""

    def __init__(self, model_analyzer: EstimatorAnalyzer, logdir: Path | str):
        self.logdir = logdir
        self.saver = SHAP_Saver(logdir=logdir)
        self.model_classes = list(model_analyzer.mapping.items())
        self.model: LGBMClassifier = LGBM_SHAP_Handler._check_model_is_lgbm(
            model_analyzer
        )

    @staticmethod
    def _check_model_is_lgbm(model_analyzer: EstimatorAnalyzer) -> LGBMClassifier:
        """Return lightgbm classifier if found, else raise ValueError."""
        model = model_analyzer.classifier
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]
        if not isinstance(model, LGBMClassifier):
            raise ValueError(
                f"Expected model to be a lightgbm classifier, but got {model} instead."
            )
        return model

    def compute_shaps(
        self,
        background_dset: SomeData,
        evaluation_dset: SomeData,
        save=True,
        name="",
        num_workers: int = 4,
    ) -> Tuple[List[np.ndarray], shap.TreeExplainer]:
        """Compute shap values of lgbm model on evaluation dataset.

        Returns shap values and explainer
        """
        explainer = shap.TreeExplainer(
            model=self.model,
            data=background_dset.signals,
            model_output="raw",
            feature_perturbation="interventional",
        )

        if save:
            self.saver.save_to_npz(
                name=name + "_explainer_background",
                background_md5s=background_dset.ids,
                background_expectation=explainer.expected_value,  # type: ignore
                classes=self.model_classes,
            )

        shap_values = LGBM_SHAP_Handler._compute_shap_values_parallel(
            explainer=explainer,
            signals=evaluation_dset.signals,
            num_workers=num_workers,
        )

        if save:
            self.saver.save_to_npz(
                name=name + "_evaluation",
                evaluation_md5s=evaluation_dset.ids,
                shap_values=shap_values,
                expected_value=explainer.expected_value,
                classes=self.model_classes,
            )

        return shap_values, explainer

    @staticmethod
    def _compute_shap_values_parallel(
        explainer: shap.TreeExplainer,
        signals: ArrayLike,
        num_workers: int,
    ) -> List[np.ndarray]:
        # Split the signals into chunks for parallel processing
        signal_chunks = np.array_split(signals, num_workers)

        # Worker function
        def worker(chunk):
            explainer_copy = copy.deepcopy(explainer)
            return explainer_copy.shap_values(X=chunk)

        # Use ThreadPoolExecutor to compute shap_values in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            shap_values_chunks = list(executor.map(worker, signal_chunks))

        # Concatenate the chunks
        if isinstance(shap_values_chunks[0], np.ndarray):  # binary case
            shap_values = list(np.concatenate(shap_values_chunks, axis=0))
        else:  # multiclass case
            shap_values = [
                np.concatenate([chunk[i] for chunk in shap_values_chunks], axis=0)
                for i in range(len(shap_values_chunks[0]))
            ]

        return shap_values


class SHAP_Analyzer:
    """SHAP_Analyzer class for analyzing SHAP values of a model.

    Attributes:
        model: The trained model.
        explainer: The SHAP explainer object used to compute SHAP values.
    """

    def __init__(self, model: LightningDenseClassifier, explainer: shap.DeepExplainer):
        self.model = model
        self.explainer = explainer

    def verify_shap_values_coherence(
        self, shap_values: List, dset: SomeData, tolerance=1e-6  # type: ignore
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
            # shap_values_class.sum(axis=1).shape = (n_samples,)
            shap_sum[:, i] = (
                shap_values_class.sum(axis=1) + self.explainer.expected_value[i]  # type: ignore
            )

        # Compute the model's output probabilities for the samples
        signals = torch.from_numpy(dset.signals).float()
        model_output_logits = self.model(signals).detach()
        probs = F.softmax(model_output_logits, dim=1).detach().numpy()
        shap_to_prob = (
            F.softmax(torch.from_numpy(shap_sum).float(), dim=1)
            .sum(dim=1)
            .detach()
            .numpy()
        )
        print(
            f"Verifying model output: Sum close to 1? {np.all(1 - probs.sum(axis=1) <= tolerance)}"
        )
        print(
            f"Verifying SHAP output: Sum close to 1 (w expected value)? {np.all(1 - shap_to_prob <= tolerance)}"
        )
        print(f"Shap sum shape, detailling all classes: {shap_sum.shape}")
        total = shap_sum.sum(axis=1)
        print(
            f"Sum of all shap values across classes, for {total.shape} samples: {total}\n"
        )
        # Compare the sum of SHAP values with the model's output probabilities
        diff = np.abs(shap_sum - model_output_logits.numpy())
        coherent = np.all(diff <= tolerance)
        print(
            f"Detailled values for shap sum, model preds and diff:\n {shap_sum}\n{model_output_logits}\n{diff}"
        )
        if not coherent:
            problematic_samples = np.argwhere(diff > tolerance)
            print(f"SHAP values are not coherent for samples:\n {problematic_samples}")

        return coherent

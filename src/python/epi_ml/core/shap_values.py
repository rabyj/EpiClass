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

from epi_ml.core.model_pytorch import LightningDenseClassifier
from epi_ml.core.types import SomeData
from epi_ml.utils.time import time_now_str

# from numpy.typing import ArrayLike


class SHAP_Handler:
    """Handle shap computations and data saving/loading."""

    def __init__(self, model: LightningDenseClassifier, logdir: Path | str):
        self.model = model
        self.model.eval()
        self.model_classes = list(self.model.mapping.items())
        self.logdir = logdir
        self.filename_template = "shap_{name}_{time}.{ext}"

    def _create_filename(self, ext: str, name="") -> Path:
        """Create a filename with the given extension and name, and a timestamp."""
        filename = self.filename_template.format(name=name, ext=ext, time=time_now_str())
        filename = Path(self.logdir) / filename
        return filename

    def compute_NN(
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
            np.savez_compressed(
                file=self._create_filename(
                    name=name + "_explainer_background", ext="npz"
                ),
                background_md5s=background_dset.ids,
                background_expectation=explainer.expected_value,  # type: ignore
                classes=self.model_classes,
            )

        signals = torch.from_numpy(evaluation_dset.signals).float()
        shap_values = self.compute_shap_values_parallel(explainer, signals, num_workers)

        if save:
            np.savez_compressed(
                file=self._create_filename(name=name + "_evaluation", ext="npz"),
                evaluation_md5s=evaluation_dset.ids,
                shap_values=shap_values,
                classes=self.model_classes,
            )

        return explainer, shap_values  # type: ignore

    @staticmethod
    def compute_shap_values_parallel(
        explainer: shap.DeepExplainer,
        signals: torch.Tensor,
        num_workers,
    ) -> List[np.ndarray]:
        """Compute SHAP values in parallel using a ThreadPoolExecutor.

        Args:
            explainer (shap.DeepExplainer): The SHAP explainer object used for computing SHAP values.
            signals (torch.Tensor): The evaluation dataset samples as a torch Tensor of shape (#samples, #features).
            num_workers (int, optional): The number of parallel threads to use for computation. Defaults to 4.

        Returns:
            List[np.ndarray]: A list of SHAP values matrices (one per output class) of shape (#samples, #features).
        """
        signal_chunks = torch.tensor_split(signals, num_workers)

        def worker(chunk: torch.Tensor) -> np.ndarray:
            explainer_copy = copy.deepcopy(explainer)
            return explainer_copy.shap_values(chunk)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            shap_values_chunks = list(executor.map(worker, signal_chunks))

        shap_values = [
            np.concatenate([chunk[i] for chunk in shap_values_chunks], axis=0)
            for i in range(len(shap_values_chunks[0]))
        ]

        return shap_values

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
                shap_values_class.sum(axis=1) + self.explainer.expected_value[i]
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

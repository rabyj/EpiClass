"""Module containing utility functions for shap analysis"""
# pylint: disable=use-dict-literal
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from scipy.special import softmax  # type: ignore


def select_shap_samples(shap_dict, n: int) -> Dict[str, List[np.ndarray]]:
    """Return a subset of shap values and their ids."""
    selected_shap_samples = {"shap": [], "ids": []}
    total_samples = len(shap_dict["ids"])
    selected_indices = np.random.choice(total_samples, n, replace=False)

    for class_shap_values in shap_dict["shap"]:
        selected_shap_samples["shap"].append(class_shap_values[selected_indices, :])

    selected_shap_samples["ids"] = [shap_dict["ids"][idx] for idx in selected_indices]

    return selected_shap_samples


def average_impact(shap_values_matrices):
    """Return average absolute shap values."""
    shap_abs = np.zeros(shap_values_matrices[0].shape)
    for matrix in shap_values_matrices:
        shap_abs += np.absolute(matrix)
    shap_abs /= len(shap_values_matrices)
    return shap_abs


def plot_feature_importance(
    sample_shap_values: np.ndarray,
    important_features: list,
    title: str,
    plot_type: str,
    logdir: str | Path,
) -> None:
    """Plot feature importance in a sample, highlighting important features using Plotly.

    Args:
        sample_shap_values (np.ndarray): The SHAP values for a single sample.
        important_features (list): List of indices corresponding to important features.
        title (str): The title for the plot.
        plot_type (str): Type of plot ("raw", "softmax", or "rank").
    """

    if plot_type == "raw":
        plot_values = sample_shap_values
    elif plot_type == "softmax":
        plot_values = softmax(sample_shap_values)
    elif plot_type == "rank":
        plot_values = np.argsort(
            np.argsort(-np.abs(sample_shap_values))
        )  # Rank based on absolute values
    else:
        raise ValueError("Invalid plot_type.")

    title = f"{title} ({plot_type})"
    # General points
    trace1 = go.Scatter(
        x=list(range(len(plot_values))),
        y=plot_values,
        mode="markers",
        marker=dict(color="blue"),
        name="All Features",
    )

    # Important points
    trace2 = go.Scatter(
        x=important_features,
        y=[plot_values[i] for i in important_features],
        mode="markers",
        marker=dict(color="red"),
        name="Important Features",
    )

    layout = go.Layout(
        title=title, xaxis=dict(title="Feature index"), yaxis=dict(title=plot_type)
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    logdir = Path(logdir)
    fig.write_image(logdir / f"{title}.png")


def n_most_important_features(sample_shaps, n):
    """Return features with highest absolute shap values."""
    return np.flip(np.argsort(np.absolute(sample_shaps)))[:n]


# def verify_subsample_coherence(
#     shap_matrices: np.ndarray, chosen_idxs: List[int], class_int: int
# ) -> None:
#     """Verify if the subsampling is coherent with the SHAP values.

#     This function calculates the mean absolute SHAP values for the samples
#     identified by chosen_idxs in each class' SHAP matrix. It then checks
#     if the class index for which the subsampling was done (class_int) has
#     the highest mean absolute SHAP value.

#     Args:
#         shap_matrices (np.ndarray): Array of SHAP matrices for each class.
#         chosen_idxs (List[int]): Indices of the samples chosen during subsampling.
#         class_int (int): The class index for which the subsampling was performed.

#     Returns:
#         None: Prints out the results.
#     """

#     # Calculate the mean of absolute SHAP values for each class for selected samples.
#     avg_abs_shap_per_class = [
#         np.mean(np.abs(shap_matrices[i][chosen_idxs, :]))
#         for i in range(len(shap_matrices))
#     ]

#     # Find the index of the class with highest average absolute SHAP value
#     highest_shap_class_idx = np.argmax(avg_abs_shap_per_class)

#     print(f"Average absolute SHAP values per class: {avg_abs_shap_per_class}")
#     print(f"Class with highest average absolute SHAP value: {highest_shap_class_idx}")

#     # Compare the index with class_int to check if they are same.
#     if highest_shap_class_idx == class_int:
#         print(
#             f"The subsampling for class index {class_int} is coherent with SHAP values."
#         )
#     else:
#         print(
#             f"Warning: The subsampling for class index {class_int} may not be coherent with SHAP values. Highest SHAP values belong to class index {highest_shap_class_idx}."
#         )

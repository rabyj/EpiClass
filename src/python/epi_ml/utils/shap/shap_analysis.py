"""Module for more complex SHAP analysis functions.

Need to disable pylint unsubscriptable-object because of incorrect report in pandas.
see: https://github.com/pylint-dev/pylint/issues/3637
"""
# pylint: disable=unsubscriptable-object
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display
from scipy.special import softmax  # type: ignore

DECILES = list(np.arange(10, 100, 10) / 100)


def feature_overlap_stats(
    feature_lists: List[List[int]], percentile_list: list[int | float]
) -> Tuple[Set[int], Set[int], Dict[int | float, List[int]], go.Figure]:
    """
    Calculate the statistics of feature overlap between multiple feature lists.

    This function takes a list of feature lists and computes feature frequency percentiles.
    It also computes the union and intersection of all features from the given feature lists.

    Args:
        feature_lists (List[List[int]]): A list of feature lists, where each inner list contains feature indices.
        percentile_list (List[int|float]: The percentile values for which the most frequent features will be returned.

    Returns:
        Tuple[Set[int], Set[int], Dict[int|float, List], go.Figure]: A tuple containing
        1) intersection of all features
        2) union of all features
        3) a dict containing the list of features present in each percentile.
        4) a plotly figure showing the histogram of feature frequency
    """
    nb_files = len(feature_lists)
    if not feature_lists:
        raise ValueError("Input list must not be empty.")

    for percentile in percentile_list:
        if percentile < 0 or percentile > 100:
            raise ValueError("Percentile values must be between 0 and 100.")

    # Most frequent features (per percentile)
    feature_counter = Counter()
    for feature_list in feature_lists:
        feature_counter.update(feature_list)

    df = pd.DataFrame.from_dict(data=feature_counter, orient="index").reset_index()
    df.columns = ["Feature", "Count"]

    # Histogram of feature frequency
    nb_features = len(feature_counter)
    nbins = int(np.sqrt(nb_features))
    hist = px.histogram(
        df,
        x="Count",
        title=f"Top N features: frequency of {nb_features} features in {nb_files} files",
        nbins=nbins,
        range_x=[0, nb_files],
    )
    hist.update_layout(xaxis_title="Nb files", yaxis_title="Feature count")

    # Feature frequency stats
    describe_percentiles = sorted([0.25, 0.5, 0.75] + [p / 100 for p in percentile_list])
    count_stats = pd.DataFrame(
        df["Count"].describe(percentiles=describe_percentiles)
    )  # pylint: disable=unsubscriptable-object
    count_stats["% of files"] = count_stats["Count"] / nb_files * 100
    count_stats["% of files"]["count"] = "nan"
    display(count_stats)

    percentile_features_dict = {}
    for percentile in percentile_list:
        # Calculate percentile count value, then select all features >= current percentile
        curr_perc = nb_files * percentile / 100
        features_above_perc = df[df["Count"] >= curr_perc]
        percentile_features_dict[percentile] = features_above_perc["Feature"].tolist()

    # Union and intersection of all features
    all_features_union: Set[int] = set()
    all_features_intersection: Set[int] = set(feature_lists[0])
    for feature_set in feature_lists:
        all_features_union.update(feature_set)
        all_features_intersection &= set(feature_set)

    return all_features_intersection, all_features_union, percentile_features_dict, hist  # type: ignore


def print_feature_overlap_stats(feature_stats: Sequence):
    """Prints the statistics of feature overlap.

    This function receives the feature statistics which include the intersection,
    union and frequent features in each quantile of features. It then prints
    these statistics for easy inspection.

    Args:
        feature_stats (Sequence): Tuple containing the intersection, union and
                                  frequent features in each quantile of features.
    """
    features_intersection, features_union, frequent_features, _ = feature_stats
    print(f"Intersection of all features: {len(features_intersection)} features")
    print(f"Fully intersecting features: {list(features_intersection)}")
    print(f"Union of all features: {len(features_union)} features\n")
    for k, v in frequent_features.items():
        print(f"Most frequent features in {k}th quantile: {len(v)} features")


def print_importance_info(feature_selection: List[int], shap_matrix: np.ndarray):
    """Prints the feature importance information.

    This function prints the feature importance information, which includes the
    average expected contribution of the selected features and one feature (if
    the importance was uniform), and statistical descriptions of the contributions
    of the selected features.

    Args:
        feature_selection (List[int]): The indices of the selected features.
        shap_matrix (np.ndarray): The SHAP values matrix.
    """
    N = len(feature_selection)
    nb_files, nb_bins = shap_matrix.shape
    print(
        f"Average expected contribution of {N} feature if uniform importance:{N/nb_bins*100:.5f}%"
    )
    print(
        f"Average expected contribution of 1 feature if uniform importance:{1/nb_bins*100:.5f}%"
    )
    print(f"Average contribution of selected features for {nb_files} files:")
    display(
        pd.DataFrame(
            softmax(shap_matrix, axis=1)[:, list(feature_selection)].sum(axis=1) * 100
        ).describe(percentiles=DECILES)
    )
    print(f"Individual contribution of selected features for {nb_files} files:")
    display(
        pd.DataFrame(
            softmax(shap_matrix, axis=1)[:, list(feature_selection)] * 100
        ).describe(percentiles=DECILES)
    )

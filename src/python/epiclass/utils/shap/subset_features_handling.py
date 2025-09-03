"""Module that defines functions useful for handling data subsets and their 'important' features.
Used on k-fold SHAP analysis results.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def read_all_feature_sets(jsons_parent_folder: Path) -> Dict[str, Dict]:
    """Return a dictionary containing the important feature sets for each subsampling/folder.

    Args:
        jsons_parent_folder (Path): The path to the folders containing feature_count.json files.

    Returns:
        Dict[str, Dict]: A dictionary containing the important feature fold counts for each subsampling/folder.
    """
    all_features_counts = {}
    for folder in jsons_parent_folder.iterdir():
        if not folder.is_dir():
            continue
        json_path = folder / "feature_count.json"

        with open(json_path, "r", encoding="utf8") as json_file:
            feature_count = json.load(json_file)

        all_features_counts[folder.name] = feature_count

    return all_features_counts


def exclude_track_subsamplings(all_features_counts: Dict[str, Dict]) -> Dict[str, Dict]:
    """Exclude track subsamplings from all_features_counts, except for (m)rna-seq."""
    to_remove_track_types = frozenset(["raw", "fc", "pval", "pos", "neg"])
    new_features_counts = {
        folder_name: feature_count
        for folder_name, feature_count in all_features_counts.items()
        if folder_name.split("_")[-1] not in to_remove_track_types
    }
    return new_features_counts


def filter_feature_sets(
    all_features_counts: Dict[str, Dict], minimum_count: int = 8
) -> Dict[str, Dict[str, Set[int]]]:
    """
    Return a dictionary containing the features that meet the minimum count threshold for each
    output class of each subsampling.

    Args:
        all_features_counts (Dict[str, Dict]): A dictionary containing the feature counts for each subsampling/folder.
        minimum_count (int, optional): The minimum count threshold for features to be included. Defaults to 8.

    Returns:
        Dict[str, Dict[str, Set[int]]]: A dictionary containing the features that meet the minimum count threshold for each
        output class of each subsampling.
    """
    filtered_class_features = defaultdict(dict)
    for subsampling_name, subsampling_feature_count in all_features_counts.items():
        for class_label, class_features in subsampling_feature_count.items():
            filtered_class_features[subsampling_name][class_label] = frozenset(
                [feature for feature, count in class_features if count >= minimum_count]
            )
    return dict(filtered_class_features)


def aggregate_feature_sets(
    filtered_class_features: Dict[str, Dict[str, Set[int]]], verbose: bool
) -> Dict[str, Set[int]]:
    """
    Aggregates and organizes feature sets from various sources based on output classes and subsamplings.

    Create union of feature sets.
    - For a given subsampling, take the union of features for each output class.
    - For a given output class, take the union of features for each assay.
    - Create global union (diagonal in intersection matrix) feature set.

    Args:
        filtered_class_features (Dict[str, Dict[str, Set[int]]]): A dictionary containing the features that meet the minimum count threshold for each
        output class of each subsampling.
        verbe (bool): The minimum count threshold for features to be included. Defaults to 8.

    Returns:
        Dict[str, Set[int]]: A dictionary containing the aggregated feature sets.
    """
    # for a given subsampling, merge features for each output class
    # for a given output class, merge features for each subsampling
    merged_output_classes = defaultdict(set)
    merged_subsamplings = defaultdict(set)
    for subsampling_name, features_per_class in filtered_class_features.items():
        for output_class, features in features_per_class.items():
            merged_output_classes[subsampling_name].update(features)
            merged_subsamplings[output_class].update(features)

    # create global union feature set
    global_union = set()
    for features in merged_output_classes.values():
        global_union.update(features)

    if verbose:
        print(f"\nGlobal union: {len(global_union)}")

        print("\nMerged subsamplings features (features per output class):")
        for output_class, features in sorted(merged_subsamplings.items()):
            print(f"{output_class}: {len(features)}")

        print("\nMerged output classes features (features per subsampling):")
        for folder_name, features in sorted(merged_output_classes.items()):
            print(f"{folder_name}: {len(features)}")

    new_sets = {}
    for output_class, features in merged_subsamplings.items():
        new_sets[f"merge_samplings_{output_class}"] = frozenset(features)
    for folder_name, features in merged_output_classes.items():
        new_sets[f"merge_output_classes_{folder_name}"] = frozenset(features)
    new_sets["global_union"] = frozenset(global_union)

    return new_sets


def flatten_feature_sets(
    feature_sets: Dict[str, Dict[str, Set[int]]]
) -> Dict[str, Set[int]]:
    """
    Flatten the feature sets from feature_sets into a single layer/depth. Merges the key names

    Args:
        feature_sets (Dict[str, Dict[str, Set[int]]]): A dictionary containing the feature sets with new key names.

    Returns:
        Dict[str, Set[int]]: A dictionary containing the flattened feature sets.
    """
    flattened_dict = {}
    for subsampling_name, features_by_class in feature_sets.items():
        for class_name, features in features_by_class.items():
            flattened_dict[f"{subsampling_name} & {class_name}"] = frozenset(features)
    return flattened_dict


def process_all_subsamplings(
    jsons_parent_folder: Path, aggregate: bool, minimum_count: int, verbose: bool
) -> Dict[str, Set[int]]:
    """Process all subsamplings and aggregate feature sets (see aggregate_feature_sets).

    Args:
        jsons_parent_folder (Path): The path to the folders containing feature_count.json files.
        aggregate (bool): Whether to aggregate feature sets.
        minimum_count (int): The minimum count threshold for features to be included.
        verbose (bool): Whether to print verbose output.

    Returns:
        Dict[str, Set[int]]: A dictionary containing the desired feature sets.
    """
    all_feature_sets = read_all_feature_sets(jsons_parent_folder)
    all_feature_sets = exclude_track_subsamplings(all_feature_sets)
    filtered_class_features = filter_feature_sets(
        all_feature_sets, minimum_count=minimum_count
    )
    flat_feature_sets = flatten_feature_sets(filtered_class_features)

    if aggregate:
        agg_sets = aggregate_feature_sets(filtered_class_features, verbose=verbose)
        return {**agg_sets, **flat_feature_sets}

    return flat_feature_sets


def collect_features_from_feature_count_file(
    path: str | Path, n: int = 8
) -> Dict[str, List[int]]:
    """Collect features from feature count file that are present in at least n splits.

    Returns:
        selected_features (Dict[str, List[int]]): A dictionary where keys are classifer output classes and values are lists of features.
    """
    feature_count_path = Path(path)
    with open(feature_count_path, "r", encoding="utf8") as f:
        feature_count_input = json.load(f)

    selected_features = defaultdict(set)
    for output_class, feature_counts in feature_count_input.items():
        for feature_idx, count in feature_counts:
            if count >= n:
                selected_features[output_class].add(feature_idx)

    selected_features = {k: sorted(v) for k, v in selected_features.items()}
    return selected_features


def collect_all_features_from_feature_count_file(
    path: str | Path, n: int = 8
) -> List[int]:
    """Collect all features from feature count file that are present in at least n splits.

    Returns:
        List[int]: A sorted list of all features present in the feature count file.
    """
    selected_features = collect_features_from_feature_count_file(path, n)
    all_features = set()
    for features in selected_features.values():
        all_features.update(features)
    return sorted(all_features)

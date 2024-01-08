"""
For every folder of a cross-validation, take already computed SHAP values and analyze them.

The following analyses are performed:
- Writing bed files of most frequent features in the N most high SHAP values (absolute value).
  e.g for N=100, the top 100 most high SHAP values (absolute value) are taken for each sample,
  counted, and features that are present in X% of the samples are written to bed.
  This is done for each class separately, using the output class SHAP matrix.
  (There are SHAP values for all features, for each class, for each sample, thus multiple matrices, one per class.)
    - For all samples of the classifier class
    - Subsampled by assay
    - Subsampled by cell type and assay
"""
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
from __future__ import annotations

import argparse
import copy
import itertools
import json
import multiprocessing
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.data_source import HDF5_RESOLUTION, EpiDataSource
from epi_ml.core.metadata import Metadata
from epi_ml.utils.bed_utils import bins_to_bed_ranges, write_to_bed
from epi_ml.utils.general_utility import get_valid_filename
from epi_ml.utils.metadata_utils import count_combinations
from epi_ml.utils.shap.shap_analysis import feature_overlap_stats
from epi_ml.utils.shap.shap_utils import (
    extract_shap_values_and_info,
    get_shap_matrix,
    n_most_important_features,
)
from epi_ml.utils.time import time_now

CELL_TYPE = "harmonized_sample_ontology_intermediate"
ASSAY = "assay_epiclass"
TRACK = "track_type"


def analyze_shap_fold(
    extract_shap_values_and_info_output: Tuple[
        np.ndarray, List[str], List[Tuple[str, str]]
    ],
    output_folder: Path,
    metadata: Metadata,
    label_category: str,
    chromsizes: List[Tuple[str, int]],
    resolution: int,
    top_N_required: int = 100,
    min_percentile: float = 80,
    overwrite: bool = False,
    copy_metadata: bool = True,
) -> Dict[str, Dict]:
    """
    Analyzes SHAP values from one fold of the training and writes BED files of most frequent important features.

    Args:
        extract_shap_values_and_info_output (Tuple[np.ndarray, List[str], List[Tuple[str, str]]]): Output of extract_shap_values_and_info.
        output_folder (Path): The directory to write the results to.
        metadata (Metadata): Metadata object containing label information.
        label_category (str): The category of labels to consider.
        chromsizes (List[Tuple[str, int]]): List with chromosome names and sizes.
        resolution (int): The resolution for binning.
        top_N_required (int, optional): The number of top SHAP values/features to consider per sample. Defaults to 100.
        min_percentile (float, optional): The percentile value for feature frequency selection (0 < x < 100). Defaults to 80.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.

    Returns:
        Dict[str, Dict]: A dictionary containing important features for each analyzed class label.
        The keys are class labels, and values are dictionaries of important features for different percentiles.
    """
    shap_matrices, eval_md5s, classes = extract_shap_values_and_info_output

    if copy_metadata:
        meta = copy.deepcopy(metadata)
    else:
        meta = metadata

    # Filter metadata to include only the samples that exist in the SHAP value archives
    for md5 in list(meta.md5s):
        if md5 not in set(eval_md5s):
            del meta[md5]

    # Since metadata gets modified in get_shap_matrix, instead
    # of copying the metadata with deepcopy internally, we just reload it
    # from the saved file beforehand, avoids many copies
    saved_eval_meta_file = (
        tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
            mode="wb", delete=False
        )
    )
    meta.save_marshal(saved_eval_meta_file.name)
    saved_eval_meta_file.close()

    if len(meta) < 5:
        print(f"Not enough samples (5) to perform analysis on {label_category}.")
        return {}
    if all(count < 5 for count in meta.label_counter(label_category).values()):
        print(f"Not enough samples (5) to perform analysis on {label_category}.")
        return {}

    # Loop over each class to perform SHAP value analysis
    important_features = {}
    for class_int, class_label in classes:
        class_int = int(class_int)
        print(f"\n\nClass: {class_label} ({class_int})")

        meta = Metadata.from_marshal(saved_eval_meta_file.name)

        # Get the SHAP matrix for the current class,
        # and only select samples that also correspond to that class
        shap_matrix, chosen_idxs = get_shap_matrix(
            meta=meta,
            shap_matrices=shap_matrices,
            eval_md5s=eval_md5s,
            label_category=label_category,
            selected_labels=[class_label],
            class_idx=class_int,
            copy_meta=False,
        )

        # this check should only be done one time but complicated to do so
        if shap_matrix.shape[1] < top_N_required:
            raise ValueError(
                f"top_N_shap was higher than the number of features per sample: {shap_matrix.shape[1]} < {top_N_required}"
            )

        if len(chosen_idxs) < 5:
            print(f"Not enough samples (5) to perform analysis on {class_label}.")
            continue

        result_bed_filename = get_valid_filename(
            f"frequent_features_f{min_percentile:.2f}_{class_label}.bed"
        )
        if not overwrite and (output_folder / result_bed_filename).is_file():
            print(f"Skipping {class_label} because {result_bed_filename} already exists.")
            continue

        # Computing statistics of feature overlap
        print(
            f"Selecting features with top {top_N_required} SHAP values for each sample of {class_label}."
        )
        top_n_features = []
        for sample in shap_matrix:
            top_n_features.append(list(n_most_important_features(sample, top_N_required)))

        interesting_quantiles = list(set([min_percentile, 90, 95, 99]))
        (
            _,
            _,
            frequent_features,
            hist_fig,
        ) = feature_overlap_stats(top_n_features, interesting_quantiles)

        important_features[class_label] = frequent_features

        hist_fig.write_image(
            file=output_folder
            / f"top{top_N_required}_feature_frequency_{class_label}.png",
            format="png",
        )

        feature_selection = frequent_features[min_percentile]

        # Convert bin indices to genomic ranges and write to a BED file
        bed_vals = bins_to_bed_ranges(
            sorted(feature_selection), chromsizes, resolution=resolution
        )
        write_to_bed(
            bed_vals,
            output_folder / result_bed_filename,
            verbose=True,
        )

    # Save important features for all classes as json
    # TODO: FIX THIS, DO NOT OVERWRITE OLD JSON
    json_path = output_folder / "important_features.json"
    with open(json_path, "w", encoding="utf8") as json_file:
        json.dump(important_features, json_file, indent=4)

    return important_features


def compare_kfold_shap_analysis(
    important_features_all_splits: Dict[str, Dict],
    resolution: int,
    chromsizes: List[Tuple[str, int]],
    output_folder: Path,
    name: str,
    chosen_percentile: float | int = 80,
    minimum_count: int = 8,
) -> Dict[str, List[Tuple[str, List[int]]]]:
    """Compare the SHAP analysis results from multiple splits and writes the results based on frequency.

    Args:
        important_features_all_splits (Dict[str, Dict]): Dictionary containing important features for each split.
        resolution (int): Resolution for binning.
        chromsizes (List[Tuple[str, int]]): List with chromosome names and sizes.
        output_folder (Path): Output directory for writing BED files.
        name (str): Name of the analysis.
        chosen_percentile (float | int, optional): The chosen percentile for selecting important features.
        minimum_count (int, optional): The minimum count of splits that a feature must be present in.

    Returns:
        Dict[str, List[Tuple[str, List[int]]]]: A dict containing the frequency of features for each class over all splits, for all given percentiles.
    """
    class_features_frequency: Dict[str, List] = {}
    class_labels = set()
    for split_dict in important_features_all_splits.values():
        class_labels.update(list(split_dict.keys()))

    percentile_labels = set()
    for split_dict in important_features_all_splits.values():
        percentile_labels.update(
            class_dict.keys() for class_dict in list(split_dict.values())
        )

    if str(chosen_percentile) not in percentile_labels:
        raise ValueError(
            f"Chosen percentile {chosen_percentile} not found in {percentile_labels}"
        )

    for class_label in class_labels:
        feature_counter = Counter()

        # Count the occurrence of each feature across all splits
        for features_dict in important_features_all_splits.values():
            current_features = features_dict.get(class_label, {}).get(
                str(chosen_percentile), []
            )
            feature_counter.update(current_features)

        class_features_frequency[class_label] = feature_counter.most_common()

    # Select features present in at least a certain count of splits (e.g., 8 out of 10)
    for class_label, feature_count_list in class_features_frequency.items():
        selected_features = {
            feature for feature, count in feature_count_list if count >= minimum_count
        }

        if not selected_features:
            print(
                f"No features meeting the required count for class {class_label}",
                file=sys.stderr,
            )
            continue

        bed_vals = bins_to_bed_ranges(
            sorted(selected_features), chromsizes, resolution=resolution
        )

        bed_filename = get_valid_filename(
            f"selected_features_{name}_f{chosen_percentile:.2f}_count{minimum_count}_{class_label}.bed"
        )

        output_folder.mkdir(exist_ok=True, parents=True)
        write_to_bed(
            bed_vals,
            output_folder / bed_filename,
            verbose=True,
        )

    return class_features_frequency


def analyze_subsamplings(
    shap_folder: Path,
    output_folder: Path,
    metadata: Metadata,
    chromsizes: List[Tuple[str, int]],
    label_category: str,
    subsample_categories: List[List[str]],
    resolution: int,
    top_N_required: int = 100,
    min_percentile: float = 80,
    overwrite: bool = False,
) -> None:
    """
    Analyzes SHAP values for given subsampling category combinations for an individual split.

    Args:
        shap_folder (Path): Path to the folder containing SHAP values.
        output_folder (Path): Path to the folder where analysis results will be stored.
        metadata (Metadata): Metadata object containing sample information.
        chromsizes: List[Tuple[str, int]]: Chromosome sizes information.
        label_category (str): Primary category for labels.
        subsample_categories (List[List[str]]): List of category combinations for subsampling.
        resolution (int): Resolution of the sample binning.
        top_N_required (int): Number of top features required. Defaults to 100.
        min_percentile (float): Minimum percentile for filtering over samples (0 < val < 100). Defaults to 80.
        overwrite (bool): Flag to overwrite existing data. Defaults to False.
    """
    # Only need to extract shap values from archives one time per split, otherwise it is VERY redundant
    extract_shap_values_and_info_output = extract_shap_values_and_info(
        shap_logdir=shap_folder,
        verbose=False,
    )

    # Create a temporary metadata file to save the current metadata state, avoids further copies
    saved_meta_file = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        mode="wb", delete=False
    )
    metadata.save_marshal(saved_meta_file.name)
    saved_meta_file.close()

    # To ignore invalid subsamplings
    valid_assay_track_combos = set(count_combinations(metadata, [ASSAY, TRACK]).keys())

    for categories in subsample_categories:
        # No subsampling case
        if not categories:
            sub_output_folder = output_folder / "mixed_samples"
            sub_output_folder.mkdir(exist_ok=True)

            if not overwrite and any(sub_output_folder.glob("*")):
                print(f"Skipping analysis for {sub_output_folder} as it is not empty")
                continue

            meta = Metadata.from_marshal(saved_meta_file.name)

            _ = analyze_shap_fold(
                extract_shap_values_and_info_output=extract_shap_values_and_info_output,
                output_folder=sub_output_folder,
                metadata=meta,
                label_category=label_category,
                chromsizes=chromsizes,
                resolution=resolution,
                top_N_required=top_N_required,
                min_percentile=min_percentile,
                copy_metadata=False,
            )

            continue

        # Subsampling case, since 'categories' is not empty
        assay_index = categories.index(ASSAY) if ASSAY in categories else None
        track_index = categories.index(TRACK) if TRACK in categories else None
        combo_labels = [list(metadata.label_counter(cat).keys()) for cat in categories]

        for label_combo in itertools.product(*combo_labels):
            # Ignore invalid subsamplings
            if assay_index is not None and track_index is not None:
                if (
                    label_combo[assay_index],
                    label_combo[track_index],
                ) not in valid_assay_track_combos:
                    continue

            combo_folder_name = "_".join(label_combo)
            print(f"\n\nSubsampling: {combo_folder_name}")

            meta = Metadata.from_marshal(saved_meta_file.name)
            for cat, label in zip(categories, label_combo):
                try:
                    meta.select_category_subsets(cat, [label])
                except KeyError as exc:
                    if "categories" in str(exc) and "[]" in str(exc):
                        break  # metadata empty

            if len(meta) < 5:
                print(
                    f"Not enough samples (5) to perform analysis on '{combo_folder_name}' subsampling"
                )
                continue
            if all(count < 5 for count in meta.label_counter(label_category).values()):
                print(
                    f"Not enough samples (5) to perform analysis on '{combo_folder_name}' subsampling"
                )
                continue

            sub_output_folder = output_folder / combo_folder_name
            sub_output_folder.mkdir(exist_ok=True)

            if not overwrite and any(sub_output_folder.glob("*")):
                print(f"Skipping analysis for {sub_output_folder} as it is not empty")
                continue

            _ = analyze_shap_fold(
                extract_shap_values_and_info_output=extract_shap_values_and_info_output,
                output_folder=sub_output_folder,
                metadata=meta,
                label_category=label_category,
                chromsizes=chromsizes,
                resolution=resolution,
                top_N_required=top_N_required,
                min_percentile=min_percentile,
                copy_metadata=False,
            )


def analyze_single_fold(
    split_folder: Path,
    metadata: Metadata,
    chromsizes: List[Tuple[str, int]],
    label_category: str,
    resolution: int,
    top_N_required: int,
    min_percentile: float,
    overwrite: bool,
) -> None:
    """Analyze SHAP values for a single fold."""
    print(f"\n\nSplit: {split_folder.name}")
    # Check if split folder has shap folder
    shap_folder = split_folder / "shap"
    if not shap_folder.exists():
        print(f"Skipping {split_folder.name} because it does not have shap folder")
        return

    # Create output folder for analysis
    analysis_folder = shap_folder / f"analysis_n{top_N_required}_f{min_percentile:.2f}"
    analysis_folder.mkdir(exist_ok=True)

    # Analyze SHAP values for the whole split
    if label_category == CELL_TYPE:
        subsample_categories = [
            [],
            [ASSAY],
            [ASSAY, TRACK],
        ]
    else:
        subsample_categories = [
            [],
            [ASSAY],
            [ASSAY, TRACK],
            [ASSAY, CELL_TYPE],
            [ASSAY, CELL_TYPE, TRACK],
        ]

    analyze_subsamplings(
        shap_folder=shap_folder,
        output_folder=analysis_folder,
        metadata=metadata,
        chromsizes=chromsizes,
        label_category=label_category,
        subsample_categories=subsample_categories,
        resolution=resolution,
        top_N_required=top_N_required,
        min_percentile=min_percentile,
        overwrite=overwrite,
    )


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument(
        "category", type=str, help="The original classifier (SHAP source) output category.",
    )
    parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file.",
        )
    parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes.",
        )
    parser.add_argument(
        "base_logdir", type=DirectoryChecker(), help="Directory where different fold directories are present",
    )
    parser.add_argument(
        "--top_N_shap", type=int, default=100, help="Number of greatest SHAP values to take into account for each sample.",
        )
    parser.add_argument(
        "--min_frequency", type=float, default=0.9, help="Minimum frequency (0.9 = 90%%) of a feature over the given samples necessary for features to be selected.",
        )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files.",
        )
    # fmt: on
    return parser.parse_args()


def main():
    """Main function."""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments()

    label_category: str = cli.category

    metadata = Metadata(cli.metadata)
    metadata.remove_missing_labels(label_category)

    chromsizes: List[Tuple[str, int]] = EpiDataSource.load_external_chrom_file(
        cli.chromsize
    )
    base_logdir: Path = cli.base_logdir
    top_N_required: int = cli.top_N_shap
    min_frequency: float = cli.min_frequency
    min_percentile = min_frequency * 100
    overwrite: bool = cli.overwrite

    # Get resolution from base_logdir
    re_pattern = "|".join(list(HDF5_RESOLUTION.keys()))
    match = re.search(pattern=re_pattern, string=str(base_logdir))
    if match is None:
        raise ValueError(
            f"Could not find resolution in {base_logdir}. Expected one of {HDF5_RESOLUTION.keys()}"
        )
    resolution = HDF5_RESOLUTION[match.group(0)]

    if top_N_required < 1:
        raise ValueError(f"top_N_shap must be >= 1, got {top_N_required}")

    if min_frequency < 0 or min_frequency > 1:
        raise ValueError(f"min_frequency must be in [0,1], got {min_frequency}")

    # Find all split folders
    split_folders = [
        x for x in base_logdir.iterdir() if (x.is_dir() and x.name.startswith("split"))
    ]

    # Prepare the arguments for each worker process
    worker_args = [
        (
            split_folder,
            metadata,
            chromsizes,
            label_category,
            resolution,
            top_N_required,
            min_percentile,
            overwrite,
        )
        for split_folder in split_folders
    ]

    available_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=available_cpus) as pool:
        pool.starmap(analyze_single_fold, worker_args)

    end = time_now()
    print(f"end {end}")
    print(f"elapsed {end - begin}")


if __name__ == "__main__":
    main()

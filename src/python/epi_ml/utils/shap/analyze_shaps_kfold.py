"""
For every folder of a cross-validation, take already computed SHAP values and analyze them.

The following analyses are performed:
- Writing bed files of most frequent features in the N most high SHAP values (absolute value).
  e.g for N=100, the top 100 most high SHAP values (absolute value) are taken for each sample,
  counted, and features that are present in X% of the samples are written to bed.
  This is done for each class separately, using the output class SHAP matrix.
  (There are SHAP values for all features, for each class, for each sample, thus multiple matrices, one per class.)
"""
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.data_source import HDF5_RESOLUTION, EpiDataSource
from epi_ml.core.metadata import Metadata
from epi_ml.utils.bed_utils import bins_to_bed_ranges, write_to_bed
from epi_ml.utils.general_utility import get_valid_filename
from epi_ml.utils.shap.shap_analysis import feature_overlap_stats
from epi_ml.utils.shap.shap_utils import (
    extract_shap_values_and_info,
    get_shap_matrix,
    n_most_important_features,
)
from epi_ml.utils.time import time_now


def analyze_shap_fold(
    shap_folder: Path,
    metadata: Metadata,
    label_category: str,
    chromsizes: List[Tuple[str, int]],
    resolution: int,
    top_N_required: int = 100,
    min_percentile: float = 80,
) -> Dict[str, Dict]:
    """
    Analyzes SHAP values from one fold of the training and writes BED files of most frequent important features.

    Args:
        shap_folder (Path): The directory of the current shap values to analyze.
        metadata (Metadata): Metadata object containing label information.
        label_category (str): The category of labels to consider.
        chromsizes (List[Tuple[str, int]]): List with chromosome names and sizes.
        resolution (int): The resolution for binning.
        top_N_required (int, optional): The number of top SHAP values/features to consider per sample. Defaults to 100.
        min_percentile (float, optional): The percentile value for feature frequency selection (0 < x < 100). Defaults to 80.

    Returns:
        Dict[str, Dict]: A dictionary containing important features for each analyzed class label.
        The keys are class labels, and values are dictionaries of important features for different percentiles.
    """
    analysis_folder = shap_folder / f"analysis_n{top_N_required}_f{min_percentile:.2f}"
    analysis_folder.mkdir(exist_ok=True)
    if any(analysis_folder.glob("*")):
        print(f"Skipping {shap_folder.name} because analysis folder is not empty")
        return {}

    # Extract shap values and md5s from archive
    shap_matrices, eval_md5s, classes = extract_shap_values_and_info(
        shap_folder, verbose=True
    )

    # Filter metadata to include only the samples that exist in the SHAP value archives
    meta = copy.deepcopy(metadata)
    for md5 in list(meta.md5s):
        if md5 not in set(eval_md5s):
            del meta[md5]

    # Loop over each class to perform SHAP value analysis
    important_features = {}
    for class_int, class_label in classes:
        class_int = int(class_int)
        print(f"\n\nClass: {class_label} ({class_int})")

        # Get the SHAP matrix for the current class,
        # and only select samples that also correspond to that class
        shap_matrix, chosen_idxs = get_shap_matrix(
            meta=meta,
            shap_matrices=shap_matrices,
            eval_md5s=eval_md5s,
            label_category=label_category,
            selected_labels=[class_label],
            class_idx=class_int,
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
        if (analysis_folder / result_bed_filename).is_file():
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
            file=analysis_folder
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
            analysis_folder / result_bed_filename,
            verbose=True,
        )

    # Save important features for all classes as json
    json_path = analysis_folder / "important_features.json"
    with open(json_path, "w", encoding="utf8") as json_file:
        json.dump(important_features, json_file, indent=4)

    return important_features


def compare_kfold_shap_analysis(
    important_features_all_splits: Dict[str, Dict],
    resolution: int,
    chromsizes: List[Tuple[str, int]],
    output_folder: Path,
    chosen_percentile: float | int = 80,
    minimum_count: int = 8,
) -> None:
    """Compare the SHAP analysis results from multiple splits and writes the results based on frequency.

    Args:
        important_features_all_splits (Dict[str, Dict]): Dictionary containing important features for each split.
        resolution (int): Resolution for binning.
        chromsizes (List[Tuple[str, int]]): List with chromosome names and sizes.
        output_folder (Path): Output directory for writing BED files.
        chosen_percentile (float | int, optional): The chosen percentile for selecting important features.
        minimum_count (int, optional): The minimum count of splits that a feature must be present in.
    """
    class_features_frequency = {}
    class_labels = list(important_features_all_splits.values())[0].keys()

    for class_label in class_labels:
        feature_counter = Counter()

        # Count the occurrence of each feature across all splits
        for features_dict in important_features_all_splits.values():
            current_features = features_dict.get(class_label, {}).get(
                chosen_percentile, []
            )
            feature_counter.update(current_features)

        class_features_frequency[class_label] = feature_counter

    # Select features present in at least a certain count of splits (e.g., 8 out of 10)
    for class_label, feature_counter in class_features_frequency.items():
        selected_features = {
            feature
            for feature, count in feature_counter.items()
            if count >= minimum_count
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
            f"selected_features_f{chosen_percentile:.2f}_count{minimum_count}_{class_label}.bed"
        )

        write_to_bed(
            bed_vals,
            output_folder / bed_filename,
            verbose=True,
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

    important_features_all_splits: Dict[str, Dict] = {}
    split_folders = [
        x for x in base_logdir.iterdir() if (x.is_dir() and x.name.startswith("split"))
    ]
    for split_folder in split_folders:
        print(f"Analyzing {split_folder.name}")

        shap_folder = split_folder / "shap"
        if not shap_folder.exists():
            print(f"Skipping {split_folder.name} because it does not have shap folder")
            continue

        important_features = analyze_shap_fold(
            shap_folder=shap_folder,
            metadata=metadata,
            label_category=label_category,
            chromsizes=chromsizes,
            resolution=resolution,
            top_N_required=top_N_required,
            min_percentile=min_percentile,
        )
        important_features_all_splits[split_folder.name] = important_features

    # Save important features for all splits as json
    global_analysis_folder = base_logdir / "shap_analysis"
    json_path = global_analysis_folder / "important_features_topn{top_N_required}.json"
    with open(json_path, "w", encoding="utf8") as json_file:
        json.dump(important_features_all_splits, json_file, indent=4)

    end = time_now()
    print(f"end {end}")
    print(f"elapsed {end - begin}")


if __name__ == "__main__":
    main()

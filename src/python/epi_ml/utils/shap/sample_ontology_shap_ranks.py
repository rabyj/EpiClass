"""Create graphs of shap values ranks for certain cell types VS other cell types.

Examine:
    - ranks on important features for an assay + cell type VS other assay + cell types
    - ranks on important features for a cell type (one output class) vs othe cell types
    - both above, but for features unique to the selection
"""
# pylint: disable=too-many-nested-blocks, too-many-branches
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.typing import ArrayLike

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.metadata import Metadata
from epi_ml.utils.shap.subset_features_handling import (
    collect_features_from_feature_count_file,
)
from epi_ml.utils.time import time_now_str

CELL_TYPE = "harmonized_sample_ontology_intermediate"
ASSAY = "assay_epiclass"
TRACK = "track_type"


def parse_arguments() -> argparse.Namespace:
    """Define CLI argument parser."""
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument(
        "shap_ranks",
        type=Path,
        help="NPZ file containing combined SHAP ranks for all splits.",
    )
    parser.add_argument(
        "global_analysis_folder",
        type=DirectoryChecker(),
        help=("Folder which contains results of global k-fold SHAP analysis (important features). "
              "Children folders should be named after subsets of data (e.g. assay/track_type)")
    )
    parser.add_argument(
        "metadata",
        type=Path,
        help="A metadata JSON file.",
    )
    parser.add_argument(
        "output_folder",
        type=DirectoryChecker(),
        help="Directory where to save outputs.",
    )
    # fmt: on
    return parser.parse_args()


def compute_iqr(data: ArrayLike) -> float:
    """Calculate the interquartile range."""
    if len(data) > 0:  # type: ignore
        q75, q25 = np.percentile(a=data, q=[75, 25])  # type: ignore
        iqr = q75 - q25
    else:
        iqr = np.nan
    return iqr


def write_stats(
    output_file: Path,
    header: str,
    data: Dict[Tuple[str, str], Dict],
    features_idx: List[int],
):
    """Write stats to a file.

    Input:
    - output_file (Path): Path to the output file.
    - header (str): Header format.
    - data (Dict[Tuple, Dict]): Dictionary of data to write.
    - features_idx (List[int]): Feature indices (for data columns names).
    """
    # Create header
    header_line = (
        "Assay\tCellType\t" + "\t".join(header.format(f=f) for f in features_idx) + "\n"
    )

    # Create content lines
    content_lines = [
        f"{subset_assay}\t{subset_ct}\t"
        + "\t".join(
            f"{feature_stats[feature_idx][0]:.2f}\t{feature_stats[feature_idx][1]:.2f}"
            for feature_idx in features_idx
        )
        + "\n"
        for (subset_assay, subset_ct), feature_stats in data.items()
    ]

    with open(output_file, "w", encoding="utf8") as f:
        f.write(header_line)
        f.writelines(content_lines)


def main():
    """Main"""
    print(f"Main starting at {time_now_str()}")
    cli = parse_arguments()
    shap_ranks: Path = cli.shap_ranks
    global_analysis_folder: Path = cli.global_analysis_folder
    metadata: Metadata = Metadata(cli.metadata)
    output_folder: Path = cli.output_folder

    # Sanity check
    for path in [shap_ranks, global_analysis_folder, output_folder]:
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

    print(f"{time_now_str()} - Loading important features.")
    # Collect important features
    important_features: Dict[str, Dict[str, List[int]]] = {}
    for folder in global_analysis_folder.iterdir():
        if not folder.is_dir():
            continue

        feature_file = folder / "feature_count.json"
        if not feature_file.exists():
            raise FileNotFoundError(f"{feature_file} does not exist.")

        file_features = collect_features_from_feature_count_file(feature_file, n=8)
        important_features[folder.name] = file_features

    # Load SHAP ranks+classes
    print(f"{time_now_str()} - Loading SHAP ranks file.")
    with np.load(shap_ranks, allow_pickle=True) as f:
        rank_data = dict(f.items())

    output_classes: List[str] = [pair[1] for pair in rank_data["classes"]]
    available_md5s: Set[str] = set(rank_data["md5s"])
    class_to_idx: Dict[str, int] = {
        pair[1]: int(pair[0]) for pair in rank_data["classes"]
    }

    # Filter metadata
    for md5 in list(metadata.md5s):
        if md5 not in available_md5s:
            del metadata[md5]

    assays: List[str] = metadata.unique_classes(ASSAY)
    cell_types: List[str] = metadata.unique_classes(CELL_TYPE)

    # metadata sanity check
    if set(output_classes) != set(cell_types):
        raise ValueError(
            f"Output classes do not match cell types in metadata:\nSHAP npz:{output_classes}\nMetadata:{cell_types}"
        )
    if len(rank_data["ranks"]) != len(output_classes):
        raise ValueError(
            f"Number of classes in SHAP npz does not match number of classes in ranks:\nSHAP npz:{len(output_classes)}\nRanks:{len(rank_data['ranks'])}"
        )

    md5_sets = {assay: {cell_type: set() for cell_type in cell_types} for assay in assays}
    for md5, dset in metadata.items:
        assay = dset[ASSAY]
        cell_type = dset[CELL_TYPE]
        md5_sets[assay][cell_type].add(md5)

    # the following code won't handle rna or wgbs properly because the track type matters more there
    # get avg+stdev rank of each bin for samples in different subsets
    print(f"{time_now_str()} - Processing ranks for important features")
    for assay in assays:
        try:
            assay_features = important_features[assay]
        except KeyError as e:
            print(f"No important features found for {assay}")
            raise e  # each assay should have important features

        # Selecting a feature set, then collect ranks for each sample subset
        for ct, features_idx in assay_features.items():
            all_subset_ranks = {
                (a, c): {f: [] for f in features_idx} for a in assays for c in cell_types
            }

            # Calculate average ranks for all (assay, ct) subsets
            print(f"{time_now_str()} - Processing feature set for ({assay},{ct})")
            for output_class in output_classes:
                class_ranks: np.ndarray = rank_data["ranks"][class_to_idx[output_class]]

                for subset_assay in assays:
                    for subset_ct in cell_types:
                        subset_md5s = md5_sets[subset_assay][subset_ct]
                        samples_idx = [
                            i
                            for i, md5 in enumerate(rank_data["md5s"])
                            if md5 in subset_md5s
                        ]

                        for i in samples_idx:
                            for feature_idx in features_idx:
                                all_subset_ranks[(subset_assay, subset_ct)][
                                    feature_idx
                                ].append(class_ranks[i, feature_idx])

            # Calculate ranks stats, (avg, stddev) and (median, iqr)
            # fmt: off
            avg_ranks = {
                (a, c): {
                    f: (np.mean(ranks), np.std(ranks))
                    for f, ranks in subset_data.items()
                }
                for (a, c), subset_data in all_subset_ranks.items()
            }

            med_ranks = {
                (a, c): {
                    f: (np.median(ranks), compute_iqr(ranks))
                    for f, ranks in subset_data.items()
                }
                for (a, c), subset_data in all_subset_ranks.items()
            }
            # fmt: on

            # Save average ranks
            output_file = output_folder / f"{assay}_{ct}_feature_set_avg_ranks.tsv"
            write_stats(
                output_file, "Feature_{f}_avg\tFeature_{f}_std", avg_ranks, features_idx
            )

            # Save median ranks
            output_file = output_folder / f"{assay}_{ct}_feature_set_median_ranks.tsv"
            write_stats(
                output_file, "Feature_{f}_med\tFeature_{f}_iqr", med_ranks, features_idx
            )

            print(
                f"{time_now_str()} - Saved ranks statistics for features from {assay} - {ct}"
            )


if __name__ == "__main__":
    main()

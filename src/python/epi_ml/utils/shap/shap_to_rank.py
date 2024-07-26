"""Script to convert SHAP values files to rank values."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.utils.shap.shap_utils import extract_shap_values_and_info


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument(
        "parent_folder", type=DirectoryChecker(), help="Folder parent of each training split. (e.g. split0)",
    )
    # fmt: on
    return parser.parse_args()


def main():
    """Main function."""

    cli = parse_arguments()

    parent_folder: Path = cli.parent_folder

    # Collect content of each SHAP archive, transfrom to ranks
    all_split_ranks: Dict[str, List[np.ndarray]] = {}
    all_split_md5s: Dict[str, List[str]] = {}
    all_split_classes: Dict[str, List] = {}
    for folder in parent_folder.glob("split*/shap"):
        if not folder.is_dir():
            continue

        split_name = folder.parent.name

        shap_matrices, eval_md5s, classes = extract_shap_values_and_info(
            folder, verbose=False
        )

        # Get ranks for each rows, for each matrix, with greatest value = rank 0
        split_ranks = [
            np.argsort(-np.abs(shap_matrix), axis=1).argsort(axis=1)
            for shap_matrix in shap_matrices
        ]
        all_split_ranks[split_name] = split_ranks

        all_split_md5s[split_name] = eval_md5s
        all_split_classes[split_name] = classes

    # Sanity check classes
    if not all(
        split_classes == all_split_classes["split0"]
        for split_classes in all_split_classes.values()
    ):
        raise ValueError("Classes differ between splits")
    classes: List[Tuple[str, str]] = all_split_classes["split0"]

    # Concatenate validation SHAP values of each split into a single matrix (one per output class)
    concat_ranks = []
    for class_idx in range(len(classes)):
        class_ranks = [
            all_split_ranks[split_name][class_idx]
            for split_name in sorted(all_split_ranks.keys())
        ]
        concat_ranks.append(np.concatenate((class_ranks), axis=0, dtype=np.int32))

    # evals md5s need to follow split order
    all_md5s = []
    for split_name in sorted(all_split_md5s.keys()):
        all_md5s.extend(all_split_md5s[split_name])

    # Sanity check md5s
    if len(all_md5s) != len(set(all_md5s)):
        raise ValueError("Some evaluation md5s are duplicated between splits")

    # Save new combined archive
    output_folder = parent_folder / "shap_ranks"
    output_folder.mkdir(exist_ok=True)
    output_file = output_folder / "all_shap_ranks.npz"

    np.savez_compressed(
        file=output_file,
        ranks=np.array(concat_ranks),
        md5s=np.array(all_md5s),
        classes=np.array(classes),
    )


if __name__ == "__main__":
    main()

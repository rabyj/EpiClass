"""Functions to prepare background and evaluation data for SHAP analysis."""
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
from __future__ import annotations

import argparse
import copy
import itertools
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import epi_ml
from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
from epi_ml.argparseutils.directorychecker import DirectoryChecker
from epi_ml.core.metadata import Metadata
from epi_ml.utils.general_utility import write_hdf5_paths_to_file
from epi_ml.utils.time import time_now

CELL_TYPE = "harmonized_sample_ontology_intermediate"
ASSAY = "assay_epiclass"
TRACK = "track_type"


def select_datasets(metadata: Metadata, n=3, seed=42) -> List[str]:
    """
    Selects a random subset of datasets for each unique trio of (track_type, assay, cell_type) found in the metadata.
    It samples 'n' datasets for each trio, if available, or fewer if a trio has less than 'n' datasets.

    Args:
    - metadata (Metadata): The metadata object containing dataset information.
    - n (int, optional): The number of datasets to sample for each unique trio. Defaults to 3.
    - seed (int, optional): The seed to use for random sampling. Defaults to 42.

    Returns:
    - List[str]: A list of md5 hashes representing the randomly selected datasets.

    Note:
    - If a trio has fewer than 'n' datasets, all datasets for that trio are included.
    """
    random.seed(seed)

    trio_files = defaultdict(list)
    for md5sum, dset in metadata.items:
        trio = (dset["track_type"], dset[ASSAY], dset[CELL_TYPE])
        trio_files[trio].append(md5sum)

    sampled_md5s = list(
        itertools.chain.from_iterable(
            [random.sample(md5_list, n) for md5_list in trio_files.values()]
        )
    )
    return sampled_md5s


def evaluate_background_ratios(
    category: str,
    metadata: Metadata,
    training_md5s: Iterable[str],
    n_samples_list: List[int],
    verbose: bool = True,
) -> Tuple[List[str], int]:
    """
    Evaluates the ratio of each class in the specified category within the background data compared to the training data.
    The comparison is performed for different sampling sizes (n_samples).

    The function identifies trios of (track_type, assay, cell_type) in the datasets. For each sampling size in n_samples,
    it calculates the absolute difference in class ratios between the training and background data. The background data
    with the minimal ratio difference is selected for use in SHAP analysis.

    Args:
    - category (str): The category of class to analyze.
    - metadata (Metadata): The metadata object containing dataset information.
    - training_md5s (List[str]): List of md5 hashes representing the training datasets.
    - n_samples_list (List[int]): List of sampling sizes to evaluate.
    - verbose (bool, optional): If True, prints detailed logs during processing. Defaults to True.

    Returns:
    - List[str]: List of md5 hashes representing the best background datasets based on minimal ratio difference.
    - int: Sampling size used to select the best background datasets.
    """
    training_metadata = copy.deepcopy(metadata)
    training_md5s_set = set(training_md5s)
    for md5 in list(training_metadata.md5s):
        if md5 not in training_md5s_set:
            del training_metadata[md5]

    trios_md5_dict = defaultdict(list)
    for dset in training_metadata.datasets:
        trios_md5_dict[(dset[ASSAY], dset[CELL_TYPE], dset[TRACK])].append(dset["md5sum"])

    if verbose:
        print(f"{len(trios_md5_dict)} entries/trios")

    training_label_counter = training_metadata.label_counter(category, verbose=False)
    total_training = sum(training_label_counter.values())

    best_background = training_metadata
    best_diff = float("inf")
    best_n_per_trio = 666
    for n_per_trio in n_samples_list:
        meta = copy.deepcopy(training_metadata)
        background_md5s = set()
        for _, md5s in trios_md5_dict.items():
            background_md5s.update(md5s[0:n_per_trio])

        # Remove md5s not in the background from meta
        for md5 in list(meta.md5s):
            if md5 not in background_md5s:
                del meta[md5]

        if verbose:
            print(f"\nn_per_trio: {n_per_trio}")

        label_counter = meta.label_counter(category, verbose=False)
        total_background = sum(label_counter.values())
        sum_diff_dict = defaultdict(float)

        for label in training_label_counter:
            ratio_training = training_label_counter[label] / total_training
            ratio_background = label_counter[label] / total_background
            diff = abs(ratio_training - ratio_background)
            sum_diff_dict[n_per_trio] += diff
            if verbose:
                print(
                    f"{label}: train:{ratio_training:.3f}, sample:{ratio_background:.3f}, diff={diff:.3f}"
                )

        current_sum_diff = sum_diff_dict[n_per_trio]
        if current_sum_diff < best_diff:
            best_diff = current_sum_diff
            best_background = meta
            best_n_per_trio = n_per_trio

        if verbose:
            print(f"sum_diff: {current_sum_diff:.3f}")

    if verbose:
        print(f"\nbest_diff (n={best_n_per_trio}): {best_diff:.3f}")

    return list(best_background.md5s), best_n_per_trio


def parse_arguments() -> argparse.Namespace:
    """Argument parser for command line."""
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument(
        "category", type=str, help="The metatada category to analyse."
        )
    parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file."
        )
    parser.add_argument(
        "base_logdir", type=DirectoryChecker(), help="Directory where different fold directories are present"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files."
        )
    parser.add_argument(
        "--sampling_sizes", nargs="+", type=int, default=[3], help="Sampling sizes to evaluate (how many times [assay, cell_type, track_type] trios are sampled)."
        )
    # fmt: on
    return parser.parse_args()


def main():
    """Selects background data for SHAP analysis for each training fold."""
    begin = time_now()
    print(f"begin {begin}")

    cli = parse_arguments()

    category = cli.category
    metadata = Metadata(cli.metadata)
    base_logdir = cli.base_logdir
    overwrite = cli.overwrite
    sampling_sizes = cli.sampling_sizes

    try:
        # Beware, sensitive to file structure
        job_template_path = (
            Path(epi_ml.__file__).parents[2] / "bash_utils" / "compute_shaps_template.sh"
        )
        if not job_template_path.exists():
            raise FileNotFoundError("Could not find compute_shaps_template.sh")
    except IndexError as exc:
        raise FileNotFoundError("Could not find compute_shaps_template.sh") from exc

    # Find all split folders
    split_folders = [
        x for x in base_logdir.iterdir() if (x.is_dir() and x.name.startswith("split"))
    ]

    job_files = []
    for split_folder in split_folders:
        split_nb = int(split_folder.name.split("split")[-1])

        # Find training and valid md5s
        training_md5_path = list(split_folder.glob(f"split{split_nb}_training_*.md5"))
        if len(training_md5_path) != 1:
            raise ValueError(f"Invalid training_md5_path: {training_md5_path}")
        training_md5_path = training_md5_path[0]

        valid_md5_path = list(split_folder.glob(f"split{split_nb}_validation_*.md5"))
        if len(valid_md5_path) != 1:
            raise ValueError(f"Invalid valid_md5_path: {valid_md5_path}")
        valid_md5_path = valid_md5_path[0]

        with open(training_md5_path, "r", encoding="utf8") as f:
            training_md5 = set(f.read().splitlines())
        with open(valid_md5_path, "r", encoding="utf8") as f:
            valid_md5 = set(f.read().splitlines())

        # Find best background selection
        best_background_md5s, best_n_per_trio = evaluate_background_ratios(
            category=category,
            metadata=metadata,
            training_md5s=training_md5,
            n_samples_list=sampling_sizes,
            verbose=True,
        )

        # Save shap selections
        shap_folder = split_folder / "shap"
        shap_folder.mkdir(exist_ok=True)

        # Background
        selection_name = f"{best_n_per_trio}pertrio"
        background_filename = shap_folder / f"shap_background_{selection_name}_hdf5.list"
        if background_filename.exists() and not overwrite:
            print(
                f"{background_filename} already exists. Skipping {split_folder.name}",
                file=sys.stderr,
            )
            continue

        write_hdf5_paths_to_file(
            md5s=sorted(best_background_md5s),
            parent=".",
            suffix="100kb_all_none",
            filepath=background_filename,
        )

        # Eval
        eval_filename = shap_folder / f"shap_eval_all_valid_split{split_nb}_hdf5.list"
        if eval_filename.exists() and not overwrite:
            print(
                f"{eval_filename} already exists. Skipping {split_folder}",
                file=sys.stderr,
            )
            continue

        write_hdf5_paths_to_file(
            md5s=sorted(valid_md5),
            parent=".",
            suffix="100kb_all_none",
            filepath=eval_filename,
        )

        # Create shap computation slurm script from template

        # Things to account for:
        # :::account:::, :::email::: (use external script or change template without saving in repo)
        job_file = shap_folder / f"{category}_shap_split{split_nb}.sh"

        if job_file.exists() and not overwrite:
            raise FileExistsError(f"{job_file} already exists")
        shutil.copy(job_template_path, job_file)

        # Replace local HOME+mount value with $HOME, for remote execution
        shap_folder, background_filename, eval_filename = [
            Path(
                str(Path(x).resolve()).replace(
                    str(Path(os.environ["HOME"]) / "mounts/narval-mount"), "$HOME"
                )
            )
            for x in [shap_folder, background_filename, eval_filename]
        ]
        replacements = {
            ":::job_name:::": f"{category}_shap_split{split_nb}",
            ":::category:::": category,
            ":::model_path:::": str(shap_folder.parent),
            ":::output_log:::": str(shap_folder),
            ":::background_list:::": str(background_filename),
            ":::eval_list:::": str(eval_filename),
        }

        # Read / replace / write
        with open(job_file, "r", encoding="utf8") as file:
            file_contents = file.read()

        for key, value in replacements.items():
            file_contents = file_contents.replace(key, value)

        with open(job_file, "w", encoding="utf8") as file:
            file.write(file_contents)

        print(f"Created {job_file}")
        job_files.append(job_file)

    print("Creating submission script.")
    submission_script = base_logdir / f"submit_{category}_shap_all_splits.sh"
    if submission_script.exists() and not overwrite:
        raise FileExistsError(f"{submission_script} already exists")

    with open(submission_script, "w", encoding="utf8") as file:
        file.write("#!/bin/bash\n")
        for i, job_file in enumerate(job_files):
            file.write(f"job{i}=$(sbatch {job_file})\n")
            file.write(f"echo $job{i}\n")
        file.write("echo 'All jobs submitted.'\n")
    print(f"Created:{submission_script}")

    end = time_now()
    print(f"end {end}")
    print(f"elapsed {end - begin}")

    print(
        """
        Don't forget to check for:
        1) correct paths for where the scripts are run from (e.g. $HOME)
        2) correct model paths (in best_checkpoint.list files)
        3) Add analysis job to submit script if desired.
        """
    )


if __name__ == "__main__":
    main()

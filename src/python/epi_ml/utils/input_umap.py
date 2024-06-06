"""Compute UMAP embedding for some input+wgbs data in epiatlas and chip-atlas datasets."""

# pylint: disable=import-error, redefined-outer-name, use-dict-literal, too-many-lines, unused-import, unused-argument, too-many-branches
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import umap
from umap.umap_ import nearest_neighbors

from epi_ml.core.hdf5_loader import Hdf5Loader


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "--load_knn",
        type=Path,
        default=None,
        help="Directory containing precomputed knn pickle file",
    )
    # fmt: on
    return arg_parser.parse_args()


def main():
    """Run the main function."""
    cli = parse_arguments()

    load_knn_dir = cli.load_knn
    if load_knn_dir is not None:
        if not load_knn_dir.exists():
            raise FileNotFoundError(f"Could not find {load_knn_dir}.")
        if not next(load_knn_dir.glob("precomputed_knn_*.pkl")):
            raise FileNotFoundError(
                f"No precomputed knn pickle files found in {load_knn_dir}."
            )

    input_basedir = Path("/lustre06/project/6007017/rabyj/epilap/input")

    chromsize_path = input_basedir / "chromsizes" / "hg38.noy.chrom.sizes"
    hdf5_loader = Hdf5Loader(chrom_file=chromsize_path, normalization=True)

    # Make temporary file list out of two filelists
    hdf5_input_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))

    hdf5_lists_dir = input_basedir / "hdf5_list"
    epiatlas_filename_list_path = (
        hdf5_lists_dir / "hg38_epiatlas-freeze-v2/100kb_all_none_dfreeze-v2.list"
    )
    chip_atlas_filename_list_path = hdf5_lists_dir / "C-A_100kb_all_none_input.list"

    for path in [epiatlas_filename_list_path, chip_atlas_filename_list_path]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")

    epiatlas_files = hdf5_loader.read_list(epiatlas_filename_list_path)
    chip_atlas_files = hdf5_loader.read_list(chip_atlas_filename_list_path)

    epiatlas_filepaths = [
        hdf5_input_dir / "epiatlas_dfreeze_100kb_all_none" / Path(path).name
        for path in epiatlas_files.values()
    ]
    chip_atlas_filepaths = [
        hdf5_input_dir / "100kb_all_none" / Path(path).name
        for path in chip_atlas_files.values()
    ]
    all_paths = epiatlas_filepaths + chip_atlas_filepaths

    hdf5_paths_list_path = hdf5_input_dir / "hdf5_paths.list"
    with open(hdf5_paths_list_path, "w", encoding="utf8") as f:
        f.writelines([str(path) + "\n" for path in all_paths])

    # Untar data from both tars into local node tmpdir, and create list of files that takes into account different folder structure for each
    chip_atlas_tar_path = Path(
        "/lustre07/scratch/rabyj/other_data/C-A/hdf5/100kb_all_none.tar"
    )
    epiatlas_tar_path = Path(
        "/lustre06/project/6007515/ihec_share/local_ihec_data/epiatlas/hg38/hdf5/epiatlas_dfreeze_100kb_all_none.tar"
    )

    for path in [chip_atlas_tar_path, epiatlas_tar_path]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path}")

    for path in [chip_atlas_tar_path, epiatlas_tar_path]:
        subprocess.run(["tar", "-xf", str(path), "-C", str(hdf5_input_dir)], check=True)

    # Load relevant files
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot read file directly with")
        hdf5_dict = hdf5_loader.load_hdf5s(
            data_file=hdf5_paths_list_path,
            verbose=True,
            strict=False,
        ).signals
    print(f"Loaded {len(hdf5_dict)}/{len(all_paths)} files.")
    file_names = list(hdf5_dict.keys())
    data = np.array(list(hdf5_dict.values()), dtype=np.float32)
    del hdf5_dict

    # UMAP parameters
    nn_default = 15
    nn_bigger = 30
    nn_biggest = 100
    embedding_params = {}
    for nn_size in [nn_default, nn_bigger, nn_biggest]:
        embedding_params[f"standard_3D_nn{nn_size}"] = {
            "n_neighbors": nn_size,
            "min_dist": 0.1,
            "n_components": 3,
            "low_memory": False,
        }
        embedding_params[f"densmap_3D_nn{nn_size}"] = {
            "n_neighbors": nn_size,
            "min_dist": 0.1,
            "n_components": 3,
            "low_memory": False,
            "densmap": True,
        }

    try:
        output_dir = chip_atlas_tar_path.parent / "umap-input" / "epiatlas_all"
        output_dir.mkdir(exist_ok=True)
    except NameError:
        output_dir = Path.home()

    nn_knn = 100
    if not load_knn_dir:
        # Compute+save knn graph
        precomputed_knn = nearest_neighbors(
            X=data,
            n_neighbors=nn_knn,
            metric="correlation",
            random_state=42,
            low_memory=False,
            metric_kwds=None,
            angular=None,
        )

        with open(output_dir / f"precomputed_knn_{nn_knn}.pkl", "wb") as f:
            pickle.dump(precomputed_knn, f)

        # Save requirements so knn pickle is never lost in the future
        dists = metadata.distributions()
        with open(output_dir / "pickle_requirements.txt", "w", encoding="utf8") as f:
            for dist in dists:
                name = dist.metadata["Name"]
                version = dist.version
                f.write(f"{name}=={version}\n")
    else:
        # Load precomputed knn graph
        with open(output_dir / f"precomputed_knn_{nn_knn}.pkl", "rb") as f:
            precomputed_knn = pickle.load(f)

    # Compute+save embeddings
    for name, params in embedding_params.items():
        embedding = umap.UMAP(
            **params, random_state=42, precomputed_knn=precomputed_knn
        ).fit_transform(X=data)

        with open(output_dir / f"embedding_{name}.pkl", "wb") as f:
            pickle.dump({"ids": file_names, "embedding": embedding, "params": params}, f)
            print(f"Saved embedding_{name}.pkl")


if __name__ == "__main__":
    main()

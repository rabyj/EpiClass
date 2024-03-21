"""Utility functions for the paper notebooks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from epi_ml.core.metadata import Metadata

ASSAY: str = "assay_epiclass"
CELL_TYPE: str = "harmonized_sample_ontology_intermediate"
ASSAY_MERGE_DICT: Dict[str, str] = {
    "mrna_seq": "rna_seq",
    "wgbs-pbat": "wgbs",
    "wgbs-standard": "wgbs",
}


class IHECColorMap:
    """Class to handle IHEC color map."""

    def __init__(self, base_fig_dir: Path):
        self.base_fig_dir = base_fig_dir
        self.ihec_colormap_name = "IHEC_EpiATLAS_IA_colors_Mar18_2024.json"
        self.ihec_color_map = self.get_IHEC_color_map(
            base_fig_dir, self.ihec_colormap_name
        )
        self.assay_color_map = self.create_assay_color_map(self.ihec_color_map)
        self.cell_type_color_map = self.create_cell_type_color_map(self.ihec_color_map)

    @staticmethod
    def get_IHEC_color_map(folder: Path, name: str) -> List[Dict]:
        """Get the IHEC color map."""
        color_map_path = folder / name
        with open(color_map_path, "r", encoding="utf8") as color_map_file:
            ihec_color_map = json.load(color_map_file)
        return ihec_color_map

    @staticmethod
    def create_assay_color_map(ihec_color_map: List[Dict]) -> Dict[str, str]:
        """Create a rbg color map for ihec core assays."""
        colors = dict(ihec_color_map[0]["experiment"][0].items())
        for name, color in list(colors.items()):
            rbg = color.split(",")
            colors[name.lower()] = f"rgb({rbg[0]},{rbg[1]},{rbg[2]})"

        colors["rna_seq"] = colors["rna-seq"]
        return colors

    @staticmethod
    def create_cell_type_color_map(ihec_color_map: List[Dict]) -> Dict[str, str]:
        """Read the rbg color map for ihec cell types."""
        colors = dict(
            ihec_color_map[3]["harmonized_sample_ontology_intermediate"][0].items()
        )
        for name, color in list(colors.items()):
            rbg = color.split(",")
            colors[name] = f"rgb({rbg[0]},{rbg[1]},{rbg[2]})"
        return colors


def merge_similar_assays(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to merge rna-seq/wgbs categories, included prediction score."""
    df = df.copy(deep=True)
    try:
        df["rna_seq"] = df["rna_seq"] + df["mrna_seq"]
        df["wgbs"] = df["wgbs-standard"] + df["wgbs-pbat"]
    except KeyError as exc:
        raise ValueError(
            "Wrong results dataframe, label category is not assay specific."
        ) from exc
    df.drop(columns=["mrna_seq", "wgbs-standard", "wgbs-pbat"], inplace=True)
    df["True class"].replace(ASSAY_MERGE_DICT, inplace=True)
    df["Predicted class"].replace(ASSAY_MERGE_DICT, inplace=True)

    try:
        df[ASSAY].replace(ASSAY_MERGE_DICT, inplace=True)
    except KeyError:
        pass

    # Recompute Max pred if it exists
    classes = df["True class"].unique()
    if "Max pred" in df.columns:
        df["Max pred"] = df[classes].max(axis=1)
    return df


class MetadataHandler:
    """Class to handle Metadata objects."""

    def __init__(self, paper_dir: Path | str):
        self.paper_dir = Path(paper_dir)
        self.data_dir = self.paper_dir / "data"

    def load_metadata(self, version: str) -> Metadata:
        """Return metadata for a specific version.

        Example of epiRR unique to v1: IHECRE00003355.2
        """
        if version not in ["v1", "v2", "v2-encode"]:
            raise ValueError("Version must be one of v1, v2, v2-encode")

        names = {
            "v1": "hg38_2023-epiatlas_dfreeze_formatted_JR.json",
            "v2": "hg38_2023-epiatlas-dfreeze-pospurge-nodup_filterCtl.json",
            "v2-encode": "hg38_2023-epiatlas-dfreeze_v2.1_w_encode_noncore_2.json",
        }
        metadata = Metadata(self.paper_dir / "data" / "metadata" / names[version])
        return metadata

    def join_metadata(self, df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
        """Join the metadata to the results dataframe."""
        metadata_df = pd.DataFrame(metadata.datasets)
        metadata_df.set_index("md5sum", inplace=True)

        diff_set = set(df.index) - set(metadata_df.index)
        if diff_set:
            err_df = pd.DataFrame(diff_set, columns=["md5sum"])
            err_df.to_csv(self.data_dir / "join_missing_md5sums.csv", index=False)
            raise AssertionError(
                f"{len(diff_set)} md5sums in the results dataframe are not present in the metadata dataframe. Saved error md5sums to join_missing_md5sums.csv."
            )

        merged_df = df.merge(metadata_df, how="left", left_index=True, right_index=True)
        if len(merged_df) != len(df):
            raise AssertionError(
                "Merged dataframe has different length than original dataframe"
            )
        return merged_df


class SplitResultsHandler:
    """Class to handle split results."""

    @staticmethod
    def gather_split_results(
        results_dir: Path, label_category: str, only_NN: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Gather split results for each classifier.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: {split_name:{classifier_name: results_df}}
        """
        all_split_dfs = {}
        for split in [f"split{i}" for i in range(10)]:
            # Get the csv paths
            if label_category == ASSAY:
                second_dir_end = ""
            elif label_category == CELL_TYPE:
                second_dir_end = "-dfreeze-v2"

            NN_csv_path = (
                results_dir
                / f"{label_category}_1l_3000n"
                / f"10fold{second_dir_end}"
                / split
                / "validation_prediction.csv"
            )
            other_csv_root = (
                results_dir / f"{label_category}" / f"predict-10fold{second_dir_end}"
            )

            if not only_NN:
                if not other_csv_root.exists():
                    raise FileNotFoundError(f"Could not find {other_csv_root}")
                other_csv_paths = other_csv_root.glob(
                    f"*/*_{split}_validation_prediction.csv"
                )

                other_csv_paths = list(other_csv_paths)
                if len(other_csv_paths) != 4:
                    raise AssertionError(
                        f"Expected 4 other_csv_paths, got {len(other_csv_paths)}"
                    )

            # Load the dataframes
            dfs = {}
            dfs["NN"] = pd.read_csv(NN_csv_path, header=0, index_col=0, low_memory=False)

            if not only_NN:
                for path in other_csv_paths:
                    name = path.name.split("_", maxsplit=1)[0]
                    dfs[name] = pd.read_csv(path, header=0, index_col=0, low_memory=False)

            # Verify that all dataframes have the same md5sums
            md5s = {}
            for key, df in dfs.items():
                md5s[key] = set(df.index)

            base_md5s = md5s["NN"]
            if not base_md5s.intersection(*list(md5s.values())) == base_md5s:
                raise AssertionError("Not all dataframes have the same md5sums")

            all_split_dfs[split] = dfs

        return all_split_dfs

    @staticmethod
    def concatenate_split_results(
        split_dfs: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """Concatenate split results for each different classifier.

        Args:
            split_dfs (Dict[str, Dict[str, pd.DataFrame]]): {split_name:{classifier_name: results_df}}

        Returns:
            Dict[str, pd.DataFrame]: {classifier_name: concatenated_df}
        """
        to_concat_dfs = defaultdict(list)
        for dfs in split_dfs.values():
            for classifier, df in dfs.items():
                to_concat_dfs[classifier].append(df)

        concatenated_dfs = {
            classifier: pd.concat(dfs, axis=0)
            for classifier, dfs in to_concat_dfs.items()
        }

        # Verify index is still md5sum
        for df in concatenated_dfs.values():
            if not isinstance(df.index[0], str):
                raise AssertionError("Index is not md5sum")

        return concatenated_dfs

    @staticmethod
    def compute_split_metrics(
        all_split_dfs: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute desired metrics for each split and classifier."""
        split_metrics = {}
        for split in [f"split{i}" for i in range(10)]:
            dfs = all_split_dfs[split]

            # Compute metrics for the split
            metrics = {}
            for key, df in dfs.items():
                # One-hot encode true and predicted classes
                classes_order = df.columns[2:]
                onehot_true = (
                    pd.get_dummies(df["True class"], dtype=int)
                    .reindex(columns=classes_order, fill_value=0)
                    .values
                )
                pred_probs = df[
                    classes_order
                ].values  # Ensure this aligns with your model's output format

                ravel_true = np.argmax(onehot_true, axis=1)
                ravel_pred = np.argmax(pred_probs, axis=1)

                metrics[key] = {
                    "Accuracy": accuracy_score(ravel_true, ravel_pred),
                    "F1_macro": f1_score(ravel_true, ravel_pred, average="macro"),
                    "AUC_micro": roc_auc_score(
                        onehot_true, pred_probs, multi_class="ovr", average="micro"
                    ),
                    "AUC_macro": roc_auc_score(
                        onehot_true, pred_probs, multi_class="ovr", average="macro"
                    ),
                }

                split_metrics[split] = metrics

        return split_metrics

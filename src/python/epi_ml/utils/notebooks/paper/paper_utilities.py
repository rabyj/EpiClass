"""Utility functions for the paper notebooks."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from epi_ml.core.metadata import Metadata

ASSAY: str = "assay_epiclass"
CELL_TYPE: str = "harmonized_sample_ontology_intermediate"
SEX: str = "harmonized_donor_sex"
LIFE_STAGE: str = "harmonized_donor_life_stage"
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
    def verify_md5_consistency(dfs: Dict[str, pd.DataFrame]) -> None:
        """Verify that all dataframes have the same md5sums.

        Args:
            dfs (Dict[str, pd.DataFrame]): {classifier_name: results_df}
                Results dataframes need to have md5sums as index.
        """
        md5s = {}
        for key, df in dfs.items():
            md5s[key] = set(df.index)

        first_key = list(dfs.keys())[0]
        base_md5s = dfs[first_key].index
        if not base_md5s.intersection(*list(md5s.values())) == base_md5s:
            raise AssertionError("Not all dataframes have the same md5sums")

    @staticmethod
    def gather_split_results_across_methods(
        results_dir: Path, label_category: str, only_NN: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Gather split results for each classifier.type

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

            # Verify md5sum consistency
            SplitResultsHandler.verify_md5_consistency(dfs)

            all_split_dfs[split] = dfs

        return all_split_dfs

    @staticmethod
    def read_split_results(parent_dir: Path) -> Dict[str, pd.DataFrame]:
        """Read split results from the given parent directory.

        Args:
            parent_dir: The parent directory containing the split results.

        Returns:
            Dict[str, pd.DataFrame]: {split_name: results_df}
        """
        csv_path_template = "split*/validation_prediction.csv"
        experiment_dict = {}
        for split_result_csv in parent_dir.glob(csv_path_template):
            split_name = split_result_csv.parent.name
            df = pd.read_csv(split_result_csv, header=0, index_col=0, low_memory=False)
            experiment_dict[split_name] = df
        return experiment_dict

    @staticmethod
    def gather_split_results_across_categories(
        parent_results_dir: Path,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Gather NN split results for each classification task in the given folder children.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: {general_name:{split_name: results_df}}
        """
        all_dfs = defaultdict(dict)

        for category_dir in parent_results_dir.iterdir():
            for experiment_dir in category_dir.iterdir():
                experiment_name = experiment_dir.name
                category_name = category_dir.name
                general_name = f"{category_name}_{experiment_name}"

                experiment_dict = SplitResultsHandler.read_split_results(experiment_dir)

                all_dfs[general_name] = experiment_dict

        return all_dfs

    @staticmethod
    def concatenate_split_results(
        split_dfs: Dict[str, Dict[str, pd.DataFrame]], concat_first_level: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Concatenate split results for each different classifier or split name based on the order.

        This method supports two structures of input dictionaries:
        1. {split_name: {classifier_name: results_df}} when concat_first_level is False.
        2. {classifier_name: {split_name: results_df}} when concat_first_level is True.

        Args:
            split_dfs: A nested dictionary of DataFrames to be concatenated. The structure
                       depends on the concat_first_level flag.
            concat_first_level: A boolean flag that indicates the structure of the split_dfs dictionary.
                                If True, the first level is the classifier name. Otherwise, the first
                                level is the split name.

        Returns:
            Dict[str, pd.DataFrame] : {classifier_name: concatenated_dataframe}

        Raises:
            AssertionError: If the index of any concatenated DataFrame is not of type str, indicating
                            an unexpected index format.
        """
        if concat_first_level:
            # Reverse the nesting of the dictionary if concatenating by the first level.
            reversed_dfs = defaultdict(dict)
            for outer_key, inner_dict in split_dfs.items():
                for inner_key, df in inner_dict.items():
                    reversed_dfs[inner_key][outer_key] = df
            split_dfs = reversed_dfs

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
        all_split_dfs: Dict[str, Dict[str, pd.DataFrame]],
        concat_first_level: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute desired metrics for each split and classifier, accommodating different dictionary structures.

        This method supports two structures of input dictionaries:
        1. {split_name: {classifier_name: results_df}} when concat_first_level is False.
        2. {classifier_name: {split_name: results_df}} when concat_first_level is True.

        Args:
            split_dfs: A nested dictionary of DataFrames to be concatenated. The structure
                       depends on the concat_first_level flag.
            concat_first_level: A boolean flag that indicates the structure of the split_dfs dictionary.
                                If True, the first level is the classifier name. Otherwise, the first
                                level is the split name.

        Returns:
            A nested dictionary with metrics computed for each classifier and split. The structure is
            {split_name: {classifier_name: {metric_name: value}}}.
        """
        split_metrics = {}

        if concat_first_level:
            # Reorganize the dictionary to always work with {split_name: {classifier_name: DataFrame}}
            temp_dict = {}
            for classifier, splits in all_split_dfs.items():
                for split, df in splits.items():
                    if split not in temp_dict:
                        temp_dict[split] = {}
                    temp_dict[split][classifier] = df
            all_split_dfs = temp_dict

        for split in [f"split{i}" for i in range(10)]:
            dfs = all_split_dfs[split]
            metrics = {}
            for task_name, df in dfs.items():
                # Ensure 'True class' labels match (e.g. int or bool class labels)
                df["True class"] = df["True class"].astype(str).str.lower()
                df.columns = list(df.columns[:2]) + [
                    label.lower()
                    for i, label in enumerate(df.columns.str.lower())
                    if i > 1
                ]

                # One hot encode true class and get predicted probabilities
                # Reindex to ensure that the order of classes is consistent
                classes_order = df.columns[2:]
                onehot_true = (
                    pd.get_dummies(df["True class"], dtype=int)
                    .reindex(columns=classes_order, fill_value=0)
                    .values
                )
                pred_probs = df[classes_order].values

                ravel_true = np.argmax(onehot_true, axis=1)
                ravel_pred = np.argmax(pred_probs, axis=1)
                try:
                    metrics[task_name] = {
                        "Accuracy": accuracy_score(ravel_true, ravel_pred),
                        "F1_macro": f1_score(ravel_true, ravel_pred, average="macro"),
                        "AUC_micro": roc_auc_score(
                            onehot_true, pred_probs, multi_class="ovr", average="micro"
                        ),
                        "AUC_macro": roc_auc_score(
                            onehot_true, pred_probs, multi_class="ovr", average="macro"
                        ),
                    }
                except ValueError as err:
                    if "Only one class" in str(err) or "multiclass format" in str(err):
                        logging.warning(
                            "Single class or incompatible format in %s for %s.",
                            split,
                            task_name,
                        )
                        metrics[task_name] = {
                            "Accuracy": np.nan,
                            "F1_macro": np.nan,
                            "AUC_micro": np.nan,
                            "AUC_macro": np.nan,
                        }
                        if not set(df["True class"].unique()).issubset(
                            set(classes_order.values)
                        ):
                            logging.error("Classes do not match columns names.")
                            logging.debug("Classes in df: %s", df["True class"].unique())
                            logging.debug("Classes in classes_order: %s", classes_order)
                            raise ValueError(
                                "Classes in df do not match columns names."
                            ) from err
                        # Attempt to compute metrics that don't require multiple classes
                        if len(np.unique(ravel_true)) == 1:
                            metrics[task_name]["Accuracy"] = accuracy_score(
                                ravel_true, ravel_pred
                            )
                            metrics[task_name]["F1_macro"] = f1_score(
                                ravel_true, ravel_pred, average="macro"
                            )
                    else:
                        err_msg = f"Unexpected error in {split} for {task_name}."
                        logging.error(err_msg)
                        logging.debug("columns: %s", df.columns)
                        logging.debug(
                            "True class values: %s", df["True class"].value_counts()
                        )
                        logging.debug(
                            "ravel_true unique values: %s",
                            set(val for val in ravel_true),
                        )
                        raise ValueError(err_msg) from err

            split_metrics[split] = metrics

        return split_metrics

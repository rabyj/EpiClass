"""Utility functions for the paper notebooks."""
# pylint: disable=too-many-branches,too-many-lines

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from IPython.display import display
except ImportError:
    display = print
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from epi_ml.core.metadata import Metadata

ASSAY: str = "assay_epiclass"
CELL_TYPE: str = "harmonized_sample_ontology_intermediate"
SEX: str = "harmonized_donor_sex"
LIFE_STAGE: str = "harmonized_donor_life_stage"
BIOMATERIAL_TYPE: str = "harmonized_biomaterial_type"
CANCER: str = "harmonized_sample_cancer_high"
DISEASE: str = "harmonized_sample_disease_high"
TRACK: str = "track_type"

ASSAY_MERGE_DICT: Dict[str, str] = {
    "mrna_seq": "rna_seq",
    "wgbs-pbat": "wgbs",
    "wgbs-standard": "wgbs",
}
ASSAY_ORDER = [
    "h3k4me3",
    "h3k27ac",
    "h3k4me1",
    "h3k36me3",
    "h3k27me3",
    "h3k9me3",
    "input",
    "rna_seq",
    "wgbs",
]

EPIATLAS_16_CT: List[str] = [
    ct.lower()
    for ct in [
        "T cell",
        "neutrophil",
        "brain",
        "monocyte",
        "lymphocyte of B lineage",
        "myeloid cell",
        "venous blood",
        "macrophage",
        "mesoderm-derived structure",
        "endoderm-derived structure",
        "colon",
        "connective tissue cell",
        "hepatocyte",
        "mammary gland epithelial cell",
        "muscle organ",
        "extraembryonic cell",
    ]
]


class IHECColorMap:
    """Class to handle IHEC color map."""

    def __init__(self, base_fig_dir: Path):
        self.base_fig_dir = base_fig_dir
        self.ihec_colormap_name = "IHEC_EpiATLAS_IA_colors_Apl01_2024.json"
        self.ihec_color_map = self.get_IHEC_color_map(
            base_fig_dir, self.ihec_colormap_name
        )
        self.assay_color_map = self.create_assay_color_map()
        self.cell_type_color_map = self.create_cell_type_color_map()
        self.sex_color_map = self.create_sex_color_map()

    @staticmethod
    def get_IHEC_color_map(folder: Path, name: str) -> List[Dict]:
        """Get the IHEC color map."""
        color_map_path = folder / name
        with open(color_map_path, "r", encoding="utf8") as color_map_file:
            ihec_color_map = json.load(color_map_file)
        return ihec_color_map

    def _create_color_map(self, label_category: str) -> Dict[str, str]:
        """Create a rbg color map from IHEC rgb strings"""
        color_dict = [elem for elem in self.ihec_color_map if label_category in elem][0][
            label_category
        ][0]
        for name, color in list(color_dict.items()):
            rbg = color.split(",")
            color_dict[name] = f"rgb({rbg[0]},{rbg[1]},{rbg[2]})"
            color_dict[name.lower()] = color_dict[name]
            color_dict[name.lower().replace("-", "_")] = color_dict[name]
        return color_dict

    def create_sex_color_map(self) -> Dict[str, str]:
        """Create a rbg color map for ihec sex label category."""
        color_dict = self._create_color_map(SEX)
        return color_dict

    def create_assay_color_map(self) -> Dict[str, str]:
        """Create a rbg color map for ihec assays."""
        category_label = "experiment"
        color_dict = self._create_color_map(category_label)
        color_dict["mrna_seq"] = color_dict["rna_seq"]
        for assay in ["wgbs-pbat", "wgbs-standard"]:
            color_dict[assay] = color_dict["wgbs"]
            color_dict[assay.replace("-", "_")] = color_dict["wgbs"]
        return color_dict

    def create_cell_type_color_map(self) -> Dict[str, str]:
        """Create the rbg color map for ihec cell types."""
        return self._create_color_map(CELL_TYPE)


def merge_similar_assays(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to merge rna-seq/wgbs categories, included prediction score.

    A ValueError is raised if the columns are not present."""
    df = df.copy(deep=True)

    if "wgbs" in df.columns:
        return df

    try:
        df["rna_seq"] = df["rna_seq"] + df["mrna_seq"]
        df["wgbs"] = df["wgbs-standard"] + df["wgbs-pbat"]
    except KeyError as exc:
        raise ValueError("Wrong results dataframe, rna or wgbs columns missing.") from exc

    df.drop(columns=["mrna_seq", "wgbs-standard", "wgbs-pbat"], inplace=True)
    df["True class"].replace(ASSAY_MERGE_DICT, inplace=True)
    df["Predicted class"].replace(ASSAY_MERGE_DICT, inplace=True)
    try:
        df[ASSAY].replace(ASSAY_MERGE_DICT, inplace=True)
    except KeyError:
        pass

    try:
        df[ASSAY].replace(ASSAY_MERGE_DICT, inplace=True)
    except KeyError:
        pass

    # Recompute Max pred if it exists
    classes = list(df["True class"].unique()) + list(df["Predicted class"].unique())
    if "Max pred" in df.columns:
        df["Max pred"] = df[classes].max(axis=1)
    return df


class MetadataHandler:
    """Class to handle Metadata objects."""

    def __init__(self, paper_dir: Path | str):
        self.paper_dir = Path(paper_dir)
        self.data_dir = self.paper_dir / "data"

        self.version_names = {
            "v1": "hg38_2023-epiatlas_dfreeze_formatted_JR.json",
            "v2": "hg38_2023-epiatlas-dfreeze-pospurge-nodup_filterCtl.json",
            "v2-encode": "hg38_2023-epiatlas-dfreeze_v2.1_w_encode_noncore_2.json",
        }

    def load_metadata(self, version: str) -> Metadata:
        """Return metadata for a specific version.

        Args:
            version (str): The version of the metadata to load.
            One of 'v1', 'v2', 'v2-encode'.

        Example of epiRR unique to v1: IHECRE00003355.2
        """
        if version not in self.version_names:
            raise ValueError(f"Version must be one of {self.version_names.keys()}")

        metadata = Metadata(
            self.paper_dir / "data" / "metadata" / self.version_names[version]
        )
        return metadata

    def load_metadata_df(self, version: str, merge_assays: bool = True) -> pd.DataFrame:
        """Load a metadata dataframe for a given version.

        merge_assays: Merge similar assays (rna 2x / wgb 2x)
        """
        metadata = self.load_metadata(version)
        metadata_df = self.metadata_to_df(metadata, merge_assays)
        return metadata_df

    @staticmethod
    def metadata_to_df(metadata: Metadata, merge_assays: bool = True) -> pd.DataFrame:
        """Convert the metadata to a dataframe.

        merge_assays: Merge similar assays (rna 2x / wgb 2x)
        """
        metadata_df = pd.DataFrame.from_records(list(metadata.datasets))
        metadata_df.set_index("md5sum", inplace=True)
        if merge_assays:
            metadata_df[ASSAY].replace(ASSAY_MERGE_DICT, inplace=True)
        return metadata_df

    @staticmethod
    def join_metadata(df: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
        """Join the metadata to the results dataframe."""
        metadata_df = pd.DataFrame(metadata.datasets)
        metadata_df.set_index("md5sum", inplace=True)
        metadata_df["md5sum"] = metadata_df.index

        diff_set = set(df.index) - set(metadata_df.index)
        if diff_set:
            err_df = pd.DataFrame(diff_set, columns=["md5sum"])
            print(err_df.to_string(), file=sys.stderr)
            raise AssertionError(
                f"{len(diff_set)} md5sums in the results dataframe are not present in the metadata dataframe. Saved error md5sums to join_missing_md5sums.csv."
            )

        merged_df = df.merge(
            metadata_df,
            how="left",
            left_index=True,
            right_index=True,
            suffixes=(None, "_delete"),
        )
        if len(merged_df) != len(df):
            raise AssertionError(
                "Merged dataframe has different length than original dataframe"
            )
        to_drop = [col for col in merged_df.columns if "_delete" in col]
        merged_df.drop(columns=to_drop, inplace=True)

        return merged_df

    @staticmethod
    def uniformize_metadata_for_plotting(
        epiatlas_metadata: Metadata,
        ca_pred_df: pd.DataFrame | None = None,
        enc_pred_df: pd.DataFrame | None = None,
        recount3_metadata: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Simplify metadata with chip-atlas metadata for plotting."""
        columns_to_keep = ["id", ASSAY, "track_type", "source", "plot_label"]
        to_concat = []

        epiatlas_df = pd.DataFrame.from_records(list(epiatlas_metadata.datasets))
        epiatlas_df["source"] = ["epiatlas"] * len(epiatlas_df)
        epiatlas_df["id"] = epiatlas_df["md5sum"]
        if "plot_label" not in epiatlas_df.columns:
            epiatlas_df["plot_label"] = [None] * len(epiatlas_df)
        epiatlas_df = epiatlas_df[columns_to_keep]

        to_concat.append(epiatlas_df)

        if ca_pred_df is not None:
            ca_df = ca_pred_df.copy(deep=True)
            ca_df["source"] = ["C-A"] * len(ca_df)
            ca_df[ASSAY] = ca_df["manual_target_consensus"]
            ca_df["track_type"] = ["raw"] * len(ca_df)
            ca_df["id"] = ca_df["Experimental-id"]
            if "plot_label" not in ca_df.columns:
                ca_df["plot_label"] = [None] * len(ca_df)

            ca_df = ca_df[columns_to_keep]
            to_concat.append(ca_df)

        if enc_pred_df is not None:
            enc_df = enc_pred_df.copy(deep=True)
            enc_df["source"] = ["encode"] * len(enc_df)
            enc_df[ASSAY] = enc_df[ASSAY]
            enc_df["track_type"] = ["pval"] * len(enc_df)
            enc_df["id"] = enc_df["FILE_accession"]
            if "plot_label" not in enc_df.columns:
                enc_df["plot_label"] = [None] * len(enc_df)

            enc_df = enc_df[columns_to_keep]
            to_concat.append(enc_df)

        if recount3_metadata is not None:
            recount3_df = recount3_metadata.copy(deep=True)
            recount3_df["source"] = ["recount3"] * len(recount3_df)
            recount3_df[ASSAY] = recount3_df["harmonized_assay"]
            recount3_df["track_type"] = ["unique_raw"] * len(recount3_df)
            recount3_df["id"] = recount3_df["ID"]
            if "plot_label" not in recount3_df.columns:
                recount3_df["plot_label"] = [None] * len(recount3_df)

            recount3_df = recount3_df[columns_to_keep]
            to_concat.append(recount3_df)

        return pd.concat(to_concat)


class SplitResultsHandler:
    """Class to handle split results."""

    @staticmethod
    def add_max_pred(df: pd.DataFrame, target_label: str = "True class") -> pd.DataFrame:
        """Add the max prediction column ("Max pred") to the results dataframe.

        The dataframe needs to not contain extra metadata columns.
        target_label: Column to ascertain output classes columns
        """
        if "Max pred" not in df.columns:
            df = df.copy(deep=True)
            classes_test = (
                df[target_label].astype(str).unique().tolist()
                + df["Predicted class"].astype(str).unique().tolist()
            )
            classes_test = list(set(classes_test))

            classes = list(df.columns[2:])
            if any(label in classes for label in ["TRUE", "FALSE"]):
                print(
                    "WARNING: Found TRUE or FALSE in pred vector columns. Changing column names."
                )
                df.rename(columns={"TRUE": "True", "FALSE": "False"}, inplace=True)
                classes = list(df.columns[2:])

            for col in ["md5sum", "split", "Same?", "Predicted class", "True class"]:
                try:
                    classes.remove(col)
                except ValueError:
                    pass
            for class_label in classes:
                if class_label not in classes_test:
                    raise ValueError(
                        f"""Dataframe contains extra metadata columns, cannot ascertain classes: {classes}.
                        Column {class_label} is not in {classes_test}"""
                    )
            df["Max pred"] = df[classes].max(axis=1)
        return df

    @staticmethod
    def verify_md5_consistency(dfs: Dict[str, pd.DataFrame]) -> None:
        """Verify that all dataframes have the same md5sums.

        Used to compare the md5sums of the same split across different classifiers methods.

        Args:
            dfs (Dict[str, pd.DataFrame]): {classifier_name: results_df}
                Results dataframes need to have md5sums as index.
        """
        md5s = {}
        for key, df in dfs.items():
            md5s[key] = set(df.index)

        first_key = list(dfs.keys())[0]
        base_md5s = set(dfs[first_key].index)
        if not base_md5s.intersection(*list(md5s.values())) == base_md5s:
            raise AssertionError("Not all dataframes have the same md5sums")

    @staticmethod
    def compute_acc_per_assay(
        split_results: Dict[str, pd.DataFrame], metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute accuracy per assay for each split.

        Args:
        - split_results: {split_name: results_df}.
        - metadata_df: The metadata dataframe.
        """

        assay_acc = defaultdict(dict)
        for split_name, split_result_df in split_results.items():
            # Merge metadata
            split_result_df = split_result_df.merge(
                metadata_df, left_index=True, right_index=True
            )

            # Compute accuracy per assay
            assay_groupby = split_result_df.groupby(ASSAY)
            for assay, assay_df in assay_groupby:
                assay_acc[assay][split_name] = np.mean(
                    assay_df["True class"].astype(str).str.lower()
                    == assay_df["Predicted class"].astype(str).str.lower()
                )

        return pd.DataFrame(assay_acc)

    @staticmethod
    def gather_split_results_across_methods(
        results_dir: Path, label_category: str, only_NN: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Gather split results for each classifier type.

        Args:
            results_dir: The directory containing the results. Child directories should be task/classifer names.
            label_category: The label category for the results.
            only_NN: A boolean flag to only gather the NN results.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: {split_name:{classifier_name: results_df}}
        """
        base_NN_path = results_dir / f"{label_category}_1l_3000n"
        if label_category == ASSAY:
            NN_csv_path_template = base_NN_path / "11c"
            if not NN_csv_path_template.exists():
                NN_csv_path_template = base_NN_path
        elif label_category == CELL_TYPE:
            NN_csv_path_template = base_NN_path
        else:
            raise ValueError("Label category not supported.")

        NN_csv_path_template = str(
            NN_csv_path_template
            / "10fold-oversampling"
            / "{split}"
            / "validation_prediction.csv"
        )
        all_split_dfs = {}
        for split in [f"split{i}" for i in range(10)]:
            # Get the csv paths
            NN_csv_path = Path(NN_csv_path_template.format(split=split))  # type: ignore
            other_csv_root = results_dir / f"{label_category}" / "predict-10fold"

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
            dfs: Dict[str, pd.DataFrame] = {}
            dfs["NN"] = pd.read_csv(NN_csv_path, header=0, index_col=0, low_memory=False)

            if not only_NN:
                for path in other_csv_paths:
                    category = path.name.split("_", maxsplit=1)[0]
                    dfs[category] = pd.read_csv(
                        path, header=0, index_col=0, low_memory=False
                    )

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
        parent_results_dir: Path, verbose: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Gather NN split results for each classification task in the given folder children.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: {general_name:{split_name: results_df}}
        """
        all_dfs = {}

        for category_dir in parent_results_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for experiment_dir in category_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue
                experiment_name = experiment_dir.name
                category_name = category_dir.name
                general_name = f"{category_name}_{experiment_name}"
                if verbose:
                    print(f"Reading {general_name} from {experiment_dir}")

                experiment_dict = SplitResultsHandler.read_split_results(experiment_dir)
                if verbose:
                    print(
                        f"Found {len(experiment_dict)} split results for {general_name}"
                    )

                all_dfs[general_name] = experiment_dict

        return all_dfs

    @staticmethod
    def concatenate_split_results(
        split_dfs: Dict[str, Dict[str, pd.DataFrame]] | Dict[str, pd.DataFrame],
        concat_first_level: bool = False,
        depth: int = 2,
    ) -> Dict[str, pd.DataFrame] | pd.DataFrame:
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
            depth: The depth of the dictionary structure. Must be 1 or 2.

        Returns:
            Dict[str, pd.DataFrame] : {classifier_name: concatenated_dataframe}

        Raises:
            AssertionError: If the index of any concatenated DataFrame is not of type str, indicating
                            an unexpected index format.
        """
        if depth not in [1, 2]:
            raise ValueError(f"Depth must be 1 or 2, got {depth}")

        # Handle basic case, code wasn't made for that, so a hack is done
        if depth == 1:
            to_concat = {"classifier": split_dfs}
            return SplitResultsHandler.concatenate_split_results(to_concat, concat_first_level=True, depth=2)["classifier"]  # type: ignore

        # Check for user error
        if depth == 2 and isinstance(next(iter(split_dfs.values())), pd.DataFrame):
            raise ValueError(
                "Depth given is two, but the input is not a nested dictionary."
            )

        if concat_first_level:
            # Reverse the nesting of the dictionary if concatenating by the first level.
            reversed_dfs = defaultdict(dict)
            for outer_key, inner_dict in split_dfs.items():
                for inner_key, df in inner_dict.items():
                    reversed_dfs[inner_key][outer_key] = df
            split_dfs = reversed_dfs

        to_concat_dfs = defaultdict(list)
        for split_name, dfs in split_dfs.items():
            for classifier, df in dfs.items():
                try:
                    df = df.assign(split=int(split_name.split("split")[-1]))
                except ValueError as e:
                    if "base 10" in str(e):
                        raise ValueError(
                            "Wrong concat_first_level value was probably used"
                        ) from e
                    raise e
                to_concat_dfs[classifier].append(df)

        concatenated_dfs = {
            classifier: pd.concat(dfs, axis=0)
            for classifier, dfs in to_concat_dfs.items()
        }

        # Verify index is still md5sum
        for df in concatenated_dfs.values():
            if not isinstance(df.index[0], str):
                raise AssertionError("Index is not md5sum")
            df["md5sum"] = df.index
            df.index.name = "md5sum"

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
        all_split_dfs = copy.deepcopy(all_split_dfs)  # avoid side-effects
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
                        "count": len(df),
                    }
                except ValueError as err:
                    if "Only one class" in str(err) or "multiclass format" in str(err):
                        if len(df["True class"].unique()) != len(classes_order):
                            missing_classes = set(classes_order.values) - set(
                                df["True class"].unique()
                            )
                            logging.warning(
                                "Cannot compute ROC AUC. At least one ground truth class missing from %s for %s: (%s)",
                                split,
                                task_name,
                                missing_classes,
                            )
                        else:
                            logging.warning(
                                "Single class or incompatible format in %s for %s. Error: %s",
                                split,
                                task_name,
                                err,
                            )
                        metrics[task_name] = {
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
                        metrics[task_name].update(
                            {
                                "Accuracy": accuracy_score(ravel_true, ravel_pred),
                                "F1_macro": f1_score(
                                    ravel_true, ravel_pred, average="macro"
                                ),
                                "count": len(df),
                            }
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

    @staticmethod
    def invert_metrics_dict(
        metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Invert metrics dict so classifier name is parent key.
        Before: {split_name:{classifier_name:{metric_name:val}}}
        After: {classifier_name:{split_name:{metric_name:val}}}
        """
        # Check for correct input
        first_keys = list(metrics.keys())
        if not all("split" in key for key in first_keys):
            raise ValueError("Wrong input dict: first level keys don't contain 'split'.")

        # Invert
        new_metrics = defaultdict(dict)
        for split_name, split_metrics in metrics.items():
            for classifier_name, classifier_metrics in split_metrics.items():
                new_metrics[classifier_name][split_name] = classifier_metrics
        return dict(new_metrics)

    @staticmethod
    def extract_count_from_metrics(metrics) -> Dict[str, int]:
        """Extract total file count from metrics dict (sums on each split).

        Returns a dict {classifier_name: count}
        """
        new_metrics = copy.deepcopy(metrics)

        # Check for correct input
        first_keys = list(new_metrics.keys())
        if all("split" in key for key in first_keys):
            new_metrics = SplitResultsHandler.invert_metrics_dict(new_metrics)

        counts = defaultdict(int)
        for classifier_name, all_split_metrics in new_metrics.items():
            for _, split_metrics in all_split_metrics.items():
                counts[classifier_name] += split_metrics["count"]  # type: ignore

        return dict(counts)

    @staticmethod
    def general_split_metrics(
        results_dir: Path,
        merge_assays: bool,
        exclude_categories: List[str] | None = None,
        exclude_names: List[str] | None = None,
        include_categories: List[str] | None = None,
        include_names: List[str] | None = None,
        return_type: str = "both",
        mislabel_corrections: Tuple[Dict[str, str], Dict[str, Dict[str, str]]]
        | None = None,
        oversampled_only: bool | None = True,
        verbose: bool | None = False,
        min_pred_score: float | None = None,
    ) -> (
        Dict[str, Dict[str, Dict[str, float]]]
        | Dict[str, Dict[str, pd.DataFrame]]
        | Tuple[
            Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, pd.DataFrame]]
        ]
    ):
        """Create the content data for figure 2a. (get metrics for each task)

        Args:
            results_dir (Path): Directory containing the results. Needs to be parent over category folders.
            merge_assays (bool): Merge similar assays (rna-seq x2, wgbs x2)
            exclude_categories (List[str]): Task categories to exclude (first level directory names).
            exclude_names (List[str]): Names of folders to exclude (ex: 7c or no-mix).
            include_categories (List[str]): Task categories to include (first level directory names).
            include_names (List[str]): Names of folders to include (ex: 7c or no-mix).
            return_type (str): Type of data to return ('metrics', 'split_results', 'both').
            mislabel_corrections (Tuple[Dict[str, str], Dict[str, Dict[str, str]]]): ({md5sum: EpiRR_no-v},{label_category: {EpiRR_no-v: corrected_label}})
            oversampled_only (bool): Only include oversampled runs.
            verbose (bool): Print additional information.
            min_pred_score (float): Minimum prediction score to consider. Affects the metrics. Defaults to None.

        Returns:
            Union[Dict[str, Dict[str, Dict[str, float]]],
                Dict[str, Dict[str, pd.DataFrame]],
                Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, pd.DataFrame]]]]
                Depending on return_type, it returns:
                - 'metrics': A metrics dictionary with the structure {split_name: {task_name: metrics_dict}}
                - 'split_results': A split results dictionary with the structure {task_name: {split_name: split_results_df}}
                - 'both': A tuple with both dictionaries described above
        """
        if return_type not in ["metrics", "split_results", "both"]:
            raise ValueError(
                f"Invalid return_type: {return_type}. Choose from 'metrics', 'split_results', or 'both'."
            )

        all_split_results = {}
        split_results_handler = SplitResultsHandler()

        if mislabel_corrections:
            md5sum_to_epirr, epirr_to_corrections = mislabel_corrections
        else:
            md5sum_to_epirr = {}
            epirr_to_corrections = {}

        for parent, _, _ in os.walk(results_dir):
            # Looking for oversampling only results
            parent = Path(parent)
            if "10fold" not in parent.name:
                continue
            if parent.name != "10fold-oversampling" and oversampled_only:
                continue

            if verbose:
                print(f"Checking {parent}")

            # Get the category + filter
            relpath = parent.relative_to(results_dir)
            category = relpath.parts[0].replace("_1l_3000n", "")
            if verbose:
                print(f"Checking category: {category}")
            if include_categories is not None:
                if not any(include_str in category for include_str in include_categories):
                    if verbose:
                        print(f"Skipping {category}: not in {include_categories}")
                    continue
            if exclude_categories is not None:
                if any(exclude_str in category for exclude_str in exclude_categories):
                    if verbose:
                        print(f"Skipping {category}: in {exclude_categories}")
                    continue

            # Get the rest of the name, ignore certain runs
            rest_of_name = list(relpath.parts[1:])
            for dir_name in ["10fold", "10fold-oversampling"]:
                try:
                    rest_of_name.remove(dir_name)
                except ValueError:
                    pass

            if len(rest_of_name) > 1:
                raise ValueError(
                    f"Too many parts in the name: {rest_of_name}. Path: {relpath}"
                )
            if rest_of_name:
                rest_of_name = rest_of_name[0]
            if verbose:
                print(f"Rest of name: {rest_of_name}")

            # Filter out certain runs
            if include_names is not None:
                if not any(name in rest_of_name for name in include_names):
                    if verbose:
                        print(
                            f"Skipping {category} {rest_of_name}: not in {include_names}"
                        )
                    continue
            if exclude_names is not None:
                if any(name in rest_of_name for name in exclude_names):
                    if verbose:
                        print(f"Skipping {category} {rest_of_name}: in {exclude_names}")
                    continue

            full_task_name = category
            if rest_of_name:
                full_task_name += f"_{rest_of_name}"

            # Get the split results
            if verbose:
                print(f"Getting split results for {full_task_name}")
            split_results = split_results_handler.read_split_results(parent)
            if not split_results:
                raise ValueError(f"No split results found in {parent}")

            if (
                ("sex" in full_task_name) or ("life_stage" in full_task_name)
            ) and mislabel_corrections:
                corrections = epirr_to_corrections[category]
                for split_name, results_df in list(split_results.items()):
                    current_true_class = results_df["True class"].to_dict()
                    new_true_class = {
                        k: corrections.get(md5sum_to_epirr[k], v)
                        for k, v in current_true_class.items()
                    }
                    results_df["True class"] = new_true_class.values()

                    split_results[split_name] = results_df

            if ("assay" in full_task_name) and merge_assays:
                for split_name, df in split_results.items():
                    try:
                        split_result_df = merge_similar_assays(df)
                    except ValueError as e:
                        print(f"Skipping {full_task_name} assay merging: {e}")
                        break
                    split_results[split_name] = split_result_df

            if min_pred_score:
                for split_name, df in split_results.items():
                    tmp_df = df.copy()
                    tmp_df = SplitResultsHandler.add_max_pred(tmp_df)
                    tmp_df = tmp_df[tmp_df["Max pred"] >= min_pred_score]
                    tmp_df = tmp_df.drop(columns=["Max pred"])
                    split_results[split_name] = tmp_df

            all_split_results[full_task_name] = split_results

        if return_type in ["metrics", "both"]:
            try:
                split_results_metrics = split_results_handler.compute_split_metrics(
                    all_split_results, concat_first_level=True
                )
            except KeyError as e:
                logging.error("KeyError: %s", e)
                logging.error("all_split_results: %s", all_split_results)
                logging.error("check folder: %s", results_dir)
                raise e

        if return_type == "metrics":
            # pylint: disable=possibly-used-before-assignment
            return split_results_metrics
        if return_type == "split_results":
            return all_split_results

        # the default return type is 'both'
        return split_results_metrics, all_split_results

    def obtain_all_feature_set_data(
        self,
        parent_folder: Path,
        merge_assays: bool,
        return_type: str,
        include_sets: List[str] | None = None,
        include_categories: List[str] | None = None,
        exclude_names: List[str] | None = None,
        oversampled_only: bool | None = True,
        verbose: bool | None = False,
    ) -> (
        Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
        | Dict[str, Dict[str, Dict[str, pd.DataFrame]]]
    ):
        """
        Obtain either metrics or split results for all feature sets based on the specified return_type.

        Args:
            parent_folder (Path): The parent folder containing all feature set folders.
                Needs to be the parent of feature set folders.
            merge_assays (bool): Whether to merge similar assays (e.g., RNA-seq x2, WGBS x2).
            return_type (str): Type of data to return, either "metrics" or "split_results".
            include_sets (List[str] | None): Feature sets to include.
            include_categories (List[str] | None): Task categories to include.
            exclude_names (List[str] | None): Names of folders to exclude (e.g., 7c or no-mix).
            oversampled_only (bool): Only include oversampled runs.
            verbose (bool): Print additional information.

        Returns:
            Union[Dict[str, Dict[str, Dict[str, Dict[str, float]]]], Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
            A dictionary containing either metrics or results for all feature sets.
            Format for metrics: {feature_set: {task_name: {split_name: metric_dict}}}
            Format for split results: {feature_set: {task_name: {split_name: results_dataframe}}}
        """
        valid_return_types = ["metrics", "split_results"]
        if return_type not in valid_return_types:
            raise ValueError(
                f"Invalid return_type: {return_type}. Choose from {valid_return_types}."
            )

        all_data = {}
        if verbose:
            print(f"Checking folder: {parent_folder}")
        for folder in parent_folder.iterdir():
            if not folder.is_dir():
                continue

            feature_set = folder.name
            if include_sets is not None and feature_set not in include_sets:
                if verbose:
                    print(f"Skipping {feature_set}: not in 'include_sets' list.")
                continue

            try:
                # Fetch either metrics or split results based on the return_type
                split_data = self.general_split_metrics(
                    results_dir=folder,
                    merge_assays=merge_assays,
                    return_type=return_type,
                    include_categories=include_categories,
                    exclude_names=exclude_names,
                    oversampled_only=oversampled_only,
                    verbose=verbose,
                )

                # Invert if metrics, otherwise keep as is
                if return_type == "metrics":
                    inverted_data = self.invert_metrics_dict(split_data)  # type: ignore
                    all_data[feature_set] = inverted_data
                elif return_type == "split_results":
                    all_data[feature_set] = split_data  # type: ignore

            except ValueError as err:
                raise ValueError(f"Problem with {feature_set}") from err

        return all_data


def create_mislabel_corrector(
    paper_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """Obtain information necessary to correct sex and life_stage mislabels.

    Returns:
    - Dict[str, str]: {md5sum: EpiRR_no-v}
    - Dict[str, Dict[str, str]]: {label_category: {EpiRR_no-v: corrected_label}}
    """
    epirr_no_v = "EpiRR_no-v"
    # Associate epirrs to md5sums
    metadata = MetadataHandler(paper_dir).load_metadata("v2")
    metadata_df = pd.DataFrame.from_records(list(metadata.datasets))
    md5sum_to_epirr = metadata_df.set_index("md5sum")[epirr_no_v].to_dict()

    # Load mislabels
    epirr_to_corrections = {}
    metadata_dir = paper_dir / "data" / "metadata" / "official" / "BadQual-mislabels"

    sex_mislabeled = pd.read_csv(metadata_dir / "official_Sex_mislabeled.csv")
    epirr_to_corrections[SEX] = sex_mislabeled.set_index(epirr_no_v)[
        "EpiClass_pred_Sex"
    ].to_dict()

    life_stage_mislabeled = pd.read_csv(
        metadata_dir / "official_Life_stage_mislabeled.csv"
    )
    epirr_to_corrections[LIFE_STAGE] = life_stage_mislabeled.set_index(epirr_no_v)[
        "EpiClass_pred_Life_stage"
    ].to_dict()

    return md5sum_to_epirr, epirr_to_corrections


def extract_data_from_files(
    parent_folder: Path,
    file_pattern: str,
    search_line: str,
    extract_pattern: str,
    type_cast: type = int,
    unique: bool = True,
) -> Dict[str, Set[Any]]:
    """
    Extracts data from files matching a specific pattern within each directory of a parent folder.

    Args:
        parent_folder (Path): The directory containing subdirectories to search.
        file_pattern (str): Glob pattern for files to search within each subdirectory.
        search_line (str): Line identifier to search for data extraction.
        extract_pattern (str): Regex pattern to extract the desired data from the identified line.
        type_cast (type): Type to cast the extracted data to.

    Returns:
        Dict[str, Set[type]]: A dictionary with subdirectory names as keys and sets of extracted data as values.
    """
    data = defaultdict(set)
    for folder in parent_folder.iterdir():
        if not folder.is_dir():
            continue
        for file in folder.rglob(file_pattern):
            with open(file, "r", encoding="utf8") as f:
                # print(f"Reading {file.name}")
                lines = [l.rstrip() for l in f if search_line in l]

            if not lines:
                # print(f"Skipping {file.name}, no relevant data found.")
                continue
            if len(lines) > 1 and unique:
                if len(set(lines)) == 1:
                    pass
                else:
                    raise ValueError(
                        f"Incorrect file reading, captured more than one unique line in {file.name}: {lines}"
                    )
            matches = [re.match(pattern=extract_pattern, string=line) for line in lines]
            try:
                extracted_data = [type_cast(match.group(1)) for match in matches]  # type: ignore
                # print(f"Extracted data: {extracted_data}")
            except AttributeError as err:
                raise ValueError(
                    f"Could not extract data from {file.name} using pattern {extract_pattern}."
                ) from err
            data[folder.name].update(extracted_data)
    return dict(data)


def extract_input_sizes_from_output_files(parent_folder: Path) -> Dict[str, Set[int]]:
    """Extracts model input sizes from output (.o) files."""
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.o",
        search_line="(1): Linear",
        extract_pattern=r".*in_features=(\d+).*",
        type_cast=int,
    )


def extract_node_jobs_from_error_files(parent_folder: Path) -> Dict[str, Set[int]]:
    """Extracts SLURM job IDs from error (.e) files."""
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.e",
        search_line="SLURM_JOB_ID",
        extract_pattern=r".*SLURM_JOB_ID\s+: (\d+)$",
        type_cast=int,
    )


def extract_experiment_keys_from_output_files(parent_folder: Path) -> Dict[str, Set[str]]:
    """Extracts experiment keys from output (.o) files."""
    search_line = "The current experiment key is"
    return extract_data_from_files(
        parent_folder=parent_folder,
        file_pattern="output_job*.o",
        search_line=search_line,
        extract_pattern=search_line + r" (\w{32})$",
        type_cast=str,
        unique=False,
    )


def add_second_highest_prediction(df: pd.DataFrame, pred_cols: List[str]) -> pd.DataFrame:
    """Return the DataFrame with a columns for the second highest prediction class.

    Adds columns:
    - '2nd pred class': The class with the second highest prediction.
    - '1rst/2nd prob diff': The difference between the highest and second highest prediction probabilities.
    - '1rst/2nd prob ratio': The ratio of the highest to second highest prediction probabilities.
    """
    # Convert the relevant columns to a numpy array
    predictions = df[pred_cols].values

    # Get the indices of the sorted values
    sorted_indices = np.argsort(predictions, axis=1)

    # The second highest will be at position -2 (second to last) in the sorted order
    second_highest_indices = sorted_indices[:, -2]

    # Map indices to column names
    second_highest_columns = np.array(pred_cols)[second_highest_indices]
    df["2nd pred class"] = second_highest_columns

    # Calculate the difference and ratio
    highest_probs = np.max(predictions, axis=1)
    second_highest_probs = predictions[
        np.arange(len(predictions)), second_highest_indices
    ]

    df["1st/2nd prob diff"] = highest_probs - second_highest_probs
    df["1st/2nd prob ratio"] = highest_probs / second_highest_probs
    return df


def display_perc(df: pd.DataFrame | pd.Series) -> None:
    """Display a DataFrame with percentages."""
    # pylint: disable=consider-using-f-string
    with pd.option_context("display.float_format", "{:.3f}".format):
        display(df)


def merge_life_stages(df: pd.DataFrame, column_name_templates: List[str]) -> pd.DataFrame:
    """Merge prenatal stages into one category, for given column names.

    New columns for LIFE_STAGE_merged will be added for each column name template.
    Args:
        df (pd.DataFrame): DataFrame to merge columns in.
        column_name_templates (List[str]): List of column name templates to merge.
            ex: ["{}", "True class ({})", "Predicted class ({})", "Max pred ({})"]
    Returns:
        pd.DataFrame: DataFrame with merged columns.
    """
    df = df.copy(deep=True)
    life_stage_merge_dict = {
        "fetal": "prenatal",
        "embryonic": "prenatal",
        "newborn": "prenatal",
    }

    for column_label in column_name_templates:
        new_cat_label = f"{LIFE_STAGE}_merged"
        new_cat_label = column_label.format(new_cat_label)

        old_cat_label = column_label.format(LIFE_STAGE)
        df[new_cat_label] = df[old_cat_label].replace(life_stage_merge_dict)

    return df


class TemporaryLogFilter:
    """Context manager for adding a filter to a logger."""

    def __init__(self, filter_obj, logger=None):
        self.logger = logger or logging.getLogger()
        self.filter = filter_obj

    def __enter__(self):
        self.logger.addFilter(self.filter)

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.removeFilter(self.filter)


def set_file_id(
    df: pd.DataFrame, input_col: str = "Unnamed: 0", output_col: str = "md5sum"
) -> pd.DataFrame:
    """Standardizes a filename column by extracting the prefix and ensuring it is the first column.

    This function renames a given column (`input_col`) by extracting the prefix (before `_`)
    and moves it to the first position in the DataFrame. If `input_col` and `output_col`
    are the same, the column is updated in place and repositioned.

    Handled filename cases:
    - recount3: sra.base_sums.SRP076599_SRR3669968.ALL_[resolution]_[filters].hdf5 -> SRR3669968
    - ENCODE/ChIP-Atlas: [file_db_accession]_[resolution]_[filters].hdf5 -> [file_db_accession]
    - EpiATLAS: [md5sum]_[resolution]_[filters].hdf5 -> [md5sum]


    Args:
        df (pd.DataFrame): The input DataFrame.
        input_col (str, optional): The name of the column to process. Defaults to "Unnamed: 0".
        output_col (str, optional): The name of the resulting column. Defaults to "md5sum".

    Returns:
        pd.DataFrame: A modified DataFrame with the processed column as the first column.

    Raises:
        KeyError: If `input_col` is not found in the DataFrame.
    """
    df = df.copy(deep=True)

    try:
        input_vals = df[input_col]
    except KeyError as err:
        raise KeyError(f"Column {input_col} not found.") from err

    if output_col in df.columns:
        df = df.drop(output_col, axis=1)

    new_ids = [
        input_val.split("_")[0]
        if input_val[0:3] != "sra"
        else input_val.split(".")[2].split("_")[1]
        for input_val in input_vals
    ]

    if len(set(new_ids)) != len(new_ids):
        raise ValueError("Produce non-unique ids. Review input column and code.")

    # Remove input_col and reinsert as the first column, regardless of whether input_col == output_col
    df[input_col] = new_ids
    df.insert(0, output_col, df.pop(input_col))

    return df


def compute_metrics(
    df: pd.DataFrame,
    cat_label: str | None = None,
    column_templates: Dict[str, str] | None = None,
    min_pred: float | None = None,
) -> Tuple[float, float, int]:
    """Compute the accuracy and f1 of the predictions on a DataFrame that
    has columns of the format 'True/Predicted class ([category])'.

    If min_pred is not None, only consider predictions with a score
    greater than or equal to min_pred.

    If min_pred is higher than the maximum prediction score in the
    DataFrame, return 0.0, 0.0, 0

    Args:
        df: DataFrame containing the predictions and true classes.
        cat_label: Label for the category being evaluated, for
        labels of the form "True class (category)".
        min_pred: Minimum prediction score to consider.

    Returns:
        Tuple of accuracy, f1 and number of samples.
    """
    if column_templates is None:
        column_templates = {
            "True": "True class ({})",
            "Predicted": "Predicted class ({})",
            "Max pred": "Max pred ({})",
        }

    true_label = "True class"
    pred_label = "Predicted class"
    max_pred_label = "Max pred"
    if cat_label:
        true_label = column_templates["True"].format(cat_label)
        pred_label = column_templates["Predicted"].format(cat_label)
        max_pred_label = column_templates["Max pred"].format(cat_label)

    sub_df = df.copy()
    if min_pred:
        try:
            sub_df = sub_df[sub_df[max_pred_label] >= min_pred]
        except KeyError as err:
            raise KeyError(
                f"Column '{max_pred_label}' not found in DataFrame and min_pred is not None."
            ) from err

    if sub_df.shape[0] == 0:
        return 0.0, 0.0, 0

    y_true = sub_df[true_label]
    y_pred = sub_df[pred_label]

    acc = (y_true == y_pred).mean()

    f1: float = f1_score(  # type: ignore
        y_true,
        y_pred,
        labels=y_pred.unique(),
        average="macro",
    )
    return acc, f1, sub_df.shape[0]


def compute_all_acc_per_assay(
    all_preds: pd.DataFrame,
    categories: List[str] | None = None,
    verbose: bool = True,
    no_epiatlas: bool = True,
    column_templates: Dict[str, str] | None = None,
    merge_assays: bool = True,
    assay_label: str = ASSAY,
) -> Dict[str, Dict]:
    """Compute accuracy for each assay.
    Checks core9 assays (core7, [m]rna_seq, wgbs-[pbat/standard]) + CTCF + non-core
    Also includes number of unknown samples.

    Args:
    - all_preds: The dataframe containing the predictions.
    - categories: List of categories to compute accuracy/f1 for.
    - verbose: Whether to print the results.
    - no_epiatlas: Whether to exclude EpiAtlas samples.
    - column_templates: Dictionary of column templates for true/predicted/max_pred columns.
        If None, the default templates will be used ([column_name] ([category]))
    - merge_assays: Whether to merge similar assays (e.g., RNA-seq x2, WGBS x2).
    - assay_label: The label to use for the assay column.

    Returns:
    - A dictionary with the accuracy for each assay.
        Format: {task_name:{assay: [(min_pred, acc, f1, nb_samples), ...], ...}, ...}
    """
    if categories is None:
        categories = [
            f"{ASSAY}_7c",
            f"{ASSAY}_11c",
            CELL_TYPE,
            SEX,
            LIFE_STAGE,
            f"{LIFE_STAGE}_merged",
            CANCER,
            BIOMATERIAL_TYPE,
        ]
    if column_templates is None:
        column_templates = {
            "True": "True class ({})",
            "Predicted": "Predicted class ({})",
            "Max pred": "Max pred ({})",
        }

    df = all_preds.copy(deep=True)
    if not (all_preds["in_epiatlas"].astype(str) == "False").all() and no_epiatlas:
        df = df[df["in_epiatlas"].astype(str) == "False"]

    df = df.fillna("unknown")
    core_assays = ASSAY_ORDER
    if "no_consensus" in df[assay_label].unique():
        core_assays.append("no_consensus")

    non_core_assays = ["ctcf", "non-core"]
    all_assays = ASSAY_ORDER + non_core_assays
    unknown_labels = ["unknown", "other"]

    # merging rna / wgbs assays
    if merge_assays:
        assay_cols = [
            ASSAY,
            column_templates["True"].format(f"{ASSAY}_11c"),
            column_templates["Predicted"].format(f"{ASSAY}_11c"),
            column_templates["True"].format(f"{ASSAY}_7c"),
            column_templates["Predicted"].format(f"{ASSAY}_7c"),
        ]
        for col in assay_cols:
            for pair in (
                ("mrna_seq", "rna_seq"),
                ("wgbs-pbat", "wgbs"),
                ("wgbs-standard", "wgbs"),
            ):
                try:
                    df[col] = df[col].str.replace(pat=pair[0], repl=pair[1], regex=False)
                except KeyError as err:
                    raise ValueError(f"Column '{col}' not found.") from err

    all_acc_per_assay = {}
    for name in categories:
        if verbose:
            print(f"Computing metrics for {name}")
        task_df = df.copy(deep=True)
        y_true_col = column_templates["True"].format(name)
        y_pred_col = column_templates["Predicted"].format(name)
        max_pred_label = column_templates["Max pred"].format(name)

        if max_pred_label not in df.columns:
            raise ValueError(f"Column '{max_pred_label}' not found.")

        unknown_mask = task_df[y_true_col].isin(unknown_labels)
        unknown_df = task_df[unknown_mask]

        # Remove unknown samples, if any
        known_df = task_df[~unknown_mask]
        known_df = known_df[known_df[y_pred_col] != "unknown"]

        if name == CELL_TYPE:
            known_df = known_df[known_df[CELL_TYPE].isin(EPIATLAS_16_CT)]

        if verbose:
            print(name, known_df.shape)
            print(known_df[y_true_col].value_counts(dropna=False))
            if not unknown_df.empty:
                print(f"Unknown {name} samples: {unknown_df.shape[0]}")
            print()

        # Metrics per assay
        acc_per_assay: Dict[str, List[Tuple[str, float, float, int]]] = {}
        for label in all_assays:
            if label in non_core_assays and name == assay_label:
                continue

            # Process known labels
            known_assay_df = known_df[known_df[assay_label] == label]
            if not known_assay_df.empty or label in known_df[assay_label].unique():
                acc_per_assay[label] = []
                for min_pred in ["0.0", "0.6", "0.8", "0.9"]:
                    acc, f1, N = compute_metrics(
                        known_assay_df,
                        cat_label=name,
                        min_pred=float(min_pred),
                        column_templates=column_templates,
                    )
                    acc_per_assay[label].append((min_pred, acc, f1, N))

        # Global metrics
        if not unknown_df.empty:
            acc_per_assay["avg-all-unknown"] = []

        for set_label in ["avg-all", "avg-core", "avg-non-core"]:
            acc_per_assay[set_label] = []

        for min_pred in ["0.0", "0.6", "0.8", "0.9"]:
            # Average across core assays
            core_df = known_df[known_df[assay_label].isin(core_assays)]
            acc, f1, N = compute_metrics(
                core_df,
                cat_label=name,
                min_pred=float(min_pred),
                column_templates=column_templates,
            )
            acc_per_assay["avg-core"].append((min_pred, acc, f1, N))

            if name == assay_label:
                continue

            # Average across non-core assays
            non_core_df = known_df[known_df[assay_label].isin(non_core_assays)]
            acc, f1, N = compute_metrics(
                non_core_df,
                cat_label=name,
                min_pred=float(min_pred),
                column_templates=column_templates,
            )
            acc_per_assay["avg-non-core"].append((min_pred, acc, f1, N))

            # Average across all assays
            acc, f1, N = compute_metrics(
                known_df,
                cat_label=name,
                min_pred=float(min_pred),
                column_templates=column_templates,
            )
            acc_per_assay["avg-all"].append((min_pred, acc, f1, N))

            # Average across all unknown
            if "avg-all-unknown" in acc_per_assay:
                high_pred_count = (unknown_df[max_pred_label] >= float(min_pred)).sum()
                acc_per_assay["avg-all-unknown"].append((min_pred, 0, 0, high_pred_count))

        all_acc_per_assay[name] = acc_per_assay

    return all_acc_per_assay


def save_acc_per_assay(
    all_acc_per_assay: pd.DataFrame,
    folders: List[Path],
    filename: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Take a dataframe containing predictions for multiple tasks, and compute accuracy for each assay.
    Saves the results to a tsv file, to given folders.

    Args:
    - all_acc_per_assay: Output of the compute_all_acc_per_assay function.
    - folders: A list of folders to save the results to.
    - filename: The name of the file to save the results to.
    - verbose: Whether to print verbose output.

    Returns:
    - A restructured dataframe with (accuracy, f1, N) for each assay
    """
    # acc per assay to table
    rows = []
    for name, acc_per_assay in all_acc_per_assay.items():
        for assay, values in acc_per_assay.items():
            for min_pred, acc, f1, nb_samples in values:
                rows.append([name, assay, min_pred, acc, f1, nb_samples])
    df_acc_per_assay = pd.DataFrame(
        rows,
        columns=["task_name", ASSAY, "min_predScore", "acc", "f1-score", "nb_samples"],
    )

    df_acc_per_assay = df_acc_per_assay.astype(
        {
            "task_name": "str",
            ASSAY: "str",
            "min_predScore": "float",
            "acc": "float",
            "f1-score": "float",
            "nb_samples": "int",
        }
    )

    # f1-score on ASSAY task, per assay, doesn't make sense
    df_acc_per_assay.loc[df_acc_per_assay["task_name"] == ASSAY, "f1-score"] = "NA"

    # acc / f1 for unknown labels is not defined
    df_acc_per_assay.loc[
        df_acc_per_assay[ASSAY].str.contains("unknown"), ["acc", "f1-score"]
    ] = "NA"

    # acc / f1 for 0 samples is not defined
    df_acc_per_assay.loc[df_acc_per_assay["nb_samples"] == 0, ["acc", "f1-score"]] = "NA"

    if verbose:
        print(f"Saving {df_acc_per_assay.shape[0]} rows")

    for folder in folders:
        path = folder / filename
        df_acc_per_assay.to_csv(
            path,
            sep="\t",
            index=False,
        )
        if verbose:
            print(f"Saved to {path}")

    return df_acc_per_assay

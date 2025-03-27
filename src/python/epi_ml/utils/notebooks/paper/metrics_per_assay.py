"""Module that defines class for computing metrics (acc, f1) per assay."""
# pylint: disable=too-many-branches
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from epi_ml.utils.notebooks.paper.paper_utilities import (
    ASSAY,
    ASSAY_ORDER,
    BIOMATERIAL_TYPE,
    CANCER,
    CELL_TYPE,
    EPIATLAS_16_CT,
    LIFE_STAGE,
    SEX,
)


class MetricsPerAssay:
    """Class for computing metrics (acc, f1) per assay and saving the results."""

    @staticmethod
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

    def compute_chunked_metrics(
        self,
        df: pd.DataFrame,
        cat_label: str,
        interval: float = 0.1,
        min_pred_column: str | None = None,
        column_templates: Dict[str, str] | None = None,
    ) -> List[Tuple[float, float, float, float, int]]:
        """
        Compute metrics chunked by prediction score intervals.

        Args:
            df: DataFrame containing predictions
            cat_label: Category label to compute metrics for
            interval: Size of prediction score interval (default: 0.1)
            min_pred_column: Column name containing prediction scores (if None, uses column template)
            column_templates: Dictionary mapping column types to template strings

        Returns:
            List of tuples: (lower_bound, upper_bound, accuracy, f1_score, sample_count)
        """
        if column_templates is None:
            column_templates = {
                "True": "True class ({})",
                "Predicted": "Predicted class ({})",
                "Max pred": "Max pred ({})",
            }

        if min_pred_column is None:
            min_pred_column = column_templates["Max pred"].format(cat_label)

        # Create chunks based on interval
        chunk_bounds = np.arange(0, 1.0 + interval, interval)
        chunked_results = []

        for i in range(len(chunk_bounds) - 1):
            lower_bound = chunk_bounds[i]
            upper_bound = chunk_bounds[i + 1]

            # Filter by prediction score range
            chunk_df = df[
                (df[min_pred_column] >= lower_bound) & (df[min_pred_column] < upper_bound)
            ]

            if not chunk_df.empty:
                acc, f1, n_samples = self.compute_metrics(
                    chunk_df,
                    cat_label=cat_label,
                    min_pred=lower_bound,
                    column_templates=column_templates,
                )
                chunked_results.append((lower_bound, upper_bound, acc, f1, n_samples))
            else:
                # Include empty chunks with zeros
                chunked_results.append((lower_bound, upper_bound, 0.0, 0.0, 0))

        return chunked_results

    def _compute_metrics_per_assay(
        self,
        all_preds: pd.DataFrame,
        categories: List[str] | None = None,
        verbose: bool = True,
        no_epiatlas: bool = True,
        column_templates: Dict[str, str] | None = None,
        merge_assays: bool = True,
        assay_label: str = ASSAY,
        chunked: bool = False,
        interval: float = 0.1,
        core_assays: List[str] | None = None,
        non_core_assays: List[str] | None = None,
        metric_function: Callable | None = None,
        metric_args: Dict[str, Any] | None = None,
    ) -> Dict[str, Dict]:
        """Compute metrics for each assay.

        Args:
            all_preds (pd.DataFrame): Dataframe containing predictions.
            categories (List[str] | None): List of categories to compute accuracy/F1 for. If None, uses default categories.
                Default: [f"{ASSAY}_7c", f"{ASSAY}_11c", CELL_TYPE, SEX, LIFE_STAGE, f"{LIFE_STAGE}_merged", CANCER, f"{CANCER}_merged"]
            verbose (bool): Whether to print the results.
            no_epiatlas (bool): Whether to exclude EpiAtlas samples.
            column_templates (Dict[str, str] | None): Column name templates. If None, uses default templates.
                Default: {'True': 'True class ({})', 'Predicted': 'Predicted class ({})', 'Max pred': 'Max pred ({})'}
            merge_assays (bool): Whether to merge similar assays.
            assay_label (str): Label to use for the assay column.
                Default: "assay_epiclass
            chunked (bool): Whether to compute chunked metrics.
            interval (float): Prediction score interval (only if chunked=True).
            core_assays (List[str] | None): List of core assays.
                Default: core11 assays (6 histones + input + 2 rna + 2 wgb)
            non_core_assays (List[str] | None): List of non-core assays.
                Default: ["ctcf", "non-core"]
            metric_function (Callable | None): Function for computing metrics.
            metric_args (Dict[str, Any] | None): Additional arguments for metric function.


        Returns:
            Dict[str, Dict]: Metrics for each assay.
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
        if metric_args is None:
            metric_args = {}

        # Set the default metric function if not provided
        if metric_function is None:
            if chunked:
                metric_function = self.compute_chunked_metrics
                metric_args["interval"] = interval
            else:
                metric_function = self.compute_metrics

        df = all_preds.copy(deep=True)
        if no_epiatlas and not (all_preds["in_epiatlas"].astype(str) == "False").all():
            df = df[df["in_epiatlas"].astype(str) == "False"]

        # target classification
        df = df.fillna("unknown")
        if core_assays is None:
            core_assays = ASSAY_ORDER.copy()
            if "no_consensus" in df[assay_label].unique():
                core_assays.append("no_consensus")

        if non_core_assays is None:
            non_core_assays = ["ctcf", "non-core"]

        all_assays = core_assays + non_core_assays

        # Define 'unknown' for expected labels
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
                if col in df.columns:  # Check if column exists
                    for pair in (
                        ("mrna_seq", "rna_seq"),
                        ("wgbs-pbat", "wgbs"),
                        ("wgbs-standard", "wgbs"),
                    ):
                        df[col] = df[col].str.replace(
                            pat=pair[0], repl=pair[1], regex=False
                        )
                else:
                    if verbose:
                        print(f"Warning: Column '{col}' not found for merging assays.")

        filter_dict = {
            "avg-all": lambda df: pd.Series(True, index=df.index),
            "avg-core": lambda df: df[assay_label].isin(core_assays),
            "avg-non-core": lambda df: df[assay_label].isin(non_core_assays),
        }

        all_metrics_per_assay = {}
        for category_name in categories:
            metric_name = "chunked metrics" if chunked else "metrics"
            if verbose:
                print(f"Computing {metric_name} for {category_name}")

            task_df = df.copy(deep=True)
            y_true_col = column_templates["True"].format(category_name)
            y_pred_col = column_templates["Predicted"].format(category_name)
            max_pred_label = column_templates["Max pred"].format(category_name)

            if max_pred_label not in df.columns:
                raise ValueError(f"Column '{max_pred_label}' not found.")

            unknown_mask = task_df[y_true_col].isin(unknown_labels)
            unknown_df = task_df[unknown_mask]

            # Remove unknown samples, if any
            known_df = task_df[~unknown_mask]
            known_df = known_df[known_df[y_pred_col] != "unknown"]

            if category_name == CELL_TYPE:
                known_df = known_df[known_df[CELL_TYPE].isin(EPIATLAS_16_CT)]

            # assumed to be ASSAY_11c/ASSAY_7c, non-core assays are removed (+ no unknown)
            if ASSAY in category_name:
                known_df = known_df[known_df[ASSAY].isin(core_assays)]

            if verbose:
                print(category_name, known_df.shape)
                print(known_df[y_true_col].value_counts(dropna=False))
                if not unknown_df.empty:
                    print(f"Unknown {category_name} samples: {unknown_df.shape[0]}")
                print()

            # Metrics per assay
            metrics_per_assay = {}

            # Process individual assays
            for label in all_assays:
                if verbose:
                    print(f"Processing assay target {label}")

                # Process known labels
                known_assay_df = known_df[known_df[assay_label] == label]
                if verbose:
                    print(f"Known {label} samples: {known_assay_df.shape[0]}")
                    print(known_assay_df[y_true_col].value_counts(dropna=False), "\n")
                    print(known_assay_df[y_pred_col].value_counts(dropna=False), "\n")

                if not known_assay_df.empty or label in known_df[assay_label].unique():
                    if chunked:
                        # Compute chunked metrics for this assay
                        metrics_per_assay[label] = metric_function(
                            known_assay_df,
                            cat_label=category_name,
                            column_templates=column_templates,
                            **metric_args,
                        )
                    else:
                        # Compute standard metrics for this assay with different thresholds
                        metrics_per_assay[label] = []
                        for min_pred in ["0.0", "0.6", "0.8", "0.9"]:
                            metric_args["min_pred"] = float(min_pred)
                            result = metric_function(
                                known_assay_df,
                                cat_label=category_name,
                                column_templates=column_templates,
                                **metric_args,
                            )
                            metrics_per_assay[label].append((min_pred, *result))

            # Compute global metrics
            if not unknown_df.empty:
                metrics_per_assay["avg-all-unknown"] = []

            set_labels = ["avg-all", "avg-core", "avg-non-core"]
            if all_assays in (core_assays, non_core_assays):
                set_labels = ["avg-all"]

            if ASSAY in category_name:
                set_labels = ["avg-core"]

            for set_label in set_labels:
                metrics_per_assay[set_label] = []

            for set_label in set_labels:
                filter_condition = filter_dict[set_label]
                filtered_df = known_df[filter_condition(known_df)]
                if filtered_df.empty:
                    continue

                if chunked:
                    metrics_per_assay[set_label] = metric_function(
                        df=filtered_df,
                        cat_label=category_name,
                        column_templates=column_templates,
                        **metric_args,
                    )
                    # Process unknown samples for chunked metrics
                    if not unknown_df.empty:
                        # Create chunks for unknown samples
                        chunk_bounds = np.arange(0, 1.0 + interval, interval)
                        unknown_chunks = []

                        for i in range(len(chunk_bounds) - 1):
                            lower_bound = chunk_bounds[i]
                            upper_bound = chunk_bounds[i + 1]

                            # Count unknown samples with prediction scores in this range
                            count = (
                                (unknown_df[max_pred_label] >= lower_bound)
                                & (unknown_df[max_pred_label] < upper_bound)
                            ).sum()

                            unknown_chunks.append(
                                (lower_bound, upper_bound, 0.0, 0.0, count)
                            )

                        metrics_per_assay["avg-all-unknown"] = unknown_chunks
                else:
                    # Standard global metrics with different thresholds
                    for min_pred in ["0.0", "0.6", "0.8", "0.9"]:
                        metric_args["min_pred"] = float(min_pred)
                        result = metric_function(
                            df=filtered_df,
                            cat_label=category_name,
                            column_templates=column_templates,
                            **metric_args,
                        )
                        metrics_per_assay[set_label].append((min_pred, *result))

                        # Average across all unknown
                        if "avg-all-unknown" in metrics_per_assay:
                            high_pred_count = (
                                unknown_df[max_pred_label] >= float(min_pred)
                            ).sum()
                            metrics_per_assay["avg-all-unknown"].append(
                                (min_pred, 0, 0, high_pred_count)
                            )

            all_metrics_per_assay[category_name] = metrics_per_assay

        if verbose:
            print(f"Computed metrics for {len(categories)} categories")
            if chunked:
                print(f"Used interval size: {interval}")

        return all_metrics_per_assay

    @wraps(_compute_metrics_per_assay)
    def compute_all_chunked_acc_per_assay(self, *args, **kwargs):
        """Compute metrics for all assays using chunked evaluation."""
        result = self._compute_metrics_per_assay(*args, chunked=True, **kwargs)
        return result

    @wraps(_compute_metrics_per_assay)
    def compute_all_acc_per_assay(self, *args, **kwargs):
        """Compute metrics for all assays using threshold-based evaluation."""
        result = self._compute_metrics_per_assay(*args, chunked=False, **kwargs)
        return result

    def save_metrics_per_assay(
        self,
        input_metrics: Dict[str, Dict],
        folders: List[Path] | Path,
        filename: str,
        chunked: bool = False,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Take metrics dictionary and save it to TSV files.
        Works with both standard metrics and chunked metrics.

        Args:
        - metrics: Output of compute_all_acc_per_assay or compute_all_chunked_acc_per_assay.
        - folders: A list of folders to save the results to.
        - filename: The name of the file to save the results to.
        - chunked: Whether the input contains chunked metrics (True) or standard metrics (False).
        - verbose: Whether to print verbose output.

        Returns:
        - A restructured dataframe with metrics for each assay
        """
        # Convert metrics to table format
        rows = []
        for name, metrics_per_assay in input_metrics.items():
            for assay, values in metrics_per_assay.items():
                for value_tuple in values:
                    if chunked:
                        # Chunked format: (lower_bound, upper_bound, acc, f1, nb_samples)
                        lower_bound, upper_bound, acc, f1, nb_samples = value_tuple
                        rows.append(
                            [name, assay, lower_bound, upper_bound, acc, f1, nb_samples]
                        )
                    else:
                        # Standard format: (min_pred, acc, f1, nb_samples)
                        min_pred, acc, f1, nb_samples = value_tuple
                        rows.append([name, assay, min_pred, acc, f1, nb_samples])

        # Create DataFrame with appropriate columns
        if chunked:
            columns = [
                "task_name",
                ASSAY,
                "pred_score_min",
                "pred_score_max",
                "acc",
                "f1-score",
                "nb_samples",
            ]
            df_metrics = pd.DataFrame(rows, columns=columns)

            # Convert columns to appropriate types
            df_metrics = df_metrics.astype(
                {
                    "task_name": "str",
                    ASSAY: "str",
                    "pred_score_min": "float",
                    "pred_score_max": "float",
                    "acc": "float",
                    "f1-score": "float",
                    "nb_samples": "int",
                }
            )
        else:
            columns = [
                "task_name",
                ASSAY,
                "min_predScore",
                "acc",
                "f1-score",
                "nb_samples",
            ]
            df_metrics = pd.DataFrame(rows, columns=columns)

            # Convert columns to appropriate types
            df_metrics = df_metrics.astype(
                {
                    "task_name": "str",
                    ASSAY: "str",
                    "min_predScore": "float",
                    "acc": "float",
                    "f1-score": "float",
                    "nb_samples": "int",
                }
            )

        # Apply post-processing rules

        # f1-score on ASSAY task, per assay, doesn't make sense
        df_metrics.loc[df_metrics["task_name"] == ASSAY, "f1-score"] = "NA"

        # metrics for unknown expected class are not defined
        df_metrics.loc[df_metrics[ASSAY] == "avg-all-unknown", ["acc", "f1-score"]] = "NA"

        # acc / f1 for 0 samples is not defined
        df_metrics.loc[df_metrics["nb_samples"] == 0, ["acc", "f1-score"]] = "NA"

        if verbose:
            print(f"Saving {df_metrics.shape[0]} rows")

        # Save to all specified folders
        if isinstance(folders, Path):
            folders = [folders]

        for folder in folders:
            path = folder / filename
            df_metrics.to_csv(
                path,
                sep="\t",
                index=False,
            )
            if verbose:
                print(f"Saved to {path}")

        return df_metrics

    def save_acc_per_assay(
        self,
        metrics: Dict[str, Dict],
        folders: List[Path] | Path,
        filename: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Take a dictionary containing standard metrics for multiple tasks and save to files.
        This is a wrapper around save_metrics_per_assay for backward compatibility.

        Args:
        - metrics: Output of the compute_all_acc_per_assay function.
        - folders: A list of folders to save the results to.
        - filename: The name of the file to save the results to.
        - verbose: Whether to print verbose output.

        Returns:
        - A restructured dataframe with (accuracy, f1, N) for each assay
        """
        return self.save_metrics_per_assay(
            input_metrics=metrics,
            folders=folders,
            filename=filename,
            chunked=False,
            verbose=verbose,
        )

    def save_chunked_acc_per_assay(
        self,
        metrics: Dict[str, Dict],
        folders: List[Path] | Path,
        filename: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Take a dictionary containing chunked metrics for multiple tasks and save to files.
        This is a wrapper around save_metrics_per_assay for backward compatibility.

        Args:
        - metrics: Output of the compute_all_chunked_acc_per_assay function.
        - folders: A list of folders to save the results to.
        - filename: The name of the file to save the results to.
        - verbose: Whether to print verbose output.

        Returns:
        - A restructured dataframe with chunked metrics for each assay
        """
        return self.save_metrics_per_assay(
            input_metrics=metrics,
            folders=folders,
            filename=filename,
            chunked=True,
            verbose=verbose,
        )

    def compute_multiple_metric_formats(
        self,
        preds: pd.DataFrame,
        folders_to_save: List[Path],
        general_filename: str,
        verbose: bool = True,
        return_df: bool = False,
        compute_fct_kwargs: Dict[str, Any] | None = None,
    ) -> None | Dict[str, pd.DataFrame]:
        """Compute and save metrics in different formats.

        Args:
            preds (pd.DataFrame): Dataframe containing predictions.
            folders_to_save (List[Path]): List of folders to save the results to.
            general_filename (str): The filename stem to use for the output files.
                Saves files will will be "<general_filename>.tsv" and "<general_filename>_chunked.tsv"
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            return_df (bool, optional): Whether to return the metrics dataframes. Defaults to False.
            compute_fct_kwargs (Dict[str, Any], optional): Keyword arguments to pass to the compute function. Defaults to None.
        """
        if return_df:
            return_dict = {}

        for filename in [f"{general_filename}.tsv", f"{general_filename}_chunked.tsv"]:
            if "chunked" in filename:
                compute_fct = self.compute_all_chunked_acc_per_assay
                save_fct = self.save_chunked_acc_per_assay
            else:
                compute_fct = self.compute_all_acc_per_assay
                save_fct = self.save_acc_per_assay

            metrics = compute_fct(  # type: ignore
                all_preds=preds,
                verbose=verbose,
                **(compute_fct_kwargs or {}),
            )
            metrics_df = save_fct(
                metrics=metrics,  # type: ignore
                folders=folders_to_save,
                filename=filename,
                verbose=verbose,
            )

            if return_df:
                return_dict[filename] = metrics_df

        if return_df:
            return return_dict

        return None

"""Utility functions for merging classification results."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sjoin(x):
    """join columns if column is not null"""
    return ";".join(x[x.notnull()].astype(str))


def merge_dataframes(
    df1: pd.DataFrame, df2: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """
    Merge two DataFrames by concatenating along the index, aligning common columns
    and appending non-common columns.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame
    df2 (pd.DataFrame): The second DataFrame

    Returns:
    pd.DataFrame: Merged DataFrame with the index name preserved.
    """
    if verbose:
        print(f"Entering shapes: {df1.shape}, {df2.shape}")
    if df1.index.name != df2.index.name:
        raise ValueError(
            f"Index names are different: {df1.index.name} != {df2.index.name}"
        )

    result = pd.merge(df1, df2, on="md5sum", how="outer", suffixes=("", "_merge"))

    result.index.name = df1.index.name
    if verbose:
        print(f"Output shape 1 (pd.merge): {result.shape}")

    dup_cols = [name for name in result.columns if name.endswith("_merge")]
    for dup_col in dup_cols:
        normal_col = dup_col[: -len("_merge")]
        result[normal_col].update(result.pop(dup_col))

    if verbose:
        print(f"Output shape 2 (merge dups): {result.shape}")

    return result


def merge_two_columns(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Return update IN-PLACE dataframe that merged values of col1 and col2, only if they are complementary."""
    df[col1].replace(np.nan, None, inplace=True)
    df[col2].replace(np.nan, None, inplace=True)
    for idx, (val1, val2) in enumerate(df[[col1, col2]].values):
        if val1 is not None and val2 is not None and val1 != val2:
            raise ValueError(
                f"Both {col1} and {col2} are not None in some rows: {idx} - {val1} != {val2}"
            )
    df[col1].update(df.pop(col2))
    return df


def remove_pred_vector(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove the prediction vector from a result dataframe.

    If the "files/epiRR" columns does not exist, it will remove everything after the "1rst/2nd prob ratio" column.
    If there is any metadata after that column, it will also be removed.
    """

    # Prediction vector should be between these columns
    col1 = "1rst/2nd prob ratio"
    col2 = "files/epiRR"

    # column and prediction are transfwred to lowercase to avoid weird mismatches (e.g. with boolean labels)
    column_names = df.columns.str.lower()

    cut_pos_1 = column_names.get_loc(col1)
    try:
        cut_pos_2 = column_names.get_loc(col2)
    except KeyError:
        # Try to determine if the dataframe is already reduced
        predict_labels = df["Predicted class"].str.lower().unique()
        cut_pos_2 = None
        for predict_label in predict_labels:
            if predict_label in column_names:
                # Not a reduced dataframe
                cut_pos_2 = df.shape[1]
                if verbose:
                    print(
                        f"No files/epiRR column, removing everything after '{col1}' column"
                    )
                break

        if cut_pos_2 is None:
            if verbose:
                print("df seems already reduced")
            return df

    df = df.drop(df.columns[cut_pos_1 + 1 : cut_pos_2], axis=1)
    df = df.drop(columns=["EpiRR", "md5sum.1"], errors="ignore")
    return df

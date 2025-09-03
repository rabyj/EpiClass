"""Test utility functions for merging classification results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from epiclass.utils.classification_merging_utils import merge_dataframes


class TestMergeDataframes:
    """Test merge_dataframes function."""

    def test_merge_on_md5sum(self):
        """Test merging on md5sum column with identical values."""
        df1 = pd.DataFrame({"md5sum": ["a1", "a2", "a3"], "value": [1, 2, 3]})
        df2 = pd.DataFrame(
            {
                "md5sum": ["a1", "a2", "a4"],
                "value": [10, 20, 40],
                "extra": ["x1", "x2", "x4"],
            }
        )

        result = merge_dataframes(df1, df2)

        expected = pd.DataFrame(
            {
                "md5sum": ["a1", "a2", "a3", "a4"],
                "value": ["1;10", "2;20", 3, 40],
                "extra": ["x1", "x2", np.nan, "x4"],
            }
        )

        # Convert to proper types for comparison
        expected["value"] = expected["value"].astype(object)

        assert_frame_equal(result, expected, check_dtype=False)

    def test_merge_on_custom_column(self):
        """Test merging on a custom column."""
        df1 = pd.DataFrame({"id": ["id1", "id2", "id3"], "data": ["d1", "d2", "d3"]})
        df2 = pd.DataFrame({"id": ["id1", "id2", "id4"], "data": ["d1-alt", "d2", "d4"]})

        result = merge_dataframes(df1, df2, on="id")

        expected = pd.DataFrame(
            {"id": ["id1", "id2", "id3", "id4"], "data": ["d1;d1-alt", "d2", "d3", "d4"]}
        )

        assert_frame_equal(result, expected)

    def test_merge_on_index(self):
        """Test merging on index."""
        df1 = pd.DataFrame(
            {"col1": [1, 2, 3]}, index=pd.Index(["a", "b", "c"], name="idx")
        )

        df2 = pd.DataFrame(
            {"col2": [10, 20, 40]}, index=pd.Index(["a", "b", "d"], name="idx")
        )

        result = merge_dataframes(df1, df2, on="index")

        expected = pd.DataFrame(
            {"col1": [1.0, 2.0, 3.0, np.nan], "col2": [10.0, 20.0, np.nan, 40.0]},
            index=pd.Index(["a", "b", "c", "d"], name="idx"),
        )

        assert_frame_equal(result, expected)

    def test_fallback_to_filename_when_md5sum_missing(self):
        """Test falling back to filename when md5sum isn't in both dataframes."""
        df1 = pd.DataFrame(
            {"filename": ["file1.txt", "file2.txt", "file3.txt"], "size": [100, 200, 300]}
        )
        df2 = pd.DataFrame(
            {
                "filename": ["file1.txt", "file2.txt", "file4.txt"],
                "modified": ["2023-01-01", "2023-01-02", "2023-01-04"],
            }
        )

        result = merge_dataframes(df1, df2)

        expected = pd.DataFrame(
            {
                "filename": ["file1.txt", "file2.txt", "file3.txt", "file4.txt"],
                "size": [100.0, 200.0, 300.0, np.nan],
                "modified": ["2023-01-01", "2023-01-02", np.nan, "2023-01-04"],
            }
        )

        assert_frame_equal(result, expected)

    def test_complex_merging_with_different_values(self):
        """Test merging with a mix of same and different values."""
        df1 = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "Bob", "Charlie", "Dave"],
                "score": [85, 90, 78, 92],
            }
        )

        df2 = pd.DataFrame(
            {
                "id": [1, 2, 3, 5],
                "name": ["Alice", "Robert", "Charlie", "Eve"],
                "age": [25, 30, 22, 28],
            }
        )

        result = merge_dataframes(df1, df2, on="id")

        expected = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob;Robert", "Charlie", "Dave", "Eve"],
                "score": [85.0, 90.0, 78.0, 92.0, np.nan],
                "age": [25.0, 30.0, 22.0, np.nan, 28.0],
            }
        )

        assert_frame_equal(result, expected, check_dtype=False)

    def test_null_values_handling(self):
        """Test how null values are handled when merging."""
        df1 = pd.DataFrame(
            {
                "key": ["a", "b", "c", "d"],
                "value1": [1, np.nan, 3, 4],
                "common": ["common1", "common2", np.nan, "common4"],
            }
        )

        df2 = pd.DataFrame(
            {
                "key": ["a", "b", "c", "e"],
                "value2": [10, 20, np.nan, 50],
                "common": ["common1", "diff2", "common3", np.nan],
            }
        )

        result = merge_dataframes(df1, df2, on="key")

        expected = pd.DataFrame(
            {
                "key": ["a", "b", "c", "d", "e"],
                "value1": [1.0, np.nan, 3.0, 4.0, np.nan],
                "common": ["common1", "common2;diff2", "common3", "common4", np.nan],
                "value2": [10.0, 20.0, np.nan, np.nan, 50.0],
            }
        )

        # When comparing frames with NaNs, use check_like=True
        assert_frame_equal(result, expected, check_dtype=False)

    def test_error_when_no_common_columns(self):
        """Test that an error is raised when no common columns exist."""
        df1 = pd.DataFrame({"col1": [1, 2, 3]})
        df2 = pd.DataFrame({"col2": [10, 20, 30]})

        with pytest.raises(ValueError):
            merge_dataframes(df1, df2, on="nonexistent_col")

    def test_different_index_names_error(self):
        """Test that an error is raised when index names differ but trying to merge on index."""
        df1 = pd.DataFrame({"col1": [1, 2, 3]}, index=pd.Index([0, 1, 2], name="index1"))
        df2 = pd.DataFrame(
            {"col2": [10, 20, 30]}, index=pd.Index([0, 1, 3], name="index2")
        )

        with pytest.raises(ValueError):
            merge_dataframes(df1, df2, on="index")

    def test_different_dtypes(self):
        """Test merging columns with different data types."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "value": [100, 200, 300]})
        df2 = pd.DataFrame(
            {"id": [1, 2, 4], "value": ["100", "220", "400"]}  # String type
        )

        result = merge_dataframes(df1, df2, on="id")

        # When merging different types, values become strings when combined
        expected = pd.DataFrame(
            {"id": [1, 2, 3, 4], "value": ["100;100", "200;220", 300, "400"]}
        )

        # Make sure our expected dataframe has proper types
        expected["value"] = expected["value"].astype(object)

        assert_frame_equal(result, expected, check_dtype=False)

    def test_mixed_numeric_string_combination(self):
        """Test the combination of numeric and string values with the semicolon separator."""
        df1 = pd.DataFrame({"id": ["ID1", "ID2", "ID3"], "mixed": [10, "text", 30.5]})

        df2 = pd.DataFrame({"id": ["ID1", "ID2", "ID4"], "mixed": ["ten", "text", False]})

        result = merge_dataframes(df1, df2, on="id")

        expected = pd.DataFrame(
            {"id": ["ID1", "ID2", "ID3", "ID4"], "mixed": ["10;ten", "text", 30.5, False]}
        )

        # Convert to proper types for comparison
        expected["mixed"] = expected["mixed"].astype(object)

        assert_frame_equal(result, expected, check_dtype=False)

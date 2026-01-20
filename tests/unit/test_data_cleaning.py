"""
Unit tests for data cleaning utilities.

Tests:
- clean_numeric
- clean_numeric_vector
- detect_outliers
- handle_outliers
- clean_dataframe
- get_cleaning_summary
- check_missing_data_impact
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.data_cleaning import (  # isort: skip
    check_missing_data_impact,
    clean_dataframe,
    clean_numeric,
    clean_numeric_vector,
    detect_outliers,
    get_cleaning_summary,
    handle_outliers,
    apply_missing_values_to_df,
)

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestNumericCleaning:
    """Tests for numeric cleaning functions."""

    def test_clean_numeric_scalar(self):
        """Test clean_numeric with various scalar inputs."""
        assert clean_numeric(">100") == 100.0
        assert clean_numeric("1,234.56") == 1234.56
        assert np.isnan(clean_numeric(None))
        assert np.isnan(clean_numeric("abc"))
        assert clean_numeric("$500") == 500.0
        assert clean_numeric("10%") == 10.0
        assert clean_numeric("(100)") == 100.0

    def test_clean_numeric_vector(self):
        """Test clean_numeric_vector with series input."""
        test_series = pd.Series([">100", "1,234.56", None, "abc", "$500"])
        result = clean_numeric_vector(test_series)

        expected = pd.Series([100.0, 1234.56, np.nan, np.nan, 500.0])
        pd.testing.assert_series_equal(
            result.reset_index(drop=True), expected, check_names=False
        )


class TestOutlierDetection:
    """Tests for outlier detection and handling."""

    def test_detect_outliers_iqr(self):
        """Test detect_outliers using IQR method."""
        test_series = pd.Series([1, 2, 3, 4, 5, 100])
        mask, stats = detect_outliers(test_series, method="iqr")

        assert mask.iloc[5]
        assert not mask.iloc[0:5].any()
        assert stats["method"] == "iqr"
        assert stats["outlier_count"] == 1

    def test_detect_outliers_zscore(self):
        """Test detect_outliers using Z-score method."""
        # Using a larger set to have meaningful z-score
        test_series = pd.Series([10, 10, 11, 10, 11, 10, 100])
        mask, stats = detect_outliers(test_series, method="zscore", threshold=2.0)

        assert mask.iloc[6]
        assert stats["method"] == "zscore"

    def test_handle_outliers_actions(self):
        """Test various outlier handling actions."""
        test_series = pd.Series([1, 2, 3, 4, 5, 100])

        # Test flag (NaN replacement)
        flagged = handle_outliers(test_series, action="flag")
        assert np.isnan(flagged.iloc[5])

        # Test winsorize
        winsorized = handle_outliers(test_series, action="winsorize")
        assert winsorized.iloc[5] < 100
        assert winsorized.iloc[5] > 5

        # Test cap
        capped = handle_outliers(test_series, action="cap")
        assert capped.iloc[5] < 100


class TestDataFrameCleaning:
    """Tests for DataFrame cleaning and reporting."""

    def test_clean_dataframe_granular(self):
        """Test clean_dataframe with mixed dirty data."""
        test_df = pd.DataFrame(
            {
                "Dirty_Numeric": [">100", "1,234", "ERROR", "500"],
                "Strictly_Text": ["A", "B", "C", "D"],
                "Mixed": ["10", "20", "bad", "worse"],
            }
        )

        cleaned_df, _report = clean_dataframe(test_df)

        # Dirty_Numeric should be float
        assert pd.api.types.is_numeric_dtype(cleaned_df["Dirty_Numeric"])
        assert cleaned_df["Dirty_Numeric"].isna().sum() == 1  # ERROR becomes NaN

        # Mixed should be float because threshold is 0.3 (2/4 = 50% > 30%)
        assert pd.api.types.is_numeric_dtype(cleaned_df["Mixed"])
        assert cleaned_df["Mixed"].isna().sum() == 2

        # Strictly_Text should stay object
        assert cleaned_df["Strictly_Text"].dtype == object

    def test_get_cleaning_summary(self):
        """Test summary generation."""
        test_df = pd.DataFrame({"a": [1, 2, 3]})
        _, report = clean_dataframe(test_df)
        summary = get_cleaning_summary(report)

        assert "DATA CLEANING SUMMARY" in summary
        assert "Original shape: (3, 1)" in summary


class TestMissingDataImpact:
    """Tests for check_missing_data_impact with missing_codes."""

    def test_check_missing_data_impact_with_coded_values(self):
        """Test that impact is accurately reported when coded values are used."""
        df_original = pd.DataFrame(
            {"age": [25, -99, 30, 999, 40], "score": [100, 85, -99, 90, 95]}
        )
        # Mock cleaned data (complete case)
        df_clean = pd.DataFrame({"age": [25.0, 40.0], "score": [100.0, 95.0]})

        var_meta = {
            "age": {"label": "Age", "missing_values": [-99, 999]},
            "score": {"label": "Score", "missing_values": [-99]},
        }

        # Case 1: Without explicit missing_codes passed to check_missing_data_impact
        # (It should still use var_meta if passed correctly through apply_missing_values_to_df elsewhere)
        # But here we test the direct call with missing_codes

        missing_codes = {"age": [-99, 999], "score": [-99]}

        result = check_missing_data_impact(
            df_original, df_clean, var_meta, missing_codes=missing_codes
        )

        # Total rows removed = 5 - 2 = 3
        assert result["rows_removed"] == 3

        # Age should have 2 missing (-99, 999)
        assert result["observations_lost"]["age"]["count"] == 2
        # Score should have 1 missing (-99)
        assert result["observations_lost"]["score"]["count"] == 1

        assert "age" in result["variables_affected"]
        assert "score" in result["variables_affected"]

    def test_apply_missing_values_dict_support(self):
        """Test that apply_missing_values_to_df supports dict-based missing_codes."""
        df = pd.DataFrame({"a": [1, -99, 3], "b": [10, -88, 30]})
        missing_codes = {"a": [-99], "b": [-88]}

        result = apply_missing_values_to_df(df, {}, missing_codes=missing_codes)

        assert result["a"].isna().sum() == 1
        assert result["b"].isna().sum() == 1
        assert np.isnan(result.loc[1, "a"])
        assert np.isnan(result.loc[1, "b"])


if __name__ == "__main__":
    pytest.main([__file__])

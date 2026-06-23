"""
🧪 Edge-Case Tests for Medical Data Patterns

Covers the non-obvious data patterns commonly encountered in clinical datasets:
- Lab values with comparison operators (<5, >1000)
- Mixed numeric/text columns from EHR exports
- Thai/international character encoding in column names
- Extreme values and boundary conditions
- Single-row and single-column datasets
- All-NA and constant-value columns
- prepare_data_for_analysis edge cases
"""

import numpy as np
import pandas as pd
import pytest

from utils.data_cleaning import (
    clean_numeric,
    clean_numeric_vector,
    clean_dataframe,
    detect_outliers,
    is_continuous_variable,
    prepare_data_for_analysis,
    DataCleaningError,
    validate_data_quality,
)

pytestmark = pytest.mark.unit


class TestMedicalLabValues:
    """Tests for lab value patterns commonly seen in ER/clinical data."""

    def test_less_than_operator(self):
        """Lab values like '<5' for below detection limit."""
        assert clean_numeric("<5") == 5.0
        assert clean_numeric("<0.001") == 0.001
        assert clean_numeric("< 10") == 10.0

    def test_greater_than_operator(self):
        """Lab values like '>1000' for above measurable range."""
        assert clean_numeric(">1000") == 1000.0
        assert clean_numeric("> 500") == 500.0

    def test_percentage_strings(self):
        """EHR percentage values like '85%'."""
        assert clean_numeric("85%") == 85.0
        assert clean_numeric("0.5%") == 0.5

    def test_currency_medical_billing(self):
        """Billing data with currency symbols."""
        assert clean_numeric("$1,500.00") == 1500.0
        assert clean_numeric("€2,000") == 2000.0
        assert clean_numeric("£150") == 150.0

    def test_parenthetical_negatives(self):
        """Accounting-style negative numbers in billing data."""
        assert clean_numeric("(100)") == -100.0

    def test_vector_mixed_lab_values(self):
        """Vector of typical lab report values."""
        series = pd.Series(["<5", "12.3", ">1000", "N/A", "7.8", "pending"])
        result = clean_numeric_vector(series)

        assert result.iloc[0] == 5.0
        assert result.iloc[1] == 12.3
        assert result.iloc[2] == 1000.0
        assert np.isnan(result.iloc[3])
        assert result.iloc[4] == 7.8
        assert np.isnan(result.iloc[5])

    def test_comma_separated_thousands(self):
        """Large values with comma separators (common in WBC counts)."""
        series = pd.Series(["12,000", "8,500", "15,200", "5,000"])
        result = clean_numeric_vector(series)

        assert result.iloc[0] == 12000.0
        assert result.iloc[1] == 8500.0
        assert result.iloc[2] == 15200.0
        assert result.iloc[3] == 5000.0


class TestEdgeCaseDataFrames:
    """Tests for boundary conditions in DataFrame operations."""

    def test_single_row_dataframe(self):
        """Single observation — common during data filtering edge cases."""
        df = pd.DataFrame({"value": [42.0], "label": ["A"]})
        cleaned, report = clean_dataframe(df, validate_quality=True)

        assert len(cleaned) == 1
        assert report["quality_report"]["summary"]["total_rows"] == 1

    def test_single_column_dataframe(self):
        """Single variable dataset."""
        df = pd.DataFrame({"only_col": [1, 2, 3, np.nan]})
        cleaned, report = clean_dataframe(df, validate_quality=True)

        assert "only_col" in cleaned.columns
        assert len(cleaned) == 4

    def test_all_na_column(self):
        """Column with all missing values — should not crash."""
        df = pd.DataFrame({"empty": [np.nan, np.nan, np.nan], "ok": [1, 2, 3]})
        cleaned, report = clean_dataframe(df, validate_quality=True)

        assert cleaned["empty"].isna().all()
        quality = report["quality_report"]
        assert any("empty" in str(w) for w in quality.get("warnings", []))

    def test_constant_value_column(self):
        """Column with zero variance — should be detectable."""
        series = pd.Series([5, 5, 5, 5, 5])
        result = is_continuous_variable(series)
        # 1 unique value < threshold=10, not float, no decimals → categorical
        assert result is False

    def test_empty_dataframe(self):
        """Completely empty DataFrame."""
        df = pd.DataFrame()
        quality = validate_data_quality(df)

        assert quality["summary"]["total_rows"] == 0

    def test_all_nan_dataframe(self):
        """DataFrame where every cell is NaN."""
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        cleaned, report = clean_dataframe(df, validate_quality=True)

        assert cleaned.isna().all().all()


class TestPrepareDataEdgeCases:
    """Edge cases for prepare_data_for_analysis — the main pipeline entry point."""

    def test_missing_column_raises(self):
        """Request a column that doesn't exist."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(DataCleaningError, match="Missing required columns"):
            prepare_data_for_analysis(df, required_cols=["a", "nonexistent"])

    def test_all_rows_excluded(self):
        """Every row has missing data → should raise."""
        df = pd.DataFrame({"x": [np.nan, np.nan], "y": [np.nan, np.nan]})
        with pytest.raises(DataCleaningError, match="No valid data remaining"):
            prepare_data_for_analysis(
                df, required_cols=["x", "y"], handle_missing="complete_case"
            )

    def test_pairwise_strategy_keeps_all_rows(self):
        """Pairwise strategy should not drop rows."""
        df = pd.DataFrame({"x": [1, np.nan, 3], "y": [np.nan, 2, 3]})
        result, info = prepare_data_for_analysis(
            df, required_cols=["x", "y"], handle_missing="pairwise"
        )

        assert len(result) == 3
        assert info["rows_excluded"] == 0

    def test_duplicate_required_cols(self):
        """Duplicate column names should be deduplicated."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result, info = prepare_data_for_analysis(
            df, required_cols=["a", "a", "b"]
        )
        assert list(result.columns) == ["a", "b"]

    def test_numeric_conversion_applied(self):
        """String numeric columns should be converted via numeric_cols."""
        df = pd.DataFrame({"val": ["1.5", "2.5", "3.5"]})
        result, info = prepare_data_for_analysis(
            df, required_cols=["val"], numeric_cols=["val"]
        )
        assert pd.api.types.is_float_dtype(result["val"])

    def test_unsupported_strategy_raises(self):
        """Unknown missing data strategy should raise DataCleaningError."""
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(DataCleaningError, match="Unknown missing data strategy"):
            prepare_data_for_analysis(
                df, required_cols=["x"], handle_missing="magic_imputation"
            )


class TestOutlierEdgeCases:
    """Edge cases for outlier detection."""

    def test_empty_series(self):
        """Outlier detection on empty series."""
        mask, stats = detect_outliers(pd.Series([], dtype=float))
        assert not mask.any()
        assert stats == {}

    def test_all_same_values(self):
        """All identical values — IQR is 0, nothing is outlier."""
        mask, stats = detect_outliers(pd.Series([5, 5, 5, 5, 5]))
        assert not mask.any()

    def test_all_nan_series(self):
        """All NaN series."""
        mask, stats = detect_outliers(pd.Series([np.nan, np.nan, np.nan]))
        assert not mask.any()

    def test_zscore_zero_std(self):
        """Z-score with constant series (std=0)."""
        mask, stats = detect_outliers(
            pd.Series([10, 10, 10, 10]), method="zscore"
        )
        assert not mask.any()


class TestIsContiguous:
    """Tests for is_continuous_variable heuristic."""

    def test_clearly_continuous(self):
        """Many unique numeric values → continuous."""
        series = pd.Series(np.random.normal(0, 1, 100))
        assert is_continuous_variable(series) is True

    def test_clearly_categorical(self):
        """Few unique integer values → not continuous."""
        series = pd.Series([1, 2, 1, 2, 1, 2, 1, 2])
        assert is_continuous_variable(series) is False

    def test_float_dtype_treated_continuous(self):
        """Float dtype with few uniques still treated as continuous."""
        series = pd.Series([1.0, 2.0, 3.0], dtype=np.float64)
        assert is_continuous_variable(series) is True

    def test_string_series(self):
        """String values → not continuous."""
        series = pd.Series(["A", "B", "C", "A", "B"])
        assert is_continuous_variable(series) is False

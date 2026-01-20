"""
🧪 Unit Tests for Formatting Utilities
File: tests/unit/test_formatting.py

Tests all functions in utils/formatting.py:
- format_p_value: P-value formatting with styling
- format_ci_html: Confidence interval HTML formatting
- get_badge_html: Badge generation
- create_missing_data_report_html: Missing data report generation
- _get_pct: Percentage extraction helper

Run with: pytest tests/unit/test_formatting.py -v
"""

import numpy as np
import pandas as pd
import pytest

from utils.formatting import (
    _get_pct,
    create_missing_data_report_html,
    format_ci_html,
    format_p_value,
    get_badge_html,
)

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================================
# Tests for format_p_value
# ============================================================================


class TestFormatPValue:
    """Tests for P-value formatting function."""

    @pytest.mark.parametrize(
        "p_value,use_style,expected_contains",
        [
            # Significant p-values (< 0.05) with styling
            (0.001, True, "0.001"),
            (0.045, True, "0.045"),
            (0.049, True, "0.049"),
            # Non-significant p-values
            (0.05, True, "0.050"),
            (0.10, True, "0.100"),
            (0.50, True, "0.500"),
            # Edge cases - very small
            (0.0001, True, "0.001"),  # Should show <0.001
            (0.00001, True, "0.001"),
            # Edge cases - very large
            (0.9999, True, "0.999"),  # Should show >0.999
            # Without styling
            (0.001, False, "0.001"),
            (0.045, False, "0.045"),
        ],
    )
    def test_format_p_value_parametrized(self, p_value, use_style, expected_contains):
        """Test P-value formatting with various inputs.

        Given: A p-value and styling preference
        When: format_p_value is called
        Then: The result contains the expected formatted value
        """
        result = format_p_value(p_value, use_style=use_style)
        assert expected_contains in result or "&lt;" in result or "&gt;" in result

    def test_format_p_value_significant_has_styling(self):
        """Test that significant P-values include HTML styling.

        Given: A significant p-value (< 0.05)
        When: format_p_value is called with use_style=True
        Then: The result includes a span tag with styling
        """
        result = format_p_value(0.01, use_style=True)
        assert "<span" in result
        assert "style=" in result

    def test_format_p_value_non_significant_no_styling(self):
        """Test that non-significant P-values have no special styling.

        Given: A non-significant p-value (>= 0.05)
        When: format_p_value is called
        Then: The result does not include a span tag
        """
        result = format_p_value(0.10, use_style=True)
        assert "<span" not in result

    @pytest.mark.parametrize(
        "invalid_value",
        [
            np.nan,
            np.inf,
            -np.inf,
            float("nan"),
        ],
    )
    def test_format_p_value_invalid_values(self, invalid_value):
        """Test handling of invalid P-values.

        Given: An invalid p-value (NaN, Inf)
        When: format_p_value is called
        Then: Returns "NA" or "-"
        """
        result_styled = format_p_value(invalid_value, use_style=True)
        result_plain = format_p_value(invalid_value, use_style=False)

        assert result_styled == "NA"
        assert result_plain == "-"


# ============================================================================
# Tests for format_ci_html
# ============================================================================


class TestFormatCIHtml:
    """Tests for confidence interval HTML formatting."""

    def test_significant_ci_excludes_null(self):
        """Test CI that excludes null value gets styling.

        Given: A CI that excludes 1.0 (e.g., 1.2-2.5)
        When: format_ci_html is called with direction='exclude'
        Then: The result includes styling for significance
        """
        result = format_ci_html("1.20 - 2.50", lower=1.2, upper=2.5, null_val=1.0)
        assert "<span" in result
        assert "style=" in result
        assert "bold" in result.lower()

    def test_non_significant_ci_includes_null(self):
        """Test CI that includes null value has no styling.

        Given: A CI that includes 1.0 (e.g., 0.8-1.5)
        When: format_ci_html is called
        Then: The result has no styling
        """
        result = format_ci_html("0.80 - 1.50", lower=0.8, upper=1.5, null_val=1.0)
        assert "<span" not in result
        assert result == "0.80 - 1.50"

    def test_significant_ci_greater_direction(self):
        """Test CI with 'greater' direction.

        Given: A CI entirely above null value
        When: format_ci_html is called with direction='greater'
        Then: The result includes significance styling
        """
        result = format_ci_html(
            "1.50 - 3.00", lower=1.5, upper=3.0, null_val=1.0, direction="greater"
        )
        assert "<span" in result

    @pytest.mark.parametrize(
        "lower,upper",
        [
            (np.nan, 2.0),
            (1.0, np.nan),
            (np.inf, 2.0),
            (1.0, np.inf),
        ],
    )
    def test_invalid_bounds_returns_original(self, lower, upper):
        """Test that invalid bounds return original string unchanged.

        Given: CI with NaN or Inf bounds
        When: format_ci_html is called
        Then: The original string is returned unchanged
        """
        ci_str = "1.00 - 2.00"
        result = format_ci_html(ci_str, lower=lower, upper=upper)
        assert result == ci_str


# ============================================================================
# Tests for get_badge_html
# ============================================================================


class TestGetBadgeHtml:
    """Tests for HTML badge generation."""

    @pytest.mark.parametrize(
        "level,expected_bg_color",
        [
            ("success", "#d4edda"),
            ("warning", "#fff3cd"),
            ("danger", "#f8d7da"),
            ("info", "#d1ecf1"),
            ("neutral", "#e2e3e5"),
        ],
    )
    def test_badge_levels(self, level, expected_bg_color):
        """Test badge colors for different levels.

        Given: A badge level (success, warning, danger, info, neutral)
        When: get_badge_html is called
        Then: The result includes the correct background color
        """
        result = get_badge_html("Test", level=level)
        assert expected_bg_color in result

    def test_badge_contains_text(self):
        """Test that badge contains the provided text.

        Given: Text to display in badge
        When: get_badge_html is called
        Then: The text is present in the output
        """
        result = get_badge_html("My Badge Text", level="info")
        assert "My Badge Text" in result

    def test_badge_structure(self):
        """Test badge HTML structure.

        Given: Any badge text and level
        When: get_badge_html is called
        Then: Output is a properly formatted span tag
        """
        result = get_badge_html("Test", level="success")
        assert result.startswith("<span")
        assert result.endswith("</span>")
        assert "style=" in result

    def test_unknown_level_uses_neutral(self):
        """Test that unknown levels fall back to neutral styling.

        Given: An unknown badge level
        When: get_badge_html is called
        Then: Neutral colors are used
        """
        result = get_badge_html("Test", level="unknown_level")
        assert "#e2e3e5" in result  # Neutral background


# ============================================================================
# Tests for create_missing_data_report_html
# ============================================================================


class TestCreateMissingDataReportHtml:
    """Tests for missing data report HTML generation."""

    @pytest.fixture
    def sample_missing_data_info(self):
        """Sample missing data info for testing."""
        return {
            "strategy": "complete-case",
            "rows_analyzed": 80,
            "rows_excluded": 20,
            "summary_before": [
                {
                    "Variable": "age",
                    "Type": "Continuous",
                    "N_Valid": 90,
                    "N_Missing": 10,
                    "Pct_Missing": "10.0%",
                },
                {
                    "Variable": "weight",
                    "Type": "Continuous",
                    "N_Valid": 70,
                    "N_Missing": 30,
                    "Pct_Missing": "30.0%",
                },
                {
                    "Variable": "group",
                    "Type": "Categorical",
                    "N_Valid": 100,
                    "N_Missing": 0,
                    "Pct_Missing": "0.0%",
                },
            ],
        }

    @pytest.fixture
    def sample_var_meta(self):
        """Sample variable metadata for testing."""
        return {
            "age": {"label": "Patient Age (years)"},
            "weight": {"label": "Body Weight (kg)"},
            "group": {"label": "Treatment Group"},
        }

    def test_report_contains_strategy(self, sample_missing_data_info, sample_var_meta):
        """Test that report includes strategy information.

        Given: Missing data info with strategy
        When: create_missing_data_report_html is called
        Then: Strategy is displayed in the output
        """
        result = create_missing_data_report_html(
            sample_missing_data_info, sample_var_meta
        )
        assert "complete-case" in result
        assert "Strategy" in result

    def test_report_contains_row_counts(
        self, sample_missing_data_info, sample_var_meta
    ):
        """Test that report includes row counts.

        Given: Missing data info with row counts
        When: create_missing_data_report_html is called
        Then: Row counts are displayed
        """
        result = create_missing_data_report_html(
            sample_missing_data_info, sample_var_meta
        )
        assert "80" in result  # rows_analyzed
        assert "20" in result  # rows_excluded

    def test_report_shows_variables_with_missing(
        self, sample_missing_data_info, sample_var_meta
    ):
        """Test that variables with missing data are shown.

        Given: Variables with missing data
        When: create_missing_data_report_html is called
        Then: Those variables appear in the report table
        """
        result = create_missing_data_report_html(
            sample_missing_data_info, sample_var_meta
        )
        assert "Patient Age" in result
        assert "Body Weight" in result
        # group has 0 missing, should not appear in missing table
        # (but might still appear if implementation shows all)

    def test_report_structure(self, sample_missing_data_info, sample_var_meta):
        """Test that report has correct HTML structure.

        Given: Valid missing data info
        When: create_missing_data_report_html is called
        Then: Output has proper HTML structure
        """
        result = create_missing_data_report_html(
            sample_missing_data_info, sample_var_meta
        )
        assert "<div" in result
        assert "<table" in result
        assert "<h4>" in result

    def test_empty_missing_data(self, sample_var_meta):
        """Test handling of empty missing data.

        Given: No missing data
        When: create_missing_data_report_html is called
        Then: Report generates without errors
        """
        empty_info = {
            "strategy": "complete-case",
            "rows_analyzed": 100,
            "rows_excluded": 0,
            "summary_before": [],
        }
        result = create_missing_data_report_html(empty_info, sample_var_meta)
        assert "Strategy" in result
        assert "100" in result

    def test_xss_protection(self, sample_var_meta):
        """Test that HTML is properly escaped to prevent XSS.

        Given: Strategy with potential XSS content
        When: create_missing_data_report_html is called
        Then: Content is HTML-escaped
        """
        malicious_info = {
            "strategy": "<script>alert('xss')</script>",
            "rows_analyzed": 100,
            "rows_excluded": 0,
            "summary_before": [],
        }
        result = create_missing_data_report_html(malicious_info, sample_var_meta)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ============================================================================
# Tests for _get_pct helper
# ============================================================================


class TestGetPct:
    """Tests for percentage extraction helper."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("50.0%", 50.0),
            ("100%", 100.0),
            ("0%", 0.0),
            ("33.33%", 33.33),
            ("75.5%", 75.5),
        ],
    )
    def test_valid_percentage_strings(self, input_str, expected):
        """Test extraction of valid percentage strings.

        Given: A valid percentage string
        When: _get_pct is called
        Then: The numeric value is returned
        """
        result = _get_pct(input_str)
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "invalid",
            "",
            None,
            "abc%",
        ],
    )
    def test_invalid_inputs_return_zero(self, invalid_input):
        """Test that invalid inputs return 0.0.

        Given: An invalid input
        When: _get_pct is called
        Then: Returns 0.0
        """
        result = _get_pct(invalid_input)
        assert result == 0.0


# ============================================================================
# Entry Point
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

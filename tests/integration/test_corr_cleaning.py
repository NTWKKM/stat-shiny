import numpy as np
import pandas as pd
import pytest

from config import CONFIG
from utils import correlation
from utils.formatting import create_missing_data_report_html


class TestCorrPipeline:
    @pytest.fixture
    def mock_data(self):
        df = pd.DataFrame(
            {
                "var1": [1.0, 2.0, np.nan, 4.0, 5.0, 999.0],
                "var2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
                "var3": [1, 2, 3, 4, 5, 999],  # Categorical-ish
                "category": ["A", "B", "A", "B", "A", "B"],
            }
        )
        return df

    @pytest.fixture
    def var_meta(self):
        return {
            "var1": {"label": "Variable 1", "type": "numeric"},
            "var2": {"label": "Variable 2", "type": "numeric"},
            "var3": {"label": "Variable 3", "type": "numeric"},
        }

    def setup_method(self):
        # Save original values to restore later
        self._orig_strategy = CONFIG.get("analysis.missing.strategy")
        self._orig_user_defined = CONFIG.get("analysis.missing.user_defined_values")

        # Reset Config
        CONFIG.update("analysis.missing.strategy", "complete-case")
        CONFIG.update("analysis.missing.user_defined_values", [999, 999.0])

    def teardown_method(self):
        # Restore original values
        CONFIG.update("analysis.missing.strategy", self._orig_strategy)
        CONFIG.update("analysis.missing.user_defined_values", self._orig_user_defined)

    def test_calculate_correlation_pipeline(self, mock_data, var_meta):
        """Test single pair correlation with pipeline integration"""
        # Strategy: complete-case (default)
        # var1 has 1 NaN, and 1 user-defined missing (999.0)
        # var2 has no missing
        # Total rows: 6.
        # Excluded: index 2 (NaN), index 5 (999.0).
        # Expected N: 4.

        results, err, fig = correlation.calculate_correlation(
            mock_data, "var1", "var2", var_meta=var_meta
        )

        assert err is None
        assert results is not None
        assert results["N"] == 4
        assert "missing_data_info" in results

        missing_info = results["missing_data_info"]
        assert missing_info["rows_original"] == 6
        assert missing_info["rows_excluded"] == 2
        assert missing_info["strategy"] == "complete-case"

        # Check HTML report generation
        html = create_missing_data_report_html(missing_info, var_meta)
        assert "Missing Data Summary" in html
        assert "<b>2</b>" in html  # 2 excluded rows

    def test_compute_correlation_matrix_pipeline(self, mock_data, var_meta):
        """Test correlation matrix with pipeline integration (pairwise strategy)"""
        # Matrix forces "pairwise" strategy (no row dropping)
        # Input rows: 6
        # var1: 2 missing (NaN, 999)
        # var2: 0 missing
        # var3: 1 missing (999)

        cols = ["var1", "var2", "var3"]

        corr_matrix, fig, summary = correlation.compute_correlation_matrix(
            mock_data, cols, var_meta=var_meta
        )

        assert corr_matrix is not None
        assert summary is not None
        assert "missing_data_info" in summary

        missing_info = summary["missing_data_info"]
        assert missing_info["strategy"] == "pairwise (matrix)"
        assert missing_info["rows_excluded"] == 0  # Should NOT drop rows
        assert missing_info["rows_original"] == 6

        # Check detailed missing counts
        details = missing_info["summary_before"]
        # var1: 1 NaN + 1 user-defined (999) = 2 missing
        var1_det = next(d for d in details if d["Variable"] == "var1")
        assert var1_det["N_Missing"] == 2

        # var3: 1 user-defined (999) = 1 missing
        var3_det = next(d for d in details if d["Variable"] == "var3")
        assert var3_det["N_Missing"] == 1

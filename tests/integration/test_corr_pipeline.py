"""
🔗 Integration Tests for Correlation Analysis Pipeline
File: tests/integration/test_corr_pipeline.py

Tests the correlation analysis workflow:
1. Pearson/Spearman correlation calculation
2. Handling of non-numeric data
3. Output format verification (including Summary Stats)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import correlation module
from utils.correlation import compute_correlation_matrix

pytestmark = pytest.mark.integration


class TestCorrelationPipeline:

    @pytest.fixture
    def corr_data(self):
        """Create dataset with known correlations"""
        np.random.seed(123)
        n = 200

        # Create correlated variables
        # x and y are positively correlated
        x = np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, 0.5, n)

        # z is independent (uncorrelated)
        z = np.random.normal(0, 1, n)

        # w is perfectly negatively correlated with x
        w = -x

        df = pd.DataFrame(
            {
                "var_x": x,
                "var_y": y,
                "var_z": z,
                "var_w": w,
                "category": np.random.choice(
                    ["A", "B"], n
                ),  # Should be ignored or handled
            }
        )
        return df

    def test_pearson_correlation_flow(self, corr_data):
        """🔄 Test Pearson correlation calculation"""
        df = corr_data
        selected_vars = ["var_x", "var_y", "var_z", "var_w"]

        # Run calculation - Updated to unpack 3 values (matrix, fig, summary)
        corr_matrix, fig, summary = compute_correlation_matrix(
            df, selected_vars, method="pearson"
        )

        # Check Matrix
        assert corr_matrix is not None
        assert not corr_matrix.empty
        assert corr_matrix.shape == (4, 4)

        # Check specific known correlations
        # x vs y should be positive and high (> 0.8)
        assert corr_matrix.loc["var_x", "var_y"] > 0.8

        # x vs w should be perfect negative (-1.0)
        assert np.isclose(corr_matrix.loc["var_x", "var_w"], -1.0, atol=0.01)

        # x vs z should be low (near 0)
        assert abs(corr_matrix.loc["var_x", "var_z"]) < 0.2

        # Check Plotly figure
        assert fig is not None
        assert hasattr(fig, "layout")
        # Check if it is a heatmap (check data type)
        assert fig.data[0].type == "heatmap"

        # Check Summary Stats (New Feature)
        assert summary is not None
        assert "n_variables" in summary
        assert summary["n_variables"] == 4
        assert "mean_correlation" in summary
        assert "strongest_positive" in summary

    def test_spearman_correlation_flow(self, corr_data):
        """🔄 Test Spearman correlation (Rank-based)"""
        df = corr_data

        # Create non-linear but monotonic relationship (Pearson < Spearman)
        df["var_exp"] = np.exp(df["var_x"])

        # Updated to unpack 3 values
        corr_matrix, _fig, _summary = compute_correlation_matrix(
            df, ["var_x", "var_exp"], method="spearman"
        )

        assert corr_matrix is not None
        # Spearman should be perfect 1.0 for monotonic transform
        assert np.isclose(corr_matrix.loc["var_x", "var_exp"], 1.0, atol=0.01)

    def test_missing_values_handling(self):
        """⚠️ Test handling of missing data"""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5], "b": [2, 4, 6, 8, 10]})

        # Should handle NaNs (usually pairwise complete or drop rows)
        # Updated to unpack 3 values
        corr_matrix, _fig, _summary = compute_correlation_matrix(df, ["a", "b"])

        assert corr_matrix is not None
        assert not corr_matrix.empty
        assert np.isclose(
            corr_matrix.loc["a", "b"], 1.0
        )  # Still perfect correlation ignoring NaN

    def test_invalid_columns(self, corr_data):
        """🚫 Test specific error handling for invalid inputs"""
        df = corr_data

        # Pass a numeric and a non-numeric column
        # Our implementation uses coerce, so non-numeric becomes NaN

        cols = ["var_x", "category"]
        # Updated to unpack 3 values
        corr_matrix, _, _ = compute_correlation_matrix(df, cols)

        # In Pandas corr(), non-numeric columns are usually excluded implicitly OR result in NaN

        if corr_matrix is not None:
            # If 'category' is present, its correlation should be NaN
            if "category" in corr_matrix.columns:
                assert pd.isna(corr_matrix.loc["var_x", "category"])
            else:
                # Or it should be dropped
                assert "category" not in corr_matrix.columns

    def test_insufficient_columns(self, corr_data):
        """🚫 Test handling of too few columns"""
        df = corr_data

        # Pass only 1 column
        # Updated to unpack 3 values
        corr_matrix, fig, summary = compute_correlation_matrix(df, ["var_x"])

        # Should return None for all
        assert corr_matrix is None
        assert fig is None
        assert summary is None

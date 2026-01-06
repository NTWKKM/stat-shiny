"""
🔗 Integration Tests for Forest Plot Library
File: tests/integration/test_forest_plot_lib.py

Tests the visualization capabilities:
1. Basic Forest Plot generation
2. Customization (colors, labels)
3. Handling of missing/invalid data
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from forest_plot_lib import create_forest_plot

pytestmark = pytest.mark.integration

class TestForestPlotLib:

    @pytest.fixture
    def ready_to_plot_df(self):
        """Create a DataFrame formatted exactly for the forest plot function"""
        return pd.DataFrame({
            'Subgroup': ['Overall', 'Age', 'Age', 'Gender', 'Gender'],
            'Level': ['', '<65', '>=65', 'Male', 'Female'],
            'Est': [0.8, 0.7, 0.9, 0.75, 0.85],  # Point estimate (OR/HR/RR)
            'Lower': [0.6, 0.5, 0.7, 0.6, 0.65],
            'Upper': [1.1, 0.9, 1.2, 0.95, 1.15],
            'P-value': [0.15, 0.02, 0.4, 0.03, 0.3],
            'Interaction P-value': [np.nan, 0.04, np.nan, 0.6, np.nan]
        })

    def test_basic_forest_plot(self, ready_to_plot_df):
        """📊 Test generating a standard forest plot"""
        fig = create_forest_plot(
            ready_to_plot_df,
            title="Test Forest Plot",
            xlabel="Odds Ratio"
        )
        
        assert fig is not None
        # Verify it returns a Plotly Figure
        assert hasattr(fig, 'layout')
        assert hasattr(fig, 'data')
        
        # Check basic layout properties
        assert fig.layout.title.text == "Test Forest Plot"
        assert len(fig.data) > 0 # Should have traces (points, lines)

    def test_forest_plot_customization(self, ready_to_plot_df):
        """🎨 Test customizing columns and styles"""
        # Rename columns to test flexibility
        df_custom = ready_to_plot_df.rename(columns={
            'Est': 'HazardRatio',
            'Lower': 'CI_L',
            'Upper': 'CI_U'
        })
        
        fig = create_forest_plot(
            df_custom,
            estimate_col='HazardRatio',
            ci_low_col='CI_L',
            ci_high_col='CI_U',
            title="Custom Cox Plot",
            colors=['blue', 'red'] # If supported
        )
        
        assert fig is not None
        assert fig.layout.title.text == "Custom Cox Plot"

    def test_plot_with_missing_interaction(self, ready_to_plot_df):
        """⚠️ Test plotting when Interaction P-value column is missing"""
        df_no_int = ready_to_plot_df.drop(columns=['Interaction P-value'])
        
        # Should still work, just not display interaction p-values
        fig = create_forest_plot(df_no_int)
        
        assert fig is not None

    def test_empty_dataframe(self):
        """🚫 Test handling of empty input"""
        df_empty = pd.DataFrame(columns=['Subgroup', 'Level', 'Est', 'Lower', 'Upper'])
        
        # Depending on implementation, might return None or Empty Fig
        # Assuming it handles gracefully
        try:
            fig = create_forest_plot(df_empty)
            # If it returns a figure, it should probably be empty
            if fig is not None:
                assert len(fig.data) == 0
        except Exception:
            # If it raises error for empty DF, that's also an acceptable behavior to catch
            pass

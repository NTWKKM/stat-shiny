"""
🔗 Integration Tests for Subgroup Analysis Pipeline
File: tests/integration/test_interaction_pipeline.py

Tests the Subgroup Analysis workflow:
1. Interaction Analysis (interaction_lib)
2. Forest Plot Generation (forest_plot_lib)
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from interaction_lib import run_subgroup_analysis
from forest_plot_lib import create_forest_plot

# Mark as integration test
pytestmark = pytest.mark.integration

class TestInteractionPipeline:

    @pytest.fixture
    def subgroup_data(self):
        """Create dataset with potential subgroup effects"""
        np.random.seed(77)
        n = 300
        
        df = pd.DataFrame({
            'outcome': np.random.binomial(1, 0.3, n),
            'treatment': np.random.choice(['Tx', 'Placebo'], n),
            'age': np.random.normal(60, 10, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'diabetes': np.random.choice(['Yes', 'No'], n)
        })
        return df

    def test_full_subgroup_analysis_flow(self, subgroup_data):
        """
        🔄 Test End-to-End Subgroup Analysis:
        Calculation -> Formatting -> Plotting
        """
        df = subgroup_data
        
        # --- Step 1: Calculate Interactions (Subgroups) ---
        # Define subgroups to analyze
        subgroups = ['gender', 'diabetes']
        
        # Run analysis (using Logistic as base)
        results_df, error_msg = run_subgroup_analysis(
            df,
            outcome_col='outcome',
            treatment_col='treatment',
            subgroup_cols=subgroups,
            model_type='logistic'  # or 'cox'
        )
        
        assert error_msg is None
        assert not results_df.empty
        
        # Check structure needed for Forest Plot
        expected_cols = ['Subgroup', 'Level', 'Est', 'Lower', 'Upper', 'P-value', 'Interaction P-value']
        for col in expected_cols:
            assert col in results_df.columns or col in str(results_df.columns)

        # --- Step 2: Generate Forest Plot ---
        # Pass the results directly to the plotter
        fig = create_forest_plot(
            results_df,
            title="Subgroup Analysis: Treatment Effect",
            xlabel="Odds Ratio (95% CI)"
        )
        
        assert fig is not None
        # Check Plotly object properties
        assert hasattr(fig, 'layout')
        assert "Subgroup Analysis" in fig.layout.title.text

    def test_interaction_invalid_inputs(self, subgroup_data):
        """🚫 Test handling of invalid subgroup inputs"""
        df = subgroup_data
        
        # Subgroup column that doesn't exist
        res_df, msg = run_subgroup_analysis(
            df, 'outcome', 'treatment', ['non_existent_col']
        )
        
        assert msg is not None
        assert res_df is None or res_df.empty

    def test_forest_plot_standalone(self):
        """📊 Test Forest Plot with manual data (Edge case check)"""
        # Manually create a dataframe structure that forest_plot_lib expects
        manual_df = pd.DataFrame({
            'Subgroup': ['Age', 'Age', 'Gender', 'Gender'],
            'Level': ['<65', '>=65', 'Male', 'Female'],
            'Est': [1.2, 0.8, 1.1, 1.0],
            'Lower': [0.9, 0.6, 0.8, 0.7],
            'Upper': [1.5, 1.1, 1.4, 1.3],
            'P-value': [0.1, 0.2, 0.3, 0.9],
            'Interaction P-value': [0.04, np.nan, 0.5, np.nan]
        })
        
        fig = create_forest_plot(manual_df)
        assert fig is not None
        # Should handle NaN interaction P-values gracefully

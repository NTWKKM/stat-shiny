"""
🔗 Integration Tests for Subgroup Analysis Module
File: tests/integration/test_subgroup_pipeline.py

Tests subgroup analysis using the SubgroupAnalysisLogit class.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# ✅ Correct Import: It's a Class, not a function
from subgroup_analysis_module import SubgroupAnalysisLogit

pytestmark = pytest.mark.integration

class TestSubgroupAnalysisPipeline:

    @pytest.fixture
    def subgroup_data(self):
        """Create dataset for subgroup analysis"""
        np.random.seed(99)
        n = 300
        
        # Subgroups
        gender = np.random.choice(['Male', 'Female'], n)
        age_group = np.random.choice(['<65', '>=65'], n)
        
        # Treatment
        treatment = np.random.choice(['A', 'B'], n)
        
        # Outcome (Binary)
        # Treatment B better in Males
        logit = -0.5 + 0.5*(treatment=='B') + 1.0*(gender=='Male')*(treatment=='B')
        prob = 1 / (1 + np.exp(-logit))
        outcome = np.random.binomial(1, prob)
        
        df = pd.DataFrame({
            'gender': gender,
            'age_group': age_group,
            'treatment': treatment,
            'outcome': outcome
        })
        return df

    def test_subgroup_analysis_logit_flow(self, subgroup_data):
        """🔄 Test Subgroup Analysis (Logistic) Flow"""
        df = subgroup_data
        
        # 1. Initialize Analyzer
        analyzer = SubgroupAnalysisLogit(
            df=df,
            outcome='outcome',
            predictors=['treatment'], # Main predictor of interest
            outcome_type='binary'
        )
        
        # 2. Run Analysis on specific subgroups
        subgroups = ['gender', 'age_group']
        results = analyzer.run_analysis(subgroups=subgroups)
        
        # 3. Verify Results Structure (Dictionary based on module code)
        assert results is not None
        assert 'summary' in results
        assert 'results_df' in results
        
        # Check DataFrame content
        res_df = results['results_df']
        assert not res_df.empty
        assert 'subgroup' in res_df.columns
        assert 'or_val' in res_df.columns  # Odds Ratio
        assert 'p_value' in res_df.columns
        
        # Should have rows for Male, Female, <65, >=65, and Overall
        # Note: Actual row count depends on implementation (sometimes Overall is separate)
        assert len(res_df) >= 4

    def test_subgroup_forest_plot_generation(self, subgroup_data):
        """📊 Test Forest Plot creation from analyzer"""
        df = subgroup_data
        analyzer = SubgroupAnalysisLogit(
            df=df,
            outcome='outcome',
            predictors=['treatment']
        )
        analyzer.run_analysis(subgroups=['gender'])
        
        # Try generating plot
        fig = analyzer.create_forest_plot(title="Test Forest Plot")
        
        assert fig is not None
        assert hasattr(fig, 'layout')
        assert "Test Forest Plot" in fig.layout.title.text

    def test_invalid_inputs(self, subgroup_data):
        """🚫 Test handling of invalid columns"""
        df = subgroup_data
        
        # Non-existent outcome
        with pytest.raises(Exception): # Assuming it raises error or handles it
            analyzer = SubgroupAnalysisLogit(
                df=df,
                outcome='non_existent',
                predictors=['treatment']
            )
            analyzer.run_analysis(['gender'])

"""
🔗 Integration Tests for Survival Analysis Pipeline
Tests the flow of survival analysis using real calculations (no mocks).
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from survival_lib import (
    calculate_median_survival,
    fit_km_logrank,
    fit_cox_ph,
    fit_nelson_aalen
)
from forest_plot_lib import create_forest_plot  # Import forest plot lib for integration

pytestmark = pytest.mark.integration

class TestSurvivalPipeline:
    
    @pytest.fixture
    def survival_data(self):
        """Create realistic survival dataset"""
        np.random.seed(42)
        n = 200
        
        # Simulate data
        df = pd.DataFrame({
            'age': np.random.normal(60, 10, n),
            'treatment': np.random.choice(['A', 'B'], n),
            'severity': np.random.choice(['Mild', 'Severe'], n)
        })
        
        # Simulate survival times (exponential)
        # Treatment B is better (longer survival)
        base_hazard = 0.02
        hazard_adj = np.where(df['treatment'] == 'B', 0.5, 1.0)
        
        df['time'] = np.random.exponential(1 / (base_hazard * hazard_adj))
        # Censoring (0=censored, 1=event)
        df['event'] = np.random.binomial(1, 0.7, n)  # 30% censoring
        
        return df

    def test_full_survival_analysis_flow(self, survival_data):
        """
        🔄 Test typical survival analysis workflow:
        1. Calculate Median Survival
        2. Plot KM Curves & Log-rank test
        3. Fit Cox Proportional Hazards Model
        """
        df = survival_data
        
        # --- Step 1: Median Survival ---
        median_res = calculate_median_survival(df, 'time', 'event', 'treatment')
        assert not median_res.empty
        assert 'Median Time (95% CI)' in median_res.columns
        assert len(median_res) == 2  # A and B groups
        
        # --- Step 2: Kaplan-Meier & Log-rank ---
        # Note: In integration test, we check if objects are created successfully
        fig, stats_df = fit_km_logrank(df, 'time', 'event', 'treatment')
        
        assert fig is not None
        # Check if Log-rank test produced a P-value
        assert not stats_df.empty
        assert 'P-value' in stats_df.columns
        p_val = stats_df['P-value'].iloc[0]
        assert 0 <= p_val <= 1
        
        # --- Step 3: Cox Regression ---
        cph, res_df, data, err = fit_cox_ph(
            df, 'time', 'event', ['age', 'treatment', 'severity']
        )
        
        assert err is None
        assert cph is not None
        assert not res_df.empty
        
        # Verify Cox results structure
        assert 'HR' in res_df.columns  # Hazard Ratio
        assert 'p' in res_df.columns   # P-value
        assert 'age' in res_df.index
        assert 'treatment' in res_df.index

    def test_cox_to_forest_plot_integration(self, survival_data):
        """
        🌲 Test Integration: Cox Regression -> Forest Plot
        Verifies that Cox PH model output can be visualized as a Forest Plot
        """
        df = survival_data
        
        # 1. Fit Cox Model
        cph, res_df, data, err = fit_cox_ph(
            df, 'time', 'event', ['age', 'treatment', 'severity']
        )
        
        assert err is None
        assert not res_df.empty
        
        # 2. Prepare Data for Forest Plot
        # fit_cox_ph creates specific column names for CIs
        # We need to map these to what create_forest_plot expects
        plot_df = pd.DataFrame({
            'Subgroup': res_df.index,
            'Level': [''] * len(res_df),  # Continuous/Binary vars often have no level label in simple Cox
            'Est': res_df['HR'],
            # Note: fit_cox_ph outputs '95% CI Lower' and '95% CI Upper'
            'Lower': res_df['95% CI Lower'],
            'Upper': res_df['95% CI Upper'],
            'P-value': res_df['p']
        })
        
        # 3. Create Forest Plot
        fig = create_forest_plot(
            plot_df,
            title="Cox Regression Results (Forest Plot)",
            xlabel="Hazard Ratio (95% CI)"
        )
        
        assert fig is not None
        assert hasattr(fig, 'layout')
        assert "Cox Regression" in fig.layout.title.text
        
        # Verify data points are plotted (traces exist)
        assert len(fig.data) > 0

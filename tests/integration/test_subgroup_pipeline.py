"""
🔗 Integration Tests for Subgroup Analysis Module
File: tests/integration/test_subgroup_pipeline.py

Tests subgroup analysis across different statistical models:
1. Logistic Regression (Binary Outcome)
2. Cox Regression (Time-to-Event)
3. Poisson Regression (Count Data)
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Assuming the main function is named 'run_subgroup_analysis' inside subgroup_analysis_module
# If the function name is different, please update the import
from subgroup_analysis_module import run_subgroup_analysis

pytestmark = pytest.mark.integration

class TestSubgroupAnalysisPipeline:

    @pytest.fixture
    def complex_dataset(self):
        """Create a comprehensive dataset suitable for all model types"""
        np.random.seed(99)
        n = 400
        
        # Demographic Subgroups
        gender = np.random.choice(['Male', 'Female'], n)
        age_group = np.random.choice(['<65', '>=65'], n)
        region = np.random.choice(['North', 'South', 'East', 'West'], n)
        
        # Exposure/Treatment
        treatment = np.random.choice(['A', 'B'], n)
        
        # --- 1. Logistic Outcome (Binary) ---
        # Interaction: Treatment B works better in 'Male'
        logit = -1 + 0.5*(treatment=='B') + 0.2*(gender=='Male') + 0.8*(treatment=='B')*(gender=='Male')
        prob = 1 / (1 + np.exp(-logit))
        binary_outcome = np.random.binomial(1, prob)
        
        # --- 2. Cox Outcome (Time-to-Event) ---
        # Baseline hazard
        h0 = 0.05
        # Interaction: Treatment B has lower hazard (better survival) in '<65'
        log_hr = -0.3*(treatment=='B') - 0.5*(treatment=='B')*(age_group=='<65')
        hr = np.exp(log_hr)
        time = np.random.exponential(1 / (h0 * hr))
        event = np.random.binomial(1, 0.8, n) # 20% censoring
        
        # --- 3. Poisson Outcome (Count) ---
        # Interaction: Treatment B reduces count more in 'North' region
        log_mu = 1.5 - 0.2*(treatment=='B') - 0.4*(treatment=='B')*(region=='North')
        mu = np.exp(log_mu)
        counts = np.random.poisson(mu)
        time_at_risk = np.random.uniform(1, 5, n)
        
        df = pd.DataFrame({
            'gender': gender,
            'age_group': age_group,
            'region': region,
            'treatment': treatment,
            'binary_outcome': binary_outcome,
            'time': time,
            'event': event,
            'counts': counts,
            'time_at_risk': time_at_risk
        })
        return df

    def test_subgroup_logistic(self, complex_dataset):
        """🔄 Test Subgroup Analysis for Logistic Regression"""
        df = complex_dataset
        subgroups = ['gender', 'age_group']
        
        # Run analysis
        results, error = run_subgroup_analysis(
            df,
            outcome_col='binary_outcome',
            treatment_col='treatment',
            subgroup_cols=subgroups,
            model_type='logistic'
        )
        
        assert error is None or error == "OK"
        assert not results.empty
        
        # Check specific columns expected in subgroup output
        expected_cols = ['Subgroup', 'Level', 'Est', 'Lower', 'Upper', 'P-value', 'Interaction P-value']
        for col in expected_cols:
            assert col in results.columns
            
        # Check values
        assert results['Est'].min() > 0  # Odds Ratios should be positive
        assert len(results) >= 4  # (Male, Female) + (<65, >=65) = 4 rows minimum

    def test_subgroup_cox(self, complex_dataset):
        """🔄 Test Subgroup Analysis for Cox Regression"""
        df = complex_dataset
        subgroups = ['age_group']
        
        # Run analysis
        results, error = run_subgroup_analysis(
            df,
            outcome_col='event',       # Event indicator
            treatment_col='treatment',
            subgroup_cols=subgroups,
            model_type='cox',
            time_col='time'            # Required for Cox
        )
        
        assert error is None or error == "OK"
        assert not results.empty
        
        # Check output structure
        assert 'Est' in results.columns # Hazard Ratio
        # Check if Interaction P-value is calculated (might be NaN if failed, but column should exist)
        assert 'Interaction P-value' in results.columns
        
        # Verify rows
        levels = df['age_group'].unique()
        assert len(results) == len(levels)

    def test_subgroup_poisson(self, complex_dataset):
        """🔄 Test Subgroup Analysis for Poisson Regression"""
        df = complex_dataset
        subgroups = ['region']
        
        # Run analysis
        results, error = run_subgroup_analysis(
            df,
            outcome_col='counts',
            treatment_col='treatment',
            subgroup_cols=subgroups,
            model_type='poisson',
            offset_col='time_at_risk'  # Required for Poisson
        )
        
        assert error is None or error == "OK"
        assert not results.empty
        
        # Check output structure
        assert 'Est' in results.columns # Rate Ratio
        
        # Check logic: 4 regions -> 4 rows
        assert len(results) == 4

    def test_invalid_inputs(self, complex_dataset):
        """🚫 Test handling of missing columns or invalid model type"""
        df = complex_dataset
        
        # Invalid model
        res, err = run_subgroup_analysis(
            df, 'binary_outcome', 'treatment', ['gender'], model_type='invalid_model'
        )
        assert err is not None
        assert res is None or res.empty
        
        # Missing column
        res, err = run_subgroup_analysis(
            df, 'binary_outcome', 'non_existent_treatment', ['gender'], model_type='logistic'
        )
        assert err is not None

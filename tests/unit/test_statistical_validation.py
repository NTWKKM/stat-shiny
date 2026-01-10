"""
Statistical Validation Tests
Compares stat-shiny implementations against known benchmarks
"""

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter

import sys
sys.path.insert(0, '../..')

from logic import run_binary_logit, analyze_outcome
from survival_lib import fit_km_logrank, fit_cox_ph, calculate_median_survival
from table_one import generate_table, calculate_smd


class TestLogisticRegression:
    """Validation against statsmodels.api.Logit"""
    
    @pytest.fixture
    def binary_data(self):
        """Synthetic binary outcome data."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
        })
        y = pd.Series((
            0.5 + 0.8 * X['x1'] - 0.5 * X['x2'] + 
            np.random.normal(0, 0.3, n) > 0
        ).astype(int))
        return y, X
    
    def test_logit_coefficients_match_statsmodels(self, binary_data):
        """Our logit coefficients should match statsmodels."""
        y, X = binary_data
        
        # Our implementation
        our_params, our_conf, our_pvals, our_status, _ = run_binary_logit(y, X)
        
        # statsmodels baseline
        X_const = sm.add_constant(X)
        sm_model = sm.Logit(y, X_const)
        sm_result = sm_model.fit(disp=0)
        
        # Check coefficients match (excluding intercept)
        assert our_status == "OK"
        assert_allclose(
            our_params.values[1:],  # Skip const
            sm_result.params.values[1:],
            rtol=1e-4
        )
    
    def test_logit_ci_coverage(self, binary_data):
        """Confidence intervals should have correct bounds."""
        y, X = binary_data
        
        our_params, our_conf, our_pvals, our_status, _ = run_binary_logit(y, X)
        
        # Lower bound < Estimate < Upper bound
        assert (our_conf.iloc[:, 0] < our_params).all()
        assert (our_params < our_conf.iloc[:, 1]).all()


class TestSurvivalAnalysis:
    """Validation against lifelines."""
    
    @pytest.fixture
    def survival_data(self):
        """Synthetic survival data."""
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.binomial(1, 0.6, n),
            'group': np.random.choice(['A', 'B'], n),
            'x1': np.random.normal(0, 1, n)
        })
        return df
    
    def test_km_median_matches_lifelines(self, survival_data):
        """Our KM median should match lifelines."""
        kmf_ours = KaplanMeierFitter()
        kmf_ours.fit(
            survival_data['T'], 
            survival_data['E'], 
            label='Test'
        )
        
        # Our implementation
        our_median_df = calculate_median_survival(
            survival_data, 'T', 'E', None
        )
        
        our_median = float(our_median_df['Median Time (95% CI)'].iloc[0].split()[0])
        
        # Compare
        assert_allclose(
            our_median,
            kmf_ours.median_survival_time_,
            rtol=0.1  # Allow 10% tolerance
        )
    
    def test_cox_hr_matches_lifelines(self, survival_data):
        """Our Cox HR should match lifelines."""
        # Create data with only numeric variables for lifelines
        cph_ours = CoxPHFitter()
        cph_data = survival_data[['T', 'E', 'x1']].copy()
        cph_ours.fit(cph_data, duration_col='T', event_col='E')
        
        # Our implementation
        our_cph, our_summary, our_data, our_status, our_stats = fit_cox_ph(
            cph_data, 'T', 'E', ['x1']
        )
        
        # Extract HR
        our_hr = our_summary['HR'].iloc[0]
        lifelines_hr = np.exp(cph_ours.params_.iloc[0])
        
        # Should be close
        assert_allclose(our_hr, lifelines_hr, rtol=0.05)


class TestTableOne:
    """Validation of Table One generation."""
    
    @pytest.fixture
    def table_one_data(self):
        """Data for table one."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'age': np.random.normal(50, 10, n),
            'sex': np.random.choice([0, 1], n),
            'treatment': np.random.choice([0, 1], n),
            'outcome': np.random.binomial(1, 0.4, n)
        })
        return df
    
    def test_smd_calculation(self, table_one_data):
        """SMD should be in [0, ~2] range."""
        smd = calculate_smd(
            table_one_data,
            'age',
            'treatment',
            0, 1,
            is_cat=False
        )
        
        # SMD is a formatted string like "0.123"
        smd_float = float(smd.replace('<b>', '').replace('</b>', ''))
        
        assert 0 <= smd_float <= 3  # Sanity check


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
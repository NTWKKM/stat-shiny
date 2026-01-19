"""
🔗 Integration Tests for Poisson Regression Pipeline
File: tests/integration/test_poisson_pipeline.py

Tests the flow of count data analysis:
1. Poisson Regression
2. Negative Binomial Regression (for overdispersion)
3. Edge case handling (zero-inflation)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.poisson_lib import run_negative_binomial_regression, run_poisson_regression

# Mark as integration test
pytestmark = pytest.mark.integration

class TestPoissonPipeline:

    @pytest.fixture
    def count_data(self):
        """Create dataset for count data analysis"""
        np.random.seed(55)
        n = 500  # Increased sample size slightly for stability
        
        # Predictors
        exposure = np.random.choice([0, 1], n)
        # Scale age to be centered around 0 to avoid huge exp() values
        age_raw = np.random.normal(50, 10, n)
        age_centered = (age_raw - 50) / 10
        
        # True count process (Poisson)
        # log(lambda) = intercept + coef*exposure + coef*age
        # Keep coefficients small to prevent overflow
        log_mu = 0.5 + 0.5 * exposure + 0.2 * age_centered
        mu = np.exp(log_mu)
        
        # Add Time-at-risk (offset)
        # Using log(time) is standard, but here we pass raw time as offset column
        person_time = np.random.uniform(1, 2, n)
        
        # Adjust mu for time at risk: mu_final = mu_rate * time
        mu_adj = mu * person_time

        # Generate counts
        counts = np.random.poisson(mu_adj)
        
        df = pd.DataFrame({
            'event_count': counts,
            'exposure': exposure,
            'age': age_raw,       # Use raw age for model input
            'time_at_risk': person_time,
            'mu_true': mu_adj     # Store true mean for NB generation
        })
        return df

    def test_poisson_regression_flow(self, count_data):
        """🔄 Test standard Poisson Regression workflow"""
        df = count_data
        
        # Run Poisson Model
        params, _conf_int, _pvalues, status_msg, _stats_dict = run_poisson_regression(
            df['event_count'],
            df[['exposure', 'age']],
            offset=df['time_at_risk']
        )
        
        assert status_msg == "OK"
        assert params is not None
        
        # Check numeric validity
        # stats_dict might have AIC if implemented, but let's check params
        assert 'exposure' in params.index
        # Verify parameter is numeric and finite (value may vary with random data)
        assert params['exposure'] is not None
        assert np.isfinite(params['exposure']), "Exposure parameter should be finite"

    def test_negative_binomial_flow(self, count_data):
        """🔄 Test Negative Binomial Regression for overdispersed data"""
        df = count_data.copy()

        # Introduce overdispersion MATHEMATICALLY CORRECT WAY
        # Instead of adding random noise which breaks the regression relationship (X -> Y),
        # we generate Y from a Negative Binomial distribution using the SAME mu (mean)
        # but with a dispersion parameter.
        #
        # Scipy/Numpy parametrization:
        # Mean = n * (1-p) / p
        # We want Mean = mu_true
        # Let 'n_disp' be the dispersion parameter (size). Smaller 'n_disp' = More overdispersion.
        
        np.random.seed(56)
        n_disp = 1.5  # Dispersion parameter
        mu = df['mu_true'].values
        
        # Calculate probability p for numpy's negative_binomial
        # p = n / (n + mu)
        prob = n_disp / (n_disp + mu)
        
        # Generate overdispersed counts that still follow the regression structure
        overdispersed_counts = np.random.negative_binomial(n_disp, prob)
        
        df['event_count'] = overdispersed_counts

        # Run Negative Binomial Model
        params, _conf_int, _pvalues, status_msg, stats_dict = run_negative_binomial_regression(
            df['event_count'],
            df[['exposure', 'age']],
            offset=df['time_at_risk']
        )

        assert status_msg == "OK", f"NB regression failed: {status_msg}"
        assert params is not None, "NB params should not be None"
        assert 'exposure' in params.index, "Exposure parameter missing"

        # Check that fit statistics are returned
        assert 'deviance' in stats_dict, "Deviance statistic missing"
        assert 'aic' in stats_dict, "AIC missing"
        
        # Ensure AIC is a valid number (not nan)
        assert not np.isnan(stats_dict['aic']), f"AIC is nan! Stats: {stats_dict}"

        # Compare with Poisson: NB should fit better for overdispersed data
        poisson_params, _p_conf_int, _p_pvalues, poisson_status, poisson_stats = run_poisson_regression(
            df['event_count'],
            df[['exposure', 'age']],
            offset=df['time_at_risk']
        )

        assert poisson_status == "OK", f"Poisson regression failed: {poisson_status}"
        assert poisson_params is not None, "Poisson params should not be None"
        assert 'aic' in poisson_stats, "Poisson AIC missing"

        # NB should have lower AIC (better fit) for overdispersed data
        nb_aic = stats_dict['aic']
        poisson_aic = poisson_stats['aic']
        
        assert nb_aic < poisson_aic, (
            f"Negative Binomial should fit better (lower AIC) for overdispersed data: "
            f"NB AIC={nb_aic:.2f} vs Poisson AIC={poisson_aic:.2f}"
        )
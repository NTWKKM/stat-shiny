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

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from poisson_lib import run_poisson_regression, run_negative_binomial_regression

# Mark as integration test
pytestmark = pytest.mark.integration

class TestPoissonPipeline:

    @pytest.fixture
    def count_data(self):
        """Create dataset for count data analysis"""
        np.random.seed(55)
        n = 200
        
        # Predictors
        exposure = np.random.choice([0, 1], n)
        age = np.random.normal(50, 10, n)
        
        # True count process (Poisson)
        # log(lambda) = intercept + coef*exposure + coef*age
        log_mu = -2 + 0.5 * exposure + 0.02 * age
        mu = np.exp(log_mu)
        
        # Generate counts
        counts = np.random.poisson(mu)
        
        # Add Time-at-risk (offset)
        person_time = np.random.uniform(1, 10, n)
        
        df = pd.DataFrame({
            'event_count': counts,
            'exposure': exposure,
            'age': age,
            'time_at_risk': person_time
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
        assert params['exposure'] != 0

    def test_negative_binomial_flow(self, count_data):
    """🔄 Test Negative Binomial Regression for overdispersed data"""
    df = count_data

    # Introduce overdispersion by adding extra variability
    np.random.seed(56)
    overdispersed_counts = df['event_count'] + np.random.negative_binomial(2, 0.3, len(df))
    df_overdispersed = df.copy()
    df_overdispersed['event_count'] = overdispersed_counts

    # Run Negative Binomial Model
    params, _conf_int, _pvalues, status_msg, stats_dict = run_negative_binomial_regression(
        df_overdispersed['event_count'],
        df_overdispersed[['exposure', 'age']],
        offset=df_overdispersed['time_at_risk']
    )

    assert status_msg == "OK", f"NB regression failed: {status_msg}"
    assert params is not None, "NB params should not be None"
    assert 'exposure' in params.index, "Exposure parameter missing"

    # Check that fit statistics are returned
    assert 'deviance' in stats_dict, "Deviance statistic missing"
    assert 'aic' in stats_dict, "AIC missing"

    # Compare with Poisson: NB should fit better for overdispersed data
    poisson_params, _p_conf_int, _p_pvalues, poisson_status, poisson_stats = run_poisson_regression(
        df_overdispersed['event_count'],
        df_overdispersed[['exposure', 'age']],
        offset=df_overdispersed['time_at_risk']
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

    def test_zero_inflation_handling(self):
        """⚠️ Test handling of data with no events"""
        df = pd.DataFrame({
            'counts': [0, 0, 0, 0, 0],
            'pred': [1, 2, 3, 4, 5],
            'time': [1, 1, 1, 1, 1]
        })
        
        # Should handle gracefully (return error or specific message)
        params, _, _, status_msg, _ = run_poisson_regression(
            df['counts'], df[['pred']], offset=df['time']
        )
        
        assert status_msg != "OK" or params is None, \
            f"Should handle zero-inflation gracefully, got status='{status_msg}', params={params is not None}"

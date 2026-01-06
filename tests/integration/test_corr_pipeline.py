"""
🔗 Integration Tests for Correlation Analysis Pipeline
File: tests/integration/test_corr_pipeline.py

Tests the correlation analysis workflow:
1. Pearson/Spearman correlation calculation
2. Handling of non-numeric data
3. Output format verification
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import correlation module (Adjust import based on your actual file structure)
# Assuming correlation.py has a function like compute_correlation_matrix or similar
from correlation import compute_correlation_matrix

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
        
        df = pd.DataFrame({
            'var_x': x,
            'var_y': y,
            'var_z': z,
            'var_w': w,
            'category': np.random.choice(['A', 'B'], n) # Should be ignored or handled
        })
        return df

    def test_pearson_correlation_flow(self, corr_data):
        """🔄 Test Pearson correlation calculation"""
        df = corr_data
        selected_vars = ['var_x', 'var_y', 'var_z', 'var_w']
        
        # Run calculation
        corr_matrix, fig = compute_correlation_matrix(
            df, 
            selected_vars,
            method='pearson'
        )
        
        assert not corr_matrix.empty
        assert corr_matrix.shape == (4, 4)
        
        # Check specific known correlations
        # x vs y should be positive and high (> 0.8)
        assert corr_matrix.loc['var_x', 'var_y'] > 0.8
        
        # x vs w should be perfect negative (-1.0)
        assert np.isclose(corr_matrix.loc['var_x', 'var_w'], -1.0, atol=0.01)
        
        # x vs z should be low (near 0)
        assert abs(corr_matrix.loc['var_x', 'var_z']) < 0.2
        
        # Check Plotly figure
        assert fig is not None
        assert hasattr(fig, 'layout')

    def test_spearman_correlation_flow(self, corr_data):
        """🔄 Test Spearman correlation (Rank-based)"""
        df = corr_data
        selected_vars = ['var_x', 'var_y']
        
        # Create non-linear but monotonic relationship (Pearson < Spearman)
        df['var_exp'] = np.exp(df['var_x'])
        
        corr_matrix, fig = compute_correlation_matrix(
            df, 
            ['var_x', 'var_exp'],
            method='spearman'
        )
        
        # Spearman should be perfect 1.0 for monotonic transform
        assert np.isclose(corr_matrix.loc['var_x', 'var_exp'], 1.0, atol=0.01)

    def test_missing_values_handling(self):
        """⚠️ Test handling of missing data"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [2, 4, 6, 8, 10]
        })
        
        # Should handle NaNs (usually pairwise complete or drop rows)
        corr_matrix, fig = compute_correlation_matrix(df, ['a', 'b'])
        
        assert not corr_matrix.empty
        assert np.isclose(corr_matrix.loc['a', 'b'], 1.0) # Still perfect correlation ignoring NaN

    def test_invalid_columns(self, corr_data):
        """🚫 Test specific error handling for invalid inputs"""
        df = corr_data
        
        # If method supports only numeric, it should filter or error out non-numeric
        try:
            corr_matrix, _ = compute_correlation_matrix(df, ['var_x', 'category'])
            # If it runs, check if category was excluded or handled
            assert 'category' not in corr_matrix.columns
        except ValueError:
            # If it raises error, that's also valid behavior
            pass

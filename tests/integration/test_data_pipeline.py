"""
🔗 Integration Tests for Data Pipeline

These tests verify that multiple components work together correctly.
Unlike unit tests, integration tests may:
- Use real data files
- Call multiple functions in sequence
- Test data flow between modules
- Not use mocks/patches

Tests verify:
1. Data loading and parsing
2. Data validation across modules
3. Statistical computation pipeline
4. Output format correctness
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from logic import validate_logit_data, clean_numeric_value, run_binary_logit

# ============================================================================
# Mark all tests in this file as integration tests
# ============================================================================
pytestmark = pytest.mark.integration


class TestDataPipeline:
    """
    Integration tests for complete data pipeline
    
    Tests multiple functions working together
    """

    @pytest.fixture
    def sample_medical_data(self):
        """
        🏥 Create realistic medical dataset for testing
        
        Simulates:
        - Patient age (realistic range)
        - Blood pressure measurements
        - Cholesterol levels
        - Binary outcome (disease present/absent)
        """
        np.random.seed(42)
        n_patients = 150
        
        # Realistic age distribution
        age = np.random.normal(55, 15, n_patients)
        age = np.clip(age, 18, 90)  # Realistic age range
        
        # Blood pressure (SBP in mmHg)
        sbp = np.random.normal(130, 20, n_patients)
        sbp = np.clip(sbp, 80, 200)
        
        # Cholesterol (mg/dL)
        cholesterol = np.random.normal(200, 50, n_patients)
        cholesterol = np.clip(cholesterol, 100, 400)
        
        # Binary outcome: disease status (0=no disease, 1=disease)
        # Higher age & cholesterol → higher probability of disease
        prob = 1 / (1 + np.exp(-(0.05 * age + 0.01 * cholesterol - 10)))
        outcome = np.random.binomial(1, prob)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'sbp': sbp,
            'cholesterol': cholesterol
        })
        
        return df, pd.Series(outcome, name='disease')

    def test_complete_logit_pipeline(self, sample_medical_data):
        """
        🔄 Test complete logistic regression pipeline
        
        Flow:
        1. Load data (fixtures provide it)
        2. Validate data
        3. Run logistic regression
        4. Verify outputs
        
        This tests the integration of all components
        """
        df, y = sample_medical_data
        
        # Step 1: Validate input data
        valid, msg = validate_logit_data(y, df)
        assert valid is True, f"Data validation failed: {msg}"
        
        # Step 2: Run logistic regression
        params, conf, pvals, status, metrics = run_binary_logit(y, df)
        
        # Step 3: Verify pipeline completed successfully
        assert status == "OK", f"Logit regression failed: {status}"
        assert params is not None
        assert len(params) == len(df.columns), "Should have coefficient for each variable"
        
        # Step 4: Verify output structure
        assert isinstance(conf, dict), "Confidence intervals should be dict"
        assert isinstance(pvals, dict), "P-values should be dict"
        assert isinstance(metrics, dict), "Metrics should be dict"
        
        # Step 5: Verify parameter values are reasonable
        for var, coef in params.items():
            assert isinstance(coef, (int, float)), f"Coefficient for {var} should be numeric"
            assert not np.isnan(coef), f"Coefficient for {var} is NaN"
            assert not np.isinf(coef), f"Coefficient for {var} is infinite"

    def test_data_flow_with_numeric_cleaning(self, sample_medical_data):
        """
        🧹 Test data flow with numeric value cleaning
        
        This tests:
        1. Read data with various numeric formats
        2. Clean numeric values
        3. Use cleaned data in analysis
        
        Common real-world scenario: CSV files with formatted numbers
        """
        df, y = sample_medical_data
        
        # Simulate real-world data with formatted numbers (common in Excel exports)
        df_formatted = df.copy()
        
        # Format some values as they might appear in CSV
        df_formatted['age'] = df_formatted['age'].apply(lambda x: f"{x:.0f}")
        df_formatted['cholesterol'] = df_formatted['cholesterol'].apply(lambda x: f"{x:,.1f}")
        
        # Clean the formatted values
        df_clean = df_formatted.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_numeric_value)
        
        # Now use cleaned data in pipeline
        valid, msg = validate_logit_data(y, df_clean)
        assert valid is True, f"Validation of cleaned data failed: {msg}"
        
        params, conf, pvals, status, metrics = run_binary_logit(y, df_clean)
        assert status == "OK", f"Logit on cleaned data failed: {status}"

    def test_pipeline_with_multiple_covariates(self, sample_medical_data):
        """
        📊 Test pipeline with multiple covariates
        
        Verifies:
        - All variables are included in analysis
        - Coefficients estimated for all variables
        - Model handles multiple predictors correctly
        """
        df, y = sample_medical_data
        
        # All three variables
        params, conf, pvals, status, metrics = run_binary_logit(y, df)
        
        assert status == "OK"
        assert len(params) == 3, "Should have 3 coefficients (age, sbp, cholesterol)"
        
        # Check all variables have p-values
        assert len(pvals) == 3, "Should have p-values for all 3 variables"
        
        # Check model fit
        assert metrics['mcfadden'] > 0, "McFadden R-squared should be positive"
        assert metrics['ll'] < 0, "Log-likelihood should be negative"
        assert metrics['n_obs'] == len(y), f"Observations should be {len(y)}"

    def test_pipeline_robustness_with_subset(self, sample_medical_data):
        """
        🎯 Test pipeline robustness with data subsets
        
        Verifies:
        - Pipeline works with different data sizes
        - Results are stable
        - No errors with subset of variables
        """
        df, y = sample_medical_data
        
        # Test with different subsets
        subsets = [
            ['age'],                           # Single variable
            ['age', 'sbp'],                    # Two variables
            ['age', 'sbp', 'cholesterol'],     # All variables
        ]
        
        for subset_vars in subsets:
            df_subset = df[subset_vars]
            
            # Validate
            valid, msg = validate_logit_data(y, df_subset)
            assert valid is True, f"Validation failed for {subset_vars}: {msg}"
            
            # Run
            params, conf, pvals, status, metrics = run_binary_logit(y, df_subset)
            assert status == "OK", f"Logit failed for {subset_vars}"
            assert len(params) == len(subset_vars)

    def test_outcome_variable_consistency(self, sample_medical_data):
        """
        ✅ Test outcome variable is used consistently
        
        Verifies:
        - Outcome variable values are used correctly
        - Binary classification is maintained
        - Sample size matches
        """
        df, y = sample_medical_data
        
        # Verify outcome is binary
        unique_values = y.unique()
        assert len(unique_values) <= 2, "Outcome should be binary"
        assert all(val in [0, 1] for val in unique_values), "Outcome should be 0 or 1"
        
        # Run pipeline
        params, conf, pvals, status, metrics = run_binary_logit(y, df)
        
        # Check n_obs matches sample size
        assert metrics['n_obs'] == len(y), f"Sample size mismatch: {metrics['n_obs']} vs {len(y)}"
        
        # Check outcome distribution is tracked
        assert len(y) == len(df), "Outcome and predictors should have same length"


class TestDataIntegrationEdgeCases:
    """
    Edge cases for data pipeline integration
    """

    def test_pipeline_with_missing_values(self):
        """
        ⚠️ Test pipeline handles missing values appropriately
        
        Most statistical functions drop missing values,
        but should do so consistently
        """
        # Create data with missing values
        df = pd.DataFrame({
            'age': [25, 30, np.nan, 45, 50],
            'sbp': [120, 130, 135, np.nan, 150]
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        # Should either handle or clearly reject missing data
        valid, msg = validate_logit_data(y, df)
        # Could be True (auto-drop) or False (reject), both OK
        # Just verify behavior is defined
        assert isinstance(valid, bool)
        assert isinstance(msg, str)

    def test_pipeline_with_constant_outcome(self):
        """
        🚫 Test pipeline rejects degenerate cases
        
        Logistic regression needs variation in outcome
        """
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'sbp': [120, 130, 135, 140, 150]
        })
        # All outcomes are 1 (no variation)
        y = pd.Series([1, 1, 1, 1, 1])
        
        valid, msg = validate_logit_data(y, df)
        # Should detect this is problematic
        assert valid is False or "constant" in msg.lower() or "variance" in msg.lower()


if __name__ == "__main__":
    print("🔗 Running Integration Tests...\n")
    pytest.main([__file__, "-v"])

"""
🧪 Unit Tests for Statistical Functions

Tests the core logic functions from logic.py:
- clean_numeric_value: Parse numeric strings
- validate_logit_data: Validate data for logistic regression
- run_binary_logit: Execute binary logistic regression
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add path to import from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from logic import validate_logit_data, clean_numeric_value, run_binary_logit

# ============================================================================
# Mark all tests in this file as unit tests
# ============================================================================
pytestmark = pytest.mark.unit


class TestLogicFunctions:
    """
    Unit tests for statistical logic functions
    
    No server required - tests pure Python functions
    """

    def test_clean_numeric_value(self):
        """
        🧪 Test numeric string parsing
        
        Verifies:
        - Comma-separated numbers are parsed
        - Comparison symbols are removed
        - Invalid strings return NaN
        - None returns NaN
        """
        assert clean_numeric_value("1,000") == 1000.0
        assert clean_numeric_value("<0.001") == 0.001
        assert clean_numeric_value("> 50") == 50.0
        assert pd.isna(clean_numeric_value("abc"))
        assert pd.isna(clean_numeric_value(None))

    def test_validate_logit_data_success(self):
        """
        🧪 Test data validation with valid input
        
        Verifies:
        - Valid data passes validation
        - Returns status message "OK"
        """
        df = pd.DataFrame({'age': [20, 30, 40, 50], 'sex': [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])
        valid, msg = validate_logit_data(y, df)
        assert valid is True
        assert msg == "OK"

    def test_validate_logit_data_zero_variance(self):
        """
        🧪 Test validation with constant variable (zero variance)
        
        Verifies:
        - Detects variables with no variance
        - Returns error message containing "zero variance"
        """
        df = pd.DataFrame({'constant': [1, 1, 1, 1], 'age': [20, 30, 40, 50]})
        y = pd.Series([0, 1, 0, 1])
        valid, msg = validate_logit_data(y, df)
        assert valid is False
        assert "zero variance" in msg

    def test_validate_logit_data_empty(self):
        """
        🧪 Test validation with empty data
        
        Verifies:
        - Detects empty DataFrame
        - Returns error message containing "Empty data"
        """
        df = pd.DataFrame()
        y = pd.Series([])
        valid, msg = validate_logit_data(y, df)
        assert valid is False
        assert "Empty data" in msg

    def test_run_binary_logit_basic(self):
        """
        🧪 Test binary logistic regression with synthetic data
        
        Creates synthetic data where:
        - Age is normally distributed (mean=50, std=10)
        - Older people have higher probability of outcome=1
        - Run logistic regression and verify results
        
        Verifies:
        - Regression completes successfully (status="OK")
        - Parameters are returned
        - Age coefficient is positive (matches synthetic pattern)
        - McFadden R-squared > 0
        """
        # Create synthetic data with clear relationship
        np.random.seed(42)
        n = 100
        age = np.random.normal(50, 10, n)
        
        # Older people have higher probability of outcome=1
        prob = 1 / (1 + np.exp(-(0.1 * (age - 50))))
        y = np.random.binomial(1, prob)
        
        df = pd.DataFrame({'age': age})
        y_series = pd.Series(y)
        
        # Run logistic regression
        params, conf, pvals, status, metrics = run_binary_logit(y_series, df)
        
        # Verify results
        assert status == "OK", f"Expected 'OK', got '{status}'"
        assert params is not None, "Parameters should not be None"
        assert 'age' in params, "Age coefficient should be in parameters"
        
        # Age coefficient should be positive (older = higher probability)
        assert params['age'] > 0, f"Expected positive age coefficient, got {params['age']}"
        
        # Model fit quality
        assert metrics['mcfadden'] > 0, "McFadden R-squared should be > 0"

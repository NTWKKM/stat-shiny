import numpy as np
import pytest
import statsmodels.api as sm

from utils.model_diagnostics_lib import run_heteroscedasticity_test, run_reset_test


@pytest.fixture
def homoscedastic_model():
    """Create a clean OLS model."""
    np.random.seed(42)
    n = 100
    X = np.random.normal(0, 1, n)
    Y = 2 * X + np.random.normal(0, 1, n)
    X_const = sm.add_constant(X)
    return sm.OLS(Y, X_const).fit()


@pytest.fixture
def heteroscedastic_model():
    """Create a model with heteroscedasticity."""
    np.random.seed(42)
    n = 200
    X = np.random.uniform(1, 10, n)
    # Variance increases with X
    Y = 2 * X + np.random.normal(0, X * 0.5, n)
    X_const = sm.add_constant(X)
    return sm.OLS(Y, X_const).fit()


def test_reset_test(homoscedastic_model):
    """Test RESET on a correctly specified model."""
    res = run_reset_test(homoscedastic_model)
    assert res is not None
    assert "p_value" in res
    # Should accept null hypothesis (p > 0.05 usually)
    # Though random noise might rarely trigger it, usually > 0.05
    assert res["p_value"] > 0.01


def test_heteroscedasticity_test_clean(homoscedastic_model):
    """Test BP test on homoscedastic data."""
    res = run_heteroscedasticity_test(homoscedastic_model)
    assert res is not None
    # Should be > 0.05
    assert res["p_value"] > 0.05


def test_heteroscedasticity_test_dirty(heteroscedastic_model):
    """Test BP test on heteroscedastic data."""
    res = run_heteroscedasticity_test(heteroscedastic_model)
    assert "p_value" in res
    # Should reject null magnitude (p < 0.05)
    assert res["p_value"] < 0.05

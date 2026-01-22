import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from utils.model_diagnostics_lib import (
    calculate_cooks_distance,
    get_diagnostic_plot_data,
)


@pytest.fixture
def sample_model():
    """Create a simple OLS model for testing."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(0, 1, n)
    # Add an outlier/influential point
    X[0] = 10
    y = 2 * X + rng.normal(0, 1, n)
    y[0] = 50  # Extreme outlier

    df = pd.DataFrame({"x": X, "y": y})
    X_with_const = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X_with_const).fit()
    return model


def test_calculate_cooks_distance_structure(sample_model):
    """Test that cook's distance returns correct structure."""
    res = calculate_cooks_distance(sample_model)
    assert "error" not in res
    assert "cooks_d" in res
    assert "p_values" in res
    assert "influential_points" in res
    assert "threshold" in res
    assert len(res["cooks_d"]) == 100


def test_cooks_distance_detects_outlier(sample_model):
    """Test that the artificial outlier is detected."""
    res = calculate_cooks_distance(sample_model)
    # Index 0 was made an extreme outlier
    assert 0 in res["influential_points"]
    assert res["n_influential"] >= 1


def test_get_diagnostic_plot_data(sample_model):
    """Test that plot data extractor works."""
    res = get_diagnostic_plot_data(sample_model)
    assert "error" not in res
    assert "fitted_values" in res
    assert "residuals" in res
    assert "std_residuals" in res
    assert len(res["fitted_values"]) == 100
    assert len(res["residuals"]) == 100

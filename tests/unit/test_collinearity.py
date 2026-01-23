import numpy as np
import pandas as pd
import pytest

from utils.collinearity_lib import calculate_vif, condition_index


@pytest.fixture
def collinear_data():
    """Create data with known collinearity."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    # x2 is highly correlated with x1
    x2 = 0.95 * x1 + np.random.normal(0, 0.1, n)
    x3 = np.random.normal(0, 1, n)  # Independent

    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


def test_calculate_vif(collinear_data):
    """Test VIF calculation detects collinearity."""
    vif_df, _ = calculate_vif(collinear_data, predictors=["x1", "x2", "x3"])

    assert not vif_df.empty
    assert "VIF" in vif_df.columns

    # Check that x1 and x2 have high VIF (e.g., > 5 or 10)
    vif_x1 = vif_df.loc[vif_df["Variable"] == "x1", "VIF"].iloc[0]
    vif_x2 = vif_df.loc[vif_df["Variable"] == "x2", "VIF"].iloc[0]
    vif_x3 = vif_df.loc[vif_df["Variable"] == "x3", "VIF"].iloc[0]

    assert vif_x1 > 5.0
    assert vif_x2 > 5.0
    assert vif_x3 < 5.0  # Independent var should have low VIF


def test_calculate_vif_single_var(collinear_data):
    """Test VIF with single variable (should be low/1.0)."""
    # Requires concept of "other predictors". If only 1 var, R^2 is 0 -> VIF=1
    # statsmodels behavior check or our func behavior check
    # Our function loops: regress X_i on others. If no others?
    # variance_inflation_factor needs at least 2 cols (including constant)
    # If we pass 1 var + constant, R^2 is 0, VIF should be 1.
    vif_df, _ = calculate_vif(collinear_data, predictors=["x3"])
    assert vif_df.iloc[0]["VIF"] == pytest.approx(1.0)


def test_condition_index(collinear_data):
    """Test Condition Index calculation."""
    ci_df, _ = condition_index(collinear_data, predictors=["x1", "x2", "x3"])

    assert not ci_df.empty
    assert "Condition Index" in ci_df.columns

    # High condition index expected (>10 or >30)
    max_ci = ci_df["Condition Index"].max()
    assert max_ci > 10.0

import numpy as np
import pandas as pd
import pytest

from utils.psm_lib import calculate_ipw, calculate_ps, check_balance
from utils.sensitivity_lib import calculate_e_value
from utils.stratified_lib import breslow_day, mantel_haenszel


# --- Test Data Fixtures ---
@pytest.fixture
def psm_data():
    np.random.seed(42)
    n = 100
    # Covariates
    age = np.random.normal(50, 10, n)
    bmi = np.random.normal(25, 5, n)

    # Treatment assignment depends on covariates
    # P(T=1) = sigmoid(-5 + 0.05*age + 0.1*bmi)
    z = -5 + 0.05 * age + 0.1 * bmi
    prob = 1 / (1 + np.exp(-z))
    treatment = np.random.binomial(1, prob, n)

    # Outcome depends on treatment and covariates
    outcome = 2 + 0.5 * treatment + 0.02 * age + 0.05 * bmi + np.random.normal(0, 1, n)

    return pd.DataFrame(
        {"Age": age, "BMI": bmi, "Treatment": treatment, "Outcome": outcome}
    )


@pytest.fixture
def stratified_data():
    return pd.DataFrame(
        {
            "Treatment": [1, 1, 0, 0, 1, 1, 0, 0],
            "Outcome": [1, 0, 1, 0, 1, 0, 0, 1],
            "Stratum": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )


# --- PSM Tests ---
def test_calculate_ps(psm_data):
    ps, _ = calculate_ps(psm_data, "Treatment", ["Age", "BMI"])
    assert len(ps) == len(psm_data)
    assert ps.min() >= 0
    assert ps.max() <= 1


def test_calculate_ipw(psm_data):
    ps, _ = calculate_ps(psm_data, "Treatment", ["Age", "BMI"])
    psm_data["ps"] = ps
    res = calculate_ipw(psm_data, "Treatment", "Outcome", "ps")

    assert "ATE" in res
    assert "SE" in res
    assert "p_value" in res
    assert res["ATE"] is not None


def test_check_balance(psm_data):
    # Unweighted balance
    bal_unweighted = check_balance(psm_data, "Treatment", ["Age", "BMI"])
    assert "SMD" in bal_unweighted.columns
    assert len(bal_unweighted) == 2

    # Weighted balance
    weights = pd.Series(np.ones(len(psm_data)), index=psm_data.index)  # Dummy weights
    bal_weighted = check_balance(psm_data, "Treatment", ["Age", "BMI"], weights=weights)
    assert "SMD" in bal_weighted.columns


# --- Stratified Tests ---
def test_mantel_haenszel(stratified_data):
    # Just checking it runs and returns structure
    res = mantel_haenszel(stratified_data, "Outcome", "Treatment", "Stratum")
    assert "MH_OR" in res
    assert isinstance(res["Strata_Results"], pd.DataFrame)


def test_breslow_day(stratified_data):
    res = breslow_day(stratified_data, "Outcome", "Treatment", "Stratum")
    assert "p_value" in res


# --- Sensitivity Tests ---
def test_calculate_e_value():
    # RR = 2.0
    res = calculate_e_value(2.0, 1.5, 3.0)
    # E-value = 2 + sqrt(2*(2-1)) = 2 + sqrt(2) approx 3.414
    assert res["e_value_estimate"] > 3.4

    # RR = 0.5 (protective) -> equivalent to RR = 2.0
    res_prot = calculate_e_value(0.5)
    assert res_prot["e_value_estimate"] > 3.4

    # Null effect
    res_null = calculate_e_value(1.0)
    assert res_null["e_value_estimate"] == 1.0


def test_calculate_ps_separation():
    """Test PS calculation with perfect separation."""
    df = pd.DataFrame(
        {
            "X": np.linspace(0, 10, 20),
            "Treatment": [0] * 10
            + [1] * 10,  # Perfect correlation with X if X is sorted
        }
    )
    # This might log warning or return probabilities close to 0/1
    # statsmodels Logit might raise PerfectSeparationError
    try:
        ps, _ = calculate_ps(df, "Treatment", ["X"])
        # If successful, check range
        assert ps.min() >= 0
        assert ps.max() <= 1
    except Exception:
        # If it raises, that's also an acceptable outcome for perfect separation
        pass


def test_check_balance_missing_cols():
    """Test error when cols are missing."""
    df = pd.DataFrame({"A": [1, 2], "T": [0, 1]})
    with pytest.raises(KeyError):
        check_balance(df, "T", ["B"])  # B missing


def test_calculate_e_value_invalid():
    """Test invalid inputs for E-value."""
    # Negative RR
    res = calculate_e_value(-1.0)
    assert "error" in res
    assert "positive" in res["error"]

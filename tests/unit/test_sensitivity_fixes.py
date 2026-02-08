import json

import numpy as np


from utils.sensitivity_lib import (
    bootstrap_confidence_interval,
    calculate_e_value,
    leave_one_out_cv,
)


def test_leave_one_out_cv_nan_handling():
    """Test that LOOCV handles NaNs correctly."""
    X = np.array([[1, 2], [3, 4], [5, np.nan], [7, 8]])
    y = np.array([10, 20, 30, 40])

    # Simple linear model
    def fit_func(X, y):
        # Just return mean for simplicity as we test data handling
        return np.mean(y)

    def predict_func(model, X):
        return np.full(X.shape[0], model)

    result = leave_one_out_cv(X, y, fit_func, predict_func)

    # Should run on 3 observations (one NaN row removed)
    assert "error" not in result
    assert result["n_samples"] == 3

    # Test with too many NaNs
    X_bad = np.array([[1, 2], [np.nan, 4], [5, np.nan], [7, 8]])
    result_error = leave_one_out_cv(X_bad, y, fit_func, predict_func)
    assert "error" in result_error
    assert "Need at least 3 non-NaN observations" in result_error["error"]


def test_bootstrap_json_serialization():
    """Test that bootstrap distribution is JSON serializable."""
    data = [1, 2, 3, 4, 5]
    result = bootstrap_confidence_interval(data, np.mean, n_iterations=100)

    # Should not raise TypeError
    _ = json.dumps(result)
    assert "bootstrap_distribution" in result
    assert isinstance(result["bootstrap_distribution"], list)


def test_e_value_or_conversion():
    """Test OR to RR conversion in E-value calculation."""
    # E-value for RR=2.0 -> 2 + sqrt(2(1)) = 3.414
    res_rr = calculate_e_value(2.0, estimate_type="RR")
    assert round(res_rr["e_value_estimate"], 3) == 3.414

    # E-value for OR=4.0 -> approx RR=sqrt(4)=2.0 -> same E-value
    res_or = calculate_e_value(4.0, estimate_type="OR")
    assert round(res_or["e_value_estimate"], 3) == 3.414

    # Ensure warning log for mismatched protective effect
    # This is harder to test directly without mocking logger, but we can ensure it runs
    res_prot = calculate_e_value(0.5, lower=0.2)
    assert res_prot["e_value_estimate"] > 1.0

    # Test protective effect with upper bound (exercises l_prime path)
    res_prot_ci = calculate_e_value(0.5, upper=0.8)
    assert res_prot_ci["e_value_estimate"] > 1.0
    assert res_prot_ci["e_value_ci_limit"] >= 1.0

    # Verify original_estimate reflects input, not converted RR
    assert res_or["original_estimate"] == 4.0


def test_e_value_validation_order():
    """Test that estimate_type is validated even if estimate is 1.0."""
    # Before fix, this would return success dict with 1.0.
    # Now it should return an error dict because ValueError is caught.
    result = calculate_e_value(1.0, estimate_type="INVALID")
    assert "error" in result
    assert "Invalid estimate_type" in result["error"]


def test_e_value_or_bounds_sanity():
    """Test sanity check for OR->RR conversion with inconsistent bounds."""
    # OR=4.0 -> approx RR=2.0
    # Lower=0.01 -> approx RR=0.1
    # Upper=100.0 -> approx RR=10.0
    # This is consistent: 0.1 <= 2.0 <= 10.0
    res_consistent = calculate_e_value(4.0, lower=0.01, upper=100.0, estimate_type="OR")
    assert res_consistent["e_value_estimate"] > 1.0

    # Inconsistent bounds after conversion (artificial example)
    # estimate=4.0 -> RR=2.0
    # lower=4.01 -> RR=2.002 (lower > estimate!) -> Should be set to None
    res_inconsistent_lower = calculate_e_value(4.0, lower=4.01, estimate_type="OR")
    assert res_inconsistent_lower["e_value_estimate"] > 1.0
    # Since lower > estimate, it should be ignored (None) for CI limit calc if it was used
    # But for RR > 1, we use lower bound for CI limit E-value.
    # If lower is None, we get default E-value of 1.0 for limit.
    assert (
        res_inconsistent_lower["e_value_ci_limit"] == 1.0
    )  # Because lower became None

    # Inconsistent upper (estimate=4.0 -> RR=2.0)
    # upper=3.99 -> RR=1.99 (upper < estimate!) -> Should be set to None
    res_inconsistent_upper = calculate_e_value(4.0, upper=3.99, estimate_type="OR")
    assert res_inconsistent_upper["e_value_estimate"] > 1.0

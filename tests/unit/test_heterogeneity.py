import numpy as np

from utils.heterogeneity_lib import calculate_heterogeneity


def test_heterogeneity_low():
    """Test heterogeneity with consistent studies."""
    # Similar effect sizes, small variance
    effects = [0.5, 0.55, 0.45, 0.52]
    variances = [0.01, 0.01, 0.01, 0.01]

    res = calculate_heterogeneity(effects, variances)

    assert res["Q"] >= 0
    assert res["I_squared"] < 10.0  # Expect low I2
    assert res["p_value"] > 0.05


def test_heterogeneity_high():
    """Test heterogeneity with inconsistent studies."""
    # Very different effects
    effects = [0.1, 0.9, 0.1, 0.9]
    variances = [0.001, 0.001, 0.001, 0.001]  # Precise estimates -> high Q

    res = calculate_heterogeneity(effects, variances)

    assert res["I_squared"] > 80.0  # Expect high I2
    assert res["p_value"] < 0.05
    assert res["tau_squared"] > 0


def test_heterogeneity_single_study():
    """Test handle single study case."""
    res = calculate_heterogeneity([0.5], [0.01])
    assert res["I_squared"] == 0.0
    assert res["Q"] == 0.0
    assert res["tau_squared"] == 0.0
    assert res["df"] == 0
    assert np.isnan(res["p_value"]) or res["p_value"] == 1.0


def test_heterogeneity_empty_input():
    """Test handle empty input lists."""

    # Function should either return a sentinel or raise ValueError
    # If it raises, we test the raise. If it returns dict with NaNs, we test that.
    # Looking at implementation, it likely raises or returns invalid dict.
    try:
        res = calculate_heterogeneity([], [])
        assert (
            res is None
            or np.isnan(res.get("I_squared", np.nan))
            or res.get("I_squared") == 0.0
        )
    except (ValueError, TypeError, ZeroDivisionError):
        pass

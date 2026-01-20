from utils.heterogeneity_lib import calculate_heterogeneity


def test_heterogeneity_low():
    """Test heterogeneity with consistent studies."""
    # Similar effect sizes, small variance
    effects = [0.5, 0.55, 0.45, 0.52]
    vars = [0.01, 0.01, 0.01, 0.01]

    res = calculate_heterogeneity(effects, vars)

    assert res["Q"] >= 0
    assert res["I_squared"] < 10.0  # Expect low I2
    assert res["p_value"] > 0.05


def test_heterogeneity_high():
    """Test heterogeneity with inconsistent studies."""
    # Very different effects
    effects = [0.1, 0.9, 0.1, 0.9]
    vars = [0.001, 0.001, 0.001, 0.001]  # Precise estimates -> high Q

    res = calculate_heterogeneity(effects, vars)

    assert res["I_squared"] > 80.0  # Expect high I2
    assert res["p_value"] < 0.05
    assert res["tau_squared"] > 0


def test_heterogeneity_single_study():
    """Test handle single study case."""
    res = calculate_heterogeneity([0.5], [0.01])
    assert res["I_squared"] == 0.0
    assert res["Q"] == 0.0

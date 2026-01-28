import numpy as np
import pandas as pd
import pytest


def test_calculate_bland_altman():
    # Create synthetic data with perfect agreement + constant bias
    # Method A: 10, 20, 30
    # Method B: 11, 21, 31 (Bias = -1)
    df = pd.DataFrame({"A": [10, 20, 30, 40, 50], "B": [11, 21, 31, 41, 51]})

    from utils.agreement_lib import AgreementAnalysis

    # FIX: Unpack tuple (stats, fig, metadata)
    stats, fig, *_ = AgreementAnalysis.bland_altman_advanced(df, "A", "B")

    assert stats["n"] == 5
    # A - B = -1
    assert stats["mean_diff"] == pytest.approx(-1.0)
    assert stats["sd_diff"] == pytest.approx(0.0)  # Constant difference

    # Limits: Mean ± 1.96*SD
    # Since SD=0, limits should equal bias
    assert stats["upper_loa"] == pytest.approx(-1.0)
    assert stats["lower_loa"] == pytest.approx(-1.0)

    # Check graph generation
    assert fig is not None
    assert "Bland-Altman" in fig.layout.title.text


def test_calculate_bland_altman_random():
    np.random.seed(42)
    # Method A vs Method B with some noise
    n = 100
    a = np.random.normal(100, 10, n)
    b = a + np.random.normal(0, 1, n)  # Mean diff approx 0, SD diff approx 1

    df = pd.DataFrame({"A": a, "B": b})
    # FIX: Unpack tuple
    from utils.agreement_lib import AgreementAnalysis

    stats, fig, *_ = AgreementAnalysis.bland_altman_advanced(df, "A", "B")

    assert stats["n"] == 100
    assert abs(stats["mean_diff"]) < 0.5  # Should be close to 0
    assert 0.8 < stats["sd_diff"] < 1.2  # Should be close to 1

    # Check LoA width approx 4 (±1.96 * 1 * 2 range?) No, Upper-Lower = 3.92*SD
    width = stats["upper_loa"] - stats["lower_loa"]
    assert 3.5 < width < 4.5

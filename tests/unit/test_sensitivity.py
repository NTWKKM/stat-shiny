import pytest

from utils.sensitivity_lib import calculate_e_value


def test_e_value_rr_gt_1():
    """Test E-value for RR > 1."""
    # Known value: RR=2 -> E-value = 2 + sqrt(2*1) = 2 + 1.414 = 3.414
    est = 2.0
    res = calculate_e_value(est)
    assert pytest.approx(res["e_value_point"], 0.01) == 3.414


def test_e_value_rr_lt_1():
    """Test E-value for RR < 1 (inverse)."""
    # RR=0.5 -> RR*=2.0 -> E-value ≈ 3.414
    est = 0.5
    res = calculate_e_value(est)
    assert pytest.approx(res["e_value_point"], 0.01) == 3.414


def test_e_value_ci():
    """Test E-value for Confidence Interval."""
    # RR=3, Lower CI=2
    # Est E-value: 3 + sqrt(3*2) = 3 + 2.45 = 5.45
    # CI E-value (limit=2): 2 + sqrt(2*1) = 3.414
    res = calculate_e_value(3.0, lower_ci=2.0)
    assert pytest.approx(res["e_value_point"], 0.01) == 5.45
    assert pytest.approx(res["e_value_ci"], 0.01) == 3.414


def test_e_value_ci_cross_null():
    """Test E-value when CI crosses 1."""
    # RR=1.5, Lower CI=0.8
    res = calculate_e_value(1.5, lower_ci=0.8)
    assert res["e_value_ci"] == 1.0

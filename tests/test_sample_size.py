import numpy as np
import pytest

from utils import sample_size_lib


def test_calculate_power_means():
    # Known value check: n=16 per group, d=1, alpha=0.05 -> power approx 0.8
    # Effect size d = |0-5|/5 = 1.0 (if sd=5)
    mean1, mean2 = 0, 5
    sd1, sd2 = 5, 5
    n1 = 17  # approx
    power = sample_size_lib.calculate_power_means(n1, n1, mean1, mean2, sd1, sd2)
    assert power > 0.8
    assert power < 0.9


def test_calculate_sample_size_means():
    # Reverse of above
    mean1, mean2 = 0, 5
    sd1, sd2 = 5, 5
    res = sample_size_lib.calculate_sample_size_means(0.8, 1.0, mean1, mean2, sd1, sd2)
    assert res["n1"] >= 16
    assert res["n2"] >= 16
    assert res["total"] == res["n1"] + res["n2"]


def test_calculate_sample_size_proportions():
    # Example: p1=0.1, p2=0.5. Large effect.
    res = sample_size_lib.calculate_sample_size_proportions(0.8, 1.0, 0.1, 0.5)
    # n approx 23
    assert 15 < res["n1"] < 35


def test_calculate_sample_size_survival():
    # Example Freedman: hr=0.5, alpha=0.05, power=0.8
    # Events approx 66 (from literature/calculators)
    res = sample_size_lib.calculate_sample_size_survival(
        power=0.8, ratio=1.0, h0=0.5, h1=None
    )
    # Allow some range due to approximation method differences
    assert 60 < res["total_events"] < 75


def test_calculate_sample_size_correlation():
    # r=0.3, power=0.8, alpha=0.05 -> N approx 85
    n = sample_size_lib.calculate_sample_size_correlation(0.8, 0.3)
    assert 80 < n < 90

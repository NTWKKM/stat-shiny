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
    assert 80 < n["total"] < 90


def test_calculate_power_survival():
    # HR=0.5, Alpha=0.05. Events approx 66 for 0.8 power.
    power = sample_size_lib.calculate_power_survival(total_events=66, ratio=1.0, h0=0.5)
    assert 0.75 < power < 0.85


def test_calculate_power_correlation():
    # r=0.3, approx N=85 for 0.8 power
    power = sample_size_lib.calculate_power_correlation(n=85, r=0.3)
    assert 0.78 < power < 0.82


def test_calculate_power_curve():
    # Test with Means
    df = sample_size_lib.calculate_power_curve(
        target_n=34,  # ~17 per group from test_calculate_power_means
        ratio=1.0,
        calc_func=sample_size_lib.calculate_power_means,
        mean1=0,
        mean2=5,
        sd1=5,
        sd2=5,
    )

    assert not df.empty
    assert "total_n" in df.columns
    assert "power" in df.columns
    assert len(df) > 10

    # Check monotonic increase
    powers = df["power"].values
    # Check if Sorted (allowing small floating point equality or NaN at start if any)
    # Just check end > start
    assert powers[-1] > powers[0]


def test_generate_methods_text():
    # Test Means (T-test)
    params_means = {
        "mean1": 0,
        "mean2": 5,
        "sd1": 5,
        "sd2": 5,
        "power": 0.8,
        "alpha": 0.05,
        "total": 34,
        "n1": 17,
        "n2": 17,
    }
    text_means = sample_size_lib.generate_methods_text(
        "Independent Means (T-test)", params_means
    )
    assert "Independent Means (T-test)" in text_means
    assert "total sample size of 34 subjects" in text_means
    assert "power of 80%" in text_means

    # Test Survival
    params_surv = {"hr": 0.5, "power": 0.8, "alpha": 0.05, "total_events": 66}
    text_surv = sample_size_lib.generate_methods_text(
        "Survival (Log-Rank)", params_surv
    )
    assert "Survival (Log-Rank)" in text_surv
    assert "total of 66 events" in text_surv

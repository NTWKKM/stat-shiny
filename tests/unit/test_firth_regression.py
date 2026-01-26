from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.logic import fit_firth_logistic, fit_standard_logistic, run_binary_logit


def test_firth_import():
    """Verify firthmodels is importable."""
    try:
        import firthmodels  # noqa: F401
    except ImportError:
        pytest.fail("firthmodels not installed")


def test_firth_separation_resolution():
    """Test that Firth solves perfect separation."""
    # Data from Heinze and Schemper (2002) / firthmodels docs
    # Case: Perfect separation
    # x=1 -> y=1 (3 cases)
    # x=0 -> y=0 (3 cases)

    data = pd.DataFrame({"const": [1.0] * 6, "x": [0, 0, 0, 1, 1, 1]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    # 1. Standard Logistic (Likely fails or warns)
    # Statsmodels usually raises PerfectSeparationError or LinAlgError
    # Our wrapper catches it.
    params_std, _, _, status_std, _ = fit_standard_logistic(y, data)

    # It might return an error string OR seemingly valid but huge params
    # If it returns "OK", check if params are huge (infinite)
    if status_std == "OK":
        # Check for numeric instability
        if np.abs(params_std["x"]) > 10:
            pass  # working as expected (diverging)
    else:
        # Status might be an error message
        pass

    # 2. Firth Logistic
    params, conf, pvals, status, metrics = fit_firth_logistic(y, data)

    assert status == "OK", f"Firth failed with status: {status}"
    assert params is not None
    assert not np.isnan(params["x"])
    assert np.abs(params["x"]) < 10.0  # Should be finite (~3.89)

    # Expected approx 3.89 based on firthmodels docs
    assert 3.5 <= params["x"] <= 4.5

    # Check P-values exist
    assert not pvals.isna().all()


def test_firth_vs_r_benchmark():
    """Compare Python firthmodels results against R logistf benchmark."""

    # Paths (relative to this test file or project root)
    # Assuming running from project root
    base_dir = Path(__file__).parent.parent.parent
    bench_dir = base_dir / "tests" / "benchmarks" / "python_results"

    data_path = bench_dir / "dataset_sex2.csv"
    res_path = bench_dir / "benchmark_firth_logistic.csv"

    if not data_path.exists() or not res_path.exists():
        pytest.skip(
            "R benchmark files not found. Run tests/benchmarks/r_scripts/test_firth.R first."
        )

    # Load Data
    sex2 = pd.read_csv(data_path)
    r_res = pd.read_csv(res_path)

    # Prepare X, y
    y = sex2["case"]
    X = sex2.drop(columns=["case"])

    # Fit Python model
    # Note: R benchmark uses Profile Likelihood CI (conf.low, conf.high)
    params, conf, pvals, status, _ = run_binary_logit(y, X, method="firth")

    assert status == "OK"

    # Compare Coefficients
    for _, row in r_res.iterrows():
        term = row["term"]
        if term == "(Intercept)":
            term_py = "const"
        else:
            term_py = term

        if term_py not in params.index:
            # Maybe strict mapping failed, skip or strictly fail?
            # R might have different dummy encoding if factors were used.
            # But sex2 is numeric/binary so should match.
            continue

        py_coef = params[term_py]
        r_coef = row["estimate"]

        # Check coefficient match (tolerance 1e-4)
        np.testing.assert_allclose(
            py_coef, r_coef, rtol=1e-4, atol=1e-4, err_msg=f"Coef mismatch for {term}"
        )

        # Check CI (if Python uses PL, it should match R's PL)
        # Our implementation tries PL first.
        # R logistf defaults to PL.
        py_low = conf.loc[term_py][0]
        py_high = conf.loc[term_py][1]

        r_low = row["conf.low"]
        r_high = row["conf.high"]

        # CI matching might be slightly looser due to optimization diffs
        np.testing.assert_allclose(
            py_low, r_low, rtol=1e-3, atol=1e-3, err_msg=f"CI Low mismatch for {term}"
        )
        np.testing.assert_allclose(
            py_high,
            r_high,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"CI High mismatch for {term}",
        )

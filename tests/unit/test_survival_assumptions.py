import importlib

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

import utils.survival_lib


@pytest.fixture(autouse=True)
def clean_survival_lib():
    """Ensure survival_lib is reloaded with real dependencies."""
    # Reload utils.survival_lib to ensure it uses the real lifelines module
    # and not any mocks that might have been injected by other tests (e.g., test_statistics.py)
    importlib.reload(utils.survival_lib)

    yield

    # Ideally we shouldn't need to clean up as reload overwrites, but good practice if needed


def test_check_cph_assumptions_structure():
    """Test that check_cph_assumptions returns the correct structure and plots."""

    # Create synthetic data
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "T": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "E": [1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            "var1": np.random.normal(0, 1, 10),
            "var2": np.random.normal(0, 1, 10),
        }
    )

    cph = CoxPHFitter()
    cph.fit(df, duration_col="T", event_col="E")

    # Run the function
    text_report, figs = utils.survival_lib.check_cph_assumptions(cph, df)

    # Check return types
    assert isinstance(text_report, str)
    assert isinstance(figs, list)

    # Check that we have multiple figures
    # We expect:
    # 1. Schoenfeld plots (one per variable + possibly global, but code iterates columns of residuals)
    #    lifelines compute_residuals('scaled_schoenfeld') returns cols for each covariate.
    #    So len(covariates) plots.
    # 2. Martingale residual plot (1 plot)
    # 3. Deviance residual plot (1 plot)
    # Total = len(covariates) + 2

    # Check for plot titles to confirm they are what we expect
    plot_titles = [f.layout.title.text for f in figs if f.layout.title.text]

    assert any("Schoenfeld" in t for t in plot_titles), "Missing Schoenfeld plots"
    assert any("Martingale" in t for t in plot_titles), "Missing Martingale plot"
    assert any("Deviance" in t for t in plot_titles), "Missing Deviance plot"

    # Check text report content
    assert "Proportional Hazards Test Results" in text_report
    # It might pass or fail depending on random data, but we check for structure
    assert "Assumption Passed" in text_report or "Assumption Violations" in text_report


from unittest.mock import MagicMock, patch


def test_check_cph_assumptions_violation_text():
    """Test check_cph_assumptions text output when proportional hazards assumption is violated."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "T": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "E": [1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            "var1": np.random.normal(0, 1, 10),
            "var2": np.random.normal(0, 1, 10),
        }
    )

    cph = CoxPHFitter()
    cph.fit(df, duration_col="T", event_col="E")

    # Mock proportional_hazard_test to return p < 0.05 for var1
    mock_res = MagicMock()
    mock_res.summary = pd.DataFrame(
        {
            "test_statistic": [12.5, 0.4],
            "p": [0.001, 0.65],
            "-log2(p)": [9.96, 0.62],
        },
        index=["var1", "var2"],
    )

    with patch("utils.survival_lib.proportional_hazard_test", return_value=mock_res):
        text_report, figs = utils.survival_lib.check_cph_assumptions(cph, df)

    assert "Assumption Violations Detected" in text_report
    assert "var1 (p=0.001)" in text_report
    assert "Suggested Remedies" in text_report
    assert len(figs) >= 3

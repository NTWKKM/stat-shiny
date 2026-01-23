import numpy as np
import pandas as pd
import pytest

# Mock CONFIG
from config import CONFIG
from utils.diag_test import analyze_roc, calculate_chi2, calculate_descriptive


@pytest.fixture(autouse=True)
def setup_config():
    # Save original values
    orig_report = CONFIG.get("analysis.missing.report_missing")
    orig_strategy = CONFIG.get("analysis.missing.strategy")
    orig_user_defined = CONFIG.get("analysis.missing.user_defined_values")

    # Update for tests
    CONFIG.update("analysis.missing.report_missing", True)
    CONFIG.update("analysis.missing.strategy", "complete-case")
    CONFIG.update("analysis.missing.user_defined_values", [])

    yield

    # Restore
    CONFIG.update("analysis.missing.report_missing", orig_report)
    CONFIG.update("analysis.missing.strategy", orig_strategy)
    CONFIG.update("analysis.missing.user_defined_values", orig_user_defined)


def test_chi2_pipeline_integration():
    """
    Test that calculate_chi2 correctly uses the unified pipeline
    and returns missing data info.
    """
    data = {
        "exposure": ["A", "A", "B", "B", np.nan, "A", "B", "A", "B", "A"],
        "outcome": ["Y", "N", "Y", "N", "Y", np.nan, "N", "Y", "N", "Y"],
    }
    df = pd.DataFrame(data)

    # Run analysis
    tab, stats_df, msg, risk_df, missing_info = calculate_chi2(
        df=df,
        col1="exposure",
        col2="outcome",
        var_meta={
            "exposure": {"type": "categorical"},
            "outcome": {"type": "categorical"},
        },
    )

    # Assertions
    assert tab is not None
    assert missing_info is not None
    assert missing_info["rows_excluded"] == 2
    assert missing_info["rows_analyzed"] == 8
    assert missing_info["strategy"] == "complete-case"


def test_roc_pipeline_integration():
    """
    Test that analyze_roc correctly uses the unified pipeline
    and returns missing data info in stats dict.
    """
    data = {
        "truth": ["1", "0", "1", "0", np.nan, "1", "0", "1", "0", "1"],
        "score": [0.9, 0.1, 0.8, 0.2, 0.5, np.nan, 0.3, 0.7, 0.4, 0.6],
    }
    df = pd.DataFrame(data)

    stats_dict, err, fig, coords = analyze_roc(
        df=df,
        truth_col="truth",
        score_col="score",
        pos_label_user="1",
        var_meta={"truth": {"type": "categorical"}, "score": {"type": "continuous"}},
    )

    assert stats_dict is not None
    assert "missing_data_info" in stats_dict
    assert stats_dict["missing_data_info"]["rows_excluded"] == 2
    assert stats_dict["missing_data_info"]["rows_analyzed"] == 8


def test_descriptive_pipeline_integration():
    """
    Test that calculate_descriptive correctly uses the unified pipeline
    and returns missing data info.
    """
    data = {"age": [20, 30, 40, np.nan, 50, 60, np.nan, 80]}
    df = pd.DataFrame(data)

    stats_df, missing_info = calculate_descriptive(
        df=df, col="age", var_meta={"age": {"type": "continuous"}}
    )

    assert stats_df is not None
    assert missing_info is not None
    assert missing_info["rows_excluded"] == 2
    assert missing_info["rows_analyzed"] == 6

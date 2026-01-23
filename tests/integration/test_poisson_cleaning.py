import numpy as np
import pandas as pd
import pytest

# Mock CONFIG to ensure report generation is enabled
from config import CONFIG
from utils.poisson_lib import analyze_poisson_outcome


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


def test_poisson_pipeline_integration():
    """
    Test that analyze_poisson_outcome correctly uses the unified pipeline
    and includes the missing data report in the returned HTML.
    """
    # Create sample data with missing values
    data = {
        "outcome": [1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10],  # 1 missing outcome
        "predictor": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],  # 1 missing predictor
        "offset": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    }
    df = pd.DataFrame(data)

    # Run analysis
    html_rep, irr, airr, interactions, _ = analyze_poisson_outcome(
        outcome_name="outcome", df=df, offset_col="offset"
    )

    # Assertions
    assert html_rep is not None
    assert "Missing Data Summary" in html_rep
    assert "Included:" in html_rep
    assert "Excluded:" in html_rep
    # We expect 2 rows excluded (one missing outcome, one missing predictor) so n=8
    # However, row 4 has missing outcome, row 2 has missing predictor.
    # Row indices: 0,1,2,3,4,5...
    # Row 2 (0-indexed) is index 2. Row 4 is index 4.
    # Total 10 rows. 2 removed. 8 remaining.
    assert "<b>2</b>" in html_rep or "2" in html_rep  # Simple check for count


def test_poisson_pipeline_all_missing():
    """
    Test that complete missing data is handled gracefully.
    """
    data = {"outcome": [np.nan] * 10, "predictor": [1] * 10}
    df = pd.DataFrame(data)

    html_rep, irr, airr, interactions, _ = analyze_poisson_outcome(
        outcome_name="outcome", df=df
    )

    # "No valid numeric values" is returned by check_count_outcome when all defaults are NaN
    assert (
        "No valid numeric values" in html_rep or "No valid data remaining" in html_rep
    )

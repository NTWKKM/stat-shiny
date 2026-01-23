import numpy as np
import pandas as pd
import pytest

from utils.repeated_measures_lib import create_trajectory_plot, run_gee, run_lmm


@pytest.fixture
def mock_longitudinal_data():
    # Create synthetic longitudinal data
    # 20 subjects, 3 time points
    np.random.seed(42)
    n_subjects = 50
    n_times = 3

    subjects = []
    times = []
    groups = []
    outcomes = []
    ages = []

    for i in range(n_subjects):
        group = "Treatment" if i < n_subjects / 2 else "Control"
        age = np.random.randint(20, 60)
        baseline = 10 + (5 if group == "Treatment" else 0) + np.random.normal(0, 2)
        slope = 2 if group == "Treatment" else 0.5

        for t in range(n_times):
            subjects.append(f"Subject_{i}")
            times.append(t)
            groups.append(group)
            ages.append(age)
            # Outcome = baseline + slope*t + error
            outcomes.append(baseline + slope * t + np.random.normal(0, 1))

    df = pd.DataFrame(
        {
            "SubjectID": subjects,
            "Time": times,
            "Group": groups,
            "Outcome": outcomes,
            "Age": ages,
        }
    )
    return df


def test_run_gee(mock_longitudinal_data):
    df = mock_longitudinal_data

    # Test Basic GEE
    results, missing_info = run_gee(
        df,
        outcome_col="Outcome",
        treatment_col="Group",
        time_col="Time",
        subject_col="SubjectID",
        covariates=["Age"],
        cov_struct="exchangeable",
    )

    assert not isinstance(results, str), f"GEE failed: {results}"
    # Check if 'Group' is part of any index name
    param_names = results.params.index.tolist()
    assert any("Group" in name for name in param_names), (
        f"Group not found in params: {param_names}"
    )
    assert any("Time" in name for name in param_names), (
        f"Time not found in params: {param_names}"
    )

    # Test with different correlation structure
    results_indep, _ = run_gee(
        df,
        outcome_col="Outcome",
        treatment_col="Group",
        time_col="Time",
        subject_col="SubjectID",
        cov_struct="independence",
    )
    assert not isinstance(results_indep, str), (
        f"GEE Independence failed: {results_indep}"
    )


def test_run_lmm(mock_longitudinal_data):
    df = mock_longitudinal_data

    # Test Basic LMM (Random Intercept)
    results, _ = run_lmm(
        df,
        outcome_col="Outcome",
        treatment_col="Group",
        time_col="Time",
        subject_col="SubjectID",
        covariates=["Age"],
        random_slope=False,
    )

    assert not isinstance(results, str), f"LMM failed: {results}"
    param_names = results.params.index.tolist()
    assert any("Group" in name for name in param_names), (
        f"Group not found in params: {param_names}"
    )

    # Test LMM with Random Slope
    results_slope, _ = run_lmm(
        df,
        outcome_col="Outcome",
        treatment_col="Group",
        time_col="Time",
        subject_col="SubjectID",
        random_slope=True,
    )
    assert not isinstance(results_slope, str), (
        f"LMM Random Slope failed: {results_slope}"
    )


def test_create_trajectory_plot(mock_longitudinal_data):
    df = mock_longitudinal_data
    fig = create_trajectory_plot(
        df, outcome_col="Outcome", time_col="Time", group_col="Group"
    )
    assert fig is not None
    assert len(fig.data) >= 2  # At least mean lines for 2 groups

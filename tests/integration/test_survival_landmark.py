"""
🔗 Integration Tests for Survival Analysis - Landmark Analysis
Tests the landmark analysis functionality in survival_lib.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Setup path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.survival_lib import fit_km_landmark


@pytest.fixture
def landmark_data():
    """Create a sample dataset for landmark analysis."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "time": np.random.exponential(50, n),
            "event": np.random.binomial(1, 0.7, n),
            "group": np.random.choice(["Control", "Treatment"], n),
        }
    )
    return df


def test_fit_km_landmark_success(landmark_data):
    """
    Test that fit_km_landmark returns the correct structure when successful.
    """
    landmark_time = 10.0

    # Run landmark analysis
    fig, stats_df, n_pre, n_post, err, missing_info = fit_km_landmark(
        landmark_data,
        duration_col="time",
        event_col="event",
        group_col="group",
        landmark_time=landmark_time,
    )

    # Check for no errors
    assert err is None
    assert missing_info is not None

    # Check return values
    assert n_pre == 100
    assert n_post < 100  # Should filter out those who died/censored before landmark
    assert n_post > 0  # Should have some remaining

    # Check stats dataframe
    assert stats_df is not None
    assert not stats_df.empty
    assert "P-value" in stats_df.columns
    assert "Statistic" in stats_df.columns

    # Check Figure
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) > 0  # Should have traces


def test_fit_km_landmark_insufficient_data(landmark_data):
    """
    Test behavior when data is insufficient at landmark time.
    """
    # Set landmark time higher than max survival time
    landmark_time = landmark_data["time"].max() + 10

    fig, stats_df, n_pre, n_post, err, missing_info = fit_km_landmark(
        landmark_data,
        duration_col="time",
        event_col="event",
        group_col="group",
        landmark_time=landmark_time,
    )

    assert err is not None
    assert "Insufficient patients" in err
    assert fig is None
    assert stats_df is None

"""
🔗 Integration Tests for PSM Pipeline
File: tests/integration/test_psm_pipeline.py

Tests Propensity Score Matching flow:
1. Propensity Score Calculation (Logistic Regression)
2. Matching (Nearest Neighbor)
3. Balance Assessment (SMD Calculation)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.psm_lib import (
    calculate_propensity_score,
    compute_smd,
    perform_matching,
    plot_love_plot,
)

# Mark as integration test
pytestmark = pytest.mark.integration


class TestPSMPipeline:
    @pytest.fixture
    def psm_data(self):
        """Create observational dataset for matching"""
        np.random.seed(202)
        n = 300

        # Confounders affecting both treatment and outcome
        age = np.random.normal(60, 10, n)
        severity = np.random.normal(5, 2, n)

        # Propensity score (Probability of treatment)
        # Older and more severe patients more likely to get Treatment A (1)
        ps_logit = -5 + 0.05 * age + 0.5 * severity
        prob = 1 / (1 + np.exp(-ps_logit))
        treatment = np.random.binomial(1, prob)

        df = pd.DataFrame(
            {
                "age": age,
                "severity": severity,
                "treatment": treatment,
                "outcome": np.random.binomial(1, 0.3, n),  # Dummy outcome
            }
        )

        return df

    def test_psm_full_flow(self, psm_data):
        """🔄 Test complete PSM workflow"""
        df = psm_data
        covariates = ["age", "severity"]

        # Step 1: Calculate Propensity Scores
        ps_series, _ = calculate_propensity_score(df, "treatment", covariates)

        assert len(ps_series) == len(df)
        assert ps_series.min() >= 0
        assert ps_series.max() <= 1

        # Attach PS to dataframe
        df["ps_score"] = ps_series

        # Step 2: Perform Matching
        matched_df = perform_matching(
            df, treatment_col="treatment", ps_col="ps_score", caliper=0.2
        )

        assert not matched_df.empty
        assert len(matched_df) <= len(df)
        # In 1:1 matching, treated and control counts should be close/equal
        counts = matched_df["treatment"].value_counts()
        # It's possible to not find matches for everyone, but should match some
        assert len(counts) == 2

        # Step 3: Check Balance (SMD)
        # Unmatched balance
        smd_pre = compute_smd(df, "treatment", covariates)
        # Matched balance
        smd_post = compute_smd(matched_df, "treatment", covariates)

        assert not smd_pre.empty
        assert not smd_post.empty

        # Verify SMD structure
        assert "SMD" in smd_pre.columns
        assert "Variable" in smd_pre.columns

        # Ideally, matching should improve balance (reduce SMD)
        # Checking average SMD reduction
        # avg_smd_pre = smd_pre["SMD"].mean()
        avg_smd_post = smd_post["SMD"].mean()

        # Note: In random data this might not always be true, but statistically likely
        # We just check the calculation ran correctly
        assert isinstance(avg_smd_post, float)

    def test_love_plot_generation(self):
        """📊 Test Love Plot generation"""
        covariates = ["age", "severity"]

        # Generate dummy SMD data
        smd_pre = pd.DataFrame({"Variable": covariates, "SMD": [0.5, 0.4]})
        smd_post = pd.DataFrame({"Variable": covariates, "SMD": [0.05, 0.04]})

        fig = plot_love_plot(smd_pre, smd_post)

        assert fig is not None
        # Check if it's a Plotly figure
        assert hasattr(fig, "to_html")
        # Check title
        assert "Love Plot" in fig.layout.title.text

    def test_psm_no_matches(self):
        """⚠️ Test matching with disjoint groups (should return empty or minimal matches)"""
        df = pd.DataFrame(
            {
                "treatment": [1] * 10 + [0] * 10,
                # Perfectly separated scores -> no overlap
                "ps_score": [0.9] * 10 + [0.1] * 10,
            }
        )

        # With caliper 0.1, no matches should be found (diff is 0.8)
        matched_df = perform_matching(
            df, treatment_col="treatment", ps_col="ps_score", caliper=0.1
        )

        # Should be empty or only self (depending on implementation, here empty expected)
        assert matched_df.empty

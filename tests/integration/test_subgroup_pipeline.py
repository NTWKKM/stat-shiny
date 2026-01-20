"""
🔗 Integration Tests for Subgroup Analysis Module
File: tests/integration/test_subgroup_pipeline.py

Tests subgroup analysis using the SubgroupAnalysisLogit class.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# ✅ Correct Import: It's a Class, not a function
from utils.subgroup_analysis_module import SubgroupAnalysisLogit

pytestmark = pytest.mark.integration


class TestSubgroupAnalysisPipeline:

    @pytest.fixture
    def subgroup_data(self):
        """Create dataset for subgroup analysis"""
        np.random.seed(99)
        n = 300

        # Subgroups
        gender = np.random.choice(["Male", "Female"], n)
        age_group = np.random.choice(["<65", ">=65"], n)

        # Treatment
        treatment = np.random.choice(["A", "B"], n)

        # Outcome (Binary)
        # Treatment B better in Males
        logit = (
            -0.5
            + 0.5 * (treatment == "B")
            + 1.0 * (gender == "Male") * (treatment == "B")
        )
        prob = 1 / (1 + np.exp(-logit))
        outcome = np.random.binomial(1, prob)

        df = pd.DataFrame(
            {
                "gender": gender,
                "age_group": age_group,
                "treatment": treatment,
                "outcome": outcome,
            }
        )
        return df

    def test_subgroup_analysis_logit_flow(self, subgroup_data):
        """🔄 Test Subgroup Analysis (Logistic) Flow"""
        df = subgroup_data

        # 1. Initialize Analyzer
        analyzer = SubgroupAnalysisLogit(df=df)

        # 2. Run Analysis on specific subgroups
        # The analyzer performs analysis per subgroup
        analyzer.analyze(
            outcome_col="outcome", treatment_col="treatment", subgroup_col="gender"
        )

        # 3. Verify Results Structure
        assert analyzer.results is not None
        assert len(analyzer.results) > 0

        # Check stats computed from results
        assert analyzer.stats is not None
        assert "n_overall" in analyzer.stats

    def test_subgroup_forest_plot_generation(self, subgroup_data):
        """📊 Test Forest Plot creation from analyzer"""
        df = subgroup_data
        analyzer = SubgroupAnalysisLogit(df=df)
        analyzer.analyze(
            outcome_col="outcome", treatment_col="treatment", subgroup_col="gender"
        )

        # Try generating plot
        fig = analyzer.create_forest_plot(title="Test Forest Plot")

        assert fig is not None
        assert hasattr(fig, "layout")
        assert "Test Forest Plot" in fig.layout.title.text

    def test_invalid_inputs(self, subgroup_data):
        """🚫 Test handling of invalid columns"""
        df = subgroup_data

        # Non-existent columns should raise ValueError
        analyzer = SubgroupAnalysisLogit(df=df)
        with pytest.raises(ValueError, match="Missing columns"):
            analyzer.analyze(
                outcome_col="non_existent",
                treatment_col="treatment",
                subgroup_col="gender",
            )

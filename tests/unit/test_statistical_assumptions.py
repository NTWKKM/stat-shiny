"""
Unit tests for utils/statistical_assumptions.py

Tests for homogeneity of variance, normality, and combined assumption checks.
"""

import numpy as np
import pandas as pd

from utils.statistical_assumptions import (
    assess_assumptions_for_anova,
    assess_assumptions_for_ttest,
    check_homogeneity_of_variance,
    check_normality_comprehensive,
    test_sphericity as check_sphericity,
)


class TestHomogeneityOfVariance:
    """Tests for variance homogeneity tests."""

    def test_levene_equal_variance(self):
        """Test Levene's test with equal variance groups."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 30)
        g2 = np.random.normal(5, 1, 30)  # Same variance, different mean

        result = check_homogeneity_of_variance(g1, g2, method="levene")

        assert "error" not in result
        assert result["assumption_met"]

    def test_levene_unequal_variance(self):
        """Test Levene's test with unequal variance groups."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(0, 5, 50)  # Much larger variance

        result = check_homogeneity_of_variance(g1, g2, method="levene")

        assert "error" not in result
        assert not result["assumption_met"]

    def test_all_methods(self):
        """Test running all variance tests together."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 30)
        g2 = np.random.normal(0, 1, 30)

        result = check_homogeneity_of_variance(g1, g2, method="all")

        assert "tests" in result
        assert "levene" in result["tests"]
        assert "bartlett" in result["tests"]
        assert "fligner" in result["tests"]
        assert "overall_recommendation" in result

    def test_three_groups(self):
        """Test with three groups."""
        g1 = [1, 2, 3, 4, 5]
        g2 = [2, 3, 4, 5, 6]
        g3 = [3, 4, 5, 6, 7]

        result = check_homogeneity_of_variance(g1, g2, g3, method="levene")

        assert "error" not in result

    def test_insufficient_data(self):
        """Test error handling for insufficient data."""
        result = check_homogeneity_of_variance([1], [2, 3])

        assert "error" in result


class TestNormality:
    """Tests for normality testing."""

    def test_normal_data(self):
        """Test with normally distributed data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        result = check_normality_comprehensive(data)

        assert "error" not in result
        assert "tests" in result
        assert "overall_assessment" in result
        # Should be approximately normal
        assert "normal" in result["overall_assessment"].lower()

    def test_non_normal_data(self):
        """Test with non-normal (uniform) data."""
        np.random.seed(42)
        data = np.random.uniform(0, 10, 100)

        result = check_normality_comprehensive(data)

        assert "error" not in result
        assert not result["tests"]["shapiro_wilk"]["normal"]

    def test_skewness_kurtosis_provided(self):
        """Test that skewness and kurtosis are reported."""
        data = np.random.normal(0, 1, 50)

        result = check_normality_comprehensive(data)

        assert "skewness" in result
        assert "kurtosis" in result


class TestTTestAssumptions:
    """Tests for combined t-test assumption checking."""

    def test_assumptions_met(self):
        """Test when all assumptions are met."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 40)
        g2 = np.random.normal(0.5, 1, 40)

        result = assess_assumptions_for_ttest(g1, g2)

        assert "error" not in result
        assert "recommended_test" in result
        assert "normality" in result
        assert "variance" in result

    def test_unequal_variance_recommendation(self):
        """Test that Welch's t-test is recommended for unequal variance."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(0, 5, 50)  # High variance

        result = assess_assumptions_for_ttest(g1, g2)

        # Should recommend Welch's t-test
        assert "welch" in result["recommended_test"].lower()

    def test_non_normal_recommendation(self):
        """Test recommendation output for small samples with outliers."""
        # Highly skewed data with outliers
        g1 = np.concatenate([np.ones(10), np.array([100])])
        g2 = np.concatenate([np.ones(10), np.array([200])])

        result = assess_assumptions_for_ttest(g1, g2)

        assert "error" not in result
        # With small samples, should give recommendation (any valid test)
        assert "recommended_test" in result
        assert len(result["recommended_test"]) > 0


class TestANOVAAssumptions:
    """Tests for combined ANOVA assumption checking."""

    def test_anova_assumptions_met(self):
        """Test when all ANOVA assumptions are met."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 30)
        g2 = np.random.normal(1, 1, 30)
        g3 = np.random.normal(2, 1, 30)

        result = assess_assumptions_for_anova(g1, g2, g3)

        assert "error" not in result
        assert result["n_groups"] == 3
        assert "recommended_test" in result

    def test_anova_unequal_variance(self):
        """Test that Welch's ANOVA is recommended for unequal variance."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 30)
        g2 = np.random.normal(0, 5, 30)  # High variance
        g3 = np.random.normal(0, 1, 30)

        result = assess_assumptions_for_anova(g1, g2, g3)

        assert "welch" in result["recommended_test"].lower()

    def test_anova_insufficient_groups(self):
        """Test error handling for insufficient groups."""
        result = assess_assumptions_for_anova([1, 2, 3])

        assert "error" in result


class TestSphericity:
    """Tests for sphericity assumption check."""

    def test_sphericity_placeholder(self):
        """Test that the function returns placeholder information."""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "t1": [1, 2, 3], "t2": [2, 3, 4], "t3": [3, 4, 5]}
        )

        result = check_sphericity(df, "id", ["t1", "t2", "t3"])

        assert "error" not in result
        assert "note" in result
        assert "pingouin" in result["note"]
        assert "recommendation" in result

    def test_sphericity_insufficient_cols(self):
        """Test error handling for insufficient within-subject columns."""
        df = pd.DataFrame({"id": [1, 2], "t1": [1, 2]})

        result = check_sphericity(df, "id", ["t1"])

        assert "error" in result

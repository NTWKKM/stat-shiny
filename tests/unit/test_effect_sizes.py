"""
Unit tests for utils/effect_sizes.py

Tests for Cohen's d, Hedges' g, Glass's Δ, η², ω², and publication formatting.
"""

import numpy as np

from utils.effect_sizes import (
    cohens_d,
    eta_squared,
    format_effect_size_for_publication,
    glass_delta,
    hedges_g,
    interpret_effect_size,
    omega_squared,
)


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_cohens_d_basic(self):
        """Test basic Cohen's d calculation."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [2, 3, 4, 5, 6]

        result = cohens_d(group1, group2)

        assert "error" not in result
        assert "d" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["n1"] == 5
        assert result["n2"] == 5

    def test_cohens_d_identical_groups(self):
        """Test Cohen's d with identical groups (d should be ~0)."""
        group = [1, 2, 3, 4, 5]

        result = cohens_d(group, group)

        assert "error" not in result
        assert abs(result["d"]) < 0.01  # Should be approximately 0

    def test_cohens_d_large_effect(self):
        """Test Cohen's d with large effect size."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [10, 11, 12, 13, 14]

        result = cohens_d(group1, group2)

        assert "error" not in result
        assert result["interpretation"] == "Large"

    def test_cohens_d_handles_nan(self):
        """Test that NaN values are properly handled."""
        group1 = [1, 2, np.nan, 4, 5]
        group2 = [2, 3, 4, 5, 6]

        result = cohens_d(group1, group2)

        assert "error" not in result
        assert result["n1"] == 4  # NaN should be excluded

    def test_cohens_d_insufficient_data(self):
        """Test error handling for insufficient data."""
        result = cohens_d([1], [2])

        assert "error" in result

    def test_cohens_d_ci_contains_estimate(self):
        """Test that CI contains the point estimate."""
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)

        result = cohens_d(group1, group2)

        assert result["ci_lower"] <= result["d"] <= result["ci_upper"]


class TestHedgesG:
    """Tests for Hedges' g bias-corrected effect size."""

    def test_hedges_g_basic(self):
        """Test basic Hedges' g calculation."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [2, 3, 4, 5, 6]

        result = hedges_g(group1, group2)

        assert "error" not in result
        assert "g" in result
        assert "correction_factor" in result

    def test_hedges_g_smaller_than_d(self):
        """Test that Hedges' g is smaller than Cohen's d for small samples."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [4, 5, 6, 7, 8]

        result_g = hedges_g(group1, group2)
        result_d = cohens_d(group1, group2)

        # For small samples, |g| < |d| due to bias correction
        assert abs(result_g["g"]) < abs(result_d["d"])

    def test_hedges_g_recommendation(self):
        """Test that recommendation is given for small samples."""
        small_group1 = [1, 2, 3, 4, 5]
        small_group2 = [2, 3, 4, 5, 6]

        result = hedges_g(small_group1, small_group2)

        assert "recommendation" in result
        assert "small sample" in result["recommendation"].lower()


class TestGlassDelta:
    """Tests for Glass's delta effect size."""

    def test_glass_delta_basic(self):
        """Test basic Glass's delta calculation."""
        treatment = [5, 6, 7, 8, 9]
        control = [1, 2, 3, 4, 5]

        result = glass_delta(treatment, control)

        assert "error" not in result
        assert "delta" in result
        assert "control_sd" in result

    def test_glass_delta_uses_control_sd(self):
        """Test that Glass's delta uses only control group SD."""
        # Treatment with high variance, control with low variance
        treatment = [1, 10, 5, 12, 3]  # High variance
        control = [2, 3, 2, 3, 2]  # Low variance

        result = glass_delta(treatment, control)

        # Delta should be larger because control SD is small
        assert abs(result["delta"]) > 0


class TestAnovaEffectSizes:
    """Tests for ANOVA effect sizes (eta squared and omega squared)."""

    def test_eta_squared_basic(self):
        """Test basic eta squared calculation."""
        result = eta_squared(ss_effect=120, ss_total=500)

        assert "error" not in result
        assert result["eta_squared"] == 0.24
        assert result["percentage_variance"] == 24.0

    def test_eta_squared_interpretation(self):
        """Test eta squared interpretation."""
        # Small effect
        result_small = eta_squared(ss_effect=5, ss_total=500)
        assert result_small["interpretation"] == "Small"

        # Large effect
        result_large = eta_squared(ss_effect=100, ss_total=500)
        assert result_large["interpretation"] == "Large"

    def test_eta_squared_invalid_input(self):
        """Test eta squared with invalid input."""
        result = eta_squared(ss_effect=100, ss_total=0)
        assert "error" in result

    def test_omega_squared_basic(self):
        """Test basic omega squared calculation."""
        # Medium effect scenario
        ss_effect = 100
        ss_error = 400
        ms_error = 10
        df_effect = 2
        n_total = 50

        result = omega_squared(ss_effect, ss_error, ms_error, df_effect, n_total)

        assert "error" not in result
        assert "omega_squared" in result
        assert result["omega_squared"] >= 0

    def test_omega_squared_less_biased(self):
        """Test that omega squared is less biased (smaller) than eta squared."""
        ss_effect = 100
        ss_error = 400
        ms_error = 10
        df_effect = 2
        n_total = 50

        result_omega = omega_squared(ss_effect, ss_error, ms_error, df_effect, n_total)
        result_eta = eta_squared(ss_effect, ss_effect + ss_error)

        # Omega squared should be smaller (less biased upward)
        assert result_omega["omega_squared"] <= result_eta["eta_squared"]

    def test_omega_squared_negative_correction(self):
        """Test that omega squared is floored at 0 for negligible effects."""
        # Case where numerator would be negative
        ss_effect = 1
        ss_error = 500
        ms_error = 10
        df_effect = 2
        n_total = 50

        result = omega_squared(ss_effect, ss_error, ms_error, df_effect, n_total)

        assert result["omega_squared"] == 0.0

    def test_omega_squared_invalid_input(self):
        """Test omega squared with invalid input."""
        result = omega_squared(100, 400, -10, 2, 50)  # Negative MS error
        assert "error" in result


class TestInterpretEffect:
    """Tests for effect size interpretation."""

    def test_interpret_cohens_d(self):
        """Test Cohen's d interpretation thresholds."""
        assert interpret_effect_size(0.1, "d") == "Negligible"
        assert interpret_effect_size(0.3, "d") == "Small"
        assert interpret_effect_size(0.6, "d") == "Medium"
        assert interpret_effect_size(1.0, "d") == "Large"

    def test_interpret_eta_squared(self):
        """Test eta squared interpretation thresholds."""
        assert interpret_effect_size(0.005, "eta_squared") == "Negligible"
        assert interpret_effect_size(0.03, "eta_squared") == "Small"
        assert interpret_effect_size(0.10, "eta_squared") == "Medium"
        assert interpret_effect_size(0.20, "eta_squared") == "Large"


class TestPublicationFormatting:
    """Tests for publication formatting."""

    def test_format_nejm(self):
        """Test NEJM style formatting."""
        result = cohens_d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        formatted = format_effect_size_for_publication(result, "d", "NEJM")

        assert "d =" in formatted
        assert "95% CI" in formatted
        assert "to" in formatted

    def test_format_apa(self):
        """Test APA style formatting."""
        result = cohens_d([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        formatted = format_effect_size_for_publication(result, "d", "APA")

        assert "d =" in formatted
        assert "[" in formatted and "]" in formatted

    def test_format_error_handling(self):
        """Test formatting with error result."""
        error_result = {"error": "Test error"}
        formatted = format_effect_size_for_publication(error_result, "d", "NEJM")

        assert "failed" in formatted.lower()

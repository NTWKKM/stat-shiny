"""
Unit tests for OR to RR conversion and FigureLegendGenerator.

Tests for Phase 3 enhancements in diag_test.py and formatting.py.
"""

from utils.diag_test import convert_or_to_rr
from utils.formatting import FigureLegendGenerator


class TestConvertORtoRR:
    """Tests for OR to RR conversion using Zhang-Yu formula."""

    def test_basic_conversion(self):
        """Test basic OR to RR conversion."""
        result = convert_or_to_rr(odds_ratio=2.5, baseline_risk=0.20)

        assert "rr" in result
        assert "error" not in result
        assert result["rr"] < 2.5  # RR should be smaller than OR when outcome is common
        assert result["rr"] > 1  # Should still indicate increased risk

    def test_conversion_with_ci(self):
        """Test conversion with confidence intervals."""
        result = convert_or_to_rr(
            odds_ratio=2.5,
            baseline_risk=0.20,
            or_ci_lower=1.5,
            or_ci_upper=4.0,
        )

        assert "rr_ci_lower" in result
        assert "rr_ci_upper" in result
        assert result["rr_ci_lower"] < result["rr"]
        assert result["rr_ci_upper"] > result["rr"]

    def test_protective_effect(self):
        """Test conversion for protective OR (< 1)."""
        result = convert_or_to_rr(odds_ratio=0.5, baseline_risk=0.15)

        assert result["rr"] < 1
        assert "lower risk" in result["interpretation"].lower()

    def test_no_effect(self):
        """Test conversion when OR = 1."""
        result = convert_or_to_rr(odds_ratio=1.0, baseline_risk=0.10)

        assert result["rr"] == 1.0
        assert "no difference" in result["interpretation"].lower()

    def test_low_baseline_risk(self):
        """Test that OR ≈ RR when baseline risk is low."""
        result = convert_or_to_rr(odds_ratio=2.0, baseline_risk=0.01)

        # When baseline risk is very low, OR ≈ RR
        assert abs(result["rr"] - 2.0) < 0.05
        assert result["warnings"] is None  # No warning for low baseline

    def test_high_baseline_risk_warning(self):
        """Test warning when baseline risk is high."""
        result = convert_or_to_rr(odds_ratio=3.0, baseline_risk=0.30)

        assert result["warnings"] is not None
        assert any("overestimates" in w.lower() for w in result["warnings"])

    def test_invalid_or(self):
        """Test error handling for invalid OR."""
        result = convert_or_to_rr(odds_ratio=-1, baseline_risk=0.20)

        assert "error" in result

    def test_invalid_baseline_risk(self):
        """Test error handling for invalid baseline risk."""
        result = convert_or_to_rr(odds_ratio=2.0, baseline_risk=1.5)

        assert "error" in result


class TestFigureLegendGenerator:
    """Tests for FigureLegendGenerator class."""

    def test_kaplan_meier_basic(self):
        """Test basic Kaplan-Meier legend."""
        legend = FigureLegendGenerator.kaplan_meier(
            outcome_var="death",
            time_var="months",
        )

        assert "Kaplan-Meier" in legend
        assert "death" in legend
        assert "censored" in legend.lower()

    def test_kaplan_meier_with_groups(self):
        """Test KM legend with stratification."""
        legend = FigureLegendGenerator.kaplan_meier(
            outcome_var="recurrence",
            time_var="years",
            group_var="treatment arm",
            n_subjects=500,
            n_events=120,
            log_rank_p=0.003,
        )

        assert "stratified by treatment arm" in legend
        assert "N = 500" in legend
        assert "120 events" in legend

    def test_forest_plot(self):
        """Test Forest plot legend."""
        legend = FigureLegendGenerator.forest_plot(
            analysis_type="subgroup analysis",
            effect_measure="hazard ratio",
            n_studies=12,
            heterogeneity_i2=45.3,
        )

        assert "Forest plot" in legend
        assert "hazard ratio" in legend
        assert "12 studies" in legend
        assert "I² = 45.3%" in legend

    def test_roc_curve(self):
        """Test ROC curve legend."""
        legend = FigureLegendGenerator.roc_curve(
            test_name="Troponin I",
            outcome_name="acute MI",
            auc=0.89,
            auc_ci=(0.84, 0.94),
        )

        assert "ROC" in legend
        assert "Troponin I" in legend
        assert "AUC" in legend
        assert "0.89" in legend

    def test_correlation_matrix(self):
        """Test correlation matrix legend."""
        legend = FigureLegendGenerator.correlation_matrix(
            n_variables=8,
            n_observations=250,
            method="Spearman",
            correction_method="Bonferroni",
        )

        assert "Correlation matrix" in legend
        assert "Spearman" in legend
        assert "8 variables" in legend
        assert "Bonferroni" in legend

    def test_custom_legend(self):
        """Test custom legend generator."""
        legend = FigureLegendGenerator.custom(
            plot_type="Scatter plot",
            description="Association between age and blood pressure",
            variables=["Age", "Systolic BP"],
            sample_size=1000,
            statistical_test="Pearson correlation",
            p_value=0.001,
        )

        assert "Scatter plot" in legend
        assert "age and blood pressure" in legend
        assert "N = 1000" in legend

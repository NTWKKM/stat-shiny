"""
🔗 Integration Tests for Advanced Features
File: tests/integration/test_advanced_features.py

Tests the integration of advanced statistical modules:
1. Negative Binomial Regression
2. Causal Inference (PSM & IPW)
3. Mediation Analysis
"""

import numpy as np
import pandas as pd
import pytest

from utils.mediation_lib import analyze_mediation
from utils.poisson_lib import analyze_poisson_outcome
from utils.psm_lib import calculate_ipw, calculate_ps, check_balance
from utils.stratified_lib import mantel_haenszel

# Mark as integration test
pytestmark = pytest.mark.integration


class TestAdvancedFeatures:
    """
    Integration tests for Advanced Inference and Causal modules
    """

    @pytest.fixture
    def causal_data(self):
        """Create realistic dataset for causal inference"""
        np.random.seed(123)
        n = 200
        # Confounders
        age = np.random.normal(60, 10, n)
        severity = np.random.normal(5, 2, n)

        # Treatment assignment (prob depends on covariates)
        z = -3 + 0.05 * age + 0.2 * severity
        prob = 1 / (1 + np.exp(-z))
        treated = np.random.binomial(1, prob)

        # Outcome (Effect of treatment + confounders)
        # Treatment effect = -0.5 (reduction in outcome)
        outcome_linear = (
            2
            + 0.02 * age
            + 0.1 * severity
            - 0.5 * treated
            + np.random.normal(0, 0.5, n)
        )

        return pd.DataFrame(
            {
                "Age": age,
                "Severity": severity,
                "Treated": treated,
                "Outcome": outcome_linear,
            }
        )

    @pytest.fixture
    def nb_data(self):
        """Create overdispersed count data for Negative Binomial"""
        np.random.seed(456)
        n = 200
        x = np.random.normal(0, 1, n)
        # True model: log(mu) = 1 + 0.5*x
        mu = np.exp(1 + 0.5 * x)
        # Gamma noise for overdispersion (mean=1, var=1/alpha)
        # alpha=0.5 -> shape=2, scale=0.5
        gamma = np.random.gamma(2, 0.5, n)
        y = np.random.poisson(mu * gamma)
        return pd.DataFrame({"x": x, "y": y})

    def test_negative_binomial_flow(self, nb_data):
        """Test full Negative Binomial analysis flow"""
        html_rep, irr_res, airr_res, int_res = analyze_poisson_outcome(
            outcome_name="y", df=nb_data, model_type="negative_binomial"
        )

        assert html_rep is not None
        assert irr_res is not None
        assert "x" in irr_res
        # Check that IRR is reasonable (should be around exp(0.5) ≈ 1.65)
        est = irr_res["x"]["irr"]
        assert 1.2 < est < 2.2
        assert "Negative Binomial" in html_rep or "NB" in str(
            irr_res
        )  # or implicit check

    def test_causal_pipeline_psm_ipw(self, causal_data):
        """Test Propensity Score Calculation -> Balance -> IPW flow"""
        df = causal_data.copy()

        # 1. Calculate PS
        ps, _ = calculate_ps(df, "Treated", ["Age", "Severity"])
        df["ps"] = ps
        assert ps.notna().all()

        # 2. Check Balance (Unweighted)
        bal_pre = check_balance(df, "Treated", ["Age", "Severity"])
        assert "SMD" in bal_pre.columns

        # 3. Calculate IPW
        ipw_res = calculate_ipw(df, "Treated", "Outcome", "ps")
        assert "ATE" in ipw_res
        assert ipw_res["ATE"] is not None

        # Check ATE is somewhat close to true effect -0.5
        # (Mock data might be noisy, but sign should be negative)
        assert ipw_res["ATE"] < 0

    def test_mediation_analysis_flow(self, causal_data):
        """Test Mediation Analysis integration"""
        # outcome: Outcome, treatment: Treated, mediator: Severity (mock)
        # Assuming Severity mediates Age -> Outcome? No, let's just test the function call structure
        # Let's say Age -> Severity -> Outcome

        res = analyze_mediation(
            data=causal_data, outcome="Outcome", treatment="Age", mediator="Severity"
        )

        assert res is not None
        assert "total_effect" in res
        assert "indirect_effect" in res
        # Check consistency
        total = res["total_effect"]
        direct = res["direct_effect"]
        indirect = res["indirect_effect"]
        assert abs(total - (direct + indirect)) < 0.001

    def test_stratified_analysis_flow(self):
        """Test Stratified Analysis (MH) flow"""
        df = pd.DataFrame(
            {
                "T": [1, 1, 0, 0, 1, 1, 0, 0] * 5,
                "Y": [1, 0, 1, 0, 1, 0, 0, 1] * 5,
                "S": ["A", "A", "A", "A", "B", "B", "B", "B"] * 5,
            }
        )

        res = mantel_haenszel(df, "Y", "T", "S")
        assert "MH_OR" in res
        assert "Strata_Results" in res
        assert not res["Strata_Results"].empty

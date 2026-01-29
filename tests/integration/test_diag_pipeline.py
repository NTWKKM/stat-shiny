"""
🔗 Integration Tests for Diagnostic & Agreement Pipeline
Tests Chi-square, ROC, and Reliability analysis flow.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.diag_test import calculate_chi2

pytestmark = pytest.mark.integration


class TestDiagnosticPipeline:
    @pytest.fixture
    def diagnostic_data(self):
        """Create dataset for diagnostic tests"""
        np.random.seed(123)
        n = 150
        df = pd.DataFrame(
            {
                "disease_status": np.random.choice([0, 1], n, p=[0.7, 0.3]),
                "risk_group": np.random.choice(["Low", "High"], n),
                "test_score": np.random.normal(0, 1, n),
                # Raters for agreement (binary)
                "doctor_A": np.random.choice([0, 1], n),
                "doctor_B": np.random.choice([0, 1], n),
                # Raters for ICC (continuous ratings)
                "rater_1": np.random.uniform(1, 5, n),
                "rater_2": np.random.uniform(1, 5, n),
                "rater_3": np.random.uniform(1, 5, n),
            }
        )
        # Add correlation for ROC
        df.loc[df["disease_status"] == 1, "test_score"] += 1.5
        # Add some correlation between raters for ICC
        df["rater_2"] = df["rater_1"] + np.random.normal(0, 0.5, n)
        df["rater_3"] = df["rater_1"] + np.random.normal(0, 0.7, n)
        # Clip to valid range
        df["rater_2"] = df["rater_2"].clip(1, 5)
        df["rater_3"] = df["rater_3"].clip(1, 5)
        return df

    def test_chi_square_flow(self, diagnostic_data):
        """🔄 Test Contingency Table & Chi-square"""
        df = diagnostic_data

        _display_tab, stats_df, _msg, risk_df, _ = calculate_chi2(
            df, "risk_group", "disease_status"
        )

        assert stats_df is not None
        assert not stats_df.empty
        assert (
            "Chi-Square" in stats_df["Value"].iloc[0]
            or "Fisher" in stats_df["Value"].iloc[0]
        )

        # Check Risk Metrics (OR/RR)
        if risk_df is not None:
            assert (
                "Odds Ratio (OR)" in risk_df["Metric"].values
                or "Risk Ratio (RR)" in risk_df["Metric"].values
            )

    def test_roc_analysis_flow(self, diagnostic_data):
        """🔄 Test ROC Curve & AUC calculation"""
        import importlib

        import utils.diag_test

        # Reload to ensure no pollution from unit test mocks
        importlib.reload(utils.diag_test)

        df = diagnostic_data

        # Use string for pos_label as per UI input
        stats, err, _fig, _coords = utils.diag_test.analyze_roc(
            df, "disease_status", "test_score", pos_label_user="1"
        )

        assert err is None
        assert stats is not None
        assert "AUC" in stats

        # AUC should be decent due to added correlation
        auc = float(stats["AUC"])
        assert 0.5 < auc <= 1.0

    def test_agreement_flow(self, diagnostic_data):
        """
        Verify Cohen's Kappa agreement analysis yields a metrics table containing "Cohen's Kappa" and no error.
        
        This test runs AgreementAnalysis.cohens_kappa on the provided fixture and asserts that the returned error is None and the resulting metrics DataFrame includes a "Cohen's Kappa" entry.
        """
        df = diagnostic_data
        from utils.agreement_lib import AgreementAnalysis

        stats_df, err, _conf_matrix, _ = AgreementAnalysis.cohens_kappa(
            df, "doctor_A", "doctor_B"
        )

        assert err is None
        assert stats_df is not None
        assert "Cohen's Kappa" in stats_df["Metric"].values

    def test_icc_flow(self, diagnostic_data):
        """🔄 Test Intraclass Correlation Coefficient (ICC)"""
        df = diagnostic_data
        from utils.agreement_lib import AgreementAnalysis

        # Select continuous rating columns for ICC
        rater_cols = ["rater_1", "rater_2", "rater_3"]

        stats_df, err, *_ = AgreementAnalysis.icc(df, rater_cols)

        assert err is None
        assert stats_df is not None
        assert not stats_df.empty
        # Check that ICC statistic is present in results
        assert "ICC" in stats_df.columns
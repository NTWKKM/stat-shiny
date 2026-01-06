"""
🔗 Integration Tests for Diagnostic & Agreement Pipeline
Tests Chi-square, ROC, and Reliability analysis flow.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from diag_test import (
    calculate_chi2,
    analyze_roc,
    calculate_kappa,
    calculate_icc
)

pytestmark = pytest.mark.integration

class TestDiagnosticPipeline:

    @pytest.fixture
    def diagnostic_data(self):
        """Create dataset for diagnostic tests"""
        np.random.seed(123)
        n = 150
        df = pd.DataFrame({
            'disease_status': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'risk_group': np.random.choice(['Low', 'High'], n),
            'test_score': np.random.normal(0, 1, n),
            # Raters for agreement
            'doctor_A': np.random.choice([0, 1], n),
            'doctor_B': np.random.choice([0, 1], n)
        })
        # Add correlation for ROC
        df.loc[df['disease_status'] == 1, 'test_score'] += 1.5
        return df

    def test_chi_square_flow(self, diagnostic_data):
        """🔄 Test Contingency Table & Chi-square"""
        df = diagnostic_data
        
        display_tab, stats_df, msg, risk_df = calculate_chi2(
            df, 'risk_group', 'disease_status'
        )
        
        assert stats_df is not None
        assert not stats_df.empty
        assert 'Chi-Square' in stats_df['Value'].iloc[0] or 'Fisher' in stats_df['Value'].iloc[0]
        
        # Check Risk Metrics (OR/RR)
        if risk_df is not None:
            assert 'Odds Ratio (OR)' in risk_df['Metric'].values or 'Risk Ratio (RR)' in risk_df['Metric'].values

    def test_roc_analysis_flow(self, diagnostic_data):
        """🔄 Test ROC Curve & AUC calculation"""
        df = diagnostic_data
        
        # Use string for pos_label as per UI input
        stats, err, fig, coords = analyze_roc(
            df, 'disease_status', 'test_score', pos_label_user='1'
        )
        
        assert err is None
        assert stats is not None
        assert 'AUC' in stats
        
        # AUC should be decent due to added correlation
        auc = float(stats['AUC'])
        assert 0.5 < auc <= 1.0

    def test_agreement_flow(self, diagnostic_data):
        """🔄 Test Cohen's Kappa Agreement"""
        df = diagnostic_data
        
        stats_df, err, conf_matrix = calculate_kappa(
            df, 'doctor_A', 'doctor_B'
        )
        
        assert err is None
        assert stats_df is not None
        assert "Cohen's Kappa" in stats_df['Statistic'].values

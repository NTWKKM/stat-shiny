"""
🧪 Unit Tests for Clinical Meta-Analysis Module

Tests:
- compute_binary_effect_sizes (OR, RR, RD)
- compute_continuous_effect_sizes (MD, SMD / Hedges' g)
- run_meta_analysis (Fixed & Random Effects, HKSJ, Prediction Interval, Heterogeneity)
- run_publication_bias_tests (Egger's test, Begg's test)
- create_meta_forest_plot
- create_contour_enhanced_funnel_plot
"""

import numpy as np
import pandas as pd
import pytest
from utils import meta_analysis_lib


@pytest.mark.unit
def test_compute_binary_effect_sizes():
    """Test 2x2 contingency table effect size calculations (OR, RR, RD)."""
    df = pd.DataFrame(
        {
            "study": ["Study 1", "Study 2"],
            "e_t": [10, 25],
            "n_t": [100, 200],
            "e_c": [20, 50],
            "n_c": [100, 200],
        }
    )

    # Odds Ratio
    res_or = meta_analysis_lib.compute_binary_effect_sizes(
        df,
        events_t_col="e_t",
        n_t_col="n_t",
        events_c_col="e_c",
        n_c_col="n_c",
        study_col="study",
        effect_measure="OR",
    )

    assert len(res_or) == 2
    assert "log_effect" in res_or.columns
    assert "se" in res_or.columns
    assert res_or.iloc[0]["effect_size"] < 1.0  # Treatment has lower event rate

    # Risk Ratio
    res_rr = meta_analysis_lib.compute_binary_effect_sizes(
        df,
        events_t_col="e_t",
        n_t_col="n_t",
        events_c_col="e_c",
        n_c_col="n_c",
        study_col="study",
        effect_measure="RR",
    )
    assert res_rr.iloc[0]["effect_size"] < 1.0


@pytest.mark.unit
def test_compute_continuous_effect_sizes():
    """Test continuous endpoint effect size calculations (MD, SMD / Hedges' g)."""
    df = pd.DataFrame(
        {
            "trial": ["Trial A", "Trial B"],
            "m_t": [120, 115],
            "s_t": [15, 12],
            "n_t": [50, 60],
            "m_c": [135, 130],
            "s_c": [14, 13],
            "n_c": [50, 60],
        }
    )

    res_smd = meta_analysis_lib.compute_continuous_effect_sizes(
        df,
        mean_t_col="m_t",
        sd_t_col="s_t",
        n_t_col="n_t",
        mean_c_col="m_c",
        sd_c_col="s_c",
        n_c_col="n_c",
        study_col="trial",
        effect_measure="SMD",
    )

    assert len(res_smd) == 2
    assert res_smd.iloc[0]["effect_size"] < 0  # Blood pressure reduction in treatment


@pytest.mark.unit
def test_run_meta_analysis_pooling_and_heterogeneity():
    """Test fixed and random effects meta-analysis on BCG vaccine trial benchmark data."""
    df = pd.DataFrame(
        {
            "study": ["Aronson", "Ferguson", "Rosenthal", "Hart", "Frimodt"],
            "e_t": [4, 6, 11, 29, 179],
            "n_t": [123, 306, 231, 247, 1697],
            "e_c": [11, 29, 29, 45, 141],
            "n_c": [139, 303, 220, 234, 1606],
        }
    )

    eff_df = meta_analysis_lib.compute_binary_effect_sizes(
        df,
        events_t_col="e_t",
        n_t_col="n_t",
        events_c_col="e_c",
        n_c_col="n_c",
        study_col="study",
        effect_measure="OR",
    )

    res = meta_analysis_lib.run_meta_analysis(eff_df, method_re="dl", use_hksj=True)

    assert "fixed_effect" in res
    assert "random_effect" in res
    assert "heterogeneity" in res
    assert res["k"] == 5

    fe = res["fixed_effect"]
    re_eff = res["random_effect"]
    het = res["heterogeneity"]

    assert fe["effect_disp"] < 1.0  # Protective effect of BCG vaccine
    assert re_eff["effect_disp"] < 1.0
    assert "I2" in het
    assert het["I2"] >= 0.0
    assert "tau2" in het
    assert het["tau2"] >= 0.0

    # Check 95% Prediction Interval exists
    pi = re_eff.get("prediction_interval", {})
    assert "pi_lower" in pi
    assert "pi_upper" in pi


@pytest.mark.unit
def test_publication_bias_tests():
    """Test Egger and Begg publication bias tests."""
    df = pd.DataFrame(
        {
            "study": [f"Study {i}" for i in range(1, 11)],
            "log_effect": [
                -0.5,
                -0.4,
                -0.6,
                -0.3,
                -0.8,
                -0.2,
                -0.45,
                -0.55,
                -0.35,
                -0.65,
            ],
            "se": [0.10, 0.15, 0.12, 0.20, 0.25, 0.30, 0.18, 0.22, 0.14, 0.16],
        }
    )

    pb = meta_analysis_lib.run_publication_bias_tests(df)

    assert "egger" in pb
    assert "begg" in pb
    assert "interpretation" in pb
    assert "intercept" in pb["egger"]
    assert "p_value" in pb["egger"]
    assert "kendall_tau" in pb["begg"]


@pytest.mark.unit
def test_create_meta_forest_and_funnel_plots():
    """Test Plotly visualizers for Forest and Contour-Enhanced Funnel plots."""
    df = pd.DataFrame(
        {
            "study": ["Study A", "Study B", "Study C"],
            "log_effect": [-0.4, -0.6, -0.3],
            "se": [0.15, 0.18, 0.12],
            "effect_size": [np.exp(-0.4), np.exp(-0.6), np.exp(-0.3)],
            "ci_lower": [
                np.exp(-0.4 - 1.96 * 0.15),
                np.exp(-0.6 - 1.96 * 0.18),
                np.exp(-0.3 - 1.96 * 0.12),
            ],
            "ci_upper": [
                np.exp(-0.4 + 1.96 * 0.15),
                np.exp(-0.6 + 1.96 * 0.18),
                np.exp(-0.3 + 1.96 * 0.12),
            ],
            "is_ratio": [True, True, True],
        }
    )

    meta_res = meta_analysis_lib.run_meta_analysis(df, method_re="dl")

    fig_forest = meta_analysis_lib.create_meta_forest_plot(meta_res)
    assert fig_forest is not None
    assert len(fig_forest.data) >= 3  # Studies + FE diamond + RE diamond

    fig_funnel = meta_analysis_lib.create_contour_enhanced_funnel_plot(meta_res)
    assert fig_funnel is not None
    assert len(fig_funnel.data) >= 4  # Contours + points

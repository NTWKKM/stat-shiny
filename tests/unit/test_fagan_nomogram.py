"""
🧪 Unit Tests for Fagan's Nomogram & Likelihood Ratio Module

Tests:
- calculate_post_test_probability (Bayes' Theorem log-odds scale)
- calculate_multilevel_likelihood_ratios (Interval LRs)
- create_fagan_nomogram_plot (Plotly 3-axis canvas)
"""

import math

import pandas as pd
import pytest

from utils import fagan_nomogram_lib


@pytest.mark.unit
def test_calculate_post_test_probability():
    """Test Bayes conversion from pre-test to post-test probability."""
    # Pre-test = 20% (0.2), LR+ = 10.0
    # Pre-odds = 0.2 / 0.8 = 0.25
    # Post-odds = 0.25 * 10 = 2.5
    # Post-prob = 2.5 / 3.5 = 0.714285 (71.4%)
    res = fagan_nomogram_lib.calculate_post_test_probability(0.20, 10.0)

    assert pytest.approx(res["post_test_prob"], 0.001) == 2.5 / 3.5
    assert pytest.approx(res["post_test_prob_pct"], 0.1) == 71.4
    assert "High Risk" in res["zone"] or "Intermediate" in res["zone"]
    assert "Strong evidence" in res["impact"]

    # Pre-test = 50% (0.5), LR- = 0.10
    # Pre-odds = 1.0, Post-odds = 0.10, Post-prob = 0.10 / 1.10 = 0.0909 (9.1%)
    res_neg = fagan_nomogram_lib.calculate_post_test_probability(0.50, 0.10)
    assert pytest.approx(res_neg["post_test_prob"], 0.001) == 0.1 / 1.1
    assert "Low Risk" in res_neg["zone"] or "Rule-Out" in res_neg["zone"]


@pytest.mark.unit
def test_calculate_multilevel_likelihood_ratios():
    """Test calculation of interval likelihood ratios for biomarker tiers."""
    df = pd.DataFrame(
        {
            "disease": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "troponin": [5, 12, 25, 45, 80, 150, 2, 4, 8, 10, 18, 30],
        }
    )

    # Cutoffs at 14 and 50 -> Tiers: <14, [14, 50), >=50
    res_df = fagan_nomogram_lib.calculate_multilevel_likelihood_ratios(
        df, outcome_col="disease", score_col="troponin", cutoffs=[14, 50], pos_label=1
    )

    assert len(res_df) == 3
    assert "Tier / Interval" in res_df.columns
    assert "Interval LR" in res_df.columns
    assert "95% CI Lower" in res_df.columns
    assert "95% CI Upper" in res_df.columns

    # High tier (>= 50) should have high LR (only diseased patients in this sample)
    high_tier_lr = res_df.iloc[-1]["Interval LR"]
    assert high_tier_lr > 1.0

    # Test string pos_label compatibility
    res_df_str = fagan_nomogram_lib.calculate_multilevel_likelihood_ratios(
        df,
        outcome_col="disease",
        score_col="troponin",
        cutoffs=[14, 50],
        pos_label="1",
    )
    assert len(res_df_str) == 3
    assert res_df_str.iloc[-1]["Interval LR"] == high_tier_lr

    # Test categorical text labels ("Yes" / "No")
    df_text = df.copy()
    df_text["disease"] = df_text["disease"].map({1: "Yes", 0: "No"})
    res_df_text = fagan_nomogram_lib.calculate_multilevel_likelihood_ratios(
        df_text,
        outcome_col="disease",
        score_col="troponin",
        cutoffs=[14, 50],
        pos_label="Yes",
    )
    assert len(res_df_text) == 3
    assert res_df_text.iloc[-1]["Interval LR"] == high_tier_lr


@pytest.mark.unit
def test_create_fagan_nomogram_plot():
    """Test 3-axis Plotly Fagan Nomogram generation and geometric collinearity."""
    pre_prob = 0.25
    lr_pos = 8.0
    lr_neg = 0.15

    fig = fagan_nomogram_lib.create_fagan_nomogram_plot(
        pre_test_prob=pre_prob,
        lr_pos=lr_pos,
        lr_neg=lr_neg,
        test_name="Cardiac Troponin I",
        multilevel_lrs=[
            {"name": "Low Tier (<14)", "lr": 0.12},
            {"name": "Intermediate (14-50)", "lr": 1.8},
            {"name": "High Tier (>50)", "lr": 14.5},
        ],
    )

    assert fig is not None
    # Traces for Positive, Negative, and 3 tiers
    assert len(fig.data) >= 5
    trace_names = [t.name for t in fig.data if hasattr(t, "name")]
    assert any("Test Positive" in str(name) for name in trace_names)
    assert any("Test Negative" in str(name) for name in trace_names)
    assert any("High Tier" in str(name) for name in trace_names)

    # Verify exact collinearity on positive trace
    pos_trace = [t for t in fig.data if "Test Positive" in str(getattr(t, "name", ""))][
        0
    ]
    xs = list(pos_trace.x)
    ys = list(pos_trace.y)
    assert xs == [0, 0.5, 1.0]

    # Midpoint of y0 and y1 must equal y_mid and equal 0.5 * ln(LR)
    y_pre, y_mid, y_post = ys[0], ys[1], ys[2]
    assert pytest.approx(y_mid, 1e-6) == (y_pre + y_post) / 2.0
    assert pytest.approx(y_mid, 1e-6) == 0.5 * math.log(lr_pos)

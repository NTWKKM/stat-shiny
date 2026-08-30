"""
🧪 Unit Tests for Modern Calibration (ICI & TRIPOD+AI Standards)

Tests:
- calculate_ici (Integrated Calibration Index, E50, E90, Emax)
- calculate_calibration_slope & calibration_intercept
- create_calibration_plot with LOESS curve
- get_calibration_report & format_calibration_html
"""

import numpy as np
import pytest

from utils import calibration_lib


@pytest.mark.unit
def test_calculate_ici_perfect_calibration():
    """Test ICI on synthetic data with near-ideal calibration."""
    rng = np.random.default_rng(42)
    n = 200
    y_pred = rng.uniform(0.05, 0.95, size=n)
    # Generate true labels according to true probabilities
    y_true = (rng.uniform(0, 1, size=n) < y_pred).astype(int)

    res = calibration_lib.calculate_ici(y_true, y_pred, span=0.75, n_bootstrap=20)

    assert "ici" in res
    assert not np.isnan(res["ici"])
    assert res["ici"] < 0.15  # Reasonably close to 0
    assert "e50" in res
    assert "e90" in res
    assert "emax" in res
    assert res["e50"] <= res["e90"] <= res["emax"]
    assert "interpretation" in res
    assert "loess_x" in res
    assert len(res["loess_x"]) > 0


@pytest.mark.unit
def test_calculate_ici_miscalibrated():
    """Test ICI on deliberately miscalibrated overconfident predictions."""
    rng = np.random.default_rng(123)
    n = 200
    # True event rate is fixed at 10%
    y_true = (rng.uniform(0, 1, size=n) < 0.10).astype(int)
    # Model mistakenly predicts very high probabilities (60-90%)
    y_pred = rng.uniform(0.60, 0.90, size=n)

    res = calibration_lib.calculate_ici(y_true, y_pred, span=0.75, n_bootstrap=10)

    assert res["ici"] > 0.40  # Massive calibration error
    assert "Poor" in res["interpretation"] or "Recalibration" in res["interpretation"]


@pytest.mark.unit
def test_calibration_slope_and_intercept():
    """Test calibration slope and calibration-in-the-large."""
    rng = np.random.default_rng(99)
    n = 150
    y_pred = rng.uniform(0.1, 0.9, size=n)
    y_true = (rng.uniform(0, 1, size=n) < y_pred).astype(int)

    res = calibration_lib.calculate_calibration_slope(y_true, y_pred)

    assert "calibration_slope" in res
    assert "calibration_intercept" in res
    assert "slope_interpretation" in res
    assert "intercept_interpretation" in res
    assert not np.isnan(res["calibration_slope"])


@pytest.mark.unit
def test_create_calibration_plot():
    """Test generation of publication Plotly calibration figure."""
    rng = np.random.default_rng(42)
    n = 100
    y_pred = rng.uniform(0.05, 0.95, size=n)
    y_true = (rng.uniform(0, 1, size=n) < y_pred).astype(int)

    fig = calibration_lib.create_calibration_plot(
        y_true, y_pred, n_bins=5, title="Validation Calibration Plot", show_loess=True
    )

    assert fig is not None
    # Check traces: Ideal line, LOESS line, Binned points, Histogram
    trace_names = [t.name for t in fig.data if hasattr(t, "name")]
    assert any("Ideal" in str(name) for name in trace_names)
    assert any("LOESS" in str(name) for name in trace_names)
    assert any("Binned" in str(name) for name in trace_names)


@pytest.mark.unit
def test_calibration_report_and_html():
    """Test complete calibration report dictionary and HTML formatting."""
    rng = np.random.default_rng(42)
    n = 100
    y_pred = rng.uniform(0.1, 0.9, size=n)
    y_true = (rng.uniform(0, 1, size=n) < y_pred).astype(int)

    report = calibration_lib.get_calibration_report(y_true, y_pred)

    assert "c_statistic" in report
    assert "ici" in report
    assert "brier" in report
    assert "calibration" in report
    assert "hosmer_lemeshow" in report

    html_table = calibration_lib.format_calibration_html(report)
    assert "TRIPOD+AI" in html_table
    assert "Integrated Calibration Index" in html_table
    assert "Calibration Slope" in html_table
    assert "Brier Score" in html_table

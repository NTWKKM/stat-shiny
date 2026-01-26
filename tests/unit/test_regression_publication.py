import numpy as np
import pandas as pd

from utils.logic import analyze_outcome, run_binary_logit


def test_publication_metrics_calculation():
    """Verify that deep diagnostics are calculated and present in results."""
    # Generate balanced synthetic data
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    # Probability with some noise
    prob = 1 / (1 + np.exp(-(0.5 * x1 + 0.8 * x2)))
    y = (prob > np.random.uniform(0, 1, n)).astype(int)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    # Run logit
    params, conf, pvals, status, metrics = run_binary_logit(df["y"], df[["x1", "x2"]])

    assert status == "OK"
    assert "aic" in metrics and not np.isnan(metrics["aic"])
    assert "bic" in metrics and not np.isnan(metrics["bic"])
    assert "auc" in metrics and not np.isnan(metrics["auc"])
    assert "hl_pvalue" in metrics and not np.isnan(metrics["hl_pvalue"])

    # AUC should be reasonable (better than random choice 0.5)
    assert metrics["auc"] > 0.5


def test_publication_report_html():
    """Verify that the HTML report contains the new diagnostic sections."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = (x1 + x2 > 0).astype(int)
    df = pd.DataFrame({"outcome": y, "x1": x1, "x2": x2})

    html_table, _, _, _ = analyze_outcome(
        "outcome", df, adv_stats={"stats.vif_enable": True}
    )

    assert "Model Diagnostics (Publication Grade)" in html_table
    assert "Discrimination (C-stat)" in html_table
    assert "Calibration (Hosmer-Lemeshow)" in html_table
    assert "AIC / BIC" in html_table
    assert "This multivariable model adjusted for" in html_table
    assert "No evidence of multicollinearity" in html_table

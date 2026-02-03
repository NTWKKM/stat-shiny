"""
📊 Calibration Library for Publication-Quality Model Validation

Provides calibration metrics and plots essential for NEJM/Lancet standards:
- Calibration plots (observed vs predicted)
- Brier score
- Calibration slope/intercept
- Net Benefit / Decision Curve Analysis

References:
    Steyerberg EW. Clinical Prediction Models. 2nd ed. Springer; 2019.
    Vickers AJ, Elkin EB. Med Decis Making. 2006;26(6):565-574.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()


# =============================================================================
# CALIBRATION METRICS
# =============================================================================


def calculate_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate Brier Score and its decomposition.

    Brier Score ranges from 0 (perfect) to 1 (worst).
    Brier < 0.25 is generally acceptable for clinical prediction.

    Args:
        y_true: Binary outcome (0/1)
        y_pred: Predicted probabilities

    Returns:
        Dictionary with Brier score and interpretation
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Remove NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        brier = brier_score_loss(y_true, y_pred)

        # Reference Brier (using prevalence as prediction)
        prevalence = np.mean(y_true)
        brier_ref = prevalence * (1 - prevalence)

        # Scaled Brier (0 = perfect, 1 = no better than reference)
        brier_scaled = 1 - (brier / brier_ref) if brier_ref > 0 else np.nan

        interpretation = (
            "Excellent"
            if brier < 0.1
            else "Good"
            if brier < 0.2
            else "Acceptable"
            if brier < 0.25
            else "Poor"
        )

        return {
            "brier_score": brier,
            "brier_scaled": brier_scaled,
            "brier_reference": brier_ref,
            "interpretation": interpretation,
            "n": len(y_true),
        }
    except Exception as e:
        logger.warning("Brier score calculation failed: %s", e)
        return {"brier_score": np.nan, "error": str(e)}


def calculate_calibration_slope(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate calibration slope and intercept via logistic regression.

    Perfect calibration: slope = 1, intercept = 0

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities

    Returns:
        Dictionary with slope, intercept, and CIs
    """
    try:
        import statsmodels.api as sm

        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Remove NaN and clip predictions
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = np.clip(y_pred[mask], 1e-6, 1 - 1e-6)

        # Logit transform of predictions
        logit_pred = np.log(y_pred / (1 - y_pred))

        # Fit logistic regression: logit(y) ~ intercept + slope * logit(pred)
        X = sm.add_constant(logit_pred)
        model = sm.Logit(y_true, X)
        result = model.fit(disp=0)

        intercept = result.params[0]
        slope = result.params[1]

        # Handle both DataFrame and numpy array from conf_int()
        ci = result.conf_int()
        if hasattr(ci, "iloc"):
            ci_slope = ci.iloc[1].tolist()
            ci_intercept = ci.iloc[0].tolist()
        else:
            # numpy array case
            ci_slope = [float(ci[1, 0]), float(ci[1, 1])]
            ci_intercept = [float(ci[0, 0]), float(ci[0, 1])]

        # Interpretation
        slope_status = (
            "✅ Well calibrated" if 0.8 <= slope <= 1.2 else "⚠️ Needs recalibration"
        )

        return {
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "slope_ci_lower": ci_slope[0],
            "slope_ci_upper": ci_slope[1],
            "intercept_ci_lower": ci_intercept[0],
            "intercept_ci_upper": ci_intercept[1],
            "slope_interpretation": slope_status,
        }
    except Exception as e:
        logger.warning("Calibration slope calculation failed: %s", e)
        return {"calibration_slope": np.nan, "error": str(e)}


def calculate_c_statistic_with_ci(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05
) -> dict:
    """
    Calculate C-statistic (AUC) with confidence interval.

    Uses DeLong method for variance estimation.

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        alpha: Significance level for CI

    Returns:
        Dictionary with C-statistic and 95% CI
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        auc = roc_auc_score(y_true, y_pred)
        n = len(y_true)

        # Variance using Hanley-McNeil approximation
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)

        q1 = auc / (2 - auc)
        q2 = (2 * auc * auc) / (1 + auc)

        var_auc = (
            auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)
        ) / (n1 * n0)

        se_auc = np.sqrt(var_auc)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lower = max(0, auc - z * se_auc)
        ci_upper = min(1, auc + z * se_auc)

        interpretation = (
            "Excellent"
            if auc >= 0.9
            else "Good"
            if auc >= 0.8
            else "Acceptable"
            if auc >= 0.7
            else "Poor"
        )

        return {
            "c_statistic": auc,
            "se": se_auc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": n,
            "interpretation": interpretation,
        }
    except Exception as e:
        logger.warning("C-statistic calculation failed: %s", e)
        return {"c_statistic": np.nan, "error": str(e)}


# =============================================================================
# CALIBRATION PLOTS
# =============================================================================


def create_calibration_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Plot",
    strategy: str = "quantile",
) -> go.Figure:
    """
    Create a calibration plot with LOWESS smooth and histogram.

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        n_bins: Number of bins for grouping
        title: Plot title
        strategy: 'uniform' or 'quantile' binning

    Returns:
        Plotly Figure object
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Use sklearn's calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy=strategy
        )

        # Create figure
        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                name="Perfect Calibration",
                showlegend=True,
            )
        )

        # Observed vs predicted points with error bars
        fig.add_trace(
            go.Scatter(
                x=mean_predicted_value,
                y=fraction_of_positives,
                mode="lines+markers",
                marker=dict(size=10, color=COLORS.get("primary", "#1E3A5F")),
                line=dict(color=COLORS.get("primary", "#1E3A5F"), width=2),
                name="Observed",
            )
        )

        # Add histogram of predictions at bottom
        fig.add_trace(
            go.Histogram(
                x=y_pred,
                y=np.zeros_like(y_pred) - 0.05,
                nbinsx=20,
                marker=dict(
                    color=COLORS.get("secondary", "#64748B"),
                    opacity=0.3,
                ),
                yaxis="y2",
                name="Distribution",
                showlegend=False,
            )
        )

        # Layout
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5),
            xaxis=dict(
                title="Mean Predicted Probability",
                range=[0, 1],
                tickformat=".1f",
            ),
            yaxis=dict(
                title="Observed Proportion",
                range=[0, 1],
                tickformat=".1f",
            ),
            yaxis2=dict(
                overlaying="y",
                side="right",
                showticklabels=False,
                range=[0, 1],
            ),
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(x=0.02, y=0.98),
            height=450,
        )

        return fig
    except Exception as e:
        logger.exception("Calibration plot creation failed: %s", e)
        # Return empty figure with error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {e}",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        return fig


# =============================================================================
# DECISION CURVE ANALYSIS
# =============================================================================


def calculate_net_benefit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Calculate Net Benefit for Decision Curve Analysis.

    Net Benefit = (TP/n) - (FP/n) * (threshold / (1 - threshold))

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        thresholds: Probability thresholds to evaluate

    Returns:
        DataFrame with threshold, net_benefit_model, net_benefit_all
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.99, 0.01)

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)

    results = []
    prevalence = np.mean(y_true)

    for thresh in thresholds:
        # Model predictions
        pred_positive = y_pred >= thresh
        tp = np.sum((pred_positive) & (y_true == 1))
        fp = np.sum((pred_positive) & (y_true == 0))

        # Net benefit for model
        if thresh < 1:
            nb_model = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        else:
            nb_model = 0

        # Net benefit for "treat all"
        if thresh < 1:
            nb_all = prevalence - (1 - prevalence) * (thresh / (1 - thresh))
        else:
            nb_all = 0

        results.append(
            {
                "threshold": thresh,
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": 0,
            }
        )

    return pd.DataFrame(results)


def create_decision_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Decision Curve Analysis",
) -> go.Figure:
    """
    Create Decision Curve Analysis plot.

    Shows net benefit across risk thresholds compared to
    'treat all' and 'treat none' strategies.

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        title: Plot title

    Returns:
        Plotly Figure object
    """
    nb_df = calculate_net_benefit(y_true, y_pred)

    fig = go.Figure()

    # Model line
    fig.add_trace(
        go.Scatter(
            x=nb_df["threshold"],
            y=nb_df["net_benefit_model"],
            mode="lines",
            line=dict(color=COLORS.get("primary", "#1E3A5F"), width=2),
            name="Model",
        )
    )

    # Treat All line
    fig.add_trace(
        go.Scatter(
            x=nb_df["threshold"],
            y=nb_df["net_benefit_all"],
            mode="lines",
            line=dict(color=COLORS.get("warning", "#F59E0B"), width=1.5, dash="dash"),
            name="Treat All",
        )
    )

    # Treat None line
    fig.add_trace(
        go.Scatter(
            x=nb_df["threshold"],
            y=nb_df["net_benefit_none"],
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            name="Treat None",
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis=dict(
            title="Threshold Probability",
            range=[0, 1],
            tickformat=".1f",
        ),
        yaxis=dict(
            title="Net Benefit",
        ),
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(x=0.7, y=0.98),
        height=400,
    )

    return fig


# =============================================================================
# HOSMER-LEMESHOW TEST (Surfacing from logic.py)
# =============================================================================


def hosmer_lemeshow_test(y_true: np.ndarray, y_pred: np.ndarray, g: int = 10) -> dict:
    """
    Perform Hosmer-Lemeshow goodness-of-fit test.

    H0: Model fits well (predicted = observed)
    p > 0.05 suggests adequate fit

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities
        g: Number of groups (typically 10)

    Returns:
        Dictionary with chi2, p-value, and interpretation
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # Sort by predicted probability and create groups
        order = np.argsort(y_pred)
        y_true_sorted = y_true[order]
        y_pred_sorted = y_pred[order]

        # Split into g groups
        groups = np.array_split(np.arange(len(y_true)), g)

        chi2 = 0.0
        df = g - 2  # Degrees of freedom

        for group_idx in groups:
            n_group = len(group_idx)
            if n_group == 0:
                continue

            obs_events = np.sum(y_true_sorted[group_idx])
            exp_events = np.sum(y_pred_sorted[group_idx])

            # Avoid division by zero
            if exp_events > 0 and exp_events < n_group:
                chi2 += ((obs_events - exp_events) ** 2) / (
                    exp_events * (1 - exp_events / n_group)
                )

        p_value = 1 - stats.chi2.cdf(chi2, df) if df > 0 else np.nan

        interpretation = (
            "✅ Good fit (p ≥ 0.05)" if p_value >= 0.05 else "⚠️ Poor fit (p < 0.05)"
        )

        return {
            "chi2": chi2,
            "df": df,
            "p_value": p_value,
            "interpretation": interpretation,
            "g": g,
        }
    except Exception as e:
        logger.warning("Hosmer-Lemeshow test failed: %s", e)
        return {"chi2": np.nan, "p_value": np.nan, "error": str(e)}


# =============================================================================
# COMPREHENSIVE CALIBRATION REPORT
# =============================================================================


def get_calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Generate comprehensive calibration report for publication.

    Combines all calibration metrics into a single report.

    Args:
        y_true: Binary outcome
        y_pred: Predicted probabilities

    Returns:
        Dictionary with all calibration metrics
    """
    return {
        "c_statistic": calculate_c_statistic_with_ci(y_true, y_pred),
        "brier": calculate_brier_score(y_true, y_pred),
        "calibration": calculate_calibration_slope(y_true, y_pred),
        "hosmer_lemeshow": hosmer_lemeshow_test(y_true, y_pred),
    }


def format_calibration_html(report: dict) -> str:
    """
    Format calibration report as HTML table.

    Args:
        report: Output from get_calibration_report()

    Returns:
        HTML string
    """
    c_stat = report.get("c_statistic", {})
    brier = report.get("brier", {})
    calib = report.get("calibration", {})
    hl = report.get("hosmer_lemeshow", {})

    html = """
    <div class="calibration-report">
        <h4>📊 Model Calibration & Discrimination</h4>
        <table class="table table-sm">
            <thead>
                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
            </thead>
            <tbody>
    """

    # C-statistic
    if "c_statistic" in c_stat:
        c_val = c_stat["c_statistic"]
        c_ci = f"({c_stat.get('ci_lower', 0):.3f}–{c_stat.get('ci_upper', 0):.3f})"
        html += f"""
                <tr>
                    <td><strong>C-statistic (AUC)</strong></td>
                    <td>{c_val:.3f} {c_ci}</td>
                    <td>{c_stat.get("interpretation", "")}</td>
                </tr>
        """

    # Brier score
    if "brier_score" in brier:
        html += f"""
                <tr>
                    <td><strong>Brier Score</strong></td>
                    <td>{brier["brier_score"]:.4f}</td>
                    <td>{brier.get("interpretation", "")}</td>
                </tr>
        """

    # Calibration slope
    if "calibration_slope" in calib:
        slope = calib["calibration_slope"]
        html += f"""
                <tr>
                    <td><strong>Calibration Slope</strong></td>
                    <td>{slope:.3f}</td>
                    <td>{calib.get("slope_interpretation", "")}</td>
                </tr>
        """

    # Hosmer-Lemeshow
    if "p_value" in hl and not np.isnan(hl.get("p_value", np.nan)):
        html += f"""
                <tr>
                    <td><strong>Hosmer-Lemeshow Test</strong></td>
                    <td>χ² = {hl["chi2"]:.2f}, p = {hl["p_value"]:.3f}</td>
                    <td>{hl.get("interpretation", "")}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    """
    return html

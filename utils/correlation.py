from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

from config import CONFIG
from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import prepare_data_for_analysis
from utils.formatting import generate_standard_report

logger = get_logger(__name__)
COLORS = get_color_palette()


def _compute_correlation_ci(
    r: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute confidence interval for correlation coefficient using Fisher's Z transformation.
    """
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)

    eps = 1e-12
    r = np.clip(r, -1 + eps, 1 - eps)

    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + confidence) / 2)

    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    return (r_lower, r_upper)


def _interpret_correlation(r: float) -> str:
    """
    Interpret correlation strength.
    """
    if not np.isfinite(r):
        return "N/A"

    abs_r = abs(r)

    if abs_r >= 0.9:
        strength = "Very Strong"
    elif abs_r >= 0.7:
        strength = "Strong"
    elif abs_r >= 0.5:
        strength = "Moderate"
    elif abs_r >= 0.3:
        strength = "Weak"
    else:
        strength = "Very Weak/Negligible"

    direction = "Positive" if r >= 0 else "Negative"

    return f"{strength} {direction}"


def compute_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
    var_meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame | None, go.Figure | None, dict[str, Any] | None]:
    """
    ENHANCED: Compute correlation matrix with summary statistics.
    Returns (corr_matrix, fig, summary) - all None if < 2 columns.
    """
    if not cols or len(cols) < 2:
        return None, None, None

    try:
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = "pairwise (matrix)"
        missing_codes = missing_cfg.get("user_defined_values", [])

        try:
            data_clean, missing_data_info = prepare_data_for_analysis(
                df,
                required_cols=cols,
                numeric_cols=cols,
                var_meta=var_meta,
                missing_codes=missing_codes,
                handle_missing="pairwise",
            )
            missing_data_info["strategy"] = strategy
            data = data_clean
        except Exception as e:
            logger.error(f"Data preparation for correlation matrix failed: {e}")
            return None, None, None

        if data.empty:
            return None, None, None

        corr_matrix = data.corr(method=method)
        corr_matrix_rounded = corr_matrix.round(3)

        # Calculate p-values
        p_values = pd.DataFrame(index=cols, columns=cols, dtype=float)

        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i == j:
                    p_values.iloc[i, j] = 1.0
                elif i > j:
                    p_values.iloc[i, j] = p_values.iloc[j, i]
                else:
                    data_pair = data[[col_i, col_j]].dropna()
                    if len(data_pair) >= 3:
                        if method == "pearson":
                            _, p = stats.pearsonr(data_pair[col_i], data_pair[col_j])
                        elif method == "spearman":
                            _, p = stats.spearmanr(data_pair[col_i], data_pair[col_j])
                        elif method == "kendall":
                            _, p = stats.kendalltau(data_pair[col_i], data_pair[col_j])
                        else:
                            raise ValueError(
                                f"Unsupported correlation method: {method}"
                            )
                        p_values.iloc[i, j] = p
                    else:
                        p_values.iloc[i, j] = np.nan

        # Summary statistics
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(mask)
        upper_triangle_flat = upper_triangle.values[mask]
        p_upper_triangle = p_values.where(mask)
        p_flat = p_upper_triangle.values[mask]

        if len(upper_triangle_flat) > 0 and not np.all(np.isnan(upper_triangle_flat)):
            mean_corr = float(np.nanmean(np.abs(upper_triangle_flat)))
            max_corr = float(np.nanmax(np.abs(upper_triangle_flat)))
            min_corr = float(np.nanmin(np.abs(upper_triangle_flat)))
        else:
            mean_corr = 0.0
            max_corr = 0.0
            min_corr = 0.0

        summary = {
            "n_variables": len(cols),
            "n_correlations": len(upper_triangle_flat),
            "mean_correlation": mean_corr,
            "max_correlation": max_corr,
            "min_correlation": min_corr,
            "n_significant": int(np.sum(p_flat < 0.05)),
            "pct_significant": (
                float(np.nansum(p_flat < 0.05) / np.sum(~np.isnan(p_flat)) * 100)
                if np.sum(~np.isnan(p_flat)) > 0
                else 0.0
            ),
            "missing_data_info": missing_data_info,
        }

        if len(upper_triangle_flat) > 0 and not np.all(np.isnan(upper_triangle_flat)):
            upper_vals = upper_triangle.values
            valid_vals = upper_vals[mask]
            if valid_vals.size == 0 or np.all(np.isnan(valid_vals)):
                summary["strongest_positive"] = "N/A"
                summary["strongest_negative"] = "N/A"
            else:
                max_idx = np.unravel_index(
                    np.nanargmax(upper_vals), upper_triangle.shape
                )
                summary["strongest_positive"] = (
                    f"{cols[max_idx[0]]} \u2194 {cols[max_idx[1]]} (r={corr_matrix.iloc[max_idx]:.3f})"
                )
                any_negative = np.any(valid_vals < 0)
                if any_negative:
                    min_idx = np.unravel_index(
                        np.nanargmin(upper_vals), upper_triangle.shape
                    )
                    summary["strongest_negative"] = (
                        f"{cols[min_idx[0]]} \u2194 {cols[min_idx[1]]} (r={corr_matrix.iloc[min_idx]:.3f})"
                    )
                else:
                    summary["strongest_negative"] = "N/A"
        else:
            summary["strongest_positive"] = "N/A"
            summary["strongest_negative"] = "N/A"

        # Create Heatmap
        colorscale = [
            [0.0, "rgb(49, 54, 149)"],
            [0.5, "rgb(255, 255, 255)"],
            [1.0, "rgb(165, 0, 38)"],
        ]

        text_values = []
        for i in range(len(cols)):
            row_text = []
            for j in range(len(cols)):
                r_val = corr_matrix_rounded.iloc[i, j]
                p_val = p_values.iloc[i, j]

                if i == j:
                    row_text.append(f"{r_val}")
                elif pd.isna(r_val):
                    row_text.append("NaN")
                elif p_val < 0.001:
                    row_text.append(f"{r_val}***")
                elif p_val < 0.01:
                    row_text.append(f"{r_val}**")
                elif p_val < 0.05:
                    row_text.append(f"{r_val}*")
                else:
                    row_text.append(f"{r_val}")
            text_values.append(row_text)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=cols,
                y=cols,
                colorscale=colorscale,
                zmin=-1,
                zmax=1,
                text=text_values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.3f}<extra></extra>",
                colorbar={
                    "title": "Correlation",
                    "tickvals": [-1, -0.5, 0, 0.5, 1],
                    "ticktext": ["-1.0", "-0.5", "0", "0.5", "1.0"],
                },
            )
        )

        fig.update_layout(
            title={
                "text": f"{method.title()} Correlation Matrix<br><sub>* p<0.05, ** p<0.01, *** p<0.001</sub>",
                "x": 0.5,
                "xanchor": "center",
            },
            height=600,
            width=700,
            xaxis={"side": "bottom"},
            yaxis={"autorange": "reversed"},
        )

        return corr_matrix_rounded, fig, summary
    except Exception as e:
        logger.error(f"Correlation matrix computation failed: {e}")
        return None, None, None


def calculate_correlation(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    method: str = "pearson",
    var_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None, go.Figure | None]:
    """
    ENHANCED: Calculate correlation between two variables with pipeline integration.
    Returns (results, error_msg, figure).
    """
    SUPPORTED_METHODS = {"pearson", "spearman", "kendall"}

    try:
        # Use existing matrix function for unified cleaning/logic
        cols = [var1, var2]

        if method not in SUPPORTED_METHODS:
            return None, f"Unsupported correlation method: {method}", None

        # Determine strategy from config (matrix forces pairwise, but single pair should honor config)
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        # Reuse prepare_data_for_analysis logic directly to honor config strategy
        data_clean, summary_info = prepare_data_for_analysis(
            df,
            required_cols=cols,
            numeric_cols=cols,
            var_meta=var_meta,
            missing_codes=missing_codes,
            handle_missing=strategy,
        )

        summary_info["strategy"] = strategy
        n = summary_info["rows_analyzed"]

        # Extract specific results for this pair
        corr_matrix = data_clean.corr(method=method)
        r_val = float(corr_matrix.loc[var1, var2])

        # statsmodels/scipy check
        if n < 3:
            return None, "Insufficient data (N < 3)", None

        if method == "pearson":
            _res = stats.pearsonr(data_clean[var1], data_clean[var2])
            p_val = _res.pvalue
        elif method == "spearman":
            _res = stats.spearmanr(data_clean[var1], data_clean[var2])
            p_val = _res.pvalue
        elif method == "kendall":
            _res = stats.kendalltau(data_clean[var1], data_clean[var2])
            p_val = _res.pvalue
        else:
            return None, f"Unsupported correlation method: {method}", None

        ci_lower, ci_upper = _compute_correlation_ci(r_val, n)

        results = {
            "Method": method.capitalize(),
            "Coefficient (r)": r_val,
            "P-value": p_val,
            "N": n,
            "95% CI Lower": ci_lower,
            "95% CI Upper": ci_upper,
            "R-squared (R²)": r_val**2,
            "Interpretation": _interpret_correlation(r_val),
            "Sample Note": f"Analysis based on {n} complete pairs.",
            "missing_data_info": summary_info,
        }

        # --- FIGURE GENERATION ---
        # Generate interactive scatter plot with trendline
        fig = px.scatter(
            data_clean,
            x=var1,
            y=var2,
            trendline="ols",
            title=f"Correlation: {var1} vs {var2} (r={r_val:.3f})",
            labels={var1: var1, var2: var2},
            template="plotly_white",
        )
        fig.update_traces(marker=dict(size=8, opacity=0.6, color=COLORS["primary"]))
        fig.update_layout(
            hovermode="closest",
            margin=dict(l=40, r=40, b=40, t=60),
        )

        return results, None, fig

    except Exception as e:
        logger.error(f"Correlation calculation failed: {str(e)}")
        return None, str(e), None


def calculate_chi2(
    df: pd.DataFrame, var1: str, var2: str, var_meta: dict[str, Any] | None = None
) -> tuple:
    """Compatibility wrapper for chi-square test."""
    from utils import diag_test

    return diag_test.calculate_chi2(df, var1, var2, var_meta=var_meta)


def generate_report(
    title: str,
    elements: list[dict[str, Any]],
    missing_data_info: dict[str, Any] | None = None,
    var_meta: dict[str, Any] | None = None,
) -> str:
    """Wrapper for backward compatibility."""
    return generate_standard_report(title, elements, missing_data_info, var_meta)

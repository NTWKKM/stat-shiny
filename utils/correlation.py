import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from typing import Any

from config import CONFIG
from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import prepare_data_for_analysis

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
        return None, None, None  # Return all None instead of dict with error

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
                handle_missing="pairwise"
            )
            missing_data_info["strategy"] = strategy
            data = data_clean
        except Exception as e:
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
                        else:
                            _, p = stats.spearmanr(data_pair[col_i], data_pair[col_j])
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
                max_idx = np.unravel_index(np.nanargmax(upper_vals), upper_triangle.shape)
                summary["strongest_positive"] = (
                    f"{cols[max_idx[0]]} ↔ {cols[max_idx[1]]} (r={corr_matrix.iloc[max_idx]:.3f})"
                )
                any_negative = np.any(valid_vals < 0)
                if any_negative:
                    min_idx = np.unravel_index(
                        np.nanargmin(upper_vals), upper_triangle.shape
                    )
                    summary["strongest_negative"] = (
                        f"{cols[min_idx[0]]} ↔ {cols[min_idx[1]]} (r={corr_matrix.iloc[min_idx]:.3f})"
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
        return None, None, None

# ✅ ADDED: Alias for compatibility with tests expecting 'calculate_correlation'
calculate_correlation = compute_correlation_matrix
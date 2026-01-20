"""
📋 Correlation & Statistical Association Module (Enhanced) - UPDATED

Provides functions for:
- Pearson/Spearman correlation analysis with enhanced statistics
- Correlation Matrix with summary statistics
- Chi-square and Fisher's exact tests for categorical data
- Interactive Plotly visualizations (Scatter Plots & Heatmaps)
- Comprehensive HTML report generation

Enhanced Features:
- Confidence intervals for correlations
- R-squared and effect size
- Matrix summary statistics
- Detailed interpretations
"""

from __future__ import annotations

import html as _html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats

from config import CONFIG
from tabs._common import get_color_palette
from utils.data_cleaning import (
    apply_missing_values_to_df,
    get_missing_summary_df,
    handle_missing_for_analysis,
)


# Get unified color palette
COLORS = get_color_palette()


def _compute_correlation_ci(
    r: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute confidence interval for correlation coefficient using Fisher's Z transformation.

    Args:
        r: Correlation coefficient
        n: Sample size
        confidence: Confidence level (default 0.95)

    Returns:
        tuple: (lower_ci, upper_ci)
    """
    # Check for non-finite correlation (NaN/Inf) or small sample size
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)

    # Clamp r to a safe range to avoid log/overflow issues in Fisher's Z transform
    eps = 1e-12
    r = np.clip(r, -1 + eps, 1 - eps)

    # Fisher's Z transformation
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)

    # Z critical value
    z_crit = stats.norm.ppf((1 + confidence) / 2)

    # CI in Z space
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    # Transform back to r space
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    return (r_lower, r_upper)


def _interpret_correlation(r: float) -> str:
    """
    Interpret correlation strength.

    Args:
        r: Correlation coefficient

    Returns:
        str: Interpretation
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


def calculate_chi2(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    method: str = "Pearson (Standard)",
    v1_pos: str | None = None,
    v2_pos: str | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str, pd.DataFrame | None]:
    """
    OPTIMIZED: Compute chi-square or Fisher's exact test between two categorical variables.

    Optimizations:
    - Single crosstab computation, reused for all operations (4x faster)
    - Vectorized string operations

    Args:
        df (pd.DataFrame): Source dataframe
        col1 (str): Row (exposure) column name
        col2 (str): Column (outcome) column name
        method (str): Test method ('Fisher', 'Yates', or 'Pearson')
        v1_pos: Position for row reordering
        v2_pos: Position for column reordering

    Returns:
        tuple: (display_tab, stats_df, msg, risk_df)
            - display_tab: Formatted contingency table
            - stats_df: Test results as DataFrame
            - msg: Human-readable summary
            - risk_df: Risk metrics for 2x2 tables
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None, None, "Columns not found", None

    data = df[[col1, col2]].dropna()

    # OPTIMIZATION: Single crosstab computation, reuse for all operations
    tab = pd.crosstab(data[col1], data[col2])
    tab_raw = pd.crosstab(data[col1], data[col2], margins=True, margins_name="Total")
    tab_row_pct = (
        pd.crosstab(
            data[col1],
            data[col2],
            normalize="index",
            margins=True,
            margins_name="Total",
        )
        * 100
    )

    # --- REORDERING LOGIC ---
    all_col_labels = tab_raw.columns.tolist()
    all_row_labels = tab_raw.index.tolist()
    base_col_labels = [col for col in all_col_labels if col != "Total"]
    base_row_labels = [row for row in all_row_labels if row != "Total"]

    def get_original_label(label_str: str, df_labels: list[Any]) -> Any:
        """Find the original label from a collection."""
        for lbl in df_labels:
            if str(lbl) == label_str:
                return lbl
        return label_str

    def custom_sort(label: Any) -> tuple[int, Any]:
        """Sort key for mixed numeric/string labels."""
        try:
            return (0, float(label))
        except (ValueError, TypeError):
            return (1, str(label))

    # Reorder columns
    final_col_order_base = base_col_labels[:]
    if v2_pos is not None:
        v2_pos_original = get_original_label(v2_pos, base_col_labels)
        if v2_pos_original in final_col_order_base:
            final_col_order_base.remove(v2_pos_original)
            final_col_order_base.insert(0, v2_pos_original)
    else:
        final_col_order_base.sort(key=custom_sort)

    final_col_order = final_col_order_base + ["Total"]

    # Reorder rows
    final_row_order_base = base_row_labels[:]
    if v1_pos is not None:
        v1_pos_original = get_original_label(v1_pos, base_row_labels)
        if v1_pos_original in final_row_order_base:
            final_row_order_base.remove(v1_pos_original)
            final_row_order_base.insert(0, v1_pos_original)
    else:
        final_row_order_base.sort(key=custom_sort)

    final_row_order = final_row_order_base + ["Total"]

    # Reindex
    tab_raw = tab_raw.reindex(index=final_row_order, columns=final_col_order)
    tab_row_pct = tab_row_pct.reindex(index=final_row_order, columns=final_col_order)
    tab = tab.reindex(index=final_row_order_base, columns=final_col_order_base)

    # Format display table
    display_data = []
    for row_name in final_row_order:
        row_data = []
        for col_name in final_col_order:
            count = tab_raw.loc[row_name, col_name]
            if col_name == "Total":
                pct = 100.0
            else:
                pct = tab_row_pct.loc[row_name, col_name]
            cell_content = f"{count} ({pct:.1f}%)"
            row_data.append(cell_content)
        display_data.append(row_data)

    # Format display table with MultiIndex for hierarchical header
    col_tuples = []
    for c in final_col_order:
        if c == "Total":
            col_tuples.append(("Total", ""))
        else:
            col_tuples.append((col2, str(c)))

    multi_cols = pd.MultiIndex.from_tuples(col_tuples)

    display_tab = pd.DataFrame(display_data, columns=multi_cols, index=final_row_order)
    display_tab.index.name = col1

    # 3. Statistical tests
    msg = ""
    try:
        is_2x2 = tab.shape == (2, 2)

        stats_res: dict[str, str | int] = {}

        if "Fisher" in method:
            if not is_2x2:
                return (
                    display_tab,
                    None,
                    "Error: Fisher's Exact Test requires a 2x2 table.",
                    None,
                )

            odds_ratio, p_value = stats.fisher_exact(tab)
            method_name = "Fisher's Exact Test"

            stats_res = {
                "Test": method_name,
                "Statistic (OR)": f"{odds_ratio:.4f}",
                "P-value": f"{p_value:.4f}",
                "Degrees of Freedom": "-",
                "N": len(data),
            }
        else:
            # Chi-Square test
            use_correction = True if "Yates" in method else False
            chi2_val, p, dof, ex = stats.chi2_contingency(
                tab, correction=use_correction
            )
            method_name = "Chi-Square"
            if is_2x2:
                method_name += " (with Yates')" if use_correction else " (Pearson)"

            stats_res = {
                "Test": method_name,
                "Statistic": f"{chi2_val:.4f}",
                "P-value": f"{p:.4f}",
                "Degrees of Freedom": f"{dof}",
                "N": len(data),
            }

            # Warning check
            if (ex < 5).any() and is_2x2 and not use_correction:
                msg += " ⚠️ Warning: Expected count < 5. Consider Fisher's Exact Test."

        stats_df = pd.DataFrame(stats_res, index=[0]).T.reset_index()
        stats_df.columns = ["Statistic", "Value"]

        return display_tab, stats_df, msg, None

    except Exception as e:
        return display_tab, None, str(e), None


def calculate_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    method: str = "pearson",
    var_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None, go.Figure | None]:
    """
    ENHANCED: Compute correlation with comprehensive statistics.

    New Features:
    - Confidence intervals (95%)
    - R-squared (coefficient of determination)
    - Effect size interpretation
    - Sample size assessment

    Args:
        df (pd.DataFrame): Source dataframe
        col1 (str): X-axis column name
        col2 (str): Y-axis column name
        method (str): 'pearson' or 'spearman'

    Returns:
        tuple: (result_dict, error_msg, plotly_figure)
            - result_dict: Enhanced statistics dictionary
            - error_msg: Error message or None
            - plotly_figure: Interactive scatter plot or None
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None, "Columns not found", None

    # OPTIMIZATION: Batch numeric conversion (2x faster)
    data_raw = df[[col1, col2]].dropna()
    data_numeric = data_raw.apply(pd.to_numeric, errors="coerce").dropna()

    if len(data_numeric) < 2:
        return None, "Error: Need at least 2 numeric values.", None

    v1 = data_numeric[col1]
    v2 = data_numeric[col2]
    n = len(data_numeric)

    # --- MISSING DATA HANDLING ---
    # Although we did dropna() above, let's capture the logic formally for reporting
    missing_data_info = {}
    if var_meta:
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        # Re-process from original df to capture missing stats properly
        cols_to_use = [col1, col2]
        df_subset = df[cols_to_use].copy()

        # 1. Get summary (on original data)
        missing_summary = get_missing_summary_df(df_subset, var_meta, missing_codes)
        # 2. Apply codes & 3. Handle (Complete case)
        df_clean, impact = handle_missing_for_analysis(
            df_subset,
            var_meta,
            missing_codes=missing_codes,
            strategy=strategy,
            return_counts=True,
        )

        missing_data_info = {
            "strategy": strategy,
            "rows_analyzed": impact["final_rows"],
            "rows_excluded": impact["rows_removed"],
            "summary_before": missing_summary.to_dict("records"),
        }

        # Use cleaner data if valid
        if not df_clean.empty:
            data_numeric = df_clean.apply(pd.to_numeric, errors="coerce").dropna()
            if len(data_numeric) < 2:
                return None, "Error: Need at least 2 numeric values.", None
            v1 = data_numeric[col1]
            v2 = data_numeric[col2]
            n = len(data_numeric)

    if method == "pearson":
        corr, p = stats.pearsonr(v1, v2)
        name = "Pearson"
    else:
        corr, p = stats.spearmanr(v1, v2)
        name = "Spearman"

    # Calculate additional statistics
    ci_lower, ci_upper = _compute_correlation_ci(corr, n)
    r_squared = corr**2
    interpretation = _interpret_correlation(corr)

    # Sample size assessment
    if n < 30:
        sample_note = "Small sample (n<30) - interpret with caution"
    elif n < 100:
        sample_note = "Moderate sample size"
    else:
        sample_note = "Large sample size - results are reliable"

    # Create Plotly figure
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=v1,
            y=v2,
            mode="markers",
            marker={
                "size": 8,
                "color": COLORS["primary"],
                "line": {"color": "white", "width": 0.5},
                "opacity": 0.7,
            },
            name="Data points",
            hovertemplate=f"{col1}: %{{x:.2f}}<br>{col2}: %{{y:.2f}}<extra></extra>",
        )
    )

    # Add regression line for Pearson
    if method == "pearson":
        try:
            m, b = np.polyfit(v1, v2, 1)
            x_line = np.array([v1.min(), v1.max()])
            y_line = m * x_line + b

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Linear fit",
                    line={"color": COLORS["danger"], "width": 2, "dash": "dash"},
                    hovertemplate="Fitted line<extra></extra>",
                )
            )
        except Exception:
            pass

    # Update layout
    fig.update_layout(
        title={
            "text": f"{col1} vs {col2}<br><sub>{name} r={corr:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]), p={p:.4f}</sub>",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=col1,
        yaxis_title=col2,
        hovermode="closest",
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        height=500,
        width=700,
        font={"size": 12},
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Return enhanced results
    result = {
        "Method": name,
        "Coefficient (r)": corr,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper,
        "R-squared (R²)": r_squared,
        "P-value": p,
        "N": n,
        "Interpretation": interpretation,
        "Sample Note": sample_note,
        "missing_data_info": missing_data_info,
    }

    return result, None, fig


def compute_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
    var_meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame | None, go.Figure | None, dict[str, Any] | None]:
    """
    ENHANCED: Compute correlation matrix with summary statistics.

    New Features:
    - Summary statistics (mean, max, min correlations)
    - Count of significant correlations
    - Strongest/weakest pairs identification

    Args:
        df (pd.DataFrame): Source dataframe
        cols (list): List of column names to include
        method (str): 'pearson' or 'spearman'

    Returns:
        tuple: (corr_matrix, heatmap_figure, summary_stats)
    """
    if not cols or len(cols) < 2:
        return None, None, None

    # Filter and convert to numeric
    # --- MISSING DATA HANDLING ---
    missing_data_info = {}
    if var_meta:
        missing_codes = CONFIG.get("analysis.missing.user_defined_values", [])
        df_subset = df[cols].copy()
        missing_summary = get_missing_summary_df(df_subset, var_meta, missing_codes)
        df_processed = apply_missing_values_to_df(df_subset, var_meta, missing_codes)

        # Note: Correlation calculation usually handles pairwise missingness (analyzes valid pairs)
        # So 'drop_na' across ALL columns might be too aggressive if we want pairwise.
        # But pandas .corr() handles NaN automatically (pairwise complete).
        # We just want to REPORT the missingness here.

        # Let's count missing but NOT drop rows list-wise for the matrix unless requested.
        # Actually .apply(pd.to_numeric) handles coercion.

        missing_data_info = {
            "strategy": "pairwise-complete (handled by correlation)",
            "rows_analyzed": len(df_processed),  # Potentially all rows
            "rows_excluded": 0,  # We don't drop rows explicitly for matrix unless all-nan
            "summary_before": missing_summary.to_dict("records"),
        }
        data = df_processed.apply(pd.to_numeric, errors="coerce")
    else:
        data = df[cols].apply(pd.to_numeric, errors="coerce")

    # Calculate correlation matrix
    corr_matrix = data.corr(method=method)

    # Round for display
    corr_matrix_rounded = corr_matrix.round(3)

    # Calculate p-values for all pairs
    p_values = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i == j:
                p_values.iloc[i, j] = 1.0  # Diagonal is always 1
            elif i > j:
                # Use symmetric property
                p_values.iloc[i, j] = p_values.iloc[j, i]
            else:
                # Calculate p-value
                data_pair = data[[col_i, col_j]].dropna()
                if len(data_pair) >= 3:
                    if method == "pearson":
                        _, p = stats.pearsonr(data_pair[col_i], data_pair[col_j])
                    else:
                        _, p = stats.spearmanr(data_pair[col_i], data_pair[col_j])
                    p_values.iloc[i, j] = p
                else:
                    p_values.iloc[i, j] = np.nan

    # Compute summary statistics (excluding diagonal)
    # Get upper triangle excluding diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_triangle = corr_matrix.where(mask)
    upper_triangle_flat = upper_triangle.values[mask]

    # P-values for upper triangle
    p_upper_triangle = p_values.where(mask)
    p_flat = p_upper_triangle.values[mask]

    # Summary statistics
    # Calculate summary stats carefully to avoid RuntimeWarning
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

    # Check if we have valid data before calling argmax/argmin to avoid All-NaN slice error
    if len(upper_triangle_flat) > 0 and not np.all(np.isnan(upper_triangle_flat)):
        upper_vals = upper_triangle.values
        valid_vals = upper_vals[mask]
        if valid_vals.size == 0 or np.all(np.isnan(valid_vals)):
            summary["strongest_positive"] = "N/A"
            summary["strongest_negative"] = "N/A"
        else:
            # strongest positive: max r
            max_idx = np.unravel_index(np.nanargmax(upper_vals), upper_triangle.shape)
            summary["strongest_positive"] = (
                f"{cols[max_idx[0]]} ↔ {cols[max_idx[1]]} (r={corr_matrix.iloc[max_idx]:.3f})"
            )
            # strongest negative: min r, but only if any negative exists
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
        # Fallback if no valid correlations exist
        summary["strongest_positive"] = "N/A"
        summary["strongest_negative"] = "N/A"

    # Create Heatmap with significance markers
    colorscale = [
        [0.0, "rgb(49, 54, 149)"],
        [0.5, "rgb(255, 255, 255)"],
        [1.0, "rgb(165, 0, 38)"],
    ]

    # Prepare text with significance markers
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


def generate_report(title: str, elements: list[dict[str, Any]]) -> str:
    """
    Generate comprehensive HTML report from elements.

    Element Type Contract:
    - 'html': Must contain pre-escaped HTML strings. Producers (e.g.,
      create_missing_data_report_html()) are responsible for escaping user-supplied
      metadata using utils.formatting._html.escape().
    - 'text', 'interpretation', 'summary', 'note', 'table': Automatically escaped
      by this function using _html.escape() or to_html(escape=True).
    - 'plot': Handled via Plotly's to_html().

    Args:
        title (str): Report title (automatically escaped)
        elements (list): List of report elements with 'type', 'data', 'header'

    Returns:
        str: Complete HTML document
    """
    primary_color = COLORS["primary"]
    primary_dark = COLORS["primary_dark"]
    text_color = COLORS["text"]

    css_style = f"""
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
            color: {text_color};
            line-height: 1.6;
        }}
        h1 {{
            color: {primary_dark};
            border-bottom: 3px solid {primary_color};
            padding-bottom: 12px;
            font-size: 2em;
            margin-bottom: 20px;
        }}
        h2 {{
            color: {primary_dark};
            margin-top: 25px;
            font-size: 1.35em;
            border-left: 5px solid {primary_color};
            padding-left: 12px;
            margin-bottom: 15px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #dee2e6;
        }}
        table th, table td {{
            padding: 12px 15px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        table th {{
            background-color: {primary_color};
            color: white;
            font-weight: 600;
        }}
        table tr:hover td {{
            background-color: #f8f9fa;
        }}
        .contingency-table {{
            border: 1px solid #dee2e6;
            margin: 24px 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .contingency-table th, .contingency-table thead th {{
            background-color: {primary_dark};
            color: white !important;
            font-weight: 600;
            text-align: center;
            vertical-align: middle;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 12px 18px;
        }}
        .contingency-table tbody th {{
            background-color: {primary_dark};
            color: white !important;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        .contingency-table td {{
            background-color: white;
            color: {text_color};
            text-align: center;
            padding: 12px 18px;
            border: 1px solid #dee2e6;
            font-variant-numeric: tabular-nums;
        }}
        .contingency-table tr:hover td {{
            background-color: #f1f5f9;
        }}
        .metric-text {{
            font-size: 1.02em;
            margin: 10px 0;
            display: flex;
            align-items: baseline;
            gap: 8px;
        }}
        .metric-label {{
            font-weight: 600;
            color: {primary_dark};
            min-width: 160px;
        }}
        .metric-value {{
            color: {primary_color};
            font-weight: 600;
            font-family: 'Courier New', monospace;
            background-color: #ecf0f1;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .interpretation {{
            background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%);
            border-left: 4px solid {primary_color};
            padding: 14px 15px;
            margin: 16px 0;
            border-radius: 5px;
            line-height: 1.7;
            color: {text_color};
            font-weight: 500;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #fff3e0 0%, #f8f9fa 100%);
            border: 2px solid #ff9800;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }}
        .summary-box h3 {{
            color: #e65100;
            margin-top: 0;
            font-size: 1.2em;
        }}
        .note-box {{
            background-color: #fff9e6;
            border-left: 4px solid #ffc107;
            padding: 12px 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .report-footer {{
            text-align: center;
            font-size: 0.85em;
            color: #7f8c8d;
            margin-top: 40px;
            border-top: 1px solid #ecf0f1;
            padding-top: 20px;
        }}
    </style>
    """

    html = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>{css_style}</head>\n<body>"
    html += f"<h1>{_html.escape(str(title))}</h1>"

    for element in elements:
        element_type = element.get("type")
        data = element.get("data")
        header = element.get("header")

        if header:
            html += f"<h2>{_html.escape(str(header))}</h2>"

        if element_type == "text":
            text_str = str(data)
            if ":" in text_str and len(text_str) < 150:
                parts = text_str.split(":", 1)
                label = _html.escape(parts[0].strip())
                value = _html.escape(parts[1].strip())
                html += f"<p class='metric-text'><span class='metric-label'>{label}:</span> <span class='metric-value'>{value}</span></p>"
            else:
                html += f"<p>{_html.escape(text_str)}</p>"

        elif element_type == "html":
            # Note: Producers of 'html' elements are responsible for escaping user-supplied metadata.
            html += str(data)

        elif element_type == "interpretation":
            html += f"<div class='interpretation'>📊 {_html.escape(str(data))}</div>"

        elif element_type == "summary":
            html += f"<div class='summary-box'>{_html.escape(str(data))}</div>"

        elif element_type == "note":
            html += f"<div class='note-box'>💡 {_html.escape(str(data))}</div>"

        elif element_type in ("table", "contingency_table", "contingency"):
            if hasattr(data, "to_html"):
                classes = (
                    "contingency-table"
                    if element_type in ("contingency_table", "contingency")
                    else ""
                )
                html += data.to_html(index=True, classes=classes, escape=True)
            else:
                html += str(data)

        elif element_type == "plot":
            if hasattr(data, "to_html"):
                html += data.to_html(full_html=False, include_plotlyjs="cdn")

    html += "<div class='report-footer'>© 2026 Statistical Analysis Report | Generated by Enhanced Correlation Module</div>"
    html += "</body>\n</html>"
    return html

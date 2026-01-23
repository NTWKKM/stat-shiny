"""
🎨 Formatting & Styling Utilities
Consolidated from logic.py, table_one.py, and diag_test.py
Driven by central configuration from config.py
"""

from __future__ import annotations

import html as _html
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG  # นำเข้าตัวสั่งการ


def format_p_value(p: float, *, use_style: bool = True) -> str:
    """
    Format P-value using settings from CONFIG.
    """
    if pd.isna(p) or not np.isfinite(p):
        return "NA" if use_style else "-"

    # ดึงค่ากำหนดจาก config.py (คนสั่งการ)
    precision = CONFIG.get("ui.table_decimal_places", 3)
    lower_bound = CONFIG.get("analysis.pvalue_bounds_lower", 0.001)
    upper_bound = CONFIG.get("analysis.pvalue_bounds_upper", 0.999)
    p_small_fmt = CONFIG.get("analysis.pvalue_format_small", "<0.001")
    p_large_fmt = CONFIG.get("analysis.pvalue_format_large", ">0.999")

    # Logic การจัดรูปแบบตามคำสั่งจาก Config
    if p < lower_bound:
        p_text = p_small_fmt if not use_style else p_small_fmt.replace("<", "&lt;")
    elif p > upper_bound:
        p_text = p_large_fmt if not use_style else p_large_fmt.replace(">", "&gt;")
    else:
        p_text = f"{p:.{precision}f}"

    if use_style:
        # ใช้ค่าความเชื่อมั่น (Significance Level) จาก Config
        sig_threshold = CONFIG.get("analysis.significance_level", 0.05)
        sig_style = CONFIG.get(
            "ui.styles.sig_p_value", "font-weight: bold; color: #d63384;"
        )
        if p < sig_threshold:
            return f'<span style="{sig_style}">{p_text}</span>'

    return p_text


def format_ci_html(
    ci_str: str,
    lower: float,
    upper: float,
    null_val: float = 1.0,
    direction: str = "exclude",
) -> str:
    """
    Format CI string with green highlighting if statistically significant.
    """
    if not np.isfinite(lower) or not np.isfinite(upper):
        return ci_str

    is_sig = False
    if direction == "exclude":
        # ไม่ครอบคลุมค่า null_val
        if (lower > null_val) or (upper < null_val):
            is_sig = True
    elif direction == "greater":
        # สูงกว่าค่า null_val
        if lower > null_val:
            is_sig = True

    if is_sig:
        # ใช้สีเขียวสำหรับค่าที่มีนัยสำคัญตาม Config
        sig_style = CONFIG.get(
            "ui.styles.sig_ci",
            "font-weight: bold; color: #198754; font-family: monospace;",
        )
        return f'<span style="{sig_style}">{ci_str}</span>'
    return ci_str


def get_badge_html(text: str, level: str = "info") -> str:
    """
    Generate HTML string for a styled badge.
    (Styles are kept consistent with original UI)
    """
    # Ensure text is safely escaped to prevent XSS
    safe_text = _html.escape(str(text))

    colors = {
        "success": {"bg": "#d4edda", "color": "#155724", "border": "#c3e6cb"},
        "warning": {"bg": "#fff3cd", "color": "#856404", "border": "#ffeeba"},
        "danger": {"bg": "#f8d7da", "color": "#721c24", "border": "#f5c6cb"},
        "info": {"bg": "#d1ecf1", "color": "#0c5460", "border": "#bee5eb"},
        "neutral": {"bg": "#e2e3e5", "color": "#383d41", "border": "#d6d8db"},
    }
    c = colors.get(level, colors["neutral"])
    style = (
        f"padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.85em; "
        f"display: inline-block; background-color: {c['bg']}; color: {c['color']}; "
        f"border: 1px solid {c['border']};"
    )
    return f'<span style="{style}">{safe_text}</span>'


def create_missing_data_report_html(missing_data_info: dict, var_meta: dict) -> str:
    """
    Generate HTML section for missing data report.

    Parameters:
        missing_data_info: Dictionary containing:
            - strategy: str (e.g., 'complete-case')
            - rows_analyzed: int
            - rows_excluded: int
            - summary_before: list of dicts with Variable, N_Missing, Pct_Missing
        var_meta: Variable metadata dictionary

    Returns:
        HTML string with formatted missing data section
    """
    html = '<div class="missing-data-section">\n'
    html += "<h4>📊 Missing Data Summary</h4>\n"

    # Strategy info (escape any user-provided content before rendering)
    strategy_raw = missing_data_info.get("strategy", "Unknown")
    strategy_badge = get_badge_html(
        strategy_raw,
        level="info" if strategy_raw == "complete_case" else "neutral",
    )

    rows_analyzed = missing_data_info.get("rows_analyzed", 0)
    rows_excluded = missing_data_info.get("rows_excluded", 0)
    total_rows = rows_analyzed + rows_excluded
    pct_excluded = (rows_excluded / total_rows * 100) if total_rows > 0 else 0
    pct_included = 100 - pct_excluded

    html += "<div style='margin-bottom: 15px; font-size: 0.95em; color: #495057;'>"
    html += "  <div style='display:flex; justify-content:space-between; margin-bottom: 8px;'>"
    html += f"    <span><strong>Strategy:</strong> {strategy_badge}</span>"
    html += f"    <span><strong>Total Rows:</strong> {total_rows:,}</span>"
    html += "  </div>"
    html += (
        "  <div>Included: <b>"
        f"{rows_analyzed:,}</b> ({pct_included:.1f}%) | Excluded: "
        "<span style='color: #dc3545;'><b>"
        f"{rows_excluded:,}</b> ({pct_excluded:.1f}%)</span></div>"
    )
    html += "</div>"

    # Variables with missing data
    summary = missing_data_info.get("summary_before", [])

    # Get threshold from config
    threshold = CONFIG.get("analysis.missing.report_threshold_pct", 50)

    if summary:
        vars_with_missing = [v for v in summary if v.get("N_Missing", 0) > 0]

        if vars_with_missing:
            html += "<h5>Variables with Missing Data:</h5>\n"
            html += '<table class="missing-table">\n'
            html += (
                "<thead><tr><th>Variable</th><th>Type</th><th>N Valid</th>"
                "<th>N Missing</th><th>% Missing</th></tr></thead>\n"
            )
            html += "<tbody>\n"

            for var in vars_with_missing:
                pct_str = var.get("Pct_Missing", "0%")
                try:
                    pct_val = float(pct_str.rstrip("%"))
                except (ValueError, AttributeError):
                    pct_val = 0

                row_class = "high-missing" if pct_val > threshold else ""
                raw_label = var_meta.get(var["Variable"], {}).get(
                    "label", var["Variable"]
                )
                var_label = _html.escape(str(raw_label))
                var_type = _html.escape(str(var.get("Type", "Unknown")))

                html += f"<tr class='{row_class}'>\n"
                html += f"<td>{var_label}</td>\n"
                html += f"<td>{var_type}</td>\n"
                html += f"<td>{var.get('N_Valid', 0):,}</td>\n"
                html += f"<td>{var.get('N_Missing', 0):,}</td>\n"
                html += f"<td>{pct_str}</td>\n"
                html += "</tr>\n"

            html += "</tbody>\n</table>\n"

    # Warnings for high missing
    high_missing = [
        v for v in summary if _get_pct(v.get("Pct_Missing", "0%")) > threshold
    ]
    if high_missing:
        html += '<div class="warning-box">\n'
        html += (
            f"<strong>⚠️ Warning:</strong> Variables with >{threshold}% missing data:\n"
        )
        html += "<ul>\n"
        for var in high_missing:
            raw_label = var_meta.get(var["Variable"], {}).get("label", var["Variable"])
            var_label = _html.escape(str(raw_label))
            html += f"<li>{var_label} ({var.get('Pct_Missing', '?')} missing)</li>\n"
        html += "</ul>\n"
        html += "</div>\n"

    html += "</div>\n"

    return html


def _get_pct(pct_str: str) -> float:
    """Helper to extract percentage as float from string like '50.0%'."""
    try:
        return float(pct_str.rstrip("%"))
    except (ValueError, AttributeError):
        return 0.0


def render_contingency_table_html(
    df: pd.DataFrame, row_var_name: str, _col_var_name: str = ""
) -> str:
    """
    Render contingency table with proper 2-row header with rowspan/colspan.
    """
    # Extract data
    col_level_0 = [col[0] for col in df.columns]
    col_level_1 = [col[1] for col in df.columns]

    # Build header HTML
    header_row_1 = "<tr>"
    header_row_1 += f'<th rowspan="2" style="vertical-align: middle;">{_html.escape(row_var_name)}</th>'

    current_group = None
    group_count = 0
    col_groups = []

    for l0, _1 in zip(col_level_0, col_level_1, strict=True):
        if l0 != current_group:
            if current_group is not None:
                col_groups.append((current_group, group_count))
            current_group = l0
            group_count = 1
        else:
            group_count += 1
    if current_group is not None:
        col_groups.append((current_group, group_count))

    for group_name, count in col_groups:
        if group_name == "Total":
            header_row_1 += '<th rowspan="2" style="vertical-align: middle;">Total</th>'
        else:
            header_row_1 += (
                f'<th colspan="{count}">{_html.escape(str(group_name))}</th>'
            )

    header_row_1 += "</tr>"

    header_row_2 = "<tr>"
    for l0, l1 in zip(col_level_0, col_level_1, strict=True):
        if l0 != "Total" and l1 != "":
            header_row_2 += f"<th>{_html.escape(str(l1))}</th>"
    header_row_2 += "</tr>"

    body_html = ""
    for idx in df.index:
        body_html += "<tr>"
        body_html += f"<th>{_html.escape(str(idx))}</th>"
        for val in df.loc[idx]:
            body_html += f"<td>{val}</td>"
        body_html += "</tr>"

    return f"""
    <table class="contingency-table">
        <thead>
            {header_row_1}
            {header_row_2}
        </thead>
        <tbody>
            {body_html}
        </tbody>
    </table>
    """


def generate_standard_report(
    title: str,
    elements: list[dict[str, Any]],
    missing_data_info: dict[str, Any] | None = None,
    var_meta: dict[str, Any] | None = None,
) -> str:
    """
    Unified report generator for all statistical modules.
    Supports: text, header, table, plot (plotly), interpretation, html, and contingency tables.

    Parameters:
        title: Main report title
        elements: List of element dicts: {"type": "...", "data": ..., "header": "..."}
        missing_data_info: Optional dict from prepare_data_for_analysis()
        var_meta: Optional variable metadata for labeling
    """
    from tabs._common import get_color_palette

    colors = get_color_palette()
    primary_color = colors["primary"]
    primary_dark = colors["primary_dark"]
    text_color = colors.get("text", "#333333")

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
        .interpretation {{
            background: linear-gradient(135deg, #ecf0f1 0%, #f8f9fa 100%);
            border-left: 4px solid {primary_color};
            padding: 14px 15px;
            margin: 16px 0;
            border-radius: 5px;
            line-height: 1.7;
            color: {text_color};
        }}
        .report-footer {{
            text-align: center;
            font-size: 0.85em;
            color: #7f8c8d;
            margin-top: 40px;
            border-top: 1px solid #ecf0f1;
            padding-top: 20px;
        }}
        .sig-p {{
            font-weight: bold;
            color: #d63384;
        }}
        /* 🎨 Navy Blue Contingency Table Theme */
        .contingency-table {{
            width: 100%;
            border-collapse: separate !important;
            border-spacing: 0;
            border: 1px solid #d1d9e6;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin: 24px 0;
        }}
        .contingency-table thead tr th {{
            background: linear-gradient(135deg, {primary_dark} 0%, #004080 100%) !important;
            color: white !important;
            font-weight: 600;
            text-align: center;
            padding: 12px 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            vertical-align: middle;
        }}
        .contingency-table tbody th {{
            background-color: #f1f8ff !important;
            color: {primary_dark} !important;
            font-weight: bold;
            text-align: center;
            border-right: 2px solid #d1d9e6;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: middle;
        }}
        .contingency-table tbody td {{
            background-color: #ffffff;
            color: #2c3e50;
            text-align: center;
            padding: 10px 15px;
            border-bottom: 1px solid #e0e0e0;
            border-right: 1px solid #f0f0f0;
        }}
        .contingency-table tbody tr:hover td {{
            background-color: #e6f3ff !important;
            transition: background-color 0.2s ease;
        }}
    </style>
    """

    html = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'>{css_style}</head>\n<body>"
    html += f"<h1>{_html.escape(str(title))}</h1>"

    # Add Model Stats at the top if provided (standard for TVC/Cox models)
    if elements and any(e.get("type") == "stats_box" for e in elements):
        # Already in elements, skip automatic box
        pass
    elif var_meta and "model_stats" in var_meta:  # Alternative pass-through
        model_stats = var_meta.get("model_stats")
        if isinstance(model_stats, dict):
            html += "<h2>📊 Model Summary</h2>"
            html += "<div class='interpretation' style='display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px;'>"
            for key, val in model_stats.items():
                html += f"<div><strong>{_html.escape(str(key))}:</strong> {_html.escape(str(val))}</div>"
            html += "</div>"

    for element in elements:
        element_type = element.get("type")
        data = element.get("data")
        header = element.get("header")

        if header:
            html += f"<h2>{_html.escape(str(header))}</h2>"

        if element_type == "text":
            html += f"<p>{_html.escape(str(data))}</p>"

        elif element_type == "header":
            html += f"<h2>{_html.escape(str(data))}</h2>"

        elif element_type == "interpretation":
            html += f"<div class='interpretation'>{_html.escape(str(data))}</div>"

        elif element_type in {"summary", "html", "raw_html"}:
            # raw_html is used in some modules
            html += str(data)

        elif element_type == "table":
            if isinstance(data, pd.DataFrame):
                d_styled = data.copy()
                # Apply P-value highlighting if column exists
                p_col = next(
                    (
                        c
                        for c in d_styled.columns
                        if c.lower() in ["p", "p-value", "p_value"]
                    ),
                    None,
                )
                if p_col:
                    p_vals = pd.to_numeric(d_styled[p_col], errors="coerce")
                    new_p_val_col = []
                    for val, pv in zip(d_styled[p_col], p_vals):
                        if not pd.isna(pv) and pv < CONFIG.get(
                            "analysis.significance_level", 0.05
                        ):
                            new_p_val_col.append(f'<span class="sig-p">{val}</span>')
                        else:
                            new_p_val_col.append(str(val))
                    d_styled[p_col] = new_p_val_col

                html += d_styled.to_html(
                    classes="table table-striped", border=0, escape=False, index=True
                )
            else:
                html += str(data)

        elif element_type in ("contingency_table", "contingency"):
            if hasattr(data, "to_html"):
                html += data.to_html(
                    classes="contingency-table", escape=False, index=True
                )
            else:
                html += str(data)

        elif element_type == "plot":
            if hasattr(data, "to_html"):
                html += data.to_html(full_html=False, include_plotlyjs="cdn")
            elif hasattr(data, "savefig"):
                import base64
                import io

                buf = io.BytesIO()
                data.savefig(buf, format="png", bbox_inches="tight")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                html += (
                    f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>'
                )

    if missing_data_info:
        html += create_missing_data_report_html(missing_data_info, var_meta or {})

    html += f"""
    <div class='report-footer'>
        © {pd.Timestamp.now().year} <a href="https://github.com/NTWKKM/" target="_blank">NTWKKM</a> | Powered by stat-shiny
    </div>
    </body>\n</html>
    """
    return html

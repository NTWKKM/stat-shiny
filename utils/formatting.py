"""
🎨 Formatting & Styling Utilities
Consolidated from logic.py, table_one.py, and diag_test.py
Driven by central configuration from config.py
"""

from __future__ import annotations
from typing import Any

import html as _html

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


class PublicationFormatter:
    """
    Format statistical results according to specific journal guidelines.
    Configured via 'analysis.publication_style'.
    """

    @staticmethod
    def format_nejm(
        coef: float, se: float | None, ci_lower: float, ci_upper: float, p_value: float
    ) -> str:
        """
        NEJM style: 3.14 (95% CI, 2.81 to 3.47); P=0.003
        """
        # NEJM uses "to" for CI (sometimes en-dash), and specific P-value formatting
        p_str = format_p_value(p_value, use_style=False)
        # NEJM usually adds P= or P<
        if "<" in p_str or ">" in p_str:
            p_display = f"P{p_str}"
        else:
            p_display = f"P={p_str}"

        return f"{coef:.2f} (95% CI, {ci_lower:.2f} to {ci_upper:.2f}); {p_display}"

    @staticmethod
    def format_jama(
        coef: float, se: float | None, ci_lower: float, ci_upper: float, p_value: float
    ) -> str:
        """
        JAMA style: 3.14 (95% CI, 2.81-3.47); P = .003
        """
        # JAMA often removes leading zero for P-values (not strictly enforced here for simplicity, reusing format_p_value)
        # JAMA usually uses hyphens for range
        p_str = format_p_value(p_value, use_style=False)
        if "<" in p_str or ">" in p_str:
            p_display = f"P{p_str}"
        else:
            p_display = f"P={p_str}"

        return f"{coef:.2f} (95% CI, {ci_lower:.2f}-{ci_upper:.2f}); {p_display}"

    @staticmethod
    def format_lancet(
        coef: float, se: float | None, ci_lower: float, ci_upper: float, p_value: float
    ) -> str:
        """
        Lancet style: 3.14 (95% CI 2.81–3.47), p=0.003
        """
        # Lancet uses en-dash and lowercase p
        p_str = format_p_value(p_value, use_style=False)
        if "<" in p_str or ">" in p_str:
            p_display = f"p{p_str}"
        else:
            p_display = f"p={p_str}"

        return f"{coef:.2f} (95% CI {ci_lower:.2f}–{ci_upper:.2f}), {p_display}"

    @staticmethod
    def format_bmj(
        coef: float, se: float | None, ci_lower: float, ci_upper: float, p_value: float
    ) -> str:
        """
        BMJ style: 3.14 (95% confidence interval 2.81 to 3.47); P=0.003
        """
        # BMJ prefers "confidence interval" spelled out sometimes, or "CI"
        # We'll stick to slightly verbose defaults if mostly requested
        p_str = format_p_value(p_value, use_style=False)
        if "<" in p_str or ">" in p_str:
            p_display = f"P{p_str}"
        else:
            p_display = f"P={p_str}"

        return f"{coef:.2f} (95% confidence interval {ci_lower:.2f} to {ci_upper:.2f}); {p_display}"

    @staticmethod
    def format(
        coef: float,
        se: float | None,
        ci_lower: float,
        ci_upper: float,
        p_value: float,
        style: str | None = None,
    ) -> str:
        """
        Dispatch formatting to the configured or requested style.
        """
        if style is None:
            style = CONFIG.get("analysis.publication_style", "nejm")

        style = style.lower()
        if style == "jama":
            return PublicationFormatter.format_jama(
                coef, se, ci_lower, ci_upper, p_value
            )
        elif style == "lancet":
            return PublicationFormatter.format_lancet(
                coef, se, ci_lower, ci_upper, p_value
            )
        elif style == "bmj":
            return PublicationFormatter.format_bmj(
                coef, se, ci_lower, ci_upper, p_value
            )
        else:
            # Default to NEJM
            return PublicationFormatter.format_nejm(
                coef, se, ci_lower, ci_upper, p_value
            )

    @staticmethod
    def format_ci(
        lower: float,
        upper: float,
        style: str | None = None,
        include_brackets: bool = True,
    ) -> str:
        """
        Format Confidence Interval (CI) according to style.
        """
        if style is None:
            style = CONFIG.get("analysis.publication_style", "nejm")
        style = style.lower()

        # Determine separator
        if style in ["nejm", "bmj"]:
            sep = " to "
        else:  # jama, lancet, default
            sep = "-"

        # Format numbers (using consistent precision)
        # We could use config precision here if needed
        l_str = f"{lower:.2f}"
        u_str = f"{upper:.2f}"

        # For 'lancet', use en-dash if possible, but standard hyphen is safer for HTML/ASCII
        if style == "lancet":
            sep = "–"  # en-dash

        res = f"{l_str}{sep}{u_str}"

        if include_brackets:
            return f"({res})"
        return res

    @staticmethod
    def format_methods_text(
        analysis_type: str,
        variables: list[str],
        adjustments: list[str] | None = None,
        assumptions_checked: str = "",
    ) -> str:
        """Auto-generate Methods section text."""
        vars_text = ", ".join(variables)
        adj_text = (
            f" while adjusting for {', '.join(adjustments)}" if adjustments else ""
        )

        # Pull p-value threshold for referencing
        alpha = CONFIG.get("analysis.significance_level", 0.05)

        text = (
            f"We used {analysis_type} to assess the association between {vars_text}{adj_text}. "
            f"{assumptions_checked} "
            f"Statistical significance was set at P < {alpha}."
        )
        return text.strip()


class MissingDataStatement:
    """Auto-generate missing data reporting text."""

    @staticmethod
    def generate(missing_info: dict[str, Any], analysis_type: str = "analysis") -> str:
        """
        Generate a textual description of missing data handling.
        """
        strategy = missing_info.get("strategy", "unknown")
        rows_excluded = missing_info.get("rows_excluded", 0)
        rows_analyzed = missing_info.get("rows_analyzed", 0)
        total = rows_excluded + rows_analyzed

        if total == 0:
            return "No data available for missingness reporting."

        pct_excluded = rows_excluded / total * 100

        # Generate detailed breakout
        summary = missing_info.get("summary_before", [])
        miss_details = []
        for item in summary:
            if item.get("N_Missing", 0) > 0:
                var = item.get("Variable")
                n_miss = item.get("N_Missing")
                pct_miss = item.get("Pct_Missing")
                miss_details.append(f"{var} (n={n_miss}, {pct_miss})")

        detail_text = ", ".join(miss_details)
        if detail_text:
            detail_text = f" Data were missing for: {detail_text}."
        else:
            detail_text = ""

        # Strategy description
        if strategy == "complete-case":
            method_text = "Analysis was conducted using complete-case analysis"
        elif strategy == "drop":
            method_text = "Rows with missing values were excluded"
        else:
            method_text = f"Missing data were handled using {strategy}"

        return (
            f"Of {total} observations, {rows_excluded} ({pct_excluded:.1f}%) "
            f"were excluded due to missing data.{detail_text} "
            f"{method_text}."
        )

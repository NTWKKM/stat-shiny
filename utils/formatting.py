"""
🎨 Formatting & Styling Utilities
Consolidated from logic.py, table_one.py, and diag_test.py
Driven by central configuration from config.py
"""

import html as _html
from typing import Optional, Union

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
    precision = CONFIG.get('ui.table_decimal_places', 3)
    lower_bound = CONFIG.get('analysis.pvalue_bounds_lower', 0.001)
    upper_bound = CONFIG.get('analysis.pvalue_bounds_upper', 0.999)
    p_small_fmt = CONFIG.get('analysis.pvalue_format_small', '<0.001')
    p_large_fmt = CONFIG.get('analysis.pvalue_format_large', '>0.999')
    
    # Logic การจัดรูปแบบตามคำสั่งจาก Config
    if p < lower_bound:
        p_text = p_small_fmt if not use_style else p_small_fmt.replace('<', '&lt;')
    elif p > upper_bound:
        p_text = p_large_fmt if not use_style else p_large_fmt.replace('>', '&gt;')
    else:
        p_text = f"{p:.{precision}f}"
        
    if use_style:
        # ใช้ค่าความเชื่อมั่น (Significance Level) จาก Config
        sig_threshold = CONFIG.get('analysis.significance_level', 0.05)
        sig_style = CONFIG.get('ui.styles.sig_p_value', 'font-weight: bold; color: #d63384;')
        if p < sig_threshold:
            return f'<span style="{sig_style}">{p_text}</span>'
            
    return p_text

def format_ci_html(ci_str: str, lower: float, upper: float, null_val: float = 1.0, direction: str = 'exclude') -> str:
    """
    Format CI string with green highlighting if statistically significant.
    """
    if not np.isfinite(lower) or not np.isfinite(upper):
        return ci_str
    
    is_sig = False
    if direction == 'exclude':
        # ไม่ครอบคลุมค่า null_val
        if (lower > null_val) or (upper < null_val):
            is_sig = True
    elif direction == 'greater':
        # สูงกว่าค่า null_val
        if lower > null_val:
            is_sig = True
            
    if is_sig:
        # ใช้สีเขียวสำหรับค่าที่มีนัยสำคัญตาม Config
        sig_style = CONFIG.get('ui.styles.sig_ci', 'font-weight: bold; color: #198754; font-family: monospace;')
        return f'<span style="{sig_style}">{ci_str}</span>'
    return ci_str

def get_badge_html(text: str, level: str = 'info') -> str:
    """
    Generate HTML string for a styled badge.
    (Styles are kept consistent with original UI)
    """
    colors = {
        'success': {'bg': '#d4edda', 'color': '#155724', 'border': '#c3e6cb'},
        'warning': {'bg': '#fff3cd', 'color': '#856404', 'border': '#ffeeba'},
        'danger':  {'bg': '#f8d7da', 'color': '#721c24', 'border': '#f5c6cb'},
        'info':    {'bg': '#d1ecf1', 'color': '#0c5460', 'border': '#bee5eb'},
        'neutral': {'bg': '#e2e3e5', 'color': '#383d41', 'border': '#d6d8db'}
    }
    c = colors.get(level, colors['neutral'])
    style = (f"padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.85em; "
             f"display: inline-block; background-color: {c['bg']}; color: {c['color']}; "
             f"border: 1px solid {c['border']};")
    return f'<span style="{style}">{text}</span>'


def create_missing_data_report_html(
    missing_data_info: dict,
    var_meta: dict
) -> str:
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
    html += '<h4>📊 Missing Data Summary</h4>\n'
    
    # Strategy info
    strategy = _html.escape(missing_data_info.get('strategy', 'Unknown'))
    rows_analyzed = missing_data_info.get('rows_analyzed', 0)
    rows_excluded = missing_data_info.get('rows_excluded', 0)
    total_rows = rows_analyzed + rows_excluded
    pct_excluded = (rows_excluded / total_rows * 100) if total_rows > 0 else 0
    pct_included = 100 - pct_excluded
    
    html += f'<p><strong>Strategy:</strong> {strategy}</p>\n'
    html += f'<p><strong>Rows Analyzed:</strong> {rows_analyzed:,} / {total_rows:,} ({pct_included:.1f}%)</p>\n'
    html += f'<p><strong>Rows Excluded:</strong> {rows_excluded:,} ({pct_excluded:.1f}%)</p>\n'
    
    # Variables with missing data
    summary = missing_data_info.get('summary_before', [])
    
    # Get threshold from config
    threshold = CONFIG.get('analysis.missing.report_threshold_pct', 50)
    
    if summary:
        vars_with_missing = [v for v in summary if v.get('N_Missing', 0) > 0]
        
        if vars_with_missing:
            html += '<h5>Variables with Missing Data:</h5>\n'
            html += '<table class="missing-table">\n'
            html += '<thead><tr><th>Variable</th><th>Type</th><th>N Valid</th><th>N Missing</th><th>% Missing</th></tr></thead>\n'
            html += '<tbody>\n'
            
            for var in vars_with_missing:
                pct_str = var.get('Pct_Missing', '0%')
                try:
                    pct_val = float(pct_str.rstrip('%'))
                except (ValueError, AttributeError):
                    pct_val = 0
                
                row_class = 'high-missing' if pct_val > threshold else ''
                raw_label = var_meta.get(var['Variable'], {}).get('label', var['Variable'])
                var_label = _html.escape(str(raw_label))
                var_type = _html.escape(str(var.get('Type', 'Unknown')))
                
                html += f"<tr class='{row_class}'>\n"
                html += f"<td>{var_label}</td>\n"
                html += f"<td>{var_type}</td>\n"
                html += f"<td>{var.get('N_Valid', 0):,}</td>\n"
                html += f"<td>{var.get('N_Missing', 0):,}</td>\n"
                html += f"<td>{pct_str}</td>\n"
                html += '</tr>\n'
            
            html += '</tbody>\n</table>\n'
    
    # Warnings for high missing
    high_missing = [v for v in summary if _get_pct(v.get('Pct_Missing', '0%')) > threshold]
    if high_missing:
        html += '<div class="warning-box">\n'
        html += f'<strong>⚠️ Warning:</strong> Variables with >{threshold}% missing data:\n'
        html += '<ul>\n'
        for var in high_missing:
            raw_label = var_meta.get(var['Variable'], {}).get('label', var['Variable'])
            var_label = _html.escape(str(raw_label))
            html += f"<li>{var_label} ({var.get('Pct_Missing', '?')} missing)</li>\n"
        html += '</ul>\n'
        html += '</div>\n'
    
    html += '</div>\n'
    
    return html


def _get_pct(pct_str: str) -> float:
    """Helper to extract percentage as float from string like '50.0%'."""
    try:
        return float(pct_str.rstrip('%'))
    except (ValueError, AttributeError):
        return 0.0

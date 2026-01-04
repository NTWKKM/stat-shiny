"""
🎨 Formatting & Styling Utilities
Consolidated from logic.py, table_one.py, and diag_test.py
Driven by central configuration from config.py
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
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

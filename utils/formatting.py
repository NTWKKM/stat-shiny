"""
🎨 Formatting & Styling Utilities
Consolidated from logic.py, table_one.py, and diag_test.py
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

def format_p_value(p: float, use_style: bool = True, precision: int = 3) -> str:
    """
    Format P-value for display with optional HTML significance highlighting.
    
    Logic:
    - If p < 0.001, display '<0.001'
    - Otherwise, display with specified precision
    - Highlighting style from diag_test.py/logic.py
    """
    if pd.isna(p) or not np.isfinite(p):
        return "NA" if use_style else "-"
    
    # Format text base on value
    if p < 0.001:
        p_text = "<0.001" if not use_style else "&lt;0.001"
    else:
        p_text = f"{p:.{precision}f}"
        
    if use_style:
        # Define style for significant p-value (Pink/Purpleish for distinction)
        sig_style = 'font-weight: bold; color: #d63384;' 
        if p < 0.05:
            return f'<span style="{sig_style}">{p_text}</span>'
            
    return p_text

def format_ci_html(ci_str: str, lower: float, upper: float, null_val: float = 1.0, direction: str = 'exclude') -> str:
    """
    Format CI string with green highlighting if statistically significant.
    (From diag_test.py)
    
    Args:
        direction='exclude': Significant if null_val is NOT in [lower, upper]
        direction='greater': Significant if lower > null_val
    """
    if not np.isfinite(lower) or not np.isfinite(upper):
        return ci_str
    
    is_sig = False
    if direction == 'exclude':
        if (lower > null_val) or (upper < null_val):
            is_sig = True
    elif direction == 'greater':
        if lower > null_val:
            is_sig = True
            
    if is_sig:
        # Green text for significant confidence intervals
        return f'<span style="font-weight: bold; color: #198754; font-family: monospace;">{ci_str}</span>'
    return ci_str

def get_badge_html(text: str, level: str = 'info') -> str:
    """
    Generate HTML string for a styled badge.
    (From diag_test.py)
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

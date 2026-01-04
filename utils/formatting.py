import numpy as np
import pandas as pd
from typing import Union

def format_p_value(p: float, use_style: bool = True) -> str:
    """
    Format P-value with optional significance highlighting.
    (Consolidated from logic.py, table_one.py, and diag_test.py)
    """
    if pd.isna(p) or not np.isfinite(p):
        return "-" if not use_style else "NA"
    
    # Logic เดิมจาก table_one.py
    if p < 0.001:
        p_text = "<0.001"
    else:
        p_text = f"{p:.3f}"
        
    if use_style:
        # ใช้ Style เดิมจาก diag_test.py/logic.py สำหรับค่าที่นัยสำคัญ
        sig_style = 'font-weight: bold; color: #d63384;' 
        if p < 0.05:
            return f'<span style="{sig_style}">{p_text}</span>'
            
    return p_text

def format_ci_html(ci_str: str, lower: float, upper: float, null_val: float = 1.0) -> str:
    """
    Format CI string with highlighting if significant (does not include null_val).
    (From diag_test.py)
    """
    if not np.isfinite(lower) or not np.isfinite(upper):
        return ci_str
    
    # ถ้าช่วงความเชื่อมั่นไม่ครอบคลุมค่า null (เช่น 1.0 สำหรับ OR/RR) ให้แสดงสีเขียว
    if (lower > null_val) or (upper < null_val):
        return f'<span style="font-weight: bold; color: #198754;">{ci_str}</span>'
    return ci_str

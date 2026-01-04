"""
🧹 Data Cleaning Utilities
Consolidated from logic.py, table_one.py, and poisson_lib.py
Driven by central configuration from config.py
"""

import pandas as pd
import numpy as np
from typing import Any, Union
from config import CONFIG  # นำเข้าตัวสั่งการ

def clean_numeric(val: Any) -> float:
    """
    Convert value to float, removing common non-numeric characters like '>', '<', and ','.
    
    This handles:
    - Strings with comparison operators (from logic.py/table_one.py)
    - Comma separators in thousands
    - Missing values (returns np.nan)
    """
    if pd.isna(val):
        return np.nan
    
    # ลบเครื่องหมาย >, <, และ comma ออกเพื่อให้แปลงเป็น float ได้
    s = str(val).strip().replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan

def robust_sort_key(x: Any) -> tuple:
    """
    Sort key placing numeric values first.
    (Consolidated from logic.py's _robust_sort_key)
    """
    try:
        if pd.isna(x):
            return (2, "")
        val = float(x)
        return (0, val)
    except (ValueError, TypeError):
        return (1, str(x))

def is_continuous_variable(series: pd.Series) -> bool:
    """
    Determine if a variable should be treated as continuous based on CONFIG.
    (Uses thresholds from config.py)
    """
    # ดึงค่าจาก config.py
    threshold = CONFIG.get('analysis.var_detect_threshold', 10)
    
    unique_count = series.nunique()
    if unique_count > threshold:
        return True
    return False

@np.vectorize
def clean_numeric_vector(val: Any) -> float:
    """
    Vectorized version of clean_numeric for performance on large series.
    (From table_one.py)
    """
    return clean_numeric(val)

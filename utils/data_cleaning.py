"""
🧹 Data Cleaning Utilities
Consolidated from logic.py, table_one.py, and poisson_lib.py
"""

import pandas as pd
import numpy as np
from typing import Any, Union

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

@np.vectorize
def clean_numeric_vector(val: Any) -> float:
    """
    Vectorized version of clean_numeric for performance on large series.
    (From table_one.py)
    """
    return clean_numeric(val)

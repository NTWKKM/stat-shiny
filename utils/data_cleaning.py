import pandas as pd
import numpy as np
from typing import Any

def clean_numeric(val: Any) -> float:
    """
    Convert value to float, removing common non-numeric characters.
    (Consolidated from logic.py and table_one.py)
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
    (From logic.py)
    """
    try:
        if pd.isna(x):
            return (2, "")
        val = float(x)
        return (0, val)
    except (ValueError, TypeError):
        return (1, str(x))

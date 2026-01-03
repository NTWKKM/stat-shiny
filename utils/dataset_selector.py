"""Shared utility for dataset selection across analysis tabs."""
import pandas as pd
from typing import Tuple


def get_dataset_for_analysis(
    df: pd.DataFrame,
    session_key: str,
    *,
    default_to_matched: bool = True,
    label_prefix: str = "📄 เลือกชุดข้อมูล:"
) -> tuple[pd.DataFrame, str]:
    """
    Choose between the original and matched DataFrame for analysis and return the selected DataFrame with a descriptive label.
    
    If a matched dataset exists in Streamlit session state (keys "is_matched" and "df_matched"), the function offers a selection between the original and matched data; otherwise it returns the original DataFrame.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame to fall back to if matched data is unavailable or not selected.
        session_key (str): Unique key used for the Streamlit radio widget to store the user's selection in session state.
        default_to_matched (bool): If True and matched data exists, the matched dataset is selected by default.
        label_prefix (str): Label text shown next to the radio widget.
    
    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing the selected DataFrame and a descriptive label indicating which dataset was chosen and its row count (e.g., "✅ Matched Data (123 rows)" or "📊 Original Data (456 rows)").
    """
    if is_matched and df_matched is not None:
        return df_matched, "Matched Data"
    return df, "Original Data"

import numpy as np
import pandas as pd
from typing import Any, List, Dict, Union, Optional
from statsmodels.stats.multitest import multipletests

from logger import get_logger
# Fix Import: Import calculate_vif as _run_vif so the wrapper works correctly
from utils.collinearity_lib import calculate_vif as _run_vif

logger = get_logger(__name__)

def apply_mcc(
    p_values: Union[List[float], np.ndarray, pd.Series], 
    method: str = "fdr_bh", 
    alpha: float = 0.05
) -> pd.Series:
    """
    Apply Multiple Comparison Correction to a list of p-values.
    
    Args:
        p_values: List, Array, or Series of p-values
        method: Method for correction (e.g., 'fdr_bh', 'bonferroni')
        alpha: Significance level
        
    Returns:
        pd.Series of adjusted p-values
    """
    # Convert input to Series for easier index and NaN handling
    if isinstance(p_values, (list, np.ndarray)):
        p_series = pd.Series(p_values)
    else:
        p_series = p_values.copy()
        
    if p_series.empty:
        return p_series
        
    # Mask NaNs: Calculate only for numeric values
    mask = p_series.notna()
    p_valid = p_series[mask]
    
    if p_valid.empty:
        return p_series
        
    try:
        # Use statsmodels for calculation
        reject, pvals_corrected, _, _ = multipletests(p_valid.values, alpha=alpha, method=method)
        
        # Place calculated values back into original positions (skipping NaNs)
        p_series.loc[mask] = pvals_corrected
    except Exception as e:
        logger.error(f"MCC calculation failed: {e}")
        # In case of error, return original values to prevent program crash
        return p_series
        
    return p_series

def get_ci_configuration(
    method: str, 
    n: int, 
    events: int, 
    n_params: int
) -> Dict[str, str]:
    """
    Configure Confidence Interval method based on sample size and events.
    Returns a dict with 'method' and 'note'.
    """
    config = {"method": method, "note": ""}
    
    # Example Logic: If profile is selected but data is large (may be slow), switch to Wald
    if method == "profile":
        if n > 5000:
            config["method"] = "wald"
            config["note"] = "Switched to Wald due to large sample size (N>5000)"
            
    return config

def calculate_vif(
    data: pd.DataFrame, 
    predictor_cols: List[str], 
    var_meta: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for predictors.
    Wrapper around collinearity_lib.calculate_vif.
    """
    # Pass var_meta to the actual implementation
    return _run_vif(data, predictor_cols, var_meta=var_meta)
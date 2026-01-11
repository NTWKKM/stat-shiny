"""
Advanced Statistics Library for stat-shiny.

This module provides utility functions for:
1. Multiple Comparison Corrections (MCC)
2. Collinearity Diagnostics (VIF)
3. Confidence Interval Configuration (Helpers)
"""

import pandas as pd
import numpy as np
import statsmodels.stats.multitest as smt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix
import logging

logger = logging.getLogger(__name__)

# --- Multiple Comparison Corrections (MCC) ---

from typing import Union

def apply_mcc(p_values: list | pd.Series | np.ndarray, method: str = 'fdr_bh', alpha: float = 0.05) -> pd.Series:
    """
    Apply Multiple Comparison Correction to a list of p-values.

    Args:
        p_values (Union[list, pd.Series, np.ndarray]): Raw p-values.
        method (str): Method for correction.
                      Options: 'bonferroni', 'sidak', 'holm-sidak', 'holm',
                               'simes-hochberg', 'hommel', 'fdr_bh' (Benjamini-Hochberg),
                               'fdr_by', 'fdr_tsbh', 'fdr_tsbky'.
                      Defaults to 'fdr_bh'.
        alpha (float): Significance level (only affects boolean reject decision,
                       but useful for consistency). Defaults to 0.05.

    Returns:
        pd.Series: Adjusted p-values.
    """
    if p_values is None or len(p_values) == 0:
        return pd.Series(dtype=float)

    # Convert to numpy array for processing, ensuring numeric type and handling NaNs
    p_vals_arr = pd.to_numeric(p_values, errors='coerce')
    
    # Mask NaNs to avoid errors in multipletests
    mask = np.isfinite(p_vals_arr)
    p_vals_clean = p_vals_arr[mask]
    
    if len(p_vals_clean) == 0:
        return pd.Series(p_vals_arr) # Return original (all NaNs/empty)

    try:
        # returns: reject, pvals_corrected, alphacSidak, alphacBonf
        _, pvals_corrected, _, _ = smt.multipletests(p_vals_clean, alpha=alpha, method=method)
        
        # Reconstruct full series with NaNs in original positions
        result = np.full_like(p_vals_arr, np.nan, dtype=float)
        result[mask] = pvals_corrected
        
        return pd.Series(result, index=p_values.index if isinstance(p_values, pd.Series) else None)
        
    except Exception:
        logger.exception("Error applying MCC method '%s'", method)
        # Fallback: return original p-values if correction fails widely
        return pd.Series(p_vals_arr, index=p_values.index if isinstance(p_values, pd.Series) else None)


# --- Collinearity Diagnostics (VIF) ---

def calculate_vif(df: pd.DataFrame, *, intercept: bool = True) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing numerical features (predictors).
                           Categorical variables should be one-hot encoded heavily beforehand 
                           or passed as design matrix.
        intercept (bool): Whether to add an intercept (constant) if not present.
                          VIF calculation requires an intercept for correct interpretation.

    Returns:
        pd.DataFrame: A DataFrame with columns ['feature', 'VIF'].
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['feature', 'VIF'])

    # Ensure all data is numeric; drop non-numeric columns silently or raise error
    # Here we attempt to coerce, drop rows with NaNs as VIF can't handle them
    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    
    if df_numeric.empty:
        return pd.DataFrame(columns=['feature', 'VIF'])

    # Drop constant predictors (VIF undefined / can explode)
    variances = df_numeric.var()
    const_predictors = variances[variances == 0].index.tolist()
    if const_predictors:
        df_numeric = df_numeric.drop(columns=const_predictors, errors="ignore")

    if intercept:
        df_numeric = df_numeric.copy()
        if "const" not in df_numeric.columns:
            df_numeric["const"] = 1.0

    vif_data = pd.DataFrame()
    vif_data["feature"] = df_numeric.columns
    
    try:
        vif_data["VIF"] = [
            variance_inflation_factor(df_numeric.values, i)
            for i in range(df_numeric.shape[1])
        ]
        
        # Filter out the constant 'const' row if we added it purely for calculation
        if 'const' in vif_data["feature"].values:
            vif_data = vif_data[vif_data["feature"] != 'const']

        return vif_data.sort_values(by='VIF', ascending=False)
        
    except Exception:
        logger.exception("Error calculating VIF")
        return pd.DataFrame(columns=['feature', 'VIF'])

# --- Confidence Interval Configuration (Helper/Placeholder) ---

def get_ci_method_params(method_name: str):
    """
    Placeholder helper to return method-specific parameters for CI calculation.
    """
    # This can be expanded later if we wrap statsmodels conf_int logic centrally.
    return {"method": method_name}

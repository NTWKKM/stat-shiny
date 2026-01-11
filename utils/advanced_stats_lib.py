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
from typing import Union
from statsmodels.stats.outliers_influence import variance_inflation_factor
from logger import get_logger

logger = get_logger(__name__)

# --- Multiple Comparison Corrections (MCC) ---

def apply_mcc(p_values: Union[list, pd.Series, np.ndarray], method: str = 'fdr_bh', alpha: float = 0.05) -> pd.Series:
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
        
    except (ValueError, RuntimeError):
        logger.exception("Error applying MCC method '%s'", method)
        # Fallback: return original p-values if correction fails widely
        return pd.Series(p_vals_arr, index=p_values.index if isinstance(p_values, pd.Series) else None)


# --- Collinearity Diagnostics (VIF) ---

def calculate_vif(df: pd.DataFrame, *, intercept: bool = True) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing numerical features (predictors).
                           Categorical variables should be one-hot encoded beforehand 
                           or passed as design matrix.
        intercept (bool): Whether to add an intercept (constant) if not present.
                          VIF calculation requires an intercept for correct interpretation.

    Returns:
        pd.DataFrame: A DataFrame with columns ['feature', 'VIF'].
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['feature', 'VIF'])

    # Select only numeric columns and drop rows with NaNs (VIF requires complete numeric data)
    non_numeric_cols = df.columns.difference(df.select_dtypes(include=[np.number]).columns)
    if len(non_numeric_cols) > 0:
        logger.debug("VIF: Dropping %d non-numeric columns: %s", len(non_numeric_cols), list(non_numeric_cols))

    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    
    if len(df_numeric) < len(df):
        logger.debug("VIF: Dropped %d rows containing NaN values", len(df) - len(df_numeric))
    
    if df_numeric.empty:
        return pd.DataFrame(columns=['feature', 'VIF'])

    # Drop constant predictors (VIF undefined / can explode)
    variances = df_numeric.var()
    const_predictors = variances[variances == 0].index.tolist()
    if const_predictors:
        df_numeric = df_numeric.drop(columns=const_predictors, errors="ignore")

    if intercept:
        if "const" not in df_numeric.columns:
            df_numeric = df_numeric.copy()
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
        
    except (ValueError, np.linalg.LinAlgError):
        logger.exception("Error calculating VIF")
        return pd.DataFrame(columns=['feature', 'VIF'])

# --- Confidence Interval Configuration (Helper/Placeholder) ---

def determine_best_ci_method(
    n_samples: int,
    n_events: Union[int, None] = None,
    n_params: int = 1,
    model_type: str = 'logistic'
) -> str:
    """
    Determine optimal CI method based on data characteristics.
    
    Rules:
    1. If events per variable (EPV) < 10 -> Suggest 'profile' (or 'firth' contextually)
    2. If sample size < 50 -> Suggest 'profile'
    3. Otherwise -> 'wald'
    
    Args:
        n_samples: Total number of observations
        n_events: Number of events (for binary/survival), None for continuous
        n_params: Number of model parameters (coefficients)
        model_type: 'logistic', 'linear', 'cox'
        
    Returns:
        str: Recommended method ('wald', 'profile')
    """
    recommended = 'wald'
    
    if model_type in ['logistic', 'cox'] and n_events is not None:
        epv = n_events / max(1, n_params)
        if epv < 10:
            recommended = 'profile'
    
    if n_samples < 50:
        recommended = 'profile'
        
    return recommended

def get_ci_configuration(method: str, n_samples: int, n_events: int = 0, n_params: int = 1, model_type: str = 'logistic') -> dict[str, str]:
    """
    Get CI configuration parameters, resolving 'auto' mode.
    """
    selected_method = method
    note = ""
    
    if method == 'auto':
        selected_method = determine_best_ci_method(n_samples, n_events, n_params, model_type)
        note = f"Auto-selected {selected_method.title()} based on sample size/events."
        
    return {
        "method": selected_method,
        "note": note
    }


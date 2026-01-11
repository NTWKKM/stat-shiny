"""
Advanced Statistics Library for stat-shiny.

This module provides utility functions for:
1. Multiple Comparison Corrections (MCC)
2. Collinearity Diagnostics (VIF)
3. Confidence Interval Configuration (Helpers)
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from logger import get_logger

logger = get_logger(__name__)

# --- Multiple Comparison Corrections (MCC) ---

def apply_mcc(p_values: list[float] | pd.Series | np.ndarray, method: str = "fdr_bh", alpha: float = 0.05) -> pd.Series:
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
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    # Convert to numpy array for processing, ensuring numeric type and handling NaNs
    p_vals_arr = pd.to_numeric(p_values, errors='coerce')
    
    # Mask NaNs to avoid errors in multipletests
    mask = np.isfinite(p_vals_arr)
    p_vals_clean = p_vals_arr[mask]
    
    if len(p_vals_clean) == 0:
        return pd.Series(
            p_vals_arr,
            index=p_values.index if isinstance(p_values, pd.Series) else None,
        )  # Return original (all NaNs/empty)

    try:
        # returns: reject, pvals_corrected, alphacSidak, alphacBonf
        _, pvals_corrected, _, _ = smt.multipletests(p_vals_clean, alpha=alpha, method=method)
        
        # Reconstruct full series with NaNs in original positions
        result = np.full_like(p_vals_arr, np.nan, dtype=float)
        result[mask] = pvals_corrected
        
        return pd.Series(result, index=p_values.index if isinstance(p_values, pd.Series) else None)
        
    except (ValueError, RuntimeError, TypeError):
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
    const_predictors = variances[variances < 1e-10].index.tolist()
    if const_predictors:
        df_numeric = df_numeric.drop(columns=const_predictors, errors="ignore")

    if intercept:
        if "const" not in df_numeric.columns:
            df_numeric = df_numeric.copy()
            df_numeric["const"] = 1.0

    features = [c for c in df_numeric.columns if c != "const"]
    if not features:
        return pd.DataFrame(columns=["feature", "VIF"])
    
    try:
        vif_vals = []
        for col in features:
            i = df_numeric.columns.get_loc(col)
            vif_vals.append(variance_inflation_factor(df_numeric.values, i))
        vif_data = pd.DataFrame({"feature": features, "VIF": vif_vals})
        return vif_data.sort_values(by="VIF", ascending=False)
        
    except (ValueError, np.linalg.LinAlgError):
        logger.exception("Error calculating VIF")
        return pd.DataFrame(columns=['feature', 'VIF'])

# --- Confidence Interval Configuration (Helper/Placeholder) ---

def determine_best_ci_method(
    n_samples: int,
    n_events: int | None = None,
    n_params: int = 1,
    model_type: str = 'logistic'
) -> str:
    """
    Choose a preferred confidence-interval method ('wald' or 'profile') based on sample size, event count, and model type.
    
    When model_type is 'logistic' or 'cox' and n_events is provided, recommends 'profile' if events-per-variable (n_events / max(1, n_params)) is less than 10. Also recommends 'profile' if n_samples is less than 50. Otherwise recommends 'wald'.
    
    Parameters:
        n_samples (int): Total number of observations; must be non-negative.
        n_events (int | None): Number of events for binary or survival models; ignored for continuous models when None.
        n_params (int): Number of model parameters (coefficients); must be non-negative.
        model_type (str): One of 'logistic', 'linear', or 'cox' that indicates the model family.
    
    Returns:
        str: The recommended CI method, either 'wald' or 'profile'.
    """
    if n_samples < 0 or n_params < 0:
        raise ValueError("n_samples and n_params must be non-negative")
    if model_type not in {"logistic", "linear", "cox"}:
        raise ValueError(f"Unsupported model_type: {model_type}")
    recommended = "wald"
    
    if model_type in ['logistic', 'cox'] and n_events is not None:
        epv = n_events / max(1, n_params)
        if epv < 10:
            recommended = 'profile'
    
    if n_samples < 50:
        recommended = 'profile'
        
    return recommended

def get_ci_configuration(method: str, n_samples: int, n_events: int = 0, n_params: int = 1, model_type: str = "logistic") -> dict[str, str]:
    """
    Resolve the confidence-interval method and return configuration details, resolving "auto" to a concrete choice based on sample/events.
    
    Parameters:
        method (str): One of "auto", "wald", or "profile". If "auto", the function selects a method based on data characteristics.
        n_samples (int): Number of observations used to inform automatic selection.
        n_events (int): Number of events (for event-based models); used when applicable to compute events-per-variable.
        n_params (int): Number of model parameters (used to compute events-per-variable when relevant).
        model_type (str): Model family influencing selection logic (e.g., "logistic", "linear", "cox").
    
    Returns:
        dict: A dictionary with keys:
            "method" (str): The selected CI method ("wald" or "profile", or the provided method if not "auto").
            "note" (str): Informational note about auto-selection when applicable, otherwise an empty string.
    
    Raises:
        ValueError: If `method` is not one of "auto", "wald", or "profile".
    """
    if method not in {"auto", "wald", "profile"}:
        raise ValueError(f"Unsupported CI method: {method}")
    selected_method = method
    note = ""
    
    if method == 'auto':
        selected_method = determine_best_ci_method(n_samples, n_events, n_params, model_type)
        note = f"Auto-selected {selected_method.title()} based on sample size/events."
        
    return {
        "method": selected_method,
        "note": note
    }

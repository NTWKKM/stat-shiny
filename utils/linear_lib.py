"""  
📈 Linear Regression (OLS) Library for Continuous Outcome Analysis

Provides comprehensive OLS regression analysis with:
- Data preparation and cleaning
- Model fitting using statsmodels
- Assumption diagnostics (normality, homoscedasticity)
- Interactive Plotly visualizations
- Robust regression support (Huber M-estimator)
- Model comparison (AIC, BIC)
- Influence diagnostics (Cook's Distance, Leverage)

This module handles continuous outcomes (e.g., blood pressure, glucose, length of stay)
and returns β coefficients with full diagnostic support.

✅ Compatible with Python 3.9+
✅ Integrated with existing data_cleaning, formatting, and plotly_html_renderer utilities
"""

from __future__ import annotations

import html as _html
import warnings
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Statsmodels imports
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import (
    prepare_data_for_analysis,
)
from utils.collinearity_lib import calculate_vif as _run_vif
from utils.formatting import (
    create_missing_data_report_html,
    format_ci_html,
    format_p_value,
)

# Internal imports


logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

# Initialize color palette
COLORS = get_color_palette()


# ==============================================================================
# Type Definitions
# ==============================================================================


class OLSCoefficient(TypedDict):
    """Type definition for a single coefficient result."""

    variable: str
    coefficient: float
    std_error: float
    t_value: float
    p_value: float
    ci_lower: float
    ci_upper: float


class OLSResult(TypedDict):
    """Type definition for complete OLS regression results."""

    model: Any  # statsmodels RegressionResults
    formula: str
    n_obs: int
    df_resid: float
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float
    residual_std_err: float
    aic: float
    bic: float
    log_likelihood: float
    durbin_watson: float
    coef_table: pd.DataFrame
    vif_table: pd.DataFrame
    residuals: pd.Series
    fitted_values: pd.Series
    standardized_residuals: np.ndarray
    leverage: np.ndarray
    cooks_distance: np.ndarray


class DiagnosticResult(TypedDict):
    """Type definition for diagnostic test results."""

    test_name: str
    statistic: float
    p_value: float
    interpretation: str
    passed: bool


# ==============================================================================
# Data Preparation Functions
# ==============================================================================


def validate_ols_inputs(
    df: pd.DataFrame, outcome_col: str, predictor_cols: list[str]
) -> tuple[bool, str]:
    """
    Validate inputs for OLS regression.

    Parameters:
        df: Input DataFrame
        outcome_col: Name of the outcome (Y) variable
        predictor_cols: List of predictor (X) variable names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"

    if not outcome_col:
        return False, "Outcome column not specified"

    if outcome_col not in df.columns:
        return False, f"Outcome column '{outcome_col}' not found in DataFrame"

    if not predictor_cols:
        return False, "At least one predictor column is required"

    missing_cols = [c for c in predictor_cols if c not in df.columns]
    if missing_cols:
        return False, f"Predictor columns not found: {missing_cols}"

    if outcome_col in predictor_cols:
        return False, "Outcome column cannot be in predictor list"

    return True, ""


def prepare_data_for_ols(
    df: pd.DataFrame,
    outcome_col: str,
    predictor_cols: list[str],
    var_meta: dict[str, Any] | None = None,
    min_sample_size: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare data for OLS regression analysis.

    Includes:
    - Deep cleaning of numeric columns
    - Application of missing value codes
    - Complete case deletion
    - Sample size validation

    Parameters:
        df: Raw DataFrame
        outcome_col: Y variable name
        predictor_cols: List of X variable names
        var_meta: Variable metadata dictionary
        min_sample_size: Minimum required sample size

    Returns:
        Tuple of (cleaned_DataFrame, missing_data_info)

    Raises:
        ValueError: If insufficient sample size or invalid inputs
    """
    var_meta = var_meta or {}

    # Validate inputs
    is_valid, error_msg = validate_ols_inputs(df, outcome_col, predictor_cols)
    if not is_valid:
        raise ValueError(error_msg)

    # Step 1: Extract relevant columns
    cols_needed = [outcome_col] + list(predictor_cols)
    # Preserve order
    cols_needed = list(dict.fromkeys(cols_needed))

    # Step 2: Unified Pipeline
    # NOTE: prepare_data_for_analysis handles numeric cleaning and missing data internally
    try:
        # Identfy numeric columns to avoid wiping categorical data
        numeric_cols = [c for c in cols_needed if pd.api.types.is_numeric_dtype(df[c])]

        df_complete, info = prepare_data_for_analysis(
            df,
            required_cols=cols_needed,
            numeric_cols=numeric_cols,
            handle_missing="complete-case",
            var_meta=var_meta
        )
    except Exception as e:
        logger.error(f"Data preparation for OLS failed: {e}")
        # Return a structure that extract_regression_results can handle or just raise
        # For OLS, we usually return a dict with 'error'
        return {"error": f"Data preparation failed: {e}"} # type: ignore
    
    final_n = info["rows_analyzed"]
    excluded_n = info["rows_excluded"]
    initial_n = info["rows_original"]

    # Step 6: Validate sample size (Legacy Logic)
    if final_n < min_sample_size:
        raise ValueError(
            f"Insufficient sample size: {final_n} complete cases "
            f"(minimum {min_sample_size} required). "
            f"Excluded {excluded_n} rows with missing data."
        )

    # Check for constant outcome
    if df_complete[outcome_col].nunique() < 2:
        raise ValueError(
            f"Outcome variable '{outcome_col}' has no variance "
            f"(all values are identical). Cannot fit regression model."
        )

    # Check for constant predictors
    constant_predictors = [
        col for col in predictor_cols if col in df_complete.columns and df_complete[col].nunique() < 2
    ]
    if constant_predictors:
        logger.warning("Dropping constant predictor(s): %s", constant_predictors)
        df_complete = df_complete.drop(columns=constant_predictors, errors="ignore")
        # Note: we don't remove from predictor_cols list here as it might be used by caller,
        # but the DF is updated.

    # Build missing data info (Extend unified info with OLS specific fields)
    info["initial_n"] = initial_n # Alias for legacy compatibility
    info["constant_predictors_dropped"] = constant_predictors

    logger.info(
        "OLS data prepared: %d/%d rows retained (%.1f%%), %d excluded",
        final_n,
        initial_n,
        100 * final_n / initial_n if initial_n > 0 else 0,
        excluded_n,
    )

    return df_complete, info


# ==============================================================================
# Model Fitting Functions
# ==============================================================================


def run_ols_regression(
    df: pd.DataFrame,
    outcome_col: str,
    predictor_cols: list[str],
    robust_se: bool = False,
    cov_type: str = "nonrobust",
) -> OLSResult:
    """
    Execute OLS regression using statsmodels.

    Parameters:
        df: Cleaned DataFrame (no missing values)
        outcome_col: Y variable name
        predictor_cols: List of X variable names
        robust_se: Whether to use heteroscedasticity-robust standard errors
        cov_type: Covariance type ('nonrobust', 'HC0', 'HC1', 'HC2', 'HC3')

    Returns:
        OLSResult dictionary with model, coefficients, diagnostics.
        Returns a dictionary with an 'error' key on failure.
    """
    # Robustness check
    is_valid, msg = validate_ols_inputs(df, outcome_col, predictor_cols)
    if not is_valid:
        return {"error": msg}  # type: ignore

    # Step 1: Build formula with Q() for variable names with spaces/special chars
    def safe_name(name: str) -> str:
        # Check if variable name needs quoting
        if not name.isidentifier() or " " in name or "-" in name:
            return f"Q('{name}')"
        return name

    predictors_str = " + ".join([safe_name(col) for col in predictor_cols])
    formula = f"{safe_name(outcome_col)} ~ {predictors_str}"

    logger.debug("OLS formula: %s", formula)

    # Step 2: Fit OLS model
    try:
        model = smf.ols(formula, data=df).fit(cov_type="HC3" if robust_se else cov_type)
    except Exception as e:
        logger.exception("Error fitting OLS model")
        raise ValueError(f"Failed to fit OLS model: {e}") from e

    # Step 3: Extract coefficients table
    coef_df = extract_coefficients(model)

    # Step 4: Calculate VIF for multicollinearity
    vif_table = calculate_vif_for_ols(df, predictor_cols)

    # Step 5: Calculate influence diagnostics
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag

    # Step 6: Durbin-Watson test for autocorrelation
    dw_stat = durbin_watson(model.resid)

    # Step 7: Build results dictionary
    results: OLSResult = {
        "model": model,
        "formula": formula,
        "n_obs": int(model.nobs),
        "df_resid": model.df_resid,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue if np.isfinite(model.fvalue) else float("nan"),
        "f_pvalue": model.f_pvalue if np.isfinite(model.f_pvalue) else float("nan"),
        "residual_std_err": np.sqrt(model.mse_resid),
        "aic": model.aic,
        "bic": model.bic,
        "log_likelihood": model.llf,
        "durbin_watson": dw_stat,
        "coef_table": coef_df,
        "vif_table": vif_table,
        "residuals": model.resid,
        "fitted_values": model.fittedvalues,
        "standardized_residuals": influence.resid_studentized_internal,
        "leverage": leverage,
        "cooks_distance": cooks_d,
    }

    return results


def run_robust_regression(
    df: pd.DataFrame,
    outcome_col: str,
    predictor_cols: list[str],
    m_estimator: str = "huber",
) -> OLSResult:
    """
    Execute Robust Regression using M-estimators (Huber or Bisquare).

    Robust regression is less sensitive to outliers than OLS.

    Parameters:
        df: Cleaned DataFrame
        outcome_col: Y variable name
        predictor_cols: List of X variable names
        m_estimator: 'huber' (default) or 'bisquare'

    Returns:
        OLSResult dictionary with model, coefficients, diagnostics
    """
    # Prepare design matrix
    X = sm.add_constant(df[predictor_cols])
    y = df[outcome_col]

    # Choose M-estimator
    if m_estimator == "bisquare":
        norm = sm.robust.norms.TukeyBiweight()
    else:
        norm = sm.robust.norms.HuberT()

    # Fit robust regression
    try:
        model = sm.RLM(y, X, M=norm).fit()
    except Exception as e:
        logger.exception("Error fitting robust regression")
        raise ValueError(f"Failed to fit robust regression: {e}") from e

    # Extract coefficients
    coef_df = pd.DataFrame(
        {
            "Variable": model.params.index.tolist(),
            "Coefficient": model.params.values,
            "Std. Error": model.bse.values,
            "T-value": model.tvalues.values,
            "P-value": model.pvalues.values,
            "CI_Lower": model.params.values - 1.96 * model.bse.values,
            "CI_Upper": model.params.values + 1.96 * model.bse.values,
        }
    )

    # Calculate VIF
    vif_table = calculate_vif_for_ols(df, predictor_cols)

    # Robust regression doesn't have all the same attributes as OLS
    results: OLSResult = {
        "model": model,
        "formula": f"{outcome_col} ~ {' + '.join(predictor_cols)}",
        "n_obs": int(model.nobs),
        "df_resid": model.df_resid,
        "r_squared": float("nan"),  # R² not well-defined for RLM
        "adj_r_squared": float("nan"),
        "f_statistic": float("nan"),
        "f_pvalue": float("nan"),
        "residual_std_err": float(model.scale),
        "aic": float("nan"),
        "bic": float("nan"),
        "log_likelihood": float("nan"),
        "durbin_watson": durbin_watson(model.resid),
        "coef_table": coef_df,
        "vif_table": vif_table,
        "residuals": model.resid,
        "fitted_values": model.fittedvalues,
        "standardized_residuals": model.resid / model.scale,
        "leverage": np.zeros(len(model.resid)),  # Not computed for RLM
        "cooks_distance": np.zeros(len(model.resid)),  # Not computed for RLM
    }

    return results


def extract_coefficients(model) -> pd.DataFrame:
    """
    Extract and format coefficient table from statsmodels regression model.

    Parameters:
        model: Fitted statsmodels RegressionResults

    Returns:
        DataFrame with columns: Variable, Coefficient, Std. Error, T-value, P-value, CI_Lower, CI_Upper
    """
    conf_int = model.conf_int()

    coef_df = pd.DataFrame(
        {
            "Variable": model.params.index.tolist(),
            "Coefficient": model.params.values,
            "Std. Error": model.bse.values,
            "T-value": model.tvalues.values,
            "P-value": model.pvalues.values,
            "CI_Lower": conf_int.iloc[:, 0].values,
            "CI_Upper": conf_int.iloc[:, 1].values,
        }
    )

    return coef_df


def calculate_vif_for_ols(df: pd.DataFrame, predictor_cols: list[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) using centralized collinearity_lib.
    
    Parameters:
        df: DataFrame with predictor columns
        predictor_cols: List of predictor column names
        
    Returns:
        DataFrame with VIF results
    """
    # Note: _run_vif now returns DataFrame only (not tuple)
    vif_df = _run_vif(df, predictor_cols)
    return vif_df
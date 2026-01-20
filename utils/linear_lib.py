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
    apply_missing_values_to_df,
    clean_numeric_vector,
    get_missing_summary_df,
)
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
    unique_cols = list(dict.fromkeys(cols_needed))  # Preserve order, remove duplicates
    df_subset = df[unique_cols].copy()

    initial_n = len(df_subset)

    # Step 2: Get missing summary BEFORE cleaning (for accurate breakdown)
    missing_summary = get_missing_summary_df(df_subset, var_meta)

    # Step 3: Apply missing value codes
    df_cleaned = apply_missing_values_to_df(df_subset, var_meta)

    # Step 4: Deep clean numeric data
    for col in df_cleaned.columns:
        df_cleaned[col] = clean_numeric_vector(df_cleaned[col])

    # Step 5: Complete case deletion
    df_complete = df_cleaned.dropna()
    final_n = len(df_complete)
    excluded_n = initial_n - final_n

    # Step 6: Validate sample size
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
        col for col in predictor_cols if df_complete[col].nunique() < 2
    ]
    if constant_predictors:
        logger.warning("Dropping constant predictor(s): %s", constant_predictors)
        df_complete = df_complete.drop(columns=constant_predictors, errors="ignore")
        predictor_cols = [c for c in predictor_cols if c not in constant_predictors]

    # Build missing data info
    missing_data_info = {
        "strategy": "complete-case",
        "rows_analyzed": final_n,
        "rows_excluded": excluded_n,
        "initial_n": initial_n,
        "summary_before": (
            missing_summary.to_dict("records") if not missing_summary.empty else []
        ),
        "constant_predictors_dropped": constant_predictors,
    }

    logger.info(
        "OLS data prepared: %d/%d rows retained (%.1f%%), %d excluded",
        final_n,
        initial_n,
        100 * final_n / initial_n if initial_n > 0 else 0,
        excluded_n,
    )

    return df_complete, missing_data_info


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
        OLSResult dictionary with model, coefficients, diagnostics
    """

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
    Calculate Variance Inflation Factor (VIF) for each predictor.

    VIF > 5-10 indicates multicollinearity.

    Parameters:
        df: DataFrame with predictor columns
        predictor_cols: List of predictor column names

    Returns:
        DataFrame with columns: Variable, VIF, Interpretation
    """
    if len(predictor_cols) < 2:
        return pd.DataFrame(
            {
                "Variable": predictor_cols,
                "VIF": [1.0] * len(predictor_cols),
                "Interpretation": ["OK (single predictor)"] * len(predictor_cols),
            }
        )

    try:
        # Add constant and prepare numeric matrix
        X = df[predictor_cols].copy()
        X = sm.add_constant(X)

        vif_data = []
        for i, col in enumerate(predictor_cols):
            col_idx = list(X.columns).index(col)
            vif = variance_inflation_factor(X.values, col_idx)

            # Handle infinite or very large VIF
            if not np.isfinite(vif):
                vif = float("inf")

            # Interpretation
            if vif < 5:
                interpretation = "✅ OK"
            elif vif < 10:
                interpretation = "⚠️ Moderate"
            else:
                interpretation = "🔴 High"

            vif_data.append(
                {"Variable": col, "VIF": vif, "Interpretation": interpretation}
            )

        return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

    except Exception as e:
        logger.warning("Error calculating VIF: %s", e)
        return pd.DataFrame(
            {
                "Variable": predictor_cols,
                "VIF": [float("nan")] * len(predictor_cols),
                "Interpretation": ["Error calculating"] * len(predictor_cols),
            }
        )


# ==============================================================================
# Stepwise Variable Selection
# ==============================================================================


def stepwise_selection(
    df: pd.DataFrame,
    outcome_col: str,
    candidate_cols: list[str],
    direction: Literal["forward", "backward", "both"] = "both",
    criterion: Literal["aic", "bic", "pvalue"] = "aic",
    p_enter: float = 0.05,
    p_remove: float = 0.10,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """
    Perform stepwise variable selection for OLS regression.

    Parameters:
        df: DataFrame with outcome and predictor columns
        outcome_col: Name of the outcome variable
        candidate_cols: List of potential predictor columns
        direction: 'forward' (add vars), 'backward' (remove vars), 'both' (stepwise)
        criterion: 'aic', 'bic' for information criteria, 'pvalue' for significance-based
        p_enter: P-value threshold for adding variables (pvalue criterion)
        p_remove: P-value threshold for removing variables (pvalue criterion)
        max_iterations: Maximum number of selection iterations

    Returns:
        Dictionary with selected variables, history, and final model
    """
    logger.info(
        "Starting stepwise selection: direction=%s, criterion=%s", direction, criterion
    )

    # Initialize
    history = []
    current_vars = [] if direction == "forward" else list(candidate_cols)
    remaining_vars = list(candidate_cols) if direction == "forward" else []

    def get_criterion_value(model, crit: str) -> float:
        """Get criterion value for model comparison."""
        if crit == "aic":
            return model.aic
        elif crit == "bic":
            return model.bic
        else:  # pvalue - return negative log-likelihood (lower is better)
            return -model.llf

    def fit_model(var_list: list[str]):
        """Fit OLS model with given variables."""
        if not var_list:
            # Null model with intercept only
            formula = f"Q('{outcome_col}') ~ 1"
        else:
            predictors = " + ".join(
                [f"Q('{v}')" if not v.isidentifier() else v for v in var_list]
            )
            formula = f"Q('{outcome_col}') ~ {predictors}"
        try:
            return smf.ols(formula, data=df).fit()
        except Exception as e:
            logger.warning("Model fitting failed: %s", e)
            return None

    # Get initial model
    current_model = fit_model(current_vars)
    if current_model is None:
        return {
            "selected_vars": [],
            "history": [],
            "error": "Initial model fitting failed",
        }

    current_criterion = get_criterion_value(current_model, criterion)

    history.append(
        {
            "iteration": 0,
            "action": "initial",
            "variable": None,
            "variables": list(current_vars),
            "criterion": current_criterion,
            "n_vars": len(current_vars),
        }
    )

    for iteration in range(1, max_iterations + 1):
        improved = False

        # FORWARD STEP: Try adding variables
        if direction in ["forward", "both"] and remaining_vars:
            best_add_var = None
            best_add_criterion = current_criterion

            for var in remaining_vars:
                test_vars = current_vars + [var]
                test_model = fit_model(test_vars)

                if test_model is not None:
                    test_criterion = get_criterion_value(test_model, criterion)

                    # For pvalue criterion, check if variable is significant
                    if criterion == "pvalue":
                        try:
                            # Find p-value for new variable
                            var_pvalue = test_model.pvalues.get(var, 1.0)
                            if var_pvalue >= p_enter:
                                continue
                        except Exception:
                            continue

                    if test_criterion < best_add_criterion:
                        best_add_criterion = test_criterion
                        best_add_var = var

            if best_add_var is not None:
                current_vars.append(best_add_var)
                remaining_vars.remove(best_add_var)
                current_criterion = best_add_criterion
                current_model = fit_model(current_vars)
                improved = True

                history.append(
                    {
                        "iteration": iteration,
                        "action": "add",
                        "variable": best_add_var,
                        "variables": list(current_vars),
                        "criterion": current_criterion,
                        "n_vars": len(current_vars),
                    }
                )

        # BACKWARD STEP: Try removing variables
        if direction in ["backward", "both"] and len(current_vars) > 0:
            best_remove_var = None
            best_remove_criterion = current_criterion

            for var in current_vars:
                test_vars = [v for v in current_vars if v != var]
                test_model = fit_model(test_vars)

                if test_model is not None:
                    test_criterion = get_criterion_value(test_model, criterion)

                    # For pvalue criterion, check if variable is non-significant
                    if criterion == "pvalue":
                        try:
                            var_pvalue = current_model.pvalues.get(var, 0.0)
                            if var_pvalue <= p_remove:
                                continue
                        except Exception:
                            continue

                    if test_criterion < best_remove_criterion:
                        best_remove_criterion = test_criterion
                        best_remove_var = var

            if best_remove_var is not None:
                current_vars.remove(best_remove_var)
                remaining_vars.append(best_remove_var)
                current_criterion = best_remove_criterion
                current_model = fit_model(current_vars)
                improved = True

                history.append(
                    {
                        "iteration": iteration,
                        "action": "remove",
                        "variable": best_remove_var,
                        "variables": list(current_vars),
                        "criterion": current_criterion,
                        "n_vars": len(current_vars),
                    }
                )

        if not improved:
            break

    logger.info("Stepwise selection complete: %d variables selected", len(current_vars))

    return {
        "selected_vars": current_vars,
        "excluded_vars": remaining_vars,
        "history": history,
        "final_model": current_model,
        "final_criterion": current_criterion,
        "n_iterations": len(history) - 1,
        "direction": direction,
        "criterion_used": criterion,
    }


def format_stepwise_history(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Format stepwise selection history as a DataFrame."""
    if not history:
        return pd.DataFrame()

    rows = []
    for step in history:
        action_emoji = {"initial": "🏁", "add": "➕", "remove": "➖"}.get(
            step["action"], "•"
        )
        rows.append(
            {
                "Step": step["iteration"],
                "Action": f"{action_emoji} {step['action'].title()}",
                "Variable": step.get("variable", "-"),
                "Criterion": (
                    f"{step['criterion']:.2f}"
                    if np.isfinite(step["criterion"])
                    else "N/A"
                ),
                "Num Vars": step["n_vars"],
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# Model Comparison
# ==============================================================================


def compare_models(
    df: pd.DataFrame, outcome_col: str, model_specs: list[dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare multiple OLS models using AIC, BIC, and adjusted R².

    Parameters:
        df: DataFrame with data
        outcome_col: Name of outcome variable
        model_specs: List of dicts with 'name' and 'predictors' keys
            Example: [
                {'name': 'Model 1', 'predictors': ['Age', 'BMI']},
                {'name': 'Model 2', 'predictors': ['Age', 'BMI', 'Cholesterol']}
            ]

    Returns:
        DataFrame with model comparison metrics
    """
    logger.info("Comparing %d models", len(model_specs))

    results = []

    for spec in model_specs:
        name = spec.get("name", "Unnamed")
        predictors = spec.get("predictors", [])

        try:
            # Build formula
            if predictors:
                pred_str = " + ".join(
                    [f"Q('{v}')" if not v.isidentifier() else v for v in predictors]
                )
                formula = f"Q('{outcome_col}') ~ {pred_str}"
            else:
                formula = f"Q('{outcome_col}') ~ 1"

            # Fit model
            model = smf.ols(formula, data=df).fit()

            # Calculate likelihood ratio statistic vs null model



            results.append(
                {
                    "Model": name,
                    "Predictors": len(predictors),
                    "N": int(model.nobs),
                    "R²": model.rsquared,
                    "Adj R²": model.rsquared_adj,
                    "AIC": model.aic,
                    "BIC": model.bic,
                    "Log-Lik": model.llf,
                    "F-stat": (
                        model.fvalue if np.isfinite(model.fvalue) else float("nan")
                    ),
                    "F p-value": (
                        model.f_pvalue if np.isfinite(model.f_pvalue) else float("nan")
                    ),
                    "RSE": np.sqrt(model.mse_resid),
                    "_model": model,  # For internal use
                }
            )

        except Exception as e:
            logger.warning("Model '%s' failed: %s", name, e)
            results.append(
                {
                    "Model": name,
                    "Predictors": len(predictors),
                    "N": 0,
                    "R²": float("nan"),
                    "Adj R²": float("nan"),
                    "AIC": float("nan"),
                    "BIC": float("nan"),
                    "Log-Lik": float("nan"),
                    "F-stat": float("nan"),
                    "F p-value": float("nan"),
                    "RSE": float("nan"),
                    "_model": None,
                }
            )

    comparison_df = pd.DataFrame(results)

    # Add rankings
    if len(comparison_df) > 1:
        comparison_df["AIC Rank"] = comparison_df["AIC"].rank()
        comparison_df["BIC Rank"] = comparison_df["BIC"].rank()
        comparison_df["Best Model"] = comparison_df["AIC"] == comparison_df["AIC"].min()

    logger.info(
        "Model comparison complete. Best AIC: %s",
        (
            comparison_df.loc[comparison_df["AIC"].idxmin(), "Model"]
            if not comparison_df.empty
            else "N/A"
        ),
    )

    return comparison_df


def format_model_comparison(comparison_df: pd.DataFrame) -> str:
    """Format model comparison results as HTML table."""
    display_cols = ["Model", "Predictors", "N", "R²", "Adj R²", "AIC", "BIC", "RSE"]
    df_display = comparison_df[
        [c for c in display_cols if c in comparison_df.columns]
    ].copy()

    # Format numeric columns
    for col in ["R²", "Adj R²"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.4f}" if np.isfinite(x) else "N/A"
            )
    for col in ["AIC", "BIC", "RSE"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.2f}" if np.isfinite(x) else "N/A"
            )

    # Highlight best model
    # Will be handled in HTML styling

    return df_display.to_html(
        index=False, escape=False, classes="table table-striped table-hover", border=0
    )


# ==============================================================================
# Bootstrap Confidence Intervals
# ==============================================================================


def bootstrap_ols(
    df: pd.DataFrame,
    outcome_col: str,
    predictor_cols: list[str],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for OLS regression coefficients.

    Uses case resampling (resample observations with replacement).

    Parameters:
        df: DataFrame with outcome and predictors
        outcome_col: Name of outcome variable
        predictor_cols: List of predictor column names
        n_bootstrap: Number of bootstrap samples (default: 1000)
        ci_level: Confidence level (default: 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with bootstrap estimates, CIs, and diagnostics
    """
    logger.info(
        "Starting bootstrap OLS: n_bootstrap=%d, ci_level=%.2f", n_bootstrap, ci_level
    )

    if random_state is not None:
        np.random.seed(random_state)

    n = len(df)
    alpha = 1 - ci_level

    # Fit original model
    def safe_name(name: str) -> str:
        if not name.isidentifier() or " " in name or "-" in name:
            return f"Q('{name}')"
        return name

    predictors_str = " + ".join([safe_name(col) for col in predictor_cols])
    formula = f"{safe_name(outcome_col)} ~ {predictors_str}"

    try:
        original_model = smf.ols(formula, data=df).fit()
    except Exception as e:
        logger.error("Original model fitting failed: %s", e)
        return {"error": str(e)}

    original_params = original_model.params.copy()
    param_names = original_params.index.tolist()

    # Bootstrap resampling
    bootstrap_params = []
    failed_samples = 0

    for i in range(n_bootstrap):
        # Resample with replacement
        sample_idx = np.random.choice(n, size=n, replace=True)
        df_sample = df.iloc[sample_idx]

        try:
            boot_model = smf.ols(formula, data=df_sample).fit()
            bootstrap_params.append(boot_model.params.values)
        except Exception:
            failed_samples += 1
            continue

    if len(bootstrap_params) < n_bootstrap * 0.5:
        logger.warning(
            "Too many failed bootstrap samples: %d/%d", failed_samples, n_bootstrap
        )
        return {"error": f"Too many failed samples ({failed_samples}/{n_bootstrap})"}

    bootstrap_array = np.array(bootstrap_params)

    # Calculate statistics
    boot_mean = np.mean(bootstrap_array, axis=0)
    boot_std = np.std(bootstrap_array, axis=0)
    boot_bias = boot_mean - original_params.values

    # Percentile CI
    ci_lower_pct = np.percentile(bootstrap_array, alpha / 2 * 100, axis=0)
    ci_upper_pct = np.percentile(bootstrap_array, (1 - alpha / 2) * 100, axis=0)

    # BCa (Bias-Corrected and Accelerated) CI - simplified version
    # Standard normal quantiles
    z_alpha_2 = stats.norm.ppf(alpha / 2)
    z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

    # Bias correction factor
    z0 = stats.norm.ppf(np.mean(bootstrap_array < original_params.values, axis=0))

    # BCa bounds (simplified, using z0 only)
    bca_lower_q = stats.norm.cdf(2 * z0 + z_alpha_2)
    bca_upper_q = stats.norm.cdf(2 * z0 + z_1_alpha_2)

    ci_lower_bca = np.array(
        [
            np.percentile(bootstrap_array[:, i], q * 100)
            for i, q in enumerate(bca_lower_q)
        ]
    )
    ci_upper_bca = np.array(
        [
            np.percentile(bootstrap_array[:, i], q * 100)
            for i, q in enumerate(bca_upper_q)
        ]
    )

    # Build results DataFrame
    results_df = pd.DataFrame(
        {
            "Variable": param_names,
            "Estimate": original_params.values,
            "Boot Mean": boot_mean,
            "Boot SE": boot_std,
            "Bias": boot_bias,
            "CI Lower (Pct)": ci_lower_pct,
            "CI Upper (Pct)": ci_upper_pct,
            "CI Lower (BCa)": ci_lower_bca,
            "CI Upper (BCa)": ci_upper_bca,
        }
    )

    logger.info(
        "Bootstrap complete: %d successful samples, %d failed",
        len(bootstrap_params),
        failed_samples,
    )

    return {
        "results": results_df,
        "bootstrap_samples": bootstrap_array,
        "n_bootstrap": len(bootstrap_params),
        "failed_samples": failed_samples,
        "ci_level": ci_level,
        "original_model": original_model,
        "param_names": param_names,
    }


def format_bootstrap_results(
    boot_results: dict[str, Any], ci_method: Literal["percentile", "bca"] = "percentile"
) -> pd.DataFrame:
    """
    Format bootstrap results for display.

    Parameters:
        boot_results: Results from bootstrap_ols()
        ci_method: 'percentile' or 'bca' for CI type

    Returns:
        DataFrame with formatted bootstrap results
    """
    if "error" in boot_results:
        return pd.DataFrame({"Error": [boot_results["error"]]})

    df = boot_results["results"].copy()
    ci_level = boot_results["ci_level"]
    ci_pct = int(ci_level * 100)

    # Choose CI columns
    if ci_method == "bca":
        ci_lower_col = "CI Lower (BCa)"
        ci_upper_col = "CI Upper (BCa)"
    else:
        ci_lower_col = "CI Lower (Pct)"
        ci_upper_col = "CI Upper (Pct)"

    # Create formatted DataFrame
    formatted = pd.DataFrame(
        {
            "Variable": df["Variable"],
            "Estimate": df["Estimate"].apply(lambda x: f"{x:.4f}"),
            "Boot SE": df["Boot SE"].apply(lambda x: f"{x:.4f}"),
            "Bias": df["Bias"].apply(lambda x: f"{x:.4f}"),
            f"{ci_pct}% CI": df.apply(
                lambda r: f"[{r[ci_lower_col]:.4f}, {r[ci_upper_col]:.4f}]", axis=1
            ),
        }
    )

    return formatted


def create_bootstrap_distribution_plot(
    boot_results: dict[str, Any], variable_idx: int = 0
) -> go.Figure:
    """
    Create histogram of bootstrap distribution for a coefficient.

    Parameters:
        boot_results: Results from bootstrap_ols()
        variable_idx: Index of coefficient to plot (0 = intercept, 1 = first predictor, etc.)

    Returns:
        Plotly Figure
    """
    if "error" in boot_results:
        return go.Figure(layout={})

    samples = boot_results["bootstrap_samples"][:, variable_idx]
    var_name = boot_results["param_names"][variable_idx]
    original_est = boot_results["original_model"].params.iloc[variable_idx]
    ci_level = boot_results["ci_level"]

    # Calculate CI bounds
    alpha = 1 - ci_level
    ci_lower = np.percentile(samples, alpha / 2 * 100)
    ci_upper = np.percentile(samples, (1 - alpha / 2) * 100)

    fig = go.Figure(layout={})

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=samples,
            nbinsx=50,
            name="Bootstrap Distribution",
            marker_color=COLORS.get("primary", "#3B82F6"),
            opacity=0.7,
        )
    )

    # Original estimate line
    fig.add_vline(
        x=original_est,
        line_dash="solid",
        line_color=COLORS.get("danger", "#EF4444"),
        line_width=2,
        annotation_text=f"Estimate: {original_est:.4f}",
        annotation_position="top",
    )

    # CI bounds
    fig.add_vline(
        x=ci_lower,
        line_dash="dash",
        line_color=COLORS.get("warning", "#F59E0B"),
        line_width=1.5,
        annotation_text=f"Lower: {ci_lower:.4f}",
        annotation_position="bottom left",
    )
    fig.add_vline(
        x=ci_upper,
        line_dash="dash",
        line_color=COLORS.get("warning", "#F59E0B"),
        line_width=1.5,
        annotation_text=f"Upper: {ci_upper:.4f}",
        annotation_position="bottom right",
    )

    # Clean variable name for display
    clean_name = var_name.replace("Q('", "").replace("')", "")

    fig.update_layout(
        title=dict(
            text=f"<b>Bootstrap Distribution: {_html.escape(clean_name)}</b>",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Coefficient Value",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig


# ==============================================================================
# Diagnostic Tests
# ==============================================================================


def run_diagnostic_tests(results: OLSResult) -> list[DiagnosticResult]:
    """
    Run comprehensive diagnostic tests on OLS results.

    Includes:
    - Shapiro-Wilk test for normality of residuals
    - Breusch-Pagan test for heteroscedasticity
    - Durbin-Watson test for autocorrelation

    Parameters:
        results: OLSResult from run_ols_regression()

    Returns:
        List of DiagnosticResult dictionaries
    """
    diagnostics: list[DiagnosticResult] = []
    residuals = results["residuals"]

    # 1. Shapiro-Wilk test for normality
    # Use sample if n > 5000 (Shapiro-Wilk limit)
    resid_sample = residuals.sample(min(5000, len(residuals)), random_state=42).values
    try:
        shapiro_stat, shapiro_p = stats.shapiro(resid_sample)
        diagnostics.append(
            {
                "test_name": "Shapiro-Wilk (Normality)",
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "interpretation": (
                    "Residuals appear normal"
                    if shapiro_p > 0.05
                    else "Residuals may not be normal"
                ),
                "passed": shapiro_p > 0.05,
            }
        )
    except Exception as e:
        logger.warning("Shapiro-Wilk test failed: %s", e)

    # 2. Breusch-Pagan test for heteroscedasticity
    try:
        model = results["model"]
        from statsmodels.stats.diagnostic import het_breuschpagan

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
            model.resid, model.model.exog
        )
        diagnostics.append(
            {
                "test_name": "Breusch-Pagan (Homoscedasticity)",
                "statistic": lm_stat,
                "p_value": lm_pvalue,
                "interpretation": (
                    "Constant variance (homoscedasticity)"
                    if lm_pvalue > 0.05
                    else "Non-constant variance (heteroscedasticity)"
                ),
                "passed": lm_pvalue > 0.05,
            }
        )
    except Exception as e:
        logger.warning("Breusch-Pagan test failed: %s", e)

    # 3. Durbin-Watson test for autocorrelation
    dw = results["durbin_watson"]
    dw_interpretation = (
        "No autocorrelation"
        if 1.5 <= dw <= 2.5
        else "Positive autocorrelation"
        if dw < 1.5
        else "Negative autocorrelation"
    )
    diagnostics.append(
        {
            "test_name": "Durbin-Watson (Autocorrelation)",
            "statistic": dw,
            "p_value": float("nan"),  # DW doesn't have a standard p-value
            "interpretation": dw_interpretation,
            "passed": 1.5 <= dw <= 2.5,
        }
    )

    # 4. Jarque-Bera test (additional normality test)
    try:
        jb_result = stats.jarque_bera(residuals)
        # Handle both scipy old (4 values) and new (2 values or JarqueBera result object)
        if hasattr(jb_result, "statistic"):
            jb_stat = jb_result.statistic
            jb_p = jb_result.pvalue
        elif len(jb_result) == 4:
            jb_stat, jb_p, _, _ = jb_result
        else:
            jb_stat, jb_p = jb_result
        diagnostics.append(
            {
                "test_name": "Jarque-Bera (Normality)",
                "statistic": jb_stat,
                "p_value": jb_p,
                "interpretation": (
                    "Normal distribution" if jb_p > 0.05 else "Non-normal distribution"
                ),
                "passed": jb_p > 0.05,
            }
        )
    except Exception as e:
        logger.warning("Jarque-Bera test failed: %s", e)

    return diagnostics


# ==============================================================================
# Visualization Functions
# ==============================================================================


def create_diagnostic_plots(results: OLSResult) -> dict[str, go.Figure]:
    """
    Create Plotly diagnostic plots for OLS assumption checking.

    Returns:
        Dictionary with plot keys:
        - 'residuals_vs_fitted': Homoscedasticity check
        - 'qq_plot': Normality check
        - 'scale_location': Variance homogeneity
        - 'residuals_vs_leverage': Influence diagnostics
    """
    plots = {}

    residuals = results["residuals"]
    fitted = results["fitted_values"]
    std_resid = results["standardized_residuals"]
    leverage = results["leverage"]
    cooks_d = results["cooks_distance"]

    # Plot 1: Residuals vs Fitted (Homoscedasticity check)
    plots["residuals_vs_fitted"] = create_residuals_vs_fitted_plot(fitted, residuals)

    # Plot 2: QQ Plot (Normality check)
    plots["qq_plot"] = create_qq_plot(residuals)

    # Plot 3: Scale-Location Plot (Variance homogeneity)
    plots["scale_location"] = create_scale_location_plot(fitted, std_resid)

    # Plot 4: Residuals vs Leverage (Influence diagnostics)
    plots["residuals_vs_leverage"] = create_residuals_vs_leverage_plot(
        leverage, std_resid, cooks_d
    )

    return plots


def create_residuals_vs_fitted_plot(
    fitted_values: pd.Series, residuals: pd.Series
) -> go.Figure:
    """
    Create Residuals vs Fitted Values plot.

    Checks for:
    - Random scatter (good - linear relationship, homoscedasticity)
    - Patterns (bad - nonlinearity, heteroscedasticity)
    """
    fig = go.Figure(layout={})

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=fitted_values,
            y=residuals,
            mode="markers",
            marker=dict(
                color=COLORS.get("primary", "#3B82F6"),
                size=6,
                opacity=0.6,
                line=dict(color="white", width=0.5),
            ),
            name="Residuals",
            hovertemplate="<b>Fitted:</b> %{x:.3f}<br><b>Residual:</b> %{y:.3f}<extra></extra>",
        )
    )

    # Add horizontal reference line at y=0
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS.get("danger", "#EF4444"),
        opacity=0.7,
        annotation_text="Reference",
        annotation_position="right",
    )

    # Add LOWESS trend line
    try:
        sorted_idx = np.argsort(fitted_values.values)
        window = max(5, len(fitted_values) // 20)
        rolling_mean = (
            pd.Series(residuals.values[sorted_idx])
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )

        fig.add_trace(
            go.Scatter(
                x=fitted_values.values[sorted_idx],
                y=rolling_mean.values,
                mode="lines",
                line=dict(color=COLORS.get("danger", "#EF4444"), width=2),
                name="Trend",
                hoverinfo="skip",
            )
        )
    except Exception:
        pass  # Skip trend line if error

    fig.update_layout(
        title=dict(text="<b>Residuals vs Fitted Values</b>", x=0.5, xanchor="center"),
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
        template="plotly_white",
        hovermode="closest",
        height=450,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_qq_plot(residuals: pd.Series) -> go.Figure:
    """
    Create Q-Q Plot for normality assessment.

    Points close to the reference line indicate normally distributed residuals.
    """
    fig = go.Figure(layout={})

    # Calculate theoretical quantiles
    quantiles = stats.probplot(residuals.values)
    theoretical_quantiles = quantiles[0][0]
    sample_quantiles = quantiles[0][1]

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode="markers",
            marker=dict(
                color=COLORS.get("primary", "#3B82F6"),
                size=6,
                opacity=0.7,
                line=dict(color="white", width=0.5),
            ),
            name="Residuals",
            hovertemplate="<b>Theoretical:</b> %{x:.3f}<br><b>Sample:</b> %{y:.3f}<extra></extra>",
        )
    )

    # Add reference line (45-degree line)
    line_min = min(theoretical_quantiles.min(), sample_quantiles.min())
    line_max = max(theoretical_quantiles.max(), sample_quantiles.max())

    fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            line=dict(color=COLORS.get("danger", "#EF4444"), width=2, dash="dash"),
            name="Normal Reference",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Normal Q-Q Plot</b>", x=0.5, xanchor="center"),
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_white",
        hovermode="closest",
        height=450,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_scale_location_plot(
    fitted_values: pd.Series, standardized_residuals: np.ndarray
) -> go.Figure:
    """
    Create Scale-Location plot (√|Standardized Residuals| vs Fitted).

    Checks for homoscedasticity (constant variance).
    A horizontal trend line indicates constant variance.
    """
    fig = go.Figure(layout={})

    # Calculate sqrt of absolute standardized residuals
    sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=fitted_values.values,
            y=sqrt_abs_resid,
            mode="markers",
            marker=dict(
                color=COLORS.get("primary", "#3B82F6"),
                size=6,
                opacity=0.6,
                line=dict(color="white", width=0.5),
            ),
            name="√|Std. Residuals|",
            hovertemplate="<b>Fitted:</b> %{x:.3f}<br><b>√|Std. Resid|:</b> %{y:.3f}<extra></extra>",
        )
    )

    # Add trend line
    try:
        sorted_idx = np.argsort(fitted_values.values)
        window = max(5, len(fitted_values) // 20)
        rolling_mean = (
            pd.Series(sqrt_abs_resid[sorted_idx])
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )

        fig.add_trace(
            go.Scatter(
                x=fitted_values.values[sorted_idx],
                y=rolling_mean.values,
                mode="lines",
                line=dict(color=COLORS.get("danger", "#EF4444"), width=2),
                name="Trend",
                hoverinfo="skip",
            )
        )
    except Exception:
        pass

    fig.update_layout(
        title=dict(text="<b>Scale-Location Plot</b>", x=0.5, xanchor="center"),
        xaxis_title="Fitted Values",
        yaxis_title="√|Standardized Residuals|",
        template="plotly_white",
        hovermode="closest",
        height=450,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_residuals_vs_leverage_plot(
    leverage: np.ndarray, standardized_residuals: np.ndarray, cooks_distance: np.ndarray
) -> go.Figure:
    """
    Create Residuals vs Leverage plot with Cook's distance contours.

    Identifies influential observations that may affect the model.
    """
    fig = go.Figure(layout={})

    # Color points by Cook's distance
    cooks_threshold = 4 / len(leverage)
    colors = np.where(
        cooks_distance > cooks_threshold,
        COLORS.get("danger", "#EF4444"),
        COLORS.get("primary", "#3B82F6"),
    )

    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=leverage,
            y=standardized_residuals,
            mode="markers",
            marker=dict(
                color=colors, size=6, opacity=0.7, line=dict(color="white", width=0.5)
            ),
            name="Observations",
            hovertemplate="<b>Leverage:</b> %{x:.4f}<br><b>Std. Residual:</b> %{y:.3f}<extra></extra>",
        )
    )

    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=2, line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_hline(y=-2, line_dash="dot", line_color="orange", opacity=0.5)

    # Add Cook's distance threshold annotation
    fig.add_annotation(
        x=max(leverage) * 0.9,
        y=max(standardized_residuals) * 0.9,
        text=f"Cook's D > {cooks_threshold:.4f} = Influential",
        showarrow=False,
        font=dict(size=10, color=COLORS.get("danger", "#EF4444")),
    )

    fig.update_layout(
        title=dict(text="<b>Residuals vs Leverage</b>", x=0.5, xanchor="center"),
        xaxis_title="Leverage (Hat Values)",
        yaxis_title="Standardized Residuals",
        template="plotly_white",
        hovermode="closest",
        height=450,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


# ==============================================================================
# Formatting Functions
# ==============================================================================


def format_ols_results(
    results: OLSResult, var_meta: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Format OLS results for presentation.

    Parameters:
        results: OLSResult from run_ols_regression()
        var_meta: Variable metadata for labels

    Returns:
        Tuple of (formatted_coefficients_df, vif_table)
    """
    var_meta = var_meta or {}
    coef_df = results["coef_table"].copy()

    # Create formatted columns
    coef_df["Coefficient_fmt"] = coef_df["Coefficient"].apply(lambda x: f"{x:.4f}")
    coef_df["SE_fmt"] = coef_df["Std. Error"].apply(lambda x: f"{x:.4f}")
    coef_df["T_fmt"] = coef_df["T-value"].apply(lambda x: f"{x:.3f}")

    # Format p-values using centralized formatter
    coef_df["P_fmt"] = coef_df["P-value"].apply(
        lambda x: format_p_value(x, use_style=True)
    )

    # Format CI
    def format_ci(row):
        ci_str = f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]"
        # For linear regression, significance is when CI doesn't cross 0
        return format_ci_html(ci_str, row["CI_Lower"], row["CI_Upper"], null_val=0.0)

    coef_df["CI_fmt"] = coef_df.apply(format_ci, axis=1)

    # Add significance indicator
    coef_df["Significant"] = coef_df["P-value"] < 0.05

    # Add variable labels from metadata
    def get_label(var_name: str) -> str:
        """Get display label for variable."""
        # Clean up Q() wrapper
        clean_name = (
            var_name.replace("Q('", "")
            .replace("')", "")
            .replace("Intercept", "(Intercept)")
        )
        meta = var_meta.get(clean_name, {})
        if isinstance(meta, dict):
            label = meta.get("label", clean_name)
        else:
            label = clean_name
        return _html.escape(str(label))

    coef_df["Label"] = coef_df["Variable"].apply(get_label)

    # Create display DataFrame
    display_df = coef_df[
        ["Label", "Coefficient_fmt", "SE_fmt", "T_fmt", "P_fmt", "CI_fmt"]
    ].copy()
    display_df.columns = ["Variable", "β", "SE", "t-value", "p-value", "95% CI"]

    return display_df, results["vif_table"]


# ==============================================================================
# Report Generation
# ==============================================================================


def generate_report(
    outcome_name: str,
    results: OLSResult,
    diagnostics: list[DiagnosticResult],
    plots: dict[str, go.Figure],
    formatted_coef: pd.DataFrame,
    vif_table: pd.DataFrame,
    missing_data_info: dict[str, Any],
    var_meta: dict[str, Any] | None = None,
    regression_type: str = "ols",
) -> str:
    """
    Generate complete HTML report for OLS regression analysis.

    Parameters:
        outcome_name: Name of the outcome variable
        results: OLSResult from regression
        diagnostics: List of diagnostic test results
        plots: Dictionary of Plotly figures
        formatted_coef: Formatted coefficients DataFrame
        vif_table: VIF table DataFrame
        missing_data_info: Missing data summary
        var_meta: Variable metadata
        regression_type: 'ols' or 'robust'

    Returns:
        HTML string with complete report
    """
    from utils.plotly_html_renderer import plotly_figure_to_html

    var_meta = var_meta or {}

    # Escape outcome name for HTML
    outcome_label = var_meta.get(outcome_name, {}).get("label", outcome_name)
    outcome_escaped = _html.escape(str(outcome_label))

    # Build HTML sections
    html_parts = []

    # --- Header Section ---
    reg_type_display = (
        "Robust Regression (Huber M-estimator)"
        if regression_type == "robust"
        else "Ordinary Least Squares (OLS)"
    )
    html_parts.append(f"""
    <div class="report-header" style="margin-bottom: 20px;">
        <h2>📈 Linear Regression Report</h2>
        <h3>Outcome: {outcome_escaped}</h3>
        <p><strong>Method:</strong> {reg_type_display}</p>
    </div>
    """)

    # --- Model Summary Section ---
    r2 = results["r_squared"]
    adj_r2 = results["adj_r_squared"]
    f_stat = results["f_statistic"]
    f_p = results["f_pvalue"]

    f_p_fmt = format_p_value(f_p, use_style=False) if np.isfinite(f_p) else "N/A"

    # Pre-compute formatted values to avoid f-string format specifier issues
    r2_fmt = f"{r2:.4f}" if np.isfinite(r2) else "N/A"
    adj_r2_fmt = f"{adj_r2:.4f}" if np.isfinite(adj_r2) else "N/A"
    rse_fmt = f"{results['residual_std_err']:.4f}"
    f_stat_fmt = f"{f_stat:.2f}" if np.isfinite(f_stat) else "N/A"
    aic_fmt = f"{results['aic']:.2f}" if np.isfinite(results["aic"]) else "N/A"
    bic_fmt = f"{results['bic']:.2f}" if np.isfinite(results["bic"]) else "N/A"
    dw_fmt = f"{results['durbin_watson']:.3f}"
    df1 = int(results["n_obs"] - results["df_resid"] - 1)
    df2 = int(results["df_resid"])

    html_parts.append(f"""
    <div class="model-summary" style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
         padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #dee2e6;">
        <h4>📊 Model Summary</h4>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
            <div class="metric-box" style="background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.85em; color: #666;">Sample Size</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #333;">{results["n_obs"]:,}</div>
            </div>
            <div class="metric-box" style="background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.85em; color: #666;">R²</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #333;">{r2_fmt}</div>
            </div>
            <div class="metric-box" style="background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.85em; color: #666;">Adjusted R²</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #333;">{adj_r2_fmt}</div>
            </div>
            <div class="metric-box" style="background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.85em; color: #666;">Residual Std. Error</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #333;">{rse_fmt}</div>
            </div>
        </div>
        <div style="margin-top: 15px; background: white; padding: 10px 15px; border-radius: 8px;">
            <strong>F-statistic:</strong> F({df1}, {df2}) = {f_stat_fmt}, 
            <strong>p-value:</strong> {f_p_fmt}
        </div>
        <div style="margin-top: 10px; background: white; padding: 10px 15px; border-radius: 8px;">
            <strong>AIC:</strong> {aic_fmt} | 
            <strong>BIC:</strong> {bic_fmt} |
            <strong>Durbin-Watson:</strong> {dw_fmt}
        </div>
    </div>
    """)

    # --- Coefficients Table Section ---
    coef_html = formatted_coef.to_html(
        index=False, escape=False, classes="table table-striped table-hover", border=0
    )

    html_parts.append(f"""
    <div class="coefficients-section" style="margin-bottom: 20px;">
        <h4>📋 Regression Coefficients</h4>
        <div class="table-responsive" style="overflow-x: auto;">
            {coef_html}
        </div>
        <p style="font-size: 0.85em; color: #666; margin-top: 10px;">
            <strong>Note:</strong> β = unstandardized coefficient. 
            Significant results (p &lt; 0.05) are highlighted.
            95% CI highlighted green when not crossing 0.
        </p>
    </div>
    """)

    # --- VIF Section ---
    if not vif_table.empty:
        vif_html = vif_table.to_html(
            index=False, escape=True, classes="table table-sm", border=0
        )

        # Check for high VIF warnings
        high_vif = (
            vif_table[vif_table["VIF"] > 5]
            if "VIF" in vif_table.columns
            else pd.DataFrame()
        )
        warning_html = ""
        if not high_vif.empty:
            vars_list = ", ".join(high_vif["Variable"].tolist())
            warning_html = f"""
            <div class="alert alert-warning" style="margin-top: 10px; padding: 10px; border-radius: 8px; background: #fff3cd; border: 1px solid #ffc107;">
                <strong>⚠️ Multicollinearity Warning:</strong> Variables with VIF &gt; 5: {_html.escape(vars_list)}
            </div>
            """

        html_parts.append(f"""
        <div class="vif-section" style="margin-bottom: 20px;">
            <h4>🔍 Multicollinearity Check (VIF)</h4>
            <div class="table-responsive" style="overflow-x: auto;">
                {vif_html}
            </div>
            <p style="font-size: 0.85em; color: #666;">
                <strong>Interpretation:</strong> VIF &lt; 5 ✅ (OK), 5-10 ⚠️ (Moderate concern), &gt; 10 🔴 (Problematic)
            </p>
            {warning_html}
        </div>
        """)

    # --- Diagnostics Section ---
    if diagnostics:
        diag_rows = []
        for d in diagnostics:
            status = "✅" if d["passed"] else "⚠️"
            p_fmt = (
                format_p_value(d["p_value"], use_style=False)
                if np.isfinite(d["p_value"])
                else "N/A"
            )
            diag_rows.append(f"""
            <tr>
                <td>{status} {_html.escape(d["test_name"])}</td>
                <td>{d["statistic"]:.4f}</td>
                <td>{p_fmt}</td>
                <td>{_html.escape(d["interpretation"])}</td>
            </tr>
            """)

        html_parts.append(f"""
        <div class="diagnostics-section" style="margin-bottom: 20px;">
            <h4>🧪 Diagnostic Tests</h4>
            <table class="table table-sm" style="width: 100%;">
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Statistic</th>
                        <th>P-value</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(diag_rows)}
                </tbody>
            </table>
        </div>
        """)

    # --- Diagnostic Plots Section ---
    html_parts.append("""
    <div class="plots-section" style="margin-bottom: 20px;">
        <h4>📈 Assumption Diagnostics</h4>
    """)

    plot_info = [
        (
            "residuals_vs_fitted",
            "Residuals vs Fitted",
            "✅ Random scatter = good (linearity + homoscedasticity) | ❌ Pattern = potential issues",
        ),
        (
            "qq_plot",
            "Normal Q-Q Plot",
            "✅ Points on line = normal residuals | ❌ Deviation at tails = non-normality",
        ),
        (
            "scale_location",
            "Scale-Location Plot",
            "✅ Horizontal trend = constant variance | ❌ Slope = heteroscedasticity",
        ),
        (
            "residuals_vs_leverage",
            "Residuals vs Leverage",
            "✅ Blue points = normal | 🔴 Red points = influential observations",
        ),
    ]

    for plot_key, plot_title, plot_desc in plot_info:
        if plot_key in plots and plots[plot_key] is not None:
            plot_html = plotly_figure_to_html(
                plots[plot_key],
                div_id=f"ols_diag_{plot_key}",
                include_plotlyjs="cdn",
                responsive=True,
            )
            html_parts.append(f"""
            <div class="plot-container" style="margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <h5>{plot_title}</h5>
                {plot_html}
                <p style="font-size: 0.85em; color: #666; margin-top: 10px;">{plot_desc}</p>
            </div>
            """)

    html_parts.append("</div>")

    # --- Missing Data Section ---
    if missing_data_info:
        missing_html = create_missing_data_report_html(missing_data_info, var_meta)
        html_parts.append(f"""
        <div class="missing-data-section" style="margin-bottom: 20px;">
            {missing_html}
        </div>
        """)

    # --- Formula Section ---
    html_parts.append(f"""
    <div class="formula-section" style="margin-bottom: 20px; padding: 15px; background: #e9ecef; border-radius: 8px; font-family: monospace;">
        <strong>Model Formula:</strong> {_html.escape(results["formula"])}
    </div>
    """)

    # --- Interpretation Guide ---
    html_parts.append("""
    <div class="interpretation-guide" style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 12px; border: 1px solid #dee2e6;">
        <h4>📚 Interpretation Guide</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
            <div>
                <h5>Coefficients (β)</h5>
                <ul>
                    <li><strong>β &gt; 0:</strong> Positive relationship - Y increases with X</li>
                    <li><strong>β &lt; 0:</strong> Negative relationship - Y decreases with X</li>
                    <li><strong>p &lt; 0.05:</strong> Statistically significant effect</li>
                    <li><strong>95% CI not crossing 0:</strong> Significant effect</li>
                </ul>
            </div>
            <div>
                <h5>Model Fit</h5>
                <ul>
                    <li><strong>R² &gt; 0.7:</strong> Strong explanatory power</li>
                    <li><strong>R² 0.4-0.7:</strong> Moderate explanatory power</li>
                    <li><strong>R² &lt; 0.4:</strong> Weak explanatory power</li>
                    <li><strong>F-test p &lt; 0.05:</strong> Model is significant overall</li>
                </ul>
            </div>
        </div>
    </div>
    """)

    return "\n".join(html_parts)


# ==============================================================================
# Main Analysis Function
# ==============================================================================


def analyze_linear_outcome(
    outcome_name: str,
    df: pd.DataFrame,
    predictor_cols: list[str] | None = None,
    var_meta: dict[str, Any] | None = None,
    exclude_cols: list[str] | None = None,
    regression_type: Literal["ols", "robust"] = "ols",
    robust_estimator: str = "huber",
    robust_se: bool = False,
) -> tuple[str, OLSResult, dict[str, go.Figure], dict[str, Any]]:
    """
    Perform complete linear regression analysis.

    This is the main entry point for linear regression analysis.

    Parameters:
        outcome_name: Name of continuous outcome column
        df: Input DataFrame
        predictor_cols: List of predictor columns (if None, uses all numeric except outcome)
        var_meta: Variable metadata dictionary
        exclude_cols: Columns to exclude from analysis
        regression_type: 'ols' for standard OLS, 'robust' for robust regression
        robust_estimator: 'huber' or 'bisquare' (only used if regression_type='robust')
        robust_se: Use heteroscedasticity-robust standard errors

    Returns:
        Tuple of (html_report, results_dict, plots_dict, missing_data_info)
    """
    var_meta = var_meta or {}
    exclude_cols = exclude_cols or []

    logger.info("Starting linear regression analysis for outcome: %s", outcome_name)

    # Step 1: Determine predictors
    if predictor_cols is None:
        # Auto-select all numeric columns except outcome
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        predictor_cols = [
            c for c in numeric_cols if c != outcome_name and c not in exclude_cols
        ]
    else:
        predictor_cols = [c for c in predictor_cols if c not in exclude_cols]

    if not predictor_cols:
        raise ValueError("No valid predictor columns found after exclusions")

    logger.debug("Using %d predictors: %s", len(predictor_cols), predictor_cols[:5])

    # Step 2: Prepare data
    df_clean, missing_data_info = prepare_data_for_ols(
        df, outcome_name, predictor_cols, var_meta
    )

    # Step 3: Fit regression model
    if regression_type == "robust":
        results = run_robust_regression(
            df_clean, outcome_name, predictor_cols, m_estimator=robust_estimator
        )
    else:
        results = run_ols_regression(
            df_clean, outcome_name, predictor_cols, robust_se=robust_se
        )

    # Step 4: Run diagnostic tests
    diagnostics = run_diagnostic_tests(results)

    # Step 5: Create diagnostic plots
    plots = create_diagnostic_plots(results)

    # Step 6: Format results
    formatted_coef, vif_table = format_ols_results(results, var_meta)

    # Step 7: Generate HTML report
    html_report = generate_report(
        outcome_name=outcome_name,
        results=results,
        diagnostics=diagnostics,
        plots=plots,
        formatted_coef=formatted_coef,
        vif_table=vif_table,
        missing_data_info=missing_data_info,
        var_meta=var_meta,
        regression_type=regression_type,
    )

    logger.info(
        "Linear regression complete: R²=%.4f, n=%d, %d predictors",
        results["r_squared"] if np.isfinite(results["r_squared"]) else 0,
        results["n_obs"],
        len(predictor_cols),
    )

    return html_report, results, plots, missing_data_info

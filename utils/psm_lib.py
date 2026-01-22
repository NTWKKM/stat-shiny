import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
from typing import Any
from config import CONFIG
from utils.data_cleaning import prepare_data_for_analysis
from logger import get_logger

logger = get_logger(__name__)

def calculate_ps(
    data: pd.DataFrame, 
    treatment: str, 
    covariates: list[str],
    var_meta: dict[str, Any] | None = None
) -> pd.Series:
    """
    Calculate propensity scores using logistic regression with unified data cleaning.
    Returns propensity_scores Series indexed to original data.
    """
    try:
        # --- MISSING DATA HANDLING ---
        missing_cfg = CONFIG.get("analysis.missing", {}) or {}
        strategy = missing_cfg.get("strategy", "complete-case")
        missing_codes = missing_cfg.get("user_defined_values", [])

        df_subset, missing_info = prepare_data_for_analysis(
            data,
            required_cols=[treatment] + covariates,
            var_meta=var_meta,
            missing_codes=missing_codes,
            handle_missing=strategy
        )

        if df_subset.empty:
             return pd.Series(dtype=float, index=data.index)

        X = df_subset[covariates]
        X = sm.add_constant(X)
        y = df_subset[treatment]

        # Simple logistic regression
        model = sm.Logit(y, X).fit(disp=0)
        ps = model.predict(X)

        # Reindex to original data (fill missing with NaN)
        ps_full = pd.Series(index=data.index, dtype=float)
        ps_full.loc[df_subset.index] = ps

        return ps_full

    except Exception as e:
        logger.error(f"Propensity score calculation failed: {str(e)}")
        return pd.Series(dtype=float, index=data.index)

# Alias for backward compatibility
calculate_propensity_score = calculate_ps

def perform_matching(
    data: pd.DataFrame,
    treatment_col: str,
    ps_col: str,
    caliper: float = 0.2,
    ratio: int = 1,
) -> pd.DataFrame:
    """
    Perform nearest-neighbor propensity score matching with caliper.
    """
    try:
        df = data.copy().dropna(subset=[ps_col, treatment_col])

        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        if treated.empty or control.empty:
            return pd.DataFrame()

        # Get propensity scores
        treated_ps = treated[[ps_col]].values
        control_ps = control[[ps_col]].values

        # Calculate distance matrix
        distances = cdist(treated_ps, control_ps, metric="euclidean")

        matched_treated_idx = []
        matched_control_idx = []
        used_controls = set()

        # Greedy matching
        for i, treat_idx in enumerate(treated.index):
            best_dist = float("inf")
            best_control = None

            for j, ctrl_idx in enumerate(control.index):
                if ctrl_idx in used_controls:
                    continue
                dist = distances[i, j]
                if dist < best_dist and dist <= caliper:
                    best_dist = dist
                    best_control = ctrl_idx

            if best_control is not None:
                matched_treated_idx.append(treat_idx)
                matched_control_idx.append(best_control)
                used_controls.add(best_control)

        if not matched_treated_idx:
            return pd.DataFrame()

        # Combine matched pairs
        matched_df = pd.concat(
            [
                df.loc[matched_treated_idx],
                df.loc[matched_control_idx],
            ]
        )

        return matched_df
    except Exception:
        return pd.DataFrame()

def calculate_ipw(
    data: pd.DataFrame, treatment: str, outcome: str, ps_col: str
) -> dict:
    """
    Calculate Average Treatment Effect (ATE) using Inverse Probability Weighting.
    """
    try:
        df = data.copy()
        T = df[treatment]
        Y = df[outcome]
        ps = df[ps_col]

        # Avoid division by zero
        ps = ps.clip(0.01, 0.99)

        # Calculate weights
        df["ipw"] = np.where(T == 1, 1 / ps, 1 / (1 - ps))

        # ATE is coefficient of T in weighted regression
        X = sm.add_constant(T)
        model = sm.WLS(Y, X, weights=df["ipw"]).fit()
        ate = model.params[treatment]
        se = model.bse[treatment]
        p_val = model.pvalues[treatment]
        conf_int = model.conf_int().loc[treatment]

        return {
            "ATE": ate,
            "SE": se,
            "p_value": p_val,
            "CI_Lower": conf_int[0],
            "CI_Upper": conf_int[1],
        }
    except Exception as e:
        return {"error": str(e)}

def check_balance(
    data: pd.DataFrame, treatment: str, covariates: list[str], weights: pd.Series = None
) -> pd.DataFrame:
    """
    Calculate Standardized Mean Differences (SMD) to check balance.
    """
    results = []

    # Ensure clean data for balance check
    df = data.dropna(subset=[treatment] + covariates)
    if df.empty:
        return pd.DataFrame(columns=["Covariate", "SMD", "Status"])

    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    for cov in covariates:
        if not pd.api.types.is_numeric_dtype(df[cov]):
            continue

        mean_t = np.average(
            treated[cov],
            weights=weights[treated.index] if weights is not None else None,
        )
        mean_c = np.average(
            control[cov],
            weights=weights[control.index] if weights is not None else None,
        )

        var_t = np.var(treated[cov])
        var_c = np.var(control[cov])
        pooled_sd = np.sqrt((var_t + var_c) / 2)

        smd = (mean_t - mean_c) / pooled_sd if pooled_sd != 0 else 0

        results.append(
            {
                "Covariate": cov,
                "Variable": cov, # For test compatibility
                "SMD": abs(smd),
                "Status": "Balanced (<0.1)" if abs(smd) < 0.1 else "Unbalanced",
            }
        )

    return pd.DataFrame(results)

def calculate_smd(data, treatment, covariates, weights=None):
    """Test-compatible alias for check_balance."""
    return check_balance(data, treatment, covariates, weights)

def plot_love_plot(smd_pre: pd.DataFrame, smd_post: pd.DataFrame) -> go.Figure:
    """
    Create a Love Plot (SMD Comparison) using Plotly.
    """
    fig = go.Figure()

    # Pre-matching
    fig.add_trace(go.Scatter(
        x=smd_pre['SMD'],
        y=smd_pre.get('Variable', smd_pre.get('Covariate')),
        mode='markers',
        name='Unadjusted',
        marker=dict(color='red', size=10, symbol='circle-open')
    ))

    # Post-matching
    fig.add_trace(go.Scatter(
        x=smd_post['SMD'],
        y=smd_post.get('Variable', smd_post.get('Covariate')),
        mode='markers',
        name='Adjusted',
        marker=dict(color='blue', size=10, symbol='circle')
    ))

    # Reference line at 0.1
    fig.add_vline(x=0.1, line_dash="dash", line_color="gray", annotation_text="Threshold (0.1)")

    fig.update_layout(
        title="Love Plot (Standardized Mean Differences)",
        xaxis_title="Absolute Standardized Mean Difference",
        yaxis_title="Covariates",
        template="plotly_white",
        height=max(400, 200 + (len(smd_pre) * 30)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

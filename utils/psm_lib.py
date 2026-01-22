import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist


def calculate_propensity_score(
    data: pd.DataFrame,
    treatment: str,
    covariates: list[str],
    var_meta: dict | None = None,
) -> tuple[pd.Series, dict]:
    """
    Calculate propensity scores using logistic regression.
    Returns (propensity_scores, missing_info_dict).
    """
    try:
        df_analysis = data.copy()
        missing_info = {"strategy": "complete_case", "rows_before": len(df_analysis)}

        # Drop rows with missing values in treatment or covariates
        cols_needed = [treatment] + covariates
        df_analysis = df_analysis.dropna(subset=cols_needed)
        missing_info["rows_after"] = len(df_analysis)
        missing_info["rows_removed"] = (
            missing_info["rows_before"] - missing_info["rows_after"]
        )

        X = df_analysis[covariates]
        X = sm.add_constant(X)
        y = df_analysis[treatment]

        # Simple logistic regression
        model = sm.Logit(y, X).fit(disp=0)
        ps = model.predict(X)

        # Reindex to original data (fill missing with NaN)
        ps_full = pd.Series(index=data.index, dtype=float)
        ps_full.loc[df_analysis.index] = ps

        return ps_full, missing_info
    except Exception as e:
        return pd.Series(dtype=float), {"error": f"Propensity score calculation failed: {str(e)}"}


def perform_matching(
    data: pd.DataFrame,
    treatment_col: str,
    ps_col: str,
    caliper: float = 0.2,
    ratio: int = 1,
) -> pd.DataFrame:
    """
    Perform nearest-neighbor propensity score matching with caliper.

    Parameters:
        data: DataFrame with propensity scores
        treatment_col: Column name for treatment indicator (1=treated, 0=control)
        ps_col: Column name for propensity scores
        caliper: Maximum allowed difference in PS for a match
        ratio: Number of control matches per treated (default 1:1)

    Returns:
        DataFrame containing matched pairs
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

        # Greedy matching: for each treated, find closest unused control within caliper
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

# ... (rest of file)

def calculate_ps(
    data: pd.DataFrame, treatment: str, covariates: list[str]
) -> pd.Series:
    """
    Calculate propensity scores using logistic regression.
    """
    try:
        X = data[covariates]
        X = sm.add_constant(X)
        y = data[treatment]

        # Simple logistic regression
        model = sm.Logit(y, X).fit(disp=0)
        return model.predict(X)
    except Exception:
        return pd.Series(dtype=float)



def calculate_ipw(
    data: pd.DataFrame, treatment: str, outcome: str, ps_col: str
) -> dict:
    """
    Calculate Average Treatment Effect (ATE) using Inverse Probability Weighting.
    Wrapper for a basic IPW estimator.
    """
    try:
        df = data.copy()
        T = df[treatment]
        Y = df[outcome]
        ps = df[ps_col]

        # Avoid division by zero
        ps = ps.clip(0.01, 0.99)

        # Calculate weights
        # weight = T / ps + (1 - T) / (1 - ps)
        df["ipw"] = np.where(T == 1, 1 / ps, 1 / (1 - ps))

        # Weighted difference (Basic ATE)
        # weighted_mean_treated = np.average(df[df[treatment]==1][outcome], weights=df[df[treatment]==1]['ipw'])
        # weighted_mean_control = np.average(df[df[treatment]==0][outcome], weights=df[df[treatment]==0]['ipw'])

        # Alternative robust estimation using OLS with weights
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
    If weights provided, calculates weighted SMD.
    """
    results = []

    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]

    for cov in covariates:
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(data[cov]):
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
                "SMD": abs(smd),
                "Status": "Balanced (<0.1)" if abs(smd) < 0.1 else "Unbalanced",
            }
        )

    return pd.DataFrame(results)

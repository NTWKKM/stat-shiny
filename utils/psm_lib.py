import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    except Exception as e:
        raise ValueError(f"Propensity score calculation failed: {str(e)}")


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

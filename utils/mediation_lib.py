import pandas as pd
from statsmodels.api import OLS, add_constant


def analyze_mediation(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    mediator: str,
    confounders: list[str] | None = None,
) -> dict[str, float]:
    """
    Perform mediation analysis using the product of coefficients method.

    Models:
    1. Total Effect: Y ~ X + C
    2. Mediator Model: M ~ X + C
    3. Outcome Model: Y ~ X + M + C

    Returns:
        Dictionary with total_effect, direct_effect, indirect_effect, proportion_mediated
    """
    # Use list for clean selection, ensuring no duplicates
    cols = [outcome, treatment, mediator]
    if confounders:
        cols.extend(confounders)
    # Remove duplicates if any
    cols = list(dict.fromkeys(cols))

    df_clean = data[cols].dropna()

    if df_clean.empty:
        raise ValueError("No valid data after removing missing values.")

    # 1. Total Effect (c): Y ~ X + C
    X_total = df_clean[[treatment]]
    if confounders:
        X_total = pd.concat([X_total, df_clean[confounders]], axis=1)
    X_total = add_constant(X_total)
    model_total = OLS(df_clean[outcome], X_total).fit()
    c_total = model_total.params[treatment]

    # 2. Mediator Model (a): M ~ X + C
    X_med = df_clean[[treatment]]
    if confounders:
        X_med = pd.concat([X_med, df_clean[confounders]], axis=1)
    X_med = add_constant(X_med)
    model_med = OLS(df_clean[mediator], X_med).fit()
    a_path = model_med.params[treatment]

    # 3. Outcome Model (b, c'): Y ~ X + M + C
    X_out = df_clean[[treatment, mediator]]
    if confounders:
        X_out = pd.concat([X_out, df_clean[confounders]], axis=1)
    X_out = add_constant(X_out)
    model_out = OLS(df_clean[outcome], X_out).fit()
    b_path = model_out.params[mediator]
    c_prime = model_out.params[treatment]  # Direct effect

    # Indirect Effect (a * b)
    indirect_effect = a_path * b_path

    # Proportion Mediated
    # Handle division by zero or very small total effect
    if abs(c_total) < 1e-9:
        prop_mediated = 0.0
    else:
        prop_mediated = indirect_effect / c_total

    return {
        "total_effect": float(c_total),
        "direct_effect": float(c_prime),
        "indirect_effect": float(indirect_effect),
        "proportion_mediated": float(prop_mediated),
        "a_path": float(a_path),
        "b_path": float(b_path),
        "n_obs": len(df_clean),
    }

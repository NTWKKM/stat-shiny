import numpy as np
import pandas as pd


def mantel_haenszel(
    data: pd.DataFrame, outcome: str, treatment: str, stratum: str
) -> dict:
    """
    Calculate Mantel-Haenszel Odds Ratio for stratified data.
    """
    try:
        # Validate inputs
        if data is None or data.empty:
             return {"error": "Input data is empty."}
        
        required_cols = [outcome, treatment, stratum]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
             return {"error": f"Missing columns: {', '.join(missing)}"}

        # Create list of 2x2 tables
        strata_levels = data[stratum].unique()

        num_sum = 0
        den_sum = 0

        results_by_stratum = []
        
        if len(strata_levels) == 0:
             return {"error": "No strata found in data."}

        for s in strata_levels:
            subset = data[data[stratum] == s]
            
            # Ensure binary outcome/treatment (0/1) for calculation, or robustly handle
            # Assuming 0/1 for now as per design
            
            a = ((subset[treatment] == 1) & (subset[outcome] == 1)).sum()
            b = ((subset[treatment] == 1) & (subset[outcome] == 0)).sum()
            c = ((subset[treatment] == 0) & (subset[outcome] == 1)).sum()
            d = ((subset[treatment] == 0) & (subset[outcome] == 0)).sum()

            n_k = a + b + c + d
            if n_k == 0:
                continue

            # MH components
            num_sum += (a * d) / n_k
            den_sum += (b * c) / n_k

            # Stratum OR
            or_k = (a * d) / (b * c) if (b * c) > 0 else np.nan

            results_by_stratum.append(
                {"Stratum": s, "OR": or_k, "a": a, "b": b, "c": c, "d": d}
            )

        mh_or = num_sum / den_sum if den_sum > 0 else np.nan

        return {"MH_OR": mh_or, "Strata_Results": pd.DataFrame(results_by_stratum)}
    except Exception as e:
        return {"error": f"Mantel-Haenszel failed: {str(e)}"}


def breslow_day(data: pd.DataFrame, outcome: str, treatment: str, stratum: str) -> dict:
    """
    Perform Breslow-Day test for homogeneity of odds ratios.
    (Simplified version or placeholder if complex implementation needed)
    """
    # Note: Full Breslow-Day is complex to implement from scratch.
    # For now, we will return a placeholder or simple interaction check using logistic regression.

    # Logistic Regression Interaction approach is often preferred in modern analysis.
    import statsmodels.api as sm

    try:
        if data is None or data.empty:
             return {"error": "Input data is empty."}
        
        if not all(col in data.columns for col in [outcome, treatment, stratum]):
             return {"error": "Missing required columns for Breslow-Day Check."}

        df = data[[outcome, treatment, stratum]].dropna()
        if df.empty:
             return {"error": "No valid data for homogeneity test."}

        # Model 1: Main effects
        X1 = df[[treatment, stratum]]
        X1 = sm.add_constant(pd.get_dummies(X1, columns=[stratum], drop_first=True))
        # Ensure treatment is clean numeric

        # Model 2: Interaction
        # This is strictly not Breslow-Day but tests the same hypothesis of homogeneity
        # Construct interaction term
        # For simplicity in this demo, treat stratum as categorical in formula

        # formula approach might be easier
        import statsmodels.formula.api as smf

        # Check for single stratum or no variation
        if df[stratum].nunique() < 2:
             return {"error": "Need at least 2 strata for homogeneity test."}

        # Check for convergence issues due to perfect separation?
        # Using try-except on fit()

        formula_h0 = f"{outcome} ~ {treatment} + C({stratum})"
        formula_h1 = f"{outcome} ~ {treatment} * C({stratum})"

        mod0 = smf.logit(formula_h0, data=df).fit(disp=0)
        mod1 = smf.logit(formula_h1, data=df).fit(disp=0)

        # Likelihood Ratio Test
        lr_stat = 2 * (mod1.llf - mod0.llf)
        
        from scipy.stats import chi2

        df_diff = mod1.df_model - mod0.df_model
        
        if df_diff <= 0:
             p_val = 1.0 # Should not happen unless models are identical
        else:
             p_val = 1 - chi2.cdf(lr_stat, df_diff)

        return {
            "test": "Likelihood Ratio Test for Homogeneity (Interaction)",
            "statistic": lr_stat,
            "p_value": p_val,
            "conclusion": "Heterogeneity detected (p<0.05)"
            if p_val < 0.05
            else "Homogeneity holds",
        }

    except Exception as e:
        return {"error": f"Error calculating homogeneity: {str(e)}"}

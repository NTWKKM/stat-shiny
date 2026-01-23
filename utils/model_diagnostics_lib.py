import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import reset_ramsey


def run_reset_test(model_results) -> dict:
    """
    Run Ramsey's RESET test for model specification.

    Args:
        model_results: A fitted statsmodels result object (e.g. from OLS)

    Returns:
        dict with 'statistic', 'p_value', 'conclusion'
    """
    try:
        # Check if model has a method/attribute required (e.g., fitted OLS)
        if model_results is None or not hasattr(model_results, "model"):
            return {"error": "Invalid model object provided."}

        # Run RESET test (power=2,3 by default generally)
        reset = reset_ramsey(model_results, degree=5)

        p_val = reset.pvalue

        return {
            "test": "Ramsey RESET",
            "statistic": reset.statistic,
            "p_value": p_val,
            "conclusion": (
                "Possible misspecification (p < 0.05)"
                if p_val < 0.05
                else "No strong evidence of misspecification"
            ),
        }
    except Exception as e:
        return {"error": f"RESET Test failed: {str(e)}"}


def run_heteroscedasticity_test(model_results) -> dict:
    """
    Run Breusch-Pagan test for heteroscedasticity.

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        dict with 'statistic', 'p_value', 'conclusion'
    """
    try:
        if model_results is None or not hasattr(model_results, "model"):
            return {"error": "Invalid model object provided."}

        # Breusch-Pagan requires residuals and exog
        resid = model_results.resid
        exog = model_results.model.exog

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, exog)

        return {
            "test": "Breusch-Pagan",
            "statistic": lm_stat,
            "p_value": lm_pvalue,
            "conclusion": (
                "Heteroscedasticity present (p < 0.05)"
                if lm_pvalue < 0.05
                else "Homoscedasticity assumption holds"
            ),
        }
    except Exception as e:
        return {"error": f"Heteroscedasticity test failed: {str(e)}"}


def calculate_cooks_distance(model_results) -> dict:
    """
    Calculate Cook's Distance for identifying influential points.

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        dict with 'cooks_d' (list), 'p_values' (list), 'influential_points' (indices)
    """
    try:
        if model_results is None:
            return {"error": "Invalid model object provided."}

        influence = model_results.get_influence()
        # cooks_d is a tuple (distance, p-value)
        c_d, p_val = influence.cooks_distance

        if c_d is None:
            return {"error": "Could not calculate Cook's distance."}

        # Threshold: 4/n rule of thumb
        n = model_results.nobs
        threshold = 4 / n if n > 0 else 1.0

        influential_indices = np.where(c_d > threshold)[0].tolist()

        return {
            "cooks_d": c_d.tolist(),
            "p_values": p_val.tolist(),
            "threshold": threshold,
            "influential_points": influential_indices,
            "n_influential": len(influential_indices),
        }
    except Exception as e:
        return {"error": f"Cooks distance calculation failed: {str(e)}"}


def get_diagnostic_plot_data(model_results) -> dict:
    """
    Extract data needed for diagnostic plots (Residuals vs Fitted, Q-Q).

    Args:
        model_results: A fitted statsmodels result object

    Returns:
        dict with 'fitted_values', 'residuals', 'std_residuals'
    """
    try:
        if model_results is None:
            return {"error": "Invalid model object provided."}

        return {
            "fitted_values": model_results.fittedvalues.tolist(),
            "residuals": model_results.resid.tolist(),
            "std_residuals": model_results.get_influence().resid_studentized_internal.tolist(),
        }
    except Exception as e:
        return {"error": f"Diagnostic plot data extraction failed: {str(e)}"}

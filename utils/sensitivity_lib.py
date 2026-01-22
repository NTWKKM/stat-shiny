import numpy as np


def calculate_e_value(
    estimate: float, lower: float = None, upper: float = None, type: str = "RR"
) -> dict:
    """
    Calculate E-value for a risk ratio (RR) or odds ratio (OR).
    If OR provided and outcome is common, E-value is approximate.

    E-value = RR + sqrt(RR * (RR - 1))
    """
    try:
        # Validate inputs
        if estimate is None or not isinstance(estimate, (int, float)):
             return {"error": "Estimate must be a number."}
        
        if estimate == 1.0:
             return {
                "original_estimate": estimate,
                "e_value_estimate": 1.0,
                "e_value_ci_limit": 1.0,
            }

        if estimate <= 0:
            return {"error": "Estimate (RR/OR) must be positive."}

        # Handle protective effects (RR < 1) by taking inverse
        if estimate < 1:
            est_prime = 1 / estimate
            # CI flips
            # If upper is provided, it becomes the lower bound in the flipped scale
            l_prime = 1 / upper if (upper and upper > 0) else None
        else:
            est_prime = estimate
            l_prime = lower

        def compute_e(val):
            if val is None or val <= 1:
                return 1.0
            return val + np.sqrt(val * (val - 1))

        e_est = compute_e(est_prime)

        # For reporting, we usually report the E-value for the estimate
        # and the E-value for the CI limit closest to the null (1).

        limit_e_val = 1.0
        if l_prime and l_prime > 1:
            limit_e_val = compute_e(l_prime)
        # If CI crosses 1, the limit E-value is 1.

        return {
            "original_estimate": estimate,
            "e_value_estimate": round(e_est, 3),
            "e_value_ci_limit": round(limit_e_val, 3),
        }
    except Exception as e:
        return {"error": f"E-value calculation failed: {str(e)}"}

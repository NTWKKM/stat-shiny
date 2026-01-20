import numpy as np


def calculate_e_value(
    estimate: float, lower: float = None, upper: float = None, type: str = "RR"
):
    """
    Calculate E-value for a risk ratio (RR) or odds ratio (OR).
    If OR provided and outcome is common, E-value is approximate.

    E-value = RR + sqrt(RR * (RR - 1))
    """
    try:
        if estimate <= 0:
            raise ValueError("Estimate (RR/OR) must be positive.")

        # Handle protective effects (RR < 1) by taking inverse
        if estimate < 1:
            est_prime = 1 / estimate
            # CI flips
            l_prime = 1 / upper if upper else None
            # u_prime = 1 / lower if lower else None # Unused
        else:
            est_prime = estimate
            l_prime = lower
            # u_prime = upper # Unused

        def compute_e(val):
            if val is None or val <= 1:
                return 1.0
            return val + np.sqrt(val * (val - 1))

        # e_lower, e_upper are calculated but not used in the return currently.
        # Keeping logic simpler as per linter request, or use them if intended.
        # The user code returned 'e_value_ci_limit' which uses l_prime.
        # e_upper is indeed unused.
        # e_lower is technically used via l_prime check logic below? No, l_prime is used.

        # Original:
        # e_est = compute_e(est_prime)
        # e_lower = compute_e(l_prime) if l_prime else None
        # e_upper = compute_e(u_prime) if u_prime else None

        e_est = compute_e(est_prime)
        # e_lower and e_upper were unused variables.

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
        return {"error": str(e)}

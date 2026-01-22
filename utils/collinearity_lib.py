import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for a list of predictors.

    Args:
        data: DataFrame containing the data
        predictors: List of column names to check for collinearity

    Returns:
        DataFrame with columns ['Variable', 'VIF', 'Tolerance'] sorted by VIF descending.
    """
    try:
        if not predictors or data is None or data.empty:
            return pd.DataFrame(columns=["Variable", "VIF", "Tolerance"])
    
        # Check if columns exist
        existing_predictors = [col for col in predictors if col in data.columns]
        if not existing_predictors:
             return pd.DataFrame(columns=["Variable", "VIF", "Tolerance"])

        # Drop rows with missing values in predictors
        X = data[existing_predictors].dropna()

        if X.empty:
            # Cannot calculate VIF with no data
            return pd.DataFrame(columns=["Variable", "VIF", "Tolerance"])

        # Ensure numeric types
        X = X.select_dtypes(include=[np.number])
        valid_predictors = X.columns.tolist()

        if not valid_predictors or len(valid_predictors) < 2:
            # VIF needs at least 2 variables to assess multicollinearity effectively
            # Though strictly 1 variable VIF is 1.0? No, VIF is about correlation with others.
            # If only 1 var, VIF is not really applicable or is 1?
            # Let's clean it up.
            
            # If only 1 variable, VIF=1 by definition if we consider it orthogonal to nothing?
            # Or undefined?
            # Often standard packages return empty or error.
            if len(valid_predictors) == 1:
                 return pd.DataFrame([{"Variable": valid_predictors[0], "VIF": 1.0, "Tolerance": 1.0}])

            return pd.DataFrame(columns=["Variable", "VIF", "Tolerance"])

        # Add constant for VIF calculation correctness (intercept)
        # statsmodels VIF requires a constant term usually if not present, but usually we just pass the matrix
        # The standard way:
        X_with_const = X.copy()
        X_with_const["const"] = 1

        vif_data = []

        # We iterate over the original valid_predictors, corresponding to columns 0 to len-1
        for i, col in enumerate(valid_predictors):
            try:
                val = variance_inflation_factor(X_with_const.values, i)
            except Exception:
                val = np.nan

            vif_data.append(
                {
                    "Variable": col,
                    "VIF": val,
                    "Tolerance": 1.0 / val if val and val != 0 else np.nan,
                }
            )

        df_vif = pd.DataFrame(vif_data)

        # Sort and return
        return df_vif.sort_values("VIF", ascending=False)
    except Exception:
        # Fallback empty
        return pd.DataFrame(columns=["Variable", "VIF", "Tolerance"])


def condition_index(data: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """
    Calculate Condition Index (CI) for assessing multicollinearity.

    CI = sqrt(max_eigenvalue / eigenvalue_i)
    Review: CI > 10 (Moderate), CI > 30 (Severe)
    """
    if not predictors:
        return pd.DataFrame()

    X = data[predictors].dropna().select_dtypes(include=[np.number])

    if X.empty:
        return pd.DataFrame()

    # Scale the data (Uncentered usually for Belsley-Kuh-Welsch, but centered is common too)
    # Standard practice: Normalize column vectors to unit length
    X_norm = X / np.sqrt((X**2).sum(axis=0))

    # Compute eigenvalues of X'X
    # Or singular values of X directly since svd(X) gives sqrt(eigenvalues of X'X)
    try:
        # Use SVD for numerical stability
        U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)

        # S contains singular values (sqrt of eigenvalues of X'X)
        max_sv = S.max()
        condition_indices = max_sv / S

        results = []
        for i, ci in enumerate(condition_indices):
            # Variance decomposition proportions could be added here for full diagnostic
            results.append(
                {"Dimension": i + 1, "Condition Index": ci, "Eigenvalue": S[i] ** 2}
            )

        return pd.DataFrame(results)

    except np.linalg.LinAlgError:
        return pd.DataFrame()

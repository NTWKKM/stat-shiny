"""
Collinearity Detection Module for stat-shiny
Calculates VIF (Variance Inflation Factor) for each covariate
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VIF for each numeric covariate.
    
    VIF Interpretation:
    - VIF = 1: No correlation with other predictors ✅
    - VIF = 1-5: Mild correlation (usually acceptable)
    - VIF > 10: Problematic collinearity ⚠️ (consider removing)
    - VIF > 30: Severe collinearity ❌ (definitely remove one)
    
    Args:
        X: DataFrame of numeric predictors (no intercept)
    
    Returns:
        DataFrame with columns: ['Variable', 'VIF', 'Flag']
    """
    from logger import get_logger
    logger = get_logger(__name__)
    
    # Start with a copy of X
    X_for_vif = X.copy()
    
    # Identify categorical columns
    cat_cols = X_for_vif.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Encode categorical variables if any
    if cat_cols:
        logger.info(f"Encoding categorical columns: {cat_cols}")
        X_for_vif = pd.get_dummies(X_for_vif, columns=cat_cols, drop_first=True, dtype=int)
    
    # Select only numeric columns
    X_numeric = X_for_vif.select_dtypes(include=[np.number]).copy()
    
    if X_numeric.empty:
        raise ValueError("No numeric columns found in X")
    
    # Check if we have enough data for VIF calculation
    if X_numeric.shape[0] <= X_numeric.shape[1]:
        logger.warning(f"Insufficient observations for VIF calculation: {X_numeric.shape[0]} rows, {X_numeric.shape[1]} predictors")
        return pd.DataFrame(columns=['Variable', 'VIF', 'Flag', 'Interpretation'])
    
    # Ensure all columns are numeric type
    for col in X_numeric.columns:
        X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
    
    # Drop any columns with all NaN after conversion
    X_numeric = X_numeric.dropna(axis=1, how='all')
    
    if X_numeric.shape[1] == 0:
        raise ValueError("No valid numeric columns after conversion")
    
    vif_data = []
    
    for i, col in enumerate(X_numeric.columns):
        try:
            vif = variance_inflation_factor(X_numeric.values, i)
            
            # Flag if problematic
            flag = ""
            if vif > 30:
                flag = "❌ SEVERE"
            elif vif > 10:
                flag = "⚠️ HIGH"
            elif vif > 5:
                flag = "⚡ MODERATE"
            
            vif_data.append({
                'Variable': col,
                'VIF': vif if not np.isinf(vif) else None,
                'Flag': flag,
                'Interpretation': _interpret_vif(vif)
            })
        except Exception as e:
            logger.warning(f"Could not calculate VIF for {col}: {e}")
            vif_data.append({
                'Variable': col,
                'VIF': None,
                'Flag': "⚠️ ERROR",
                'Interpretation': str(e)
            })
    
    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False, na_position='last')


def _interpret_vif(vif: float) -> str:
    """Plain-English interpretation of VIF value."""
    if pd.isna(vif) or np.isinf(vif):
        return "Cannot calculate (may indicate perfect collinearity)"
    elif vif == 1:
        return "Independent predictor"
    elif vif < 5:
        return "Acceptable level of collinearity"
    elif vif < 10:
        return "Moderate collinearity - consider removing if possible"
    else:
        return "Severe collinearity - strongly recommend removal"


def generate_vif_report_html(vif_df: pd.DataFrame) -> str:
    """Generate HTML report with VIF results and recommendations."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h2 { color: #1B7E8F; }
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
            th { background-color: #0D4D57; color: white; }
            .high { color: red; font-weight: bold; }
            .moderate { color: orange; }
            .warning-box { 
                background-color: #fff3cd; 
                border-left: 4px solid #ffc107; 
                padding: 10px; 
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h2>📊 Collinearity Diagnostic Report</h2>
        
        <div class="warning-box">
            <strong>What is VIF?</strong> Variance Inflation Factor measures how much a 
            predictor's variance is inflated due to collinearity with other predictors. 
            Higher = more problematic.
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>VIF</th>
                    <th>Flag</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in vif_df.iterrows():
        vif_val = f"{row['VIF']:.2f}" if pd.notna(row['VIF']) else "N/A"
        class_attr = "high" if "SEVERE" in str(row['Flag']) else ("moderate" if "MODERATE" in str(row['Flag']) else "")
        
        html += f"""
                <tr>
                    <td><strong>{row['Variable']}</strong></td>
                    <td class="{class_attr}">{vif_val}</td>
                    <td>{row['Flag']}</td>
                    <td>{row['Interpretation']}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <div class="warning-box">
            <strong>Recommendations:</strong>
            <ul>
                <li>If VIF > 10: Consider removing the variable with highest VIF</li>
                <li>Repeat analysis after removal to check VIF improvements</li>
                <li>Consult domain expertise - sometimes collinearity is unavoidable</li>
                <li>Consider: Ridge regression or elastic net if multicollinearity severe</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html


# Integration with existing logic.py
def enhance_logistic_regression_with_vif(
    y: pd.Series, 
    X: pd.DataFrame, 
    **kwargs
) -> tuple:
    """
    Wrapper that adds VIF diagnostics to logistic regression output.
    
    Usage:
        params, conf_int, pvals, status, stats_metrics, vif_report = \\
            enhance_logistic_regression_with_vif(y, X)
    """
    from logic import run_binary_logit
    
    # Get VIF diagnostics first
    vif_df = calculate_vif(X)
    vif_html = generate_vif_report_html(vif_df)
    
    # Run logistic regression as usual
    params, conf_int, pvals, status, stats_metrics = run_binary_logit(y, X, **kwargs)
    
    return params, conf_int, pvals, status, stats_metrics, vif_html
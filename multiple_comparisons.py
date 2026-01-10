"""
Multiple Comparison Correction Methods for stat-shiny
Controls family-wise error rate and false discovery rate when conducting multiple tests.

USAGE:
    from multiple_comparisons import MultipleComparisonCorrection, generate_mcc_report_html
    
    mcc = MultipleComparisonCorrection()
    
    # Bonferroni correction (conservative)
    results, threshold = mcc.bonferroni(p_values=[0.01, 0.03, 0.05])
    
    # Holm correction (less conservative, recommended)
    results, threshold = mcc.holm(p_values=[0.01, 0.03, 0.05])
    
    # Benjamini-Hochberg (FDR control, more powerful)
    results, threshold = mcc.benjamini_hochberg(p_values=[0.01, 0.03, 0.05])
    
    # Generate HTML report
    html = generate_mcc_report_html("Holm", results)

METHOD COMPARISON:
    - Bonferroni: Conservative, most protective against false positives
      * Recommended when: n_tests < 10
      * Controls: Family-wise error rate (FWER)
      * Drawback: Can be too conservative, may miss true effects
    
    - Holm: Less conservative than Bonferroni, still strong control
      * Recommended when: Most applications (default choice)
      * Controls: Family-wise error rate (FWER)
      * Advantage: More powerful while maintaining strong control
    
    - Benjamini-Hochberg: Controls False Discovery Rate (FDR), more powerful
      * Recommended when: Exploratory analyses, many tests
      * Controls: False Discovery Rate (FDR)
      * Interpretation: ~5% of significant results are expected to be false positives
      * Advantage: Most powerful, good for large-scale testing
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

class MultipleComparisonCorrection:
    """
    Implements multiple comparison correction methods to control for false positives
    when conducting multiple statistical tests.
    
    When running multiple tests, the chance of finding at least one false positive
    increases with the number of tests. These corrections adjust the significance
    threshold to control this inflation.
    
    Available Methods:
        - bonferroni: Conservative correction (n_tests < 10)
        - holm: Sequential correction (recommended for most applications)
        - benjamini_hochberg: FDR control (exploratory analyses, many tests)
    """
    
    @staticmethod
    def bonferroni(p_values: list, alpha: float = 0.05) -> pd.DataFrame:
        """
        Bonferroni correction: p_threshold = alpha / n_tests
        Conservative but simple. Recommended when n_tests < 10.
        """
        n_tests = len(p_values)
        threshold = alpha / n_tests
        
        results = pd.DataFrame({
            'Test #': range(1, n_tests + 1),
            'P-value': p_values,
            'Adjusted P': [min(p * n_tests, 1.0) for p in p_values],
            'Significant (α=0.05)': [p < threshold for p in p_values]
        })
        
        return results, threshold
    
    @staticmethod
    def holm(p_values: list, alpha: float = 0.05) -> pd.DataFrame:
        """
        Holm-Bonferroni: Sequential correction. Less conservative than Bonferroni.
        Recommended for most applications.
        """
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        adjusted_p = sorted_p * (n_tests - np.arange(n_tests))
        adjusted_p = np.minimum(adjusted_p, 1.0)
        
        # Unsort back to original order
        result_p = np.empty_like(adjusted_p)
        result_p[sorted_indices] = adjusted_p
        
        results = pd.DataFrame({
            'Test #': range(1, n_tests + 1),
            'P-value': p_values,
            'Adjusted P (Holm)': result_p,
            'Significant (α=0.05)': result_p < alpha
        })
        
        return results, alpha
    
    @staticmethod
    def benjamini_hochberg(p_values: list, fdr: float = 0.05) -> pd.DataFrame:
        """
        Benjamini-Hochberg: Controls False Discovery Rate (not Family-Wise Error).
        More powerful than Bonferroni. Recommended for exploratory analyses.
        
        Interpretation: ~5% of significant results are expected to be false positives.
        """
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]
        
        # BH critical value
        ranks = np.arange(1, n_tests + 1)
        bh_critical = (ranks / n_tests) * fdr
        
        # Find largest i where P(i) <= (i/m) * FDR
        threshold = 0
        for i in range(n_tests - 1, -1, -1):
            if sorted_p[i] <= bh_critical[i]:
                threshold = bh_critical[i]
                break
        
        # Unsort
        significant = np.zeros(n_tests, dtype=bool)
        significant[sorted_indices[sorted_p <= threshold]] = True
        
        results = pd.DataFrame({
            'Test #': range(1, n_tests + 1),
            'P-value': p_values,
            'Threshold': [f"{bh_critical[i]:.4f}" for i in range(n_tests)],
            f'Significant (FDR={fdr})': significant
        })
        
        return results, threshold


def generate_mcc_report_html(method: str, results_df: pd.DataFrame) -> str:
    """Generate HTML report comparing correction methods."""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #1B7E8F; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ padding: 10px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #0D4D57; color: white; }}
            .significant {{ color: red; font-weight: bold; }}
            .info-box {{ 
                background-color: #e7f3ff; 
                border-left: 4px solid #2196F3; 
                padding: 10px; 
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <h2>🔬 Multiple Comparison Correction Results</h2>
        <h3>Method: {method}</h3>
        
        <div class="info-box">
            <strong>What this means:</strong> When running multiple statistical tests, 
            the chance of finding a false positive increases. This correction adjusts 
            the threshold to control for this inflation.
        </div>
        
        {results_df.to_html(classes='table', border=0)}
        
        <div class="info-box">
            <strong>Method Comparison:</strong>
            <ul>
                <li><strong>Bonferroni:</strong> Conservative, most protective against false positives</li>
                <li><strong>Holm:</strong> Less conservative than Bonferroni, still strong control</li>
                <li><strong>Benjamini-Hochberg:</strong> Controls False Discovery Rate (FDR), more powerful</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html
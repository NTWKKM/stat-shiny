# 🚀 Development Roadmap: stat-shiny Next Steps

**Quick Implementation Guide for Tier-1 Enhancements**

---

## Phase 1: Collinearity Diagnostics (VIF) - 2-3 Hours ⭐ START HERE

### Why: Users need to detect redundant predictors

**File to create:** `collinearity_check.py`

```python
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
    
    # Drop non-numeric columns
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    
    if X_numeric.empty:
        raise ValueError("No numeric columns found in X")
    
    # Encode categorical variables
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        logger.info(f"Encoding categorical columns: {cat_cols}")
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        X_numeric = X_encoded.select_dtypes(include=[np.number])
    
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
```

### Integration Steps:

1. **Add to `logic.py` after imports:**
   ```python
   from collinearity_check import enhance_logistic_regression_with_vif
   ```

2. **Modify `analyze_outcome()` to include VIF:**
   ```python
   # In analyze_outcome(), after multivariate model fitting:
   vif_report = ""
   try:
       vif_df = calculate_vif(data[covariate_cols])
       vif_report = generate_vif_report_html(vif_df)
   except Exception as e:
       logger.warning(f"Could not calculate VIF: {e}")
   
   # Append to final HTML
   html_table += f"<div style='margin-top: 30px;'>{vif_report}</div>"
   ```

3. **Test (add to `tests/test_statistics.py`):**
   ```python
   def test_vif_calculation():
       """Test VIF calculation against known values."""
       X = pd.DataFrame({
           'x1': [1, 2, 3, 4, 5],
           'x2': [2, 4, 6, 8, 10],  # Perfectly correlated with x1
           'x3': [1, 1, 2, 2, 3]
       })
       
       vif = calculate_vif(X)
       
       # x2 should have very high VIF (almost perfect correlation with x1)
       assert vif[vif['Variable'] == 'x2']['VIF'].values[0] > 50
   ```

---

## Phase 2: Multiple Comparison Corrections - 2 Hours

### Why: Prevents false discoveries when testing >1 analysis

**File to create:** `multiple_comparisons.py`

```python
"""
Multiple Comparison Correction Methods
For landmark survival analysis, subgroup analyses, etc.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

class MultipleComparisonCorrection:
    """
    Implements Bonferroni, Holm, Benjamini-Hochberg corrections.
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
```

### Integration with survival_lib.py:

```python
# In fit_km_landmark() function, add:

if n_landmarks > 1:
    from multiple_comparisons import MultipleComparisonCorrection
    
    mcc = MultipleComparisonCorrection()
    p_values = [stats_data['P-value'] for _ in range(n_landmarks)]
    
    # Apply Holm correction
    corrected_df, _ = mcc.holm(p_values)
    
    logger.warning(
        f"Multiple landmark times detected ({n_landmarks} tests). "
        f"Recommended significance threshold: 0.05/{n_landmarks} = {0.05/n_landmarks:.4f}"
    )
    
    # Add to output
    return fig, pd.concat([pd.DataFrame([stats_data]), corrected_df]), ...
```

---

## Phase 3: Test Coverage - 3-4 Hours ⭐ HIGHEST ROI

### Create comprehensive test file: `tests/test_statistical_validation.py`

```python
"""
Statistical Validation Tests
Compares stat-shiny implementations against known benchmarks
"""

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import statsmodels.api as sm
from lifelines import KaplanMeierFitter, CoxPHFitter

import sys
sys.path.insert(0, '..')

from logic import run_binary_logit, analyze_outcome
from survival_lib import fit_km_logrank, fit_cox_ph, calculate_median_survival
from table_one import generate_table, calculate_smd


class TestLogisticRegression:
    """Validation against statsmodels.api.Logit"""
    
    @pytest.fixture
    def binary_data(self):
        """Synthetic binary outcome data."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
        })
        y = pd.Series((
            0.5 + 0.8 * X['x1'] - 0.5 * X['x2'] + 
            np.random.normal(0, 0.3, n) > 0
        ).astype(int))
        return y, X
    
    def test_logit_coefficients_match_statsmodels(self, binary_data):
        """Our logit coefficients should match statsmodels."""
        y, X = binary_data
        
        # Our implementation
        our_params, our_conf, our_pvals, our_status, _ = run_binary_logit(y, X)
        
        # statsmodels baseline
        X_const = sm.add_constant(X)
        sm_model = sm.Logit(y, X_const)
        sm_result = sm_model.fit(disp=0)
        
        # Check coefficients match (excluding intercept)
        assert our_status == "OK"
        assert_allclose(
            our_params.values[1:],  # Skip const
            sm_result.params.values[1:],
            rtol=1e-4
        )
    
    def test_logit_ci_coverage(self, binary_data):
        """Confidence intervals should have correct bounds."""
        y, X = binary_data
        
        our_params, our_conf, our_pvals, our_status, _ = run_binary_logit(y, X)
        
        # Lower bound < Estimate < Upper bound
        assert (our_conf.iloc[:, 0] < our_params).all()
        assert (our_params < our_conf.iloc[:, 1]).all()


class TestSurvivalAnalysis:
    """Validation against lifelines."""
    
    @pytest.fixture
    def survival_data(self):
        """Synthetic survival data."""
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'T': np.random.exponential(10, n),
            'E': np.random.binomial(1, 0.6, n),
            'group': np.random.choice(['A', 'B'], n),
            'x1': np.random.normal(0, 1, n)
        })
        return df
    
    def test_km_median_matches_lifelines(self, survival_data):
        """Our KM median should match lifelines."""
        kmf_ours = KaplanMeierFitter()
        kmf_ours.fit(
            survival_data['T'], 
            survival_data['E'], 
            label='Test'
        )
        
        # Our implementation
        our_median_df = calculate_median_survival(
            survival_data, 'T', 'E', None
        )
        
        our_median = float(our_median_df['Median Time (95% CI)'].iloc[0].split()[0])
        
        # Compare
        assert_allclose(
            our_median,
            kmf_ours.median_survival_time_,
            rtol=0.1  # Allow 10% tolerance
        )
    
    def test_cox_hr_matches_lifelines(self, survival_data):
        """Our Cox HR should match lifelines."""
        cph_ours = CoxPHFitter()
        cph_ours.fit(survival_data, duration_col='T', event_col='E')
        
        # Our implementation
        cph_data = survival_data.copy()
        cph_data['x1_encoded'] = cph_data['group'].map({'A': 0, 'B': 1})
        
        our_cph, our_summary, our_data, our_status, our_stats = fit_cox_ph(
            cph_data, 'T', 'E', ['x1', 'x1_encoded']
        )
        
        # Extract HR
        our_hr = our_summary['HR'].iloc[0]
        lifelines_hr = np.exp(cph_ours.params_.iloc[0])
        
        # Should be close
        assert_allclose(our_hr, lifelines_hr, rtol=0.05)


class TestTableOne:
    """Validation of Table One generation."""
    
    @pytest.fixture
    def table_one_data(self):
        """Data for table one."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'age': np.random.normal(50, 10, n),
            'sex': np.random.choice([0, 1], n),
            'treatment': np.random.choice([0, 1], n),
            'outcome': np.random.binomial(1, 0.4, n)
        })
        return df
    
    def test_smd_calculation(self, table_one_data):
        """SMD should be in [0, ~2] range."""
        smd = calculate_smd(
            table_one_data,
            'age',
            'treatment',
            0, 1,
            is_cat=False
        )
        
        # SMD is a formatted string like "0.123"
        smd_float = float(smd.replace('<b>', '').replace('</b>', ''))
        
        assert 0 <= smd_float <= 3  # Sanity check


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Run tests:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest tests/test_statistical_validation.py -v --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## Phase 4: Documentation Updates - 1-2 Hours

### Add to docstrings in key functions:

```python
# In logic.py - analyze_outcome() docstring:

"""
Perform comprehensive logistic regression analysis.

STATISTICAL ASSUMPTIONS & LIMITATIONS:
- Outcome must be binary (0/1 or False/True)
- Uses listwise deletion for missing data (assumes MCAR)
- Univariate screening threshold: p < 0.20 (arbitrary, see Bursac et al. 2008)
- Firth regression used automatically if: perfect separation, n < 50, or <20 events
- Interaction terms tested only if explicitly specified in interaction_pairs

INTERPRETATION GUIDE:
- OR > 1: Increased odds of outcome
- OR < 1: Decreased odds of outcome  
- 95% CI contains 1.0: Not statistically significant (p ≥ 0.05)
- aOR: Adjusted for all variables in multivariate model

COLLINEARITY:
- Check VIF output before interpreting coefficients
- VIF > 10 indicates problematic multicollinearity
- Consider removing high-VIF variables

MULTIPLE COMPARISONS:
- If testing >1 interaction: Apply Bonferroni or Holm correction
- Unadjusted p-values have inflation
"""
```

---

## Priority Order for Implementation

1. **Week 1:** VIF collinearity check (highest impact)
2. **Week 2:** Test validation suite (quality assurance)
3. **Week 3:** Multiple comparison corrections
4. **Week 4:** Documentation & deployment prep

---

## Quick Testing Checklist

```bash
# Before each commit:

□ Run tests: pytest tests/ -v
□ Check coverage: pytest --cov=.
□ Lint code: flake8 *.py
□ Type check: mypy logic.py survival_lib.py
□ Manual test: Upload small CSV, verify results
□ Check error handling: Try bad data, confirm friendly message
```

---

## Success Metrics

After implementing these enhancements:

- ✅ Users get warning if VIF > 10 (prevents false interpretations)
- ✅ Test coverage rises from 40% → 75%+ (catches regressions)
- ✅ Multiple comparisons handled automatically (prevents p-hacking)
- ✅ Docstrings document all assumptions (reproducibility)

---

## Questions?

If you get stuck on any phase:
1. Check Python docs for `variance_inflation_factor`, `scipy.stats` functions
2. Reference lifelines docs for Cox/KM validation
3. Review statsmodels Logit for coefficient comparison

Good luck! 🚀
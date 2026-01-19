# Implementation Plan: Linear Regression (OLS) Module
## stat-shiny Continuous Outcome Analysis

**Current Date:** January 19, 2026  
**Repository:** https://github.com/NTWKKM/stat-shiny/tree/patch  
**Module Status:** NEW  
**Priority:** HIGH (Critical gap in continuous outcome analysis)

---

## 📋 Executive Summary

### Problem Statement
- ✅ Current system: Binary (Logit) + Count (Poisson) regression
- ❌ Missing: Continuous outcome (Linear Regression/OLS)
- 🔴 Impact: Cannot analyze continuous clinical variables (blood pressure, glucose, LOS)

### Solution
- Add **Linear Regression (OLS) Module** as subtab in `tabs/tab_logit.py`
- Create new statistical library: `utils/linear_lib.py`
- Implement assumption checking (Normality, Homoscedasticity)
- Interactive Plotly visualizations

### Expected Outcome
Medical researchers can analyze continuous outcomes with full diagnostic support

---

## 📊 Architecture Overview

```
END-TO-END WORKFLOW: Linear Regression Module

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: USER INTERFACE & INPUT (Frontend)                     │
│ File: tabs/tab_regression.py                                   │
│ - Display "Linear Regression" subtab in tab_logit.py          │
│ - Smart filtering: Show only numeric columns in dropdowns      │
│ - User selects: Outcome (Y) + Predictors (X)                  │
│ - Trigger: Click "🚀 Run Linear Regression" button            │
└─────────────────────────────────────────────────────────────────┘
                             ⬇
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: DATA PREPARATION & CLEANING (Data Layer)              │
│ File: utils/linear_lib.py + utils/data_cleaning.py            │
│ - Receive: Raw DataFrame with potential data quality issues    │
│ - Deep Clean: Apply clean_numeric_vector()                    │
│   • Convert ">10" → 10.0                                      │
│   • Convert "1,200" → 1200.0                                  │
│   • Handle empty strings → NaN                                 │
│ - Remove: Complete case deletion (.dropna())                  │
│ - Output: DataFrame ready for statistical modeling            │
└─────────────────────────────────────────────────────────────────┘
                             ⬇
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: STATISTICAL CALCULATION (Engine Layer)                │
│ File: utils/linear_lib.py                                     │
│ Tool: statsmodels.formula.api.ols()                          │
│ - Build formula: "BP ~ Age + BMI + Income"                   │
│ - Use Q() for variable names with spaces                     │
│ - Calculate: β coefficients, SE, t-stats, p-values, R²       │
│ - Extract: CI (Lower/Upper), RSE, F-statistic, adjusted R²   │
└─────────────────────────────────────────────────────────────────┘
                             ⬇
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: FORMATTING & PRESENTATION (Output Layer)              │
│ File: utils/linear_lib.py + utils/formatting.py              │
│ - Format p-values: 0.0000023 → "<0.001" with color highlight │
│ - Format CI: Color green if doesn't cross 0 (significant)    │
│ - Format coefficients: Round to appropriate decimals          │
│ - Create summary DataFrame ready for display                 │
└─────────────────────────────────────────────────────────────────┘
                             ⬇
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: VISUALIZATION & DIAGNOSTICS (Graph Layer)             │
│ File: utils/linear_lib.py + tabs/tab_regression.py           │
│ - Generate Plotly plots:                                      │
│   • Residuals vs Fitted (homoscedasticity check)             │
│   • QQ Plot (normality check)                                │
│   • Scale-Location Plot (optional, advanced)                │
│ - HTML Render: Convert to self-contained HTML with CDN      │
│ - Tool: utils/plotly_html_renderer.py                       │
└─────────────────────────────────────────────────────────────────┘
                             ⬇
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6: RENDERING (Display Layer)                             │
│ File: tabs/tab_regression.py                                   │
│ - Display results table with formatted values                 │
│ - Show diagnostic plots in tabs/expandable sections           │
│ - Enable download of results (CSV) and plots (PNG)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ File Structure & Changes

### NEW FILES
```
stat-shiny/
├── utils/
│   └── linear_lib.py                    [NEW] Statistical engine for Linear Regression
├── tabs/
│   └── tab_regression.py                [NEW] UI for Linear Regression module
```

### MODIFIED FILES
```
stat-shiny/
├── tabs/
│   └── tab_logit.py                     [UPDATE] Add subtab for Linear Regression
├── app.py                                [UPDATE] Register new tab in navigation
```

### EXISTING DEPENDENCIES (No changes needed)
```
stat-shiny/
├── utils/
│   ├── data_cleaning.py                 [EXISTING] clean_numeric_vector()
│   ├── formatting.py                    [EXISTING] format_p_value(), format_ci_html()
│   └── plotly_html_renderer.py          [EXISTING] plotly_figure_to_html()
```

---

## 📝 Phase-by-Phase Implementation Details

### PHASE 1: User Interface & Input

**File:** `tabs/tab_regression.py`

**Key Requirements:**
1. Create subtab named "Linear Regression" 
2. Add to tab_logit.py (alongside Binary Logit, Poisson)
3. UI Elements:
   - Outcome (Y) Dropdown: Filter numeric columns only
   - Predictor (X) Multi-select: Filter numeric columns only
   - Reference category selector (if needed for binary predictors)
   - Button: "🚀 Run Linear Regression"
4. Smart filtering logic:
   ```python
   numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
   ```

**Pseudocode:**
```python
def create_linear_regression_tab(df):
    numeric_cols = get_numeric_columns(df)
    
    # UI Layout
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            outcome_var = st.selectbox("Select Outcome (Y)", numeric_cols)
        with col2:
            predictor_vars = st.multiselect("Select Predictors (X)", numeric_cols)
        
        if st.button("🚀 Run Linear Regression"):
            results = run_linear_regression(df, outcome_var, predictor_vars)
            display_results(results)
```

---

### PHASE 2: Data Preparation & Cleaning

**File:** `utils/linear_lib.py` (NEW)

**Key Functions:**
1. `prepare_data_for_ols()` - Main orchestrator
2. `clean_ols_data()` - Deep cleaning pipeline
3. `validate_ols_inputs()` - Input validation

**Flow:**
```python
def prepare_data_for_ols(df, outcome_col, predictor_cols):
    """
    Prepare data for OLS regression analysis
    
    Args:
        df: Raw DataFrame
        outcome_col: Y variable name
        predictor_cols: List of X variable names
    
    Returns:
        Cleaned DataFrame, sample size, missing count
    """
    # Step 1: Extract relevant columns
    cols_needed = [outcome_col] + predictor_cols
    df_subset = df[cols_needed].copy()
    
    # Step 2: Deep clean numeric data
    df_cleaned = clean_ols_data(df_subset)
    
    # Step 3: Remove missing values
    n_before = len(df_cleaned)
    df_cleaned = df_cleaned.dropna()
    n_after = len(df_cleaned)
    
    # Step 4: Validation
    if n_after < 10:
        raise ValueError(f"Not enough complete cases (n={n_after})")
    
    return df_cleaned, n_before, n_after
```

**Cleaning Logic:**
```python
def clean_ols_data(df):
    """
    Deep clean numeric columns for OLS
    Handles: ">10", "1,200", empty strings, etc.
    """
    from utils.data_cleaning import clean_numeric_vector
    
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = clean_numeric_vector(df_clean[col])
    
    return df_clean
```

---

### PHASE 3: Statistical Calculation (The Engine)

**File:** `utils/linear_lib.py`

**Key Function:** `run_ols_regression()`

**Implementation:**
```python
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

def run_ols_regression(df, outcome_col, predictor_cols):
    """
    Execute OLS regression using statsmodels
    
    Args:
        df: Cleaned DataFrame
        outcome_col: Y variable name
        predictor_cols: List of X variable names
    
    Returns:
        Dictionary with model, coefficients, diagnostics
    """
    # Step 1: Build formula with Q() for variable names with spaces
    predictors_str = " + ".join([f"Q('{col}')" for col in predictor_cols])
    formula = f"Q('{outcome_col}') ~ {predictors_str}"
    
    # Step 2: Fit OLS model
    model = smf.ols(formula, data=df).fit()
    
    # Step 3: Extract results
    results_dict = {
        'model': model,
        'formula': formula,
        'n_obs': model.nobs,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'residual_std_err': np.sqrt(model.mse_resid),
        'coef_table': extract_coefficients(model),
        'residuals': model.resid,
        'fitted_values': model.fittedvalues,
        'residuals_standardized': model.resid_pearson,
    }
    
    # Step 4: Calculate VIF for multicollinearity
    vif_data = calculate_vif(df, predictor_cols)
    results_dict['vif_table'] = vif_data
    
    return results_dict
```

**Extract Coefficients:**
```python
def extract_coefficients(model):
    """
    Extract and format coefficient table from model
    """
    coef_df = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std. Error': model.bse.values,
        'T-value': model.tvalues.values,
        'P-value': model.pvalues.values,
        'CI_Lower': model.conf_int()[0].values,
        'CI_Upper': model.conf_int()[1].values,
    })
    
    return coef_df
```

**Calculate VIF (Multicollinearity Check):**
```python
def calculate_vif(df, predictor_cols):
    """
    Calculate Variance Inflation Factor for diagnostics
    VIF > 5-10 indicates multicollinearity
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = predictor_cols
    vif_data["VIF"] = [
        variance_inflation_factor(df[predictor_cols].values, i) 
        for i in range(len(predictor_cols))
    ]
    
    return vif_data
```

---

### PHASE 4: Formatting & Presentation

**File:** `utils/linear_lib.py` + `utils/formatting.py`

**Key Function:** `format_ols_results()`

```python
def format_ols_results(results_dict):
    """
    Format raw model results for presentation
    """
    from utils.formatting import format_p_value, format_ci_html
    
    coef_df = results_dict['coef_table'].copy()
    
    # Format p-values: 0.0000023 -> "<0.001"
    coef_df['P-value_formatted'] = coef_df['P-value'].apply(format_p_value)
    
    # Format coefficients to 3 decimals
    coef_df['Coefficient_formatted'] = coef_df['Coefficient'].apply(lambda x: f"{x:.3f}")
    coef_df['SE_formatted'] = coef_df['Std. Error'].apply(lambda x: f"{x:.3f}")
    coef_df['T_formatted'] = coef_df['T-value'].apply(lambda x: f"{x:.3f}")
    
    # Format CI with HTML color
    coef_df['CI_formatted'] = coef_df.apply(
        lambda row: format_ci_html(row['CI_Lower'], row['CI_Upper']),
        axis=1
    )
    
    # Mark significance
    coef_df['Significant'] = coef_df['P-value'] < 0.05
    
    # Create display table
    display_df = coef_df[[
        'Variable',
        'Coefficient_formatted',
        'SE_formatted',
        'T_formatted',
        'P-value_formatted',
        'CI_formatted'
    ]].copy()
    
    display_df.columns = ['Variable', 'β', 'SE', 't-value', 'p-value', '95% CI']
    
    return display_df, results_dict['vif_table']
```

---

### PHASE 5: Visualization & Diagnostics

**File:** `utils/linear_lib.py`

**Key Diagnostics:**

```python
def create_diagnostic_plots(results_dict):
    """
    Create Plotly diagnostic plots for assumption checking
    
    Returns:
        Dictionary with 3 plots: residuals_vs_fitted, qq_plot, scale_location
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy import stats
    
    model = results_dict['model']
    residuals = results_dict['residuals']
    fitted = results_dict['fitted_values']
    
    plots = {}
    
    # Plot 1: Residuals vs Fitted (Homoscedasticity check)
    plots['residuals_vs_fitted'] = create_residuals_vs_fitted_plot(
        fitted, residuals
    )
    
    # Plot 2: QQ Plot (Normality check)
    plots['qq_plot'] = create_qq_plot(residuals)
    
    # Plot 3: Scale-Location Plot (Variance homogeneity)
    plots['scale_location'] = create_scale_location_plot(
        fitted, residuals
    )
    
    return plots
```

**Residuals vs Fitted Plot:**
```python
def create_residuals_vs_fitted_plot(fitted_values, residuals):
    """
    Residuals vs Fitted Values plot
    Check for: Random scatter (good), Patterns (bad - nonlinearity)
    """
    import plotly.graph_objects as go
    from scipy.signal import savgol_filter
    
    fig = go.Figure()
    
    # Add scatter
    fig.add_trace(go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='blue', size=5, opacity=0.6),
        name='Residuals',
        hovertemplate='<b>Fitted:</b> %{x:.3f}<br><b>Residual:</b> %{y:.3f}<extra></extra>'
    ))
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
    
    # Add smooth trend line (LOWESS-like)
    sorted_idx = np.argsort(fitted_values)
    fig.add_trace(go.Scatter(
        x=fitted_values[sorted_idx],
        y=residuals.iloc[sorted_idx].rolling(window=5, center=True).mean(),
        mode='lines',
        line=dict(color='red', width=2),
        name='Trend',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Residuals vs Fitted Values',
        xaxis_title='Fitted Values',
        yaxis_title='Residuals',
        template='plotly_white',
        hovermode='closest',
        height=500,
        width=700
    )
    
    return fig
```

**QQ Plot (Normality Check):**
```python
def create_qq_plot(residuals):
    """
    Q-Q Plot for normality assessment
    Points close to line = normally distributed
    """
    import plotly.graph_objects as go
    from scipy import stats
    
    # Calculate theoretical quantiles
    quantiles = stats.probplot(residuals)
    theoretical_quantiles = quantiles[0][0]
    sample_quantiles = quantiles[0][1]
    
    # Create line
    line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    line_y = np.array([sample_quantiles.min(), sample_quantiles.max()])
    
    fig = go.Figure()
    
    # Add scatter
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        marker=dict(color='blue', size=6, opacity=0.7),
        name='Residuals',
        hovertemplate='<b>Theory:</b> %{x:.3f}<br><b>Sample:</b> %{y:.3f}<extra></extra>'
    ))
    
    # Add reference line
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Normal Line',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Normal Q-Q Plot',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        template='plotly_white',
        hovermode='closest',
        height=500,
        width=700
    )
    
    return fig
```

**Scale-Location Plot (Variance Homogeneity):**
```python
def create_scale_location_plot(fitted_values, residuals):
    """
    Scale-Location plot (sqrt of standardized residuals)
    Checks for homoscedasticity
    """
    import plotly.graph_objects as go
    from scipy.stats import zscore
    
    # Standardized residuals
    standardized_residuals = zscore(residuals)
    sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
    
    fig = go.Figure()
    
    # Add scatter
    fig.add_trace(go.Scatter(
        x=fitted_values,
        y=sqrt_abs_resid,
        mode='markers',
        marker=dict(color='blue', size=5, opacity=0.6),
        name='|Standardized Residuals|^0.5',
        hovertemplate='<b>Fitted:</b> %{x:.3f}<br><b>√|Std. Resid|:</b> %{y:.3f}<extra></extra>'
    ))
    
    # Add trend line
    sorted_idx = np.argsort(fitted_values)
    fig.add_trace(go.Scatter(
        x=fitted_values[sorted_idx],
        y=sqrt_abs_resid.iloc[sorted_idx].rolling(window=5, center=True).mean(),
        mode='lines',
        line=dict(color='red', width=2),
        name='Trend',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Scale-Location Plot',
        xaxis_title='Fitted Values',
        yaxis_title='√|Standardized Residuals|',
        template='plotly_white',
        hovermode='closest',
        height=500,
        width=700
    )
    
    return fig
```

---

### PHASE 6: Rendering & Display

**File:** `tabs/tab_regression.py`

**Main Display Function:**

```python
def display_ols_results(results_dict, formatted_results, vif_table):
    """
    Display all OLS results and diagnostics in Streamlit
    """
    import streamlit as st
    from utils.plotly_html_renderer import plotly_figure_to_html
    
    # Section 1: Model Summary
    with st.container():
        st.subheader("📊 Model Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sample Size", results_dict['n_obs'])
        with col2:
            st.metric("R²", f"{results_dict['r_squared']:.4f}")
        with col3:
            st.metric("Adj. R²", f"{results_dict['adj_r_squared']:.4f}")
        with col4:
            st.metric("RSE", f"{results_dict['residual_std_err']:.4f}")
        
        # F-test
        st.write(f"""
        **Overall F-test:** F({results_dict['f_statistic']:.2f}) = {results_dict['f_pvalue']:.2e}
        """)
    
    # Section 2: Coefficient Table
    st.subheader("📋 Coefficients")
    st.dataframe(
        formatted_results,
        use_container_width=True,
        hide_index=True
    )
    
    # Section 3: Multicollinearity Check (VIF)
    st.subheader("🔍 Multicollinearity Check (VIF)")
    st.dataframe(
        vif_table,
        use_container_width=True,
        hide_index=True
    )
    st.write("**Interpretation:** VIF < 5 (good), 5-10 (moderate concern), > 10 (problematic)")
    
    # Section 4: Diagnostic Plots
    st.subheader("📈 Assumption Diagnostics")
    
    tab1, tab2, tab3 = st.tabs([
        "Residuals vs Fitted",
        "Q-Q Plot",
        "Scale-Location"
    ])
    
    with tab1:
        st.plotly_chart(
            results_dict['plots']['residuals_vs_fitted'],
            use_container_width=True
        )
        st.write("""
        **What to look for:**
        - ✅ Random scatter around red line = good (homoscedasticity)
        - ❌ Funnel pattern = variance increases with fitted value (heteroscedasticity)
        - ❌ Curved pattern = nonlinear relationship (need transformation)
        """)
    
    with tab2:
        st.plotly_chart(
            results_dict['plots']['qq_plot'],
            use_container_width=True
        )
        st.write("""
        **What to look for:**
        - ✅ Points close to red line = normally distributed residuals
        - ❌ Points deviate at tails = heavy-tailed or skewed distribution
        - ❌ S-shape = non-normal distribution
        """)
    
    with tab3:
        st.plotly_chart(
            results_dict['plots']['scale_location'],
            use_container_width=True
        )
        st.write("""
        **What to look for:**
        - ✅ Horizontal trend line = constant variance (homoscedasticity)
        - ❌ Upward/downward trend = variance changes with fitted value
        """)
    
    # Section 5: Download Results
    st.subheader("📥 Download Results")
    
    # CSV download
    csv = formatted_results.to_csv(index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv,
        file_name="linear_regression_results.csv",
        mime="text/csv"
    )
```

---

## 🛠️ Development Checklist

### Phase 1: Core Engine Development
- [ ] Create `utils/linear_lib.py` with basic structure
- [ ] Implement `prepare_data_for_ols()` with validation
- [ ] Implement `run_ols_regression()` with statsmodels
- [ ] Test with sample medical data (blood pressure, age, BMI)
- [ ] Verify formula construction with special characters

### Phase 2: Formatting & Output
- [ ] Implement `format_ols_results()`
- [ ] Integrate with existing `utils/formatting.py`
- [ ] Test p-value formatting (especially <0.001)
- [ ] Test CI formatting with color highlighting
- [ ] Add VIF calculation for multicollinearity

### Phase 3: Diagnostics & Visualization
- [ ] Create `create_diagnostic_plots()`
- [ ] Implement residuals vs fitted plot
- [ ] Implement QQ plot
- [ ] Implement scale-location plot
- [ ] Test all plots with various data distributions

### Phase 4: UI & Integration
- [ ] Create `tabs/tab_regression.py`
- [ ] Add subtab to `tabs/tab_logit.py`
- [ ] Implement numeric column filtering
- [ ] Add run button with error handling
- [ ] Test end-to-end with sample data

### Phase 5: Testing & Validation
- [ ] Unit tests for data cleaning
- [ ] Unit tests for statistical calculations
- [ ] Integration tests end-to-end
- [ ] Test with edge cases (small samples, missing data, single predictor)
- [ ] Cross-check results with R lm() or Python sm.OLS
- [ ] Performance testing with large datasets

### Phase 6: Documentation & Deployment
- [ ] Code documentation with docstrings
- [ ] User guide with interpretation tips
- [ ] Add help text for diagnostic plots
- [ ] Deploy to patch branch for testing
- [ ] Collect feedback and iterate

---

## 💡 Future Enhancements (Phase 2)

- [ ] Robust regression (Huber M-estimator)
- [ ] Weighted regression for survey data
- [ ] Variable selection (step-wise, LASSO)
- [ ] Residual diagnostics (leverage, influence)
- [ ] Non-linear transformations (log, sqrt, polynomial)
- [ ] Interaction effects visualization
- [ ] Model comparison (AIC, BIC)
- [ ] Bootstrap confidence intervals
- [ ] Power analysis for sample size calculation

---

## 🧪 Testing Strategy

### Unit Tests

```python
# tests/test_linear_lib.py
import pytest
import pandas as pd
import numpy as np
from utils.linear_lib import prepare_data_for_ols, run_ols_regression

class TestDataPreparation:
    def test_clean_numeric_conversion(self):
        """Test conversion of ">10" to 10.0"""
        pass
    
    def test_missing_value_handling(self):
        """Test dropna() removes incomplete cases"""
        pass
    
    def test_sample_size_validation(self):
        """Test minimum sample size requirement"""
        pass

class TestOLSRegression:
    def test_single_predictor(self):
        """Test simple linear regression"""
        pass
    
    def test_multiple_predictors(self):
        """Test multiple regression"""
        pass
    
    def test_variable_names_with_spaces(self):
        """Test Q() formula construction"""
        pass
    
    def test_coefficient_extraction(self):
        """Verify coefficients match statsmodels"""
        pass

class TestFormatting:
    def test_p_value_formatting(self):
        """Test p-value display <0.001 for small values"""
        pass
    
    def test_ci_highlighting(self):
        """Test CI color based on significance"""
        pass
```

### Integration Tests

```python
# tests/test_linear_regression_ui.py
def test_full_workflow():
    """
    End-to-end test: Data → Cleaning → Analysis → Display
    """
    # 1. Load sample medical data
    df = load_sample_medical_data()
    
    # 2. Run linear regression
    results = run_ols_regression(df, 'BP', ['Age', 'BMI'])
    
    # 3. Format results
    formatted, vif = format_ols_results(results)
    
    # 4. Verify output structure
    assert 'Coefficient' in formatted.columns
    assert 'p-value' in formatted.columns
    assert len(vif) == 2  # 2 predictors
    
    # 5. Cross-check with known values
    # (Compare against R lm() or manual calculation)
```

---

## 📊 Expected Results & Sample Output

### Input
```python
df = {
    'SystolicBP': [120, 135, 128, 142, 138, ...],
    'Age': [45, 52, 48, 65, 58, ...],
    'BMI': [24.5, 28.3, 25.1, 30.2, 27.8, ...],
    'Income': [35000, 52000, 41000, 68000, 55000, ...]
}

outcome = 'SystolicBP'
predictors = ['Age', 'BMI', 'Income']
```

### Output: Coefficient Table
```
Variable          β        SE      t-value    p-value      95% CI
─────────────────────────────────────────────────────────────
Intercept        50.123   12.456   4.024     <0.001       [25.892 - 74.354]
Age              0.523    0.145    3.603     <0.001       [0.239 - 0.807]
BMI              1.245    0.321    3.876     <0.001       [0.616 - 1.874]
Income           0.0001   0.0002   0.512     0.609        [-0.0003 - 0.0005]
```

### Model Summary
```
Sample Size: 248
R²: 0.6234
Adj. R²: 0.6145
RSE: 8.234
F(3, 244) = 70.23, p < 0.001
```

### VIF Table
```
Variable    VIF
─────────────────
Age         2.145
BMI         1.892
Income      1.456
```

---

## 🚀 Deployment & Rollout

### Branch Strategy
```
main (stable)
└── patch (testing/development)
    └── feature/linear-regression (your feature branch)
        ├── utils/linear_lib.py
        ├── tabs/tab_regression.py
        └── tests/test_linear_lib.py
```

### Testing Timeline
1. **Local Testing** (2-3 days): Unit + integration tests
2. **Code Review** (1 day): Review with team/collaborators
3. **Beta Testing** (3-5 days): Get feedback from medical researchers
4. **Deployment to patch** (1 day): Merge to patch branch
5. **Production Ready** (1-2 weeks): After validation, merge to main

### Documentation
- [ ] Add section to README.md about Linear Regression
- [ ] Include interpretation guide for diagnostics
- [ ] Provide examples with medical datasets
- [ ] Add FAQ for common issues

---

## 📚 References & Best Practices

### Statistical Resources
- Hosmer, Lemeshow, Sturdivant (2013). *Applied Logistic Regression* (3rd ed.)
- James, Witten, Hastie, Tibshirani (2013). *An Introduction to Statistical Learning*
- Assumption checking: https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faqwhat-does-the-p-value-represent/

### Implementation References
- statsmodels documentation: https://www.statsmodels.org/stable/regression.html
- Plotly diagnostics: https://plotly.com/python/qq-plots/
- Streamlit best practices: https://docs.streamlit.io/

### Quality Assurance
- Cross-validate results against R `lm()` function
- Test edge cases: collinearity, outliers, non-normal residuals
- Performance: Benchmark with n=10,000+ observations
- Browser compatibility: Test plot rendering across browsers

---

## 📞 Support & Questions

For questions about implementation:
- Review statsmodels documentation for OLS specifics
- Check existing `utils/data_cleaning.py` for cleaning patterns
- Refer to `utils/formatting.py` for output formatting examples
- Test diagnostic plots with `plotly` official examples

---


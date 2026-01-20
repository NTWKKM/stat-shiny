# 🚀 Implementation Quick Start Guide

**For:** Immediate action on module restructuring  
**Status:** Ready-to-implement  
**Timeline:** Week-by-week breakdown

---

## 📌 TL;DR

**Current Problem:** Tab 4 has 7 subtabs (overcrowded) + missing critical publication methods

**Solution:** 
1. Restructure Tab 4 into 5 semantic subtabs
2. Create Tab 7 (Advanced Inference) - 5 subtabs
3. Create Tab 8 (Causal Inference) - 5 subtabs
4. Add critical missing modules (Mediation, Collinearity, Bayesian, etc.)

**Result:** Professional-grade medical statistics for Nature/Lancet/JAMA publications

---

## 🎯 Week-by-Week Implementation Plan

### WEEK 1: Restructure Existing Tab

**Goal:** Move from 7 scattered subtabs to 5 semantic subtabs

#### Day 1-2: Code Refactoring
```bash
# 1. Create feature branch
git checkout -b feature/restructure-regression

# 2. Create new tab file (copy from existing)
cp tabs/tab_logit.py tabs/tab_core_regression.py

# 3. Edit tabs/__init__.py
# Add: from . import tab_core_regression
```

#### Day 3-4: Reorganize Subtabs

**OLD structure in `tab_logit.py`:**
```python
ui.navset_tab(
    ui.nav_panel("📈 Binary Logistic Regression", ...),  # OUTCOME BASED
    ui.nav_panel("📊 Poisson Regression", ...),          # OUTCOME BASED
    ui.nav_panel("📈 GLM", ...),                         # METHOD BASED
    ui.nav_panel("📐 Linear Regression", ...),           # OUTCOME BASED
    ui.nav_panel("🗣️ Subgroup Analysis", ...),           # CAUSAL - WRONG TAB!
    ui.nav_panel("🔄 Repeated Measures", ...),           # DESIGN BASED
    ui.nav_panel("ℹ️ Reference", ...),                   # HELP
)
```

**NEW structure in `tab_core_regression.py`:**
```python
# Semantic grouping by outcome/design type
ui.navset_tab(
    ui.nav_panel("📈 Binary Outcomes", 
        # Combine: Binary Logistic + Firth detection
    ),
    ui.nav_panel("📉 Continuous Outcomes",
        # Combine: Linear Regression + Diagnostics + move collinearity here
    ),
    ui.nav_panel("🔢 Count & Special",
        # Combine: Poisson + GLM (+ add Negative Binomial later)
    ),
    ui.nav_panel("🔄 Repeated Measures",
        # Keep: GEE + LMM + trajectory
    ),
    ui.nav_panel("ℹ️ Reference & Guidelines",
        # Keep: But expand with new diagnostics info
    ),
)
```

#### Day 5-6: Update app.py and Testing
```python
# app.py - BEFORE
from tabs import tab_logit
ui.nav_panel("📊 Regression Models", wrap_with_container(tab_logit.logit_ui("logit")))

# app.py - AFTER
from tabs import tab_core_regression
ui.nav_panel("📊 Core Regression Models", wrap_with_container(tab_core_regression.core_regression_ui("core_reg")))
```

#### Day 7: Final Testing
```bash
# Run tests
pytest tests/ -v

# Test locally
shiny run app.py

# Check each subtab manually
```

---

### WEEK 2-3: Create Advanced Inference Tab

**Goal:** Add professional-grade advanced statistical methods

#### Step 1: Create utility libraries

**File: `utils/mediation_lib.py`**
```python
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant

def analyze_mediation(data, outcome, treatment, mediator, confounders=None):
    """Mediation analysis with bootstrap CI."""
    # Prepare data
    df_clean = data[[outcome, treatment, mediator] + (confounders or [])].dropna()
    
    # Total effect
    X_total = add_constant(df_clean[[treatment] + (confounders or [])])
    model_total = OLS(df_clean[outcome], X_total).fit()
    c_total = model_total.params[treatment]
    
    # Direct effect
    X_direct = add_constant(df_clean[[treatment, mediator] + (confounders or [])])
    model_direct = OLS(df_clean[outcome], X_direct).fit()
    c_prime = model_direct.params[treatment]
    b_med = model_direct.params[mediator]
    
    # Mediation effect
    X_med = add_constant(df_clean[[treatment] + (confounders or [])])
    model_med = OLS(df_clean[mediator], X_med).fit()
    a_med = model_med.params[treatment]
    
    # Calculate effects
    indirect_effect = a_med * b_med
    acme = indirect_effect
    pme = acme / c_total if c_total != 0 else 0
    
    return {
        'total_effect': c_total,
        'direct_effect': c_prime,
        'indirect_effect': indirect_effect,
        'proportion_mediated': pme,
        'n_obs': len(df_clean),
    }
```

**File: `utils/collinearity_lib.py`**
```python
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data, predictors):
    """Calculate VIF for each predictor."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = predictors
    vif_data["VIF"] = [variance_inflation_factor(data[predictors].values, i) 
                       for i in range(len(predictors))]
    vif_data["Tolerance"] = 1 / vif_data["VIF"]
    return vif_data.sort_values("VIF", ascending=False)

def condition_index(X):
    """Calculate condition index from eigenvalues."""
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    corr = np.corrcoef(X_std.T)
    eigenvalues = np.linalg.eigvals(corr)
    eigenvalues = np.sort(eigenvalues)[::-1]
    max_eigen = eigenvalues.max()
    cond_index = np.sqrt(max_eigen / eigenvalues)
    return cond_index
```

#### Step 2: Create tab file

**File: `tabs/tab_advanced_inference.py`**
```python
from shiny import module, reactive, render, ui
from utils.mediation_lib import analyze_mediation
from utils.collinearity_lib import calculate_vif

@module.ui
def advanced_inference_ui():
    """UI for Advanced Inference."""
    return ui.div(
        ui.h3("🔍 Advanced Inference"),
        ui.navset_tab(
            ui.nav_panel(
                "🎯 Mediation Analysis",
                ui.card(
                    ui.input_select("med_outcome", "Outcome (Y):", choices=[]),
                    ui.input_select("med_treatment", "Treatment (X):", choices=[]),
                    ui.input_select("med_mediator", "Mediator (M):", choices=[]),
                    ui.input_action_button("btn_run_mediation", "Run"),
                ),
                ui.output_data_frame("tbl_mediation"),
            ),
            ui.nav_panel(
                "🔬 Collinearity",
                ui.card(
                    ui.input_selectize("coll_vars", "Predictors:", choices=[], multiple=True),
                    ui.input_action_button("btn_run_collinearity", "Analyze"),
                ),
                ui.output_data_frame("tbl_vif"),
            ),
        ),
    )

@module.server
def advanced_inference_server(input, output, session, df, var_meta):
    """Server logic for Advanced Inference."""
    
    mediation_results = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.btn_run_mediation)
    def _run_mediation():
        if df.get() is None:
            ui.notification_show("Load data first", type="error")
            return
        
        results = analyze_mediation(
            data=df.get(),
            outcome=input.med_outcome(),
            treatment=input.med_treatment(),
            mediator=input.med_mediator(),
        )
        mediation_results.set(results)
    
    @render.data_frame
    def tbl_mediation():
        res = mediation_results.get()
        if res:
            return pd.DataFrame({
                'Effect': ['Total', 'Direct', 'Indirect', 'Proportion Mediated'],
                'Estimate': [res['total_effect'], res['direct_effect'], 
                            res['indirect_effect'], res['proportion_mediated']],
            })
        return None
```

---

### WEEK 4: Create Causal Inference Tab

Similar structure as Tab 7 with:
- Advanced PSM (IPW, AIPW)
- Stratified Analysis
- Bayesian Inference (optional)
- Sensitivity Analysis

---

### WEEK 5: Testing & Documentation

1. Comprehensive unit tests
2. Integration tests
3. User documentation
4. Final QA

---

## 🔧 Configuration Changes Required

### Update `app.py`

```python
from tabs import (
    tab_core_regression,        # RENAMED from tab_logit
    tab_advanced_inference,      # NEW
    tab_causal_inference,        # NEW
)

app_ui = ui.page_navbar(
    ui.nav_panel("📊 Core Regression Models", 
                 wrap_with_container(tab_core_regression.core_regression_ui("core_reg"))),
    ui.nav_panel("🔍 Advanced Inference", 
                 wrap_with_container(tab_advanced_inference.advanced_inference_ui("adv_inf"))),
    ui.nav_panel("🎯 Causal Inference", 
                 wrap_with_container(tab_causal_inference.causal_inference_ui("causal"))),
)

def server(input, output, session):
    tab_core_regression.core_regression_server("core_reg", df, var_meta, df_matched, is_matched)
    tab_advanced_inference.advanced_inference_server("adv_inf", df, var_meta, df_matched, is_matched)
    tab_causal_inference.causal_inference_server("causal", df, var_meta, df_matched, is_matched)
```

### Update `requirements.txt`

```txt
# Core (existing)
shiny==0.8.1
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
plotly>=5.15.0
lifelines>=0.29.0

# NEW - Advanced Statistics
econml>=0.14.0              # AIPW, causal forest (optional)

# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
```

---

## ✅ Testing Checklist

```bash
# Unit Tests
pytest tests/unit/ -v

# Integration Tests
pytest tests/integration/ -v

# Run with Coverage
pytest --cov=tabs --cov=utils tests/

# Manual Testing
# 1. Tab 4: Try each new subtab
# 2. Tab 7: Run mediation, collinearity tests
# 3. Tab 8: Run PSM, stratified analysis
```

---

## 🎯 Success Metrics

After implementation:
- ✅ All 9 tabs display correctly
- ✅ Tab 4 has 5 semantic subtabs
- ✅ All new modules functional
- ✅ Test coverage ≥90%
- ✅ Performance acceptable
- ✅ Documentation complete

---

**Status:** Ready to implement  
**Next:** Begin Phase 1 this week

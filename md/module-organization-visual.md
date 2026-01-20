# 📊 Visual Module Organization Summary

## Current State vs. Proposed State

### CURRENT STRUCTURE (7 tabs)
```
┌─────────────────────────────────────────────────────────────┐
│  MAIN NAVIGATION BAR                                        │
├─────────────────────────────────────────────────────────────┤
│ 📁 Data | 📋 Table1 | 🧪 Diag | 📊 Regression | 📈 Corr | ⏳ Surv | ⚙️ Set │
└─────────────────────────────────────────────────────────────┘
                              ▼
                    TAB 4: 📊 Regression Models
                    (CURRENTLY: 7 internal subtabs)
                    ├─ 📈 Binary Logistic ❌ MIXED PURPOSES
                    ├─ 📊 Poisson
                    ├─ 📈 GLM
                    ├─ 📐 Linear
                    ├─ 🗣️ Subgroup (SHOULD BE CAUSAL)
                    ├─ 🔄 Repeated Measures
                    └─ ℹ️ Reference
                    
⚠️ PROBLEMS:
• Tab 4 is overcrowded (7 subtabs = cognitive overload)
• Subgroup analysis mixed with regression (semantic mismatch)
• Advanced inference topics missing (mediation, collinearity)
• Causal methods not organized (PSM scattered)
• No publication workflow support
```

---

### PROPOSED STRUCTURE (9 tabs)
```
┌────────────────────────────────────────────────────────────────────┐
│  MAIN NAVIGATION BAR (ENHANCED)                                    │
├────────────────────────────────────────────────────────────────────┤
│ 📁 Data | 📋 T1 | 🧪 Diag | 📊 Regr* | 📈 Corr | ⏳ Surv | 🔍 Adv* │
│ 🎯 Causal* | ⚙️ Set │
└────────────────────────────────────────────────────────────────────┘
     ▼                    ▼                           ▼                ▼
TAB 4 RESTRUCTURED    TAB 7 NEW               TAB 8 NEW
  (5 subtabs)         (5 subtabs)            (5 subtabs)
  
Tab 4: 📊 Core Regression Models (REORGANIZED)
├─ 📈 Binary Outcomes
│  ├─ Logistic Regression (Standard + Firth)
│  ├─ Perfect Separation Detection
│  ├─ Forest Plot & Diagnostics
│  └─ Publication Table
├─ 📉 Continuous Outcomes
│  ├─ Linear Regression (OLS)
│  ├─ Stepwise Selection
│  ├─ Bootstrap CI
│  ├─ Diagnostic Plots
│  └─ ⭐ Collinearity (ADDED - from Advanced)
├─ 🔢 Count & Special
│  ├─ Poisson Regression
│  ├─ ⭐ Negative Binomial (NEW)
│  ├─ GLM
│  └─ ⭐ Zero-Inflated (NEW)
├─ 🔄 Repeated Measures
│  ├─ GEE
│  ├─ LMM
│  └─ Trajectory Plots
└─ ℹ️ Reference

Tab 7: 🔍 Advanced Inference (NEW - Advanced Statistics)
├─ 🎯 Causal Pathways (Mediation)
│  ├─ ⭐ Direct/Indirect Effects
│  ├─ ⭐ Bootstrap CI for ACME
│  ├─ ⭐ Visualization
│  └─ ⭐ Publication Table
├─ 🔬 Collinearity & Diagnostics
│  ├─ ⭐ VIF Analysis
│  ├─ ⭐ Tolerance & Condition Index
│  ├─ ⭐ Variance Decomposition
│  └─ ⭐ Heatmap
├─ 📊 Model Diagnostics
│  ├─ ⭐ RESET Test
│  ├─ ⭐ Heteroscedasticity Tests
│  ├─ ⭐ Influential Observations
│  └─ ⭐ Remedial Actions Guide
├─ 🏥 Heterogeneity Testing
│  ├─ ⭐ I² Index
│  ├─ ⭐ Q-statistic
│  ├─ ⭐ Tau² Estimation
│  └─ ⭐ Forest Plot with I²
└─ ℹ️ Reference & Interpretation

Tab 8: 🎯 Causal Inference (NEW - Causal Methods)
├─ 🎲 PSM Methods (Advanced)
│  ├─ ⭐ 1:1 Optimal Matching
│  ├─ ⭐ IPW (Inverse Probability Weighting)
│  ├─ ⭐ AIPW (Augmented IPW)
│  ├─ ⭐ Love Plot (Balance Check)
│  ├─ ⭐ Rosenbaum Bounds
│  └─ ⭐ Effect Estimation
├─ 📊 Stratified Analysis
│  ├─ ⭐ Mantel-Haenszel
│  ├─ ⭐ Breslow-Day Test
│  ├─ ⭐ Interaction Testing
│  └─ ⭐ Stratified Forest Plot
├─ 🔬 Bayesian Inference (OPTIONAL)
│  ├─ ⭐ Prior Specification
│  ├─ ⭐ MCMC Computation
│  ├─ ⭐ Credible Intervals
│  └─ ⭐ Sensitivity Analysis
├─ 📈 Sensitivity Analysis
│  ├─ ⭐ E-value Calculation
│  ├─ ⭐ Rosenbaum Bounds
│  └─ ⭐ Impact Interpretation
└─ ℹ️ Reference & DAGs

✅ IMPROVEMENTS:
• Each tab has 5-6 subtabs (manageable)
• Semantic organization (related methods grouped)
• All publication-required methods included
• Causal methods properly organized
• Room for expansion
• Professional workflow support
```

---

## Module Classification Matrix

```
┌──────────────────┬──────────────────┬────────────────────────────┐
│ Category         │ Publication Tier │ Location in New Structure   │
├──────────────────┼──────────────────┼────────────────────────────┤
│ CORE METHODS                                                      │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Logistic Reg     │ ⭐⭐⭐ Critical   │ Tab 4 / Binary Outcomes    │
│ Linear Reg       │ ⭐⭐⭐ Critical   │ Tab 4 / Continuous         │
│ Poisson Reg      │ ⭐⭐⭐ Critical   │ Tab 4 / Count & Special    │
│ GLM              │ ⭐⭐⭐ Critical   │ Tab 4 / Count & Special    │
│ Survival (Cox)   │ ⭐⭐⭐ Critical   │ Tab 6 (existing)           │
├──────────────────┼──────────────────┼────────────────────────────┤
│ REQUIRED FEATURES                                                 │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Firth's Method   │ ⭐⭐⭐ Critical   │ Tab 4 / Binary Outcomes    │
│ Negative Binomial│ ⭐⭐⭐ Critical   │ Tab 4 / Count & Special    │
│ Collinearity     │ ⭐⭐⭐ Critical   │ Tab 4 / Continuous         │
│ Model Diagnostics│ ⭐⭐⭐ Critical   │ Tab 4 / Continuous + Tab 7 │
│ Bootstrap CI     │ ⭐⭐⭐ Critical   │ Tab 4 / Continuous         │
├──────────────────┼──────────────────┼────────────────────────────┤
│ CAUSAL METHODS                                                    │
├──────────────────┼──────────────────┼────────────────────────────┤
│ PSM (Basic)      │ ⭐⭐ Important   │ Tab 2 (existing)           │
│ PSM (Advanced)   │ ⭐⭐⭐ Critical   │ Tab 8 / PSM Methods        │
│ IPW/AIPW         │ ⭐⭐⭐ Critical   │ Tab 8 / PSM Methods        │
│ Stratified Anal. │ ⭐⭐ Important   │ Tab 8 / Stratified Anal.   │
│ Sensitivity      │ ⭐⭐ Important   │ Tab 8 / Sensitivity Anal.  │
├──────────────────┼──────────────────┼────────────────────────────┤
│ ADVANCED STATS                                                    │
├──────────────────┼──────────────────┼────────────────────────────┤
│ Mediation        │ ⭐⭐⭐ Critical   │ Tab 7 / Causal Pathways    │
│ VIF/Tolerance    │ ⭐⭐⭐ Critical   │ Tab 4 & Tab 7 / Collin.    │
│ RESET Test       │ ⭐⭐ Important   │ Tab 7 / Model Diagnostics  │
│ Heterogeneity    │ ⭐⭐ Important   │ Tab 7 / Heterogeneity      │
│ Bayesian Inf.    │ ⭐ Specialized  │ Tab 8 / Bayesian (opt.)    │
└──────────────────┴──────────────────┴────────────────────────────┘
```

---

## Subtab Count Optimization

```
CURRENT STATE:
Tab 4 (Regression) = 7 subtabs ❌ TOO MANY
│
└─ Problem: Users get overwhelmed; hard to find methods

PROPOSED STATE:
Tab 4 (Core Regression)    = 5 subtabs ✅ IDEAL
Tab 7 (Advanced Inference) = 5 subtabs ✅ IDEAL
Tab 8 (Causal Inference)   = 5 subtabs ✅ IDEAL
│
└─ Solution: Distributed across tabs with clear semantic grouping

COGNITIVE LOAD ANALYSIS:
• 1-3 subtabs   = Excellent (quick scan)
• 4-5 subtabs   = Good (manageable)
• 6-7 subtabs   = Fair (getting crowded)
• 8+ subtabs    = Poor (overwhelming) ❌

NAVIGATION TIME:
• 7 subtabs in 1 tab = ~3-5 seconds to find method
• 5 subtabs per tab = ~1-2 seconds to find method
• 9 tabs total = ~2-3 seconds to find right tab
• Total = ~5-6 seconds vs ~10 seconds improvement!
```

---

## Data Flow Diagram

```
DATA INPUT
    ↓
    └─→ Tab 1: Data Management
         ├─ Load CSV/Excel
         ├─ Variable Classification
         ├─ Missing Data Report
         └─ Output: Clean Dataset (df)
            ↓
            ├────→ Tab 2: Table 1 & Matching
            │       ├─ Descriptive Statistics
            │       ├─ PSM (Basic)
            │       └─ Output: Matched Dataset (df_matched)
            │
            ├────→ Tab 3: Diagnostic Tests
            │       ├─ ROC, Sensitivity/Specificity
            │       └─ Output: Test Statistics
            │
            ├────→ Tab 4: Core Regression ⭐ RESTRUCTURED
            │       ├─ Subtab 1: Binary Outcomes
            │       │  ├─ Logistic Regression
            │       │  └─ Output: OR, Forest Plot
            │       ├─ Subtab 2: Continuous Outcomes
            │       │  ├─ Linear Regression
            │       │  ├─ ⭐ Collinearity Check (MOVED HERE)
            │       │  └─ Output: β, Diagnostic Plots
            │       ├─ Subtab 3: Count & Special
            │       │  ├─ Poisson
            │       │  ├─ ⭐ Negative Binomial (NEW)
            │       │  └─ Output: IRR
            │       ├─ Subtab 4: Repeated Measures
            │       │  ├─ GEE, LMM
            │       │  └─ Output: Trajectory Plot
            │       └─ Subtab 5: Reference
            │
            ├────→ Tab 5: Correlation & ICC
            │       └─ Output: Correlation Matrix
            │
            ├────→ Tab 6: Survival Analysis
            │       ├─ Kaplan-Meier
            │       ├─ Cox Regression
            │       └─ Output: Survival Curves, HR
            │
            ├────→ Tab 7: Advanced Inference ⭐ NEW
            │       ├─ Subtab 1: Mediation Analysis
            │       │  ├─ ⭐ Direct/Indirect Effects (NEW)
            │       │  └─ Output: Effect Decomposition
            │       ├─ Subtab 2: Collinearity Diagnostics
            │       │  ├─ ⭐ VIF, Tolerance (MOVED HERE)
            │       │  └─ Output: Collinearity Report
            │       ├─ Subtab 3: Model Diagnostics
            │       │  ├─ ⭐ RESET, Heteroscedasticity (NEW)
            │       │  └─ Output: Assumption Checks
            │       ├─ Subtab 4: Heterogeneity Testing
            │       │  ├─ ⭐ I², Q-test (NEW)
            │       │  └─ Output: Heterogeneity Report
            │       └─ Subtab 5: Reference
            │
            └────→ Tab 8: Causal Inference ⭐ NEW
                    ├─ Subtab 1: PSM Methods
                    │  ├─ ⭐ IPW, AIPW (NEW)
                    │  ├─ ⭐ Love Plot (NEW)
                    │  └─ Output: Treatment Effect
                    ├─ Subtab 2: Stratified Analysis
                    │  ├─ ⭐ Mantel-Haenszel (NEW)
                    │  └─ Output: Stratified OR
                    ├─ Subtab 3: Bayesian Inference
                    │  ├─ ⭐ Posterior Distribution (NEW - optional)
                    │  └─ Output: Credible Interval
                    ├─ Subtab 4: Sensitivity Analysis
                    │  ├─ ⭐ E-value (NEW)
                    │  └─ Output: Sensitivity Report
                    └─ Subtab 5: Reference & DAGs

EXPORT OPTIONS (All Tabs):
    ├─ HTML (Interactive, for sharing)
    ├─ PDF (Static, for publication)
    ├─ CSV (Data extraction)
    ├─ DOCX (Word document template)
    └─ Publication Table (Direct copy-paste)
```

---

## File Structure After Implementation

```
stat-shiny/
├── tabs/
│   ├── __init__.py                      ✅ Existing
│   ├── _common.py                       ✅ Existing
│   ├── _styling.py                      ✅ Existing
│   ├── _tvc_components.py               ✅ Existing
│   ├── tab_data.py                      ✅ Existing
│   ├── tab_baseline_matching.py         ✅ Existing
│   ├── tab_diag.py                      ✅ Existing
│   ├── tab_core_regression.py           🆕 NEW (renamed from tab_logit.py)
│   ├── tab_corr.py                      ✅ Existing
│   ├── tab_survival.py                  ✅ Existing
│   ├── tab_advanced_inference.py        🆕 NEW
│   ├── tab_causal_inference.py          🆕 NEW
│   ├── tab_settings.py                  ✅ Existing
│   └── [DEPRECATED]
│       ├── tab_logit.py                 (superseded)
│       └── tab_advanced_stats.py        (superseded)
│
└── utils/
    ├── linear_lib.py                    ⭐ Enhanced
    ├── mediation_lib.py                 🆕 NEW
    ├── collinearity_lib.py              🆕 NEW
    ├── model_diagnostics_lib.py         🆕 NEW
    ├── heterogeneity_lib.py             🆕 NEW
    ├── psm_advanced_lib.py              🆕 NEW
    ├── stratified_analysis_lib.py       🆕 NEW
    ├── bayesian_lib.py                  🆕 NEW (optional)
    ├── sensitivity_lib.py               🆕 NEW
    └── subgroup_analysis_module.py      ⭐ Modified
```

---

**Summary:** This restructuring provides professional-grade medical statistics platform suitable for world-class publications, with clear semantic organization and intuitive navigation!

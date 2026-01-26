# 📊 Medical Stat Tool (stat-shiny) - Comprehensive Optimization & Development Roadmap

**Document Version:** 5.1 (Enterprise Production Hardening - Firth Update)  
**Date:** January 24, 2026  
**Status:** 🚀 **Phase 2B - UI Standardization & Advanced Feature Development** **Target Architecture:** Enterprise-Grade Medical Publication Standard  
**Maintainer:** NTWKKM

---

## 📑 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Architecture & Health](#2-current-system-architecture-health)
3. [Verified Component Status](#3-verified-component-status)
4. [Phase 2 - Deep Dive Implementation Plan](#4-phase-2-deep-dive-implementation-plan)
5. [Phase 3 - Advanced Features Roadmap](#5-phase-3-advanced-features-roadmap)
6. [Quality Assurance & Validation Strategy](#6-quality-assurance--validation-strategy)
7. [Performance Optimization & DevOps](#7-performance-optimization--devops)
8. [Development Timeline & Priorities](#8-development-timeline--priorities)

---

## <a id="1-executive-summary"></a>1. Executive Summary

### 🎯 Achievement Status

The **stat-shiny** project has successfully reached **Production-Ready State** with:

✅ **Modular Architecture:** Complete separation of statistical logic from UI (MVC pattern)  
✅ **7-Tab Navigation Structure:** Fully implemented in `app.py` with proper navigation menus  
✅ **Data Integrity Pipeline:** Robust `utils/data_cleaning.py` with vectorized processing  
✅ **Statistical Engine:** Comprehensive `utils/logic.py`, `utils/tvc_lib.py`, and supporting libraries  
✅ **HTML-First Export:** `utils/plotly_html_renderer.py` for offline-capable reports  
✅ **CI/CD Foundation:** Docker, requirements management, and deployment configuration  

### 📋 Current Phase Focus

**Phase 2B:** UI Refinement & Clinical Validation (Current - Weeks 1-8)

- [ ] Statistical accuracy validation against R benchmarks (Updated for `logistf` & `coxphf`)
- [ ] Enhanced UI/UX for clinical workflows
- [ ] Comprehensive test suite development
- [ ] Performance profiling for large datasets (50k+ rows)

**Phase 3:** Advanced Features & Production Hardening (Weeks 9-16)

- [ ] Batch report generation system
- [ ] AI-powered clinical interpretation integration
- [ ] Advanced caching and data processing
- [ ] Enterprise deployment features

---

## <a id="2-current-system-architecture-health"></a>2. Current System Architecture & Health

### 2.1 Proven Hybrid Architecture

```text

┌─────────────────────────────────────────────────────────────────┐
│                         Shiny Web Layer                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  app.py (Router & Session Manager)                      │  │
│  │  - 7-Tab Navigation Structure (Verified ✅)             │  │
│  │  - Reactive State Management                            │  │
│  │  - Module Orchestration                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
↓ (Pass Reactive Values)
┌─────────────────────────────────────────────────────────────────┐
│                   UI Component Modules (tabs/)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ tab_home.py            | Landing & quick-start           │  │
│  │ tab_baseline_matching  | Table 1 & PSM (Primary)         │  │
│  │ tab_data.py            | Data management & cleaning      │  │
│  │ tab_diag/corr/agree    | General statistics group        │  │
│  │ tab_core_regression    | Basic & advanced regression     │  │
│  │ tab_survival.py        | Survival analysis (TVC ready)   │  │
│  │ tab_advanced_inference | Mediation & causal inference    │  │
│  │ tab_causal_inference   | ATE, IPW, AIPW methods          │  │
│  │ tab_sample_size.py     | Power calculation & sample size │  │
│  │ tab_settings.py        | Configuration & UI styling      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
↓ (Unidirectional Flow)
┌─────────────────────────────────────────────────────────────────┐
│              Statistical Logic Engine (utils/)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ data_cleaning.py       | Vectorized medical data prep    │  │
│  │ logic.py               | Regression & model fits         │  │
│  │ tvc_lib.py             | Time-varying covariate prep     │  │
│  │ table_one.py           | Baseline characteristics        │  │
│  │ forest_plot_lib.py     | Effect size visualization       │  │
│  │ plotly_html_renderer   | Single-file HTML export         │  │
│  │ ui_helpers.py          | Reusable UI components          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
↓ (External Dependencies)
┌─────────────────────────────────────────────────────────────────┐
│                    Third-Party Libraries                        │
│  Statsmodels | Scikit-learn | Lifelines | Pandas | Plotly     │
│  **firthmodels (jzluo)** │
└─────────────────────────────────────────────────────────────────┘

```

### 2.2 Data Flow Architecture

```text

User Upload (CSV/Excel)
↓
[tap_data.py] → Data Preview & Validation
↓
[utils/data_cleaning.py] → Vectorized Cleaning & QA Report
├─ Smart Numeric Conversion (handles "<5", ">100")
├─ Missing Data Strategy (complete-case, patterns)
├─ Outlier Detection & Flagging
└─ Type Casting with Validation
↓
[Reactive State - df, var_meta] → Shared across all tabs
↓
┌─ [tab_baseline_matching] → Table 1 generation & PSM
├─ [tab_diag] → Sensitivity/Specificity analysis
├─ [tab_corr] → Association matrices
├─ [tab_agreement] → ICC/Kappa calculations
├─ [tab_core_regression] → Logistic/Poisson regression (Firth Support)
├─ [tab_survival] → KM curves, Cox models, Penalized Cox
├─ [tab_advanced_inference] → Mediation analysis
├─ [tab_causal_inference] → Propensity scoring
└─ [tab_sample_size] → Power calculations
↓
[utils/plotly_html_renderer.py] → Single-File HTML Reports
↓
Downloaded Report (Offline-Ready)

```

### 2.3 Critical Dependencies & Versions

```python
# Core Web Framework
shiny==1.1.4                    # UI/Server framework
shinywidgets==0.1.4             # Enhanced widgets

# Statistical Libraries
pandas==2.2.3                   # Data manipulation
numpy==1.26.4                   # Numerical computing
scipy==1.14.1                   # Scientific computing
statsmodels==0.14.2             # GLM, survival analysis
scikit-learn==1.5.2             # ML algorithms, preprocessing
lifelines==0.31.4               # Survival analysis (Kaplan-Meier, Cox)

# Visualization
plotly==6.1.0                   # Interactive charts

# Advanced Features
fastavro==1.10.2               # Data serialization
python-multipart==0.0.7        # File upload handling
python-multipart==0.0.7        # Async support

# Specialized Statistics (UPDATED)
# Bias-reduced regression (source: jzluo/firthmodels)
# See docs/firthmodels.md for details
firthmodels @ git+https://github.com/jzluo/firthmodels.git
---

## <a id="3-verified-component-status"></a>3. Verified Component Status

### 3.1 Tab Components - Current Implementation

| Tab | Corresponding Module | Status | Health | Known Gaps |
| --- | --- | --- | --- | --- |
| **🏠 Home** | `tab_home.py` | ✅ **Ready** | 🟢 **Excellent** | None - fully functional |
| **📋 Table 1** | `tab_baseline_matching.py` + `utils/table_one.py` | ✅ **Promoted** | 🟢 **Excellent** | PSM covariate selection UI could be enhanced |
| **📁 Data** | `tab_data.py` + `utils/data_cleaning.py` | ✅ **Excellent** | 🟢 **Production-Ready** | Large dataset preview (>100k) needs optimization |
| **📊 General** | `tab_diag.py`, `tab_corr.py`, `tab_agreement.py` | ✅ **Feature-Rich** | 🟢 **Good** | Need consolidated UI helper framework |
| **🔬 Advanced** | `tab_core_regression.py`, `tab_survival.py` | ✅ **Complex** | 🟡 **Needs Validation** | **New:** Firth integration needs `logistf` validation |
| **🏥 Clinical** | `tab_sample_size.py`, `tab_causal_inference.py` | ✅ **Good** | 🟢 **Solid** | IPW weight diagnostics visualization needed |
| **⚙️ Settings** | `tab_settings.py` | ✅ **Ready** | 🟢 **Working** | Theme persistence could use database backing |

### 3.2 Utility Modules - Implementation Quality

| Module | Purpose | Status | LOC | Test Coverage | Priority |
| --- | --- | --- | --- | --- | --- |
| `data_cleaning.py` | Medical data vectorization | ✅ **Mature** | ~400 | 🟡 **30%** | HIGH - needs unit tests |
| `logic.py` | Statistical calculations | ✅ **Update Needed** | ~600 | 🟡 **25%** | HIGH - Add `firthmodels` methods |
| `tvc_lib.py` | Time-varying covariate prep | ✅ **Complete** | ~250 | 🔴 **5%** | HIGH - needs comprehensive tests |
| `table_one.py` | Baseline characteristics | ✅ **Stable** | ~300 | 🟡 **40%** | MEDIUM - improving |
| `forest_plot_lib.py` | Effect visualization | ✅ **Feature-Complete** | ~200 | 🟡 **35%** | MEDIUM |
| `plotly_html_renderer.py` | HTML export engine | ✅ **Production** | ~350 | 🟢 **60%** | MEDIUM - well-tested |
| `ui_helpers.py` | Reusable UI components | ✅ **Developed** | ~180 | 🔴 **0%** | MEDIUM - new module |

---

## <a id="4-phase-2-deep-dive-implementation-plan"></a>4. Phase 2 - Deep Dive Implementation Plan

### 4.1 Statistical Validation Against R (CRITICAL PATH)

#### 4.1.1 Logistic & Firth Regression Benchmarking (UPDATED)

**Current Status:** Basic implementation in `utils/logic.py`. `firthmodels` integration pending validation against correct R libraries.
**Priority:** CRITICAL
**Timeline:** Weeks 1-2

**Validation Plan:** detailed in docs/updated-R-benchmark-script.md

1. **Generate Benchmarks (R):** Run `tests/benchmarks/r_scripts/test_firth.R` to create `benchmark_firth_logistic.csv` and `benchmark_firth_cox.csv`.
2. **Verify Python Implementation:**

**Python Test Implementation:**

```python
# File: tests/unit/test_firth_regression.py

import pytest
import numpy as np
import pandas as pd
from firthmodels import FirthLogisticRegression
# Note: FirthCoxPH import depends on library version/structure
from utils.logic import fit_firth_logistic

def test_firth_logistic_vs_r():
    """Verify Python Firth logistic matches R logistf() output"""
    
    # Load test data & R benchmark
    # (Assuming data sync mechanism exists or data is embedded)
    r_benchmark = pd.read_csv("tests/benchmarks/python_results/benchmark_firth_logistic.csv")
    df = pd.read_csv("tests/benchmarks/python_results/dataset_sex2.csv")
    
    # Fit Python model using firthmodels
    # Note: jzluo/firthmodels uses sklearn style
    X = df[['age', 'oc', 'vic', 'vicl', 'vis', 'dia']]
    y = df['case']
    
    # Use wald=False to match R's default Profile Likelihood CIs
    model = FirthLogisticRegression(wald=False) 
    model.fit(X, y)
    
    # Compare coefficients (Estimate)
    assert np.allclose(model.coef_, r_benchmark['estimate'], atol=1e-4)
    
    # Compare P-values
    # Note: Accessing p-values depends on firthmodels API specifics (summary() or attribute)
    # assert np.allclose(model.pvalues_, r_benchmark['p.value'], atol=1e-4)

```

#### 4.1.2 Survival Analysis Validation

**Current Status:** Implemented with Lifelines
**Priority:** HIGH
**Timeline:** Weeks 2-3

**Key Tests Required:**

```python
# File: tests/unit/test_survival_models.py

def test_km_curves_vs_r():
    """Compare Kaplan-Meier curves with R survival package"""
    # Test: Time, event pairs should produce identical survival probabilities
    pass

def test_cox_model_vs_r():
    """Verify Cox PH hazard ratios match R coxph()"""
    # Validate coefficients, log-likelihood, concordance index
    pass

```

### 4.2 UI/UX Refinement (Current - In Progress)

#### 4.2.1 Navigation Standardization ✅ (Completed in app.py)

**Status:** ✅ **COMPLETE** - No changes needed

#### 4.2.2 Responsive Layout Enhancement (Priority: MEDIUM)

**Current Issues:**

- Data preview tables don't render well on tablets (iPad)
- Results tables need horizontal scroll for mobile

**Improvements Needed:**

1. **Mobile-First CSS Breakpoints**
2. **Data Table Virtualization**
3. **Plotly Responsiveness**

#### 4.2.3 UI Helper Framework Consolidation

**Goal:** Reduce code duplication across tab modules by enhancing `utils/ui_helpers.py`.

### 4.3 Enhanced Error Handling & Validation

#### 4.3.1 Comprehensive Error Tracking

**Needed:** Structured error collection (`AnalysisErrorTracker`) to capture errors with context and display user-friendly suggestions.

---

## <a id="5-phase-3-advanced-features-roadmap"></a>5. Phase 3 - Advanced Features Roadmap

### 5.1 Batch Report Generation System

**Timeline:** Weeks 9-12
**Priority:** MEDIUM
**Concept:** `BatchReportGenerator` class to compile Table 1, Regression Results, and Plots into a single HTML dossier.

### 5.2 AI-Powered Clinical Interpretation Integration

**Timeline:** Weeks 13-14
**Priority:** MEDIUM
**Concept:** Integrate LLM (e.g., Claude/GPT-4) to generate plain-English clinical interpretations of statistical outputs.

### 5.3 Advanced Performance Optimization

**Timeline:** Weeks 15-16
**Priority:** HIGH

#### 5.3.1 In-Memory Caching Strategy

**Concept:** `DataFrameCache` using hashing to cache expensive computations (e.g., survival fits) based on data content and parameters.

#### 5.3.2 Vectorized Data Processing

**Concept:** Enhance `utils/data_cleaning.py` to use fully vectorized operations for missing data handling and type conversion.

---

## 6. Quality Assurance & Validation Strategy

### 6.1 Comprehensive Test Suite Structure

```text
tests/
├── unit/
│   ├── test_data_cleaning.py          # utils/data_cleaning.py tests
│   ├── test_logistic_regression.py    # utils/logic.py - logistic
│   ├── test_firth_regression.py       # [NEW] Firthmodels validation
│   ├── test_survival_models.py        # Kaplan-Meier, Cox
│   ├── test_agreement_metrics.py      # ICC, Kappa, Bland-Altman
│   ├── test_table_one.py              # Baseline characteristics
│   ├── test_tvc_lib.py                # Time-varying covariate handling
│   ├── test_causal_inference.py       # ATE, IPW, AIPW
│   └── test_forest_plots.py           # Effect visualization
│
├── benchmarks/
│   ├── r_scripts/
│   │   ├── test_glm.R                 # Standard Logistic benchmark
│   │   ├── test_firth.R               # [NEW] Firth/Penalized Cox benchmark
│   │   └── generate_benchmarks.R      # Master script
│   │
│   ├── python_results/
│   │   ├── firth_benchmark.csv        # [NEW]
│   │   └── ...
│   │
│   └── test_r_vs_python.py            # Main validation tests
...

```

### 6.2 CI/CD

*Standard GitHub Actions pipeline for testing, linting, and deployment.*

---

## 7. Performance Optimization & DevOps

### 7.1 Docker Optimization

**Action:** Multi-stage build to reduce image size (~46% reduction target).

### 7.2 Production Deployment Checklist

**Includes:** Security (env vars, HTTPS), Performance (caching, indexes), Monitoring (sentry), and Documentation.

---

## 8. Development Timeline & Priorities

### 8.1 Phase 2B Timeline (Weeks 1-8)

```text
Week 1-2: Statistical Validation (CRITICAL PATH)
├─ Logistic regression R benchmarks
├─ Firth/Penalized Cox R benchmarks (NEW)
├─ Python test suite implementation
├─ Coefficient comparison validation
└─ Deliverable: test_models_vs_r.py (PASSING)

Week 2-3: Survival Analysis Validation
...

```

### 8.3 Priority Matrix

| Feature | Phase | Timeline | Complexity | Business Value | Priority |
| --- | --- | --- | --- | --- | --- |
| R Validation (Std) | 2 | 2w | HIGH | CRITICAL | 🔴 CRITICAL |
| R Validation (Firth) | 2 | 1w | HIGH | CRITICAL | 🔴 CRITICAL |
| Mobile UI | 2 | 1-2w | MEDIUM | HIGH | 🔴 HIGH |
| Error Handling | 2 | 1-2w | LOW | MEDIUM | 🟡 MEDIUM |

---

**Document prepared:** January 24, 2026

**Last updated:** January 24, 2026 v5.1

**Maintainer:** NTWKKM (Thailand)

**Repository:** [NTWKKM/stat-shiny](https://github.com/NTWKKM/stat-shiny)

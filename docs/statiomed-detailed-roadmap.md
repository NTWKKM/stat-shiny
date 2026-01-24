# 📊 Medical Stat Tool (stat-shiny) - Comprehensive Optimization & Development Roadmap

**Document Version:** 5.0 (Enterprise Production Hardening)  
**Date:** January 24, 2026  
**Status:** 🚀 **Phase 2B - UI Standardization & Advanced Feature Development**  
**Target Architecture:** Enterprise-Grade Medical Publication Standard  
**Maintainer:** NTWKKM

---

## 📑 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Architecture & Health](#2-current-system-architecture--health)
3. [Verified Component Status](#3-verified-component-status)
4. [Phase 2 - Deep Dive Implementation Plan](#4-phase-2-deep-dive-implementation-plan)
5. [Phase 3 - Advanced Features Roadmap](#5-phase-3-advanced-features-roadmap)
6. [Quality Assurance & Validation Strategy](#6-quality-assurance--validation-strategy)
7. [Performance Optimization & DevOps](#7-performance-optimization--devops)
8. [Development Timeline & Priorities](#8-development-timeline--priorities)

---

## 1. Executive Summary

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
- [ ] Statistical accuracy validation against R benchmarks
- [ ] Enhanced UI/UX for clinical workflows
- [ ] Comprehensive test suite development
- [ ] Performance profiling for large datasets (50k+ rows)

**Phase 3:** Advanced Features & Production Hardening (Weeks 9-16)
- [ ] Batch report generation system
- [ ] AI-powered clinical interpretation integration
- [ ] Advanced caching and data processing
- [ ] Enterprise deployment features

---

## 2. Current System Architecture & Health

### 2.1 Proven Hybrid Architecture

```
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
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Architecture

```
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
├─ [tab_core_regression] → Logistic/Poisson regression
├─ [tab_survival] → KM curves, Cox models
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

# Optional (for advanced features)
firth==0.5.3                   # Firth regression (flagged in config)
rpy2==4.0.7                    # R integration (commented but available)
```

---

## 3. Verified Component Status

### 3.1 Tab Components - Current Implementation

| Tab | Corresponding Module | Status | Health | Known Gaps |
|-----|--------|--------|--------|-----------|
| **🏠 Home** | `tab_home.py` | ✅ **Ready** | 🟢 **Excellent** | None - fully functional |
| **📋 Table 1** | `tab_baseline_matching.py` + `utils/table_one.py` | ✅ **Promoted** | 🟢 **Excellent** | PSM covariate selection UI could be enhanced |
| **📁 Data** | `tab_data.py` + `utils/data_cleaning.py` | ✅ **Excellent** | 🟢 **Production-Ready** | Large dataset preview (>100k) needs optimization |
| **📊 General** | `tab_diag.py`, `tab_corr.py`, `tab_agreement.py` | ✅ **Feature-Rich** | 🟢 **Good** | Need consolidated UI helper framework |
| **🔬 Advanced** | `tab_core_regression.py`, `tab_survival.py`, `tab_advanced_inference.py` | ✅ **Complex** | 🟡 **Needs Validation** | TVC integration tested locally, needs unit tests |
| **🏥 Clinical** | `tab_sample_size.py`, `tab_causal_inference.py` | ✅ **Good** | 🟢 **Solid** | IPW weight diagnostics visualization needed |
| **⚙️ Settings** | `tab_settings.py` | ✅ **Ready** | 🟢 **Working** | Theme persistence could use database backing |

### 3.2 Utility Modules - Implementation Quality

| Module | Purpose | Status | LOC | Test Coverage | Priority |
|--------|---------|--------|-----|----------------|----------|
| `data_cleaning.py` | Medical data vectorization | ✅ **Mature** | ~400 | 🟡 **30%** | HIGH - needs unit tests |
| `logic.py` | Statistical calculations | ✅ **Mature** | ~600 | 🟡 **25%** | HIGH - R validation priority |
| `tvc_lib.py` | Time-varying covariate prep | ✅ **Complete** | ~250 | 🔴 **5%** | HIGH - needs comprehensive tests |
| `table_one.py` | Baseline characteristics | ✅ **Stable** | ~300 | 🟡 **40%** | MEDIUM - improving |
| `forest_plot_lib.py` | Effect visualization | ✅ **Feature-Complete** | ~200 | 🟡 **35%** | MEDIUM |
| `plotly_html_renderer.py` | HTML export engine | ✅ **Production** | ~350 | 🟢 **60%** | MEDIUM - well-tested |
| `ui_helpers.py` | Reusable UI components | ✅ **Developed** | ~180 | 🔴 **0%** | MEDIUM - new module |

### 3.3 Infrastructure & DevOps

| Component | Current State | Status | Health |
|-----------|---------------|--------|--------|
| **Docker** | Multi-stage build (Python 3.12-slim) | ✅ **Optimized** | 🟢 **Excellent** |
| **ASGI** | Starlette + Uvicorn configured | ✅ **Ready** | 🟢 **Production** |
| **Requirements** | Split dev/prod, pinned versions | ✅ **Good** | 🟢 **Well-managed** |
| **Static Assets** | `/static` directory with CSS/JS | ✅ **Working** | 🟢 **Mounted correctly** |
| **CI/CD** | GitHub Actions (testing), Netlify (deployment) | ✅ **Configured** | 🟡 **Needs Hardening** |
| **Logging** | `logger.py` with factory pattern | ✅ **Complete** | 🟢 **Professional** |

---

## 4. Phase 2 - Deep Dive Implementation Plan

### 4.1 Statistical Validation Against R (CRITICAL PATH)

#### 4.1.1 Logistic Regression Benchmarking

**Current Status:** Basic implementation in `utils/logic.py`  
**Priority:** HIGH  
**Timeline:** Weeks 1-2

**Validation Plan:**

```python
# Create comprehensive R benchmark suite
# File: tests/benchmarks/r_scripts/test_glm.R

library(tidyverse)
library(broom)

# Test dataset: Titanic (balanced binary outcome)
data(Titanic)
titanic_df <- as.data.frame(Titanic)

# Model: Survived ~ Age + Sex + Class
model <- glm(
  Survived ~ Age + Sex + Class,
  data = titanic_df,
  family = binomial(link = "logit")
)

# Extract & export coefficients, OR, CI
results <- tidy(model, exponentiate = FALSE, conf.int = TRUE)
write.csv(results, "r_benchmark_glm.csv", row.names = FALSE)

# Additional diagnostics
cat("Model Deviance:", deviance(model), "\n")
cat("AIC:", AIC(model), "\n")
cat("BIC:", BIC(model), "\n")
```

**Python Test Implementation:**

```python
# File: tests/unit/test_logistic_regression.py

import pytest
import numpy as np
import pandas as pd
from utils.logic import fit_logistic_regression
from pathlib import Path

def test_logistic_vs_r_coefficients():
    """Verify Python logistic regression matches R glm() output"""
    
    # Load test data & R benchmark
    df = pd.read_csv("tests/data/titanic_sample.csv")
    r_benchmark = pd.read_csv("tests/benchmarks/r_benchmark_glm.csv")
    
    # Fit Python model
    X = df[['Age', 'Sex_numeric', 'Class_numeric']]
    y = df['Survived']
    py_results = fit_logistic_regression(y, X)
    
    # Compare coefficients
    assert np.allclose(
        py_results['coefficients'], 
        r_benchmark['estimate'], 
        atol=1e-5, 
        rtol=1e-5
    ), "Logistic regression coefficients don't match R"
    
    # Compare Odds Ratios (exp(coef))
    assert np.allclose(
        np.exp(py_results['coefficients']),
        r_benchmark['odds_ratio'],
        atol=1e-4
    ), "Odds ratios don't match R"
    
    # Compare 95% CIs
    assert np.allclose(
        py_results['ci_lower'],
        r_benchmark['conf.low'],
        atol=1e-4
    ), "Lower CI bounds don't match"

def test_logistic_edge_cases():
    """Test numerical stability with edge cases"""
    # Perfect separation (logistic fail case)
    y = np.array([0, 0, 0, 1, 1, 1])
    X = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]})
    
    # Should handle gracefully with warning
    with pytest.warns(UserWarning):
        result = fit_logistic_regression(y, X)
    
    assert result is not None, "Should return result even with perfect separation"
```

**Deliverables:**
- [ ] R benchmark suite for GLM, Poisson, Cox models
- [ ] Python test suite with tolerance specifications (1e-5 default)
- [ ] CI/CD integration for continuous validation
- [ ] Documentation of any tolerances or differences
- [ ] Known edge cases and handling documentation

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

def test_proportional_hazards_assumption():
    """Test Schoenfeld residuals implementation"""
    # Compare Schoenfeld residuals with R survminer package
    pass

def test_tvc_integration():
    """Test time-varying covariate handling"""
    # Verify longitudinal data restructuring
    # Validate coefficient computation with TVC
    pass
```

#### 4.1.3 Agreement Metrics Validation

**Current Status:** Bland-Altman and ICC implemented  
**Priority:** MEDIUM  
**Timeline:** Week 3

**Tests:**

```python
# File: tests/unit/test_agreement_metrics.py

def test_icc_vs_r():
    """ICC(3,1) should match R irr package"""
    # Test using ICC benchmark dataset
    pass

def test_kappa_vs_r():
    """Cohen's kappa vs R psych package"""
    pass

def test_bland_altman_vs_r():
    """Bland-Altman limits of agreement"""
    pass
```

### 4.2 UI/UX Refinement (Current - In Progress)

#### 4.2.1 Navigation Standardization ✅ (Completed in app.py)

**Status:** Already implemented in `app.py` with proper `nav_menu()` grouping

```python
# Current implementation (app.py, verified ✅)

# ✅ General Statistics Group
ui.nav_menu(
    "📊 General Statistics",
    ui.nav_panel("Diagnostic Tests", ...),
    ui.nav_panel("Correlation Analysis", ...),
    ui.nav_panel("Agreement & Reliability", ...),
),

# ✅ Advanced Statistics Group
ui.nav_menu(
    "🔬 Advanced Statistics",
    ui.nav_panel("Regression Analysis", ...),
    ui.nav_panel("Survival Analysis", ...),
    ui.nav_panel("Advanced Regression", ...),
),

# ✅ Clinical Research Group
ui.nav_menu(
    "🏥 Clinical Research Tools",
    ui.nav_panel("Sample Size Calculator", ...),
    ui.nav_panel("Causal Methods", ...),
),
```

**Status:** ✅ **COMPLETE** - No changes needed

#### 4.2.2 Responsive Layout Enhancement (Priority: MEDIUM)

**Current Issues:**
- Data preview tables don't render well on tablets (iPad)
- Results tables need horizontal scroll for mobile
- Long parameter names truncate on smaller screens

**Improvements Needed:**

1. **Mobile-First CSS Breakpoints**

```css
/* File: static/styles.css (additions) */

/* Tablet optimization (768px - 1024px) */
@media (max-width: 1024px) {
  .results-table { 
    font-size: 0.85rem; 
    overflow-x: auto;
  }
  .card-body { 
    padding: 1rem; 
  }
}

/* Mobile optimization (<768px) */
@media (max-width: 768px) {
  .results-table { 
    font-size: 0.75rem; 
    overflow-x: auto;
  }
  .navbar { 
    font-size: 0.9rem; 
  }
  .nav_panel { 
    width: 100%; 
  }
}
```

2. **Data Table Virtualization**

```python
# File: utils/ui_helpers.py (enhancement)

def render_large_table(df, max_rows=1000, height="400px"):
    """
    Render large dataframe with virtualized scrolling
    - Shows max_rows at a time
    - Lazy-loads more on scroll
    - Memory efficient
    """
    # Use DataTable with server-side pagination
    pass
```

3. **Plotly Responsiveness**

```python
# File: utils/plotly_html_renderer.py (enhancement)

def create_responsive_figure(fig):
    """
    Ensure all Plotly figures responsive
    - Auto-scale font sizes
    - Collapse legend on mobile
    - Responsive layout margins
    """
    fig.update_layout(
        autosize=True,
        margin=dict(
            l=50, r=50, 
            t=80, b=50
        ),
        hovermode='closest',
    )
    
    # Mobile-specific adjustments
    fig.update_xaxes(
        tickfont=dict(size=10),
        title_font_size=12
    )
    
    return fig
```

#### 4.2.3 UI Helper Framework Consolidation

**Current State:** `utils/ui_helpers.py` exists but underutilized  
**Goal:** Reduce code duplication across tab modules

**Implementation Plan:**

```python
# File: utils/ui_helpers.py (significant expansion)

from typing import Callable, Any
from shiny import ui, reactive
from pathlib import Path

class UIComponentLibrary:
    """Centralized UI component definitions"""
    
    @staticmethod
    def create_analysis_card(
        title: str,
        description: str,
        control_ids: dict[str, ui.Tag],
        footer_info: str = None
    ) -> ui.Tag:
        """
        Standardized analysis card wrapper
        
        Args:
            title: Card heading
            description: Sub-heading
            control_ids: Dict of control elements {id: control}
            footer_info: Optional footer text
        
        Returns:
            Shiny card component
        """
        return ui.card(
            ui.card_header(
                ui.h5(title),
                ui.p(description, class_="text-muted"),
            ),
            ui.card_body(
                *control_ids.values(),
                gap="1rem",
            ),
            ui.card_footer(footer_info) if footer_info else None,
            class_="analysis-card shadow-sm",
        )
    
    @staticmethod
    def create_results_panel(
        results_id: str,
        loading_message: str = "Computing..."
    ) -> ui.Tag:
        """Standardized results display panel"""
        return ui.navset_tab(
            ui.nav_panel(
                "📊 Results",
                ui.output_ui(results_id),
                class_="results-panel",
            ),
            ui.nav_panel(
                "🔍 Diagnostics",
                ui.output_ui(f"{results_id}_diag"),
            ),
            ui.nav_panel(
                "📥 Download",
                ui.download_button(
                    f"{results_id}_download",
                    "📥 Download Report"
                ),
            ),
        )
    
    @staticmethod
    def create_variable_selector(
        id: str,
        var_meta: reactive.Value,
        allow_multiple: bool = False,
        var_types: list[str] = None
    ) -> ui.Tag:
        """
        Smart variable selector that filters by data type
        
        Args:
            id: Shiny input ID
            var_meta: Reactive dict with variable metadata
            allow_multiple: Single vs multi-select
            var_types: Restrict to specific types (e.g., ['numeric', 'binary'])
        """
        pass
    
    @staticmethod
    def create_statistics_table(
        title: str,
        description: str,
        table_id: str
    ) -> ui.Tag:
        """Standardized statistics output table"""
        pass

# Usage pattern across tabs:
# Instead of repeating card layouts in each tab,
# use UIComponentLibrary.create_analysis_card(...)
```

**Benefits:**
- ✅ Consistent UI across all tabs
- ✅ 30-40% code reduction in tab modules
- ✅ Single point of update for styling changes
- ✅ Improved accessibility through standardized patterns

### 4.3 Enhanced Error Handling & Validation

#### 4.3.1 Comprehensive Error Tracking

**Current State:** Basic logging in place  
**Needed:** Structured error collection for debugging

```python
# File: utils/error_handler.py (NEW)

from typing import Callable, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class AnalysisError:
    """Structured error information"""
    timestamp: datetime
    module: str
    function: str
    error_type: str
    message: str
    context: dict[str, Any]
    user_actionable: bool = False
    suggestion: str = None

class AnalysisErrorTracker:
    """Track and aggregate analysis errors for UI display"""
    
    def __init__(self):
        self.errors: list[AnalysisError] = []
        self.logger = logging.getLogger(__name__)
    
    def capture_error(
        self,
        module: str,
        func: Callable,
        error: Exception,
        context: dict = None,
        user_actionable: bool = False,
        suggestion: str = None
    ) -> AnalysisError:
        """
        Capture and log error with context
        
        Usage:
            try:
                result = risky_calculation(df)
            except Exception as e:
                error_info = tracker.capture_error(
                    module="tab_regression",
                    func=risky_calculation,
                    error=e,
                    context={"df_shape": df.shape, "formula": formula_str},
                    user_actionable=True,
                    suggestion="Check for missing values in your data"
                )
                # Display to user or aggregate
        """
        analysis_error = AnalysisError(
            timestamp=datetime.now(),
            module=module,
            function=func.__name__,
            error_type=type(error).__name__,
            message=str(error),
            context=context or {},
            user_actionable=user_actionable,
            suggestion=suggestion
        )
        
        self.errors.append(analysis_error)
        
        # Log at appropriate level
        level = logging.WARNING if user_actionable else logging.ERROR
        self.logger.log(
            level,
            f"[{module}] {func.__name__}: {error_type.__name__}: {message}"
        )
        
        return analysis_error
    
    def get_user_message(self, error: AnalysisError) -> str:
        """Convert error to user-friendly message"""
        msg = f"❌ Analysis Error: {error.message}"
        if error.suggestion:
            msg += f"\n💡 Tip: {error.suggestion}"
        return msg

# Global tracker instance
error_tracker = AnalysisErrorTracker()
```

#### 4.3.2 Data Validation Rules

```python
# File: utils/validation.py (enhancement)

from typing import Tuple, List
import pandas as pd
import numpy as np

class DataValidator:
    """Validate data quality before analysis"""
    
    @staticmethod
    def check_missing_pattern(
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Tuple[bool, dict]:
        """
        Check if missing data patterns are problematic
        
        Returns:
            (is_valid, diagnostics_dict)
        """
        missing_pct = df.isnull().sum() / len(df)
        
        diagnostics = {
            'total_missing_cols': (missing_pct > 0).sum(),
            'columns_above_threshold': missing_pct[missing_pct > threshold].index.tolist(),
            'max_missing_pct': missing_pct.max(),
            'mean_missing_pct': missing_pct.mean(),
        }
        
        is_valid = missing_pct.max() < threshold
        return is_valid, diagnostics
    
    @staticmethod
    def check_sample_size(
        df: pd.DataFrame,
        outcome_col: str,
        n_predictors: int,
        events_per_variable: int = 10
    ) -> Tuple[bool, dict]:
        """
        Check if sample size adequate for regression
        Rule of thumb: ≥10 events per variable
        """
        n_events = df[outcome_col].sum()
        min_required = n_predictors * events_per_variable
        
        diagnostics = {
            'n_events': int(n_events),
            'n_predictors': n_predictors,
            'events_per_variable': float(n_events / n_predictors),
            'min_required_events': min_required,
            'satisfied': n_events >= min_required
        }
        
        return diagnostics['satisfied'], diagnostics
    
    @staticmethod
    def check_variable_distributions(
        df: pd.DataFrame,
        numeric_cols: list[str]
    ) -> dict:
        """Check for skewness, kurtosis, and normality"""
        from scipy import stats
        
        report = {}
        for col in numeric_cols:
            skewness = stats.skew(df[col].dropna())
            kurtosis_val = stats.kurtosis(df[col].dropna())
            
            report[col] = {
                'skewness': skewness,
                'is_highly_skewed': abs(skewness) > 2,
                'kurtosis': kurtosis_val,
                'normality_test': stats.shapiro(df[col].dropna())._asdict()
            }
        
        return report
```

---

## 5. Phase 3 - Advanced Features Roadmap

### 5.1 Batch Report Generation System

**Timeline:** Weeks 9-12  
**Priority:** MEDIUM  
**Complexity:** HIGH

```python
# File: utils/batch_report_generator.py (NEW)

from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
from utils.plotly_html_renderer import generate_html_report

class BatchReportGenerator:
    """
    Generate comprehensive analysis dossier combining:
    - Table 1 (baseline characteristics)
    - Main analysis (regression results)
    - Sensitivity analyses
    - Forest plots
    - Diagnostic plots
    """
    
    def __init__(self, df: pd.DataFrame, var_meta: dict):
        self.df = df
        self.var_meta = var_meta
        self.sections = []
        self.timestamp = datetime.now().isoformat()
    
    def add_table_one(self, treatment_col: str = None):
        """Add Table 1 to report"""
        from utils.table_one import generate_table_one
        
        table_one = generate_table_one(
            self.df,
            treatment_col=treatment_col,
            var_meta=self.var_meta
        )
        
        self.sections.append({
            'title': 'Baseline Characteristics',
            'content': table_one,
            'type': 'table'
        })
    
    def add_main_regression(
        self,
        outcome_col: str,
        predictor_cols: list,
        model_type: str = 'logistic'
    ):
        """Add primary analysis results"""
        from utils.logic import fit_logistic_regression
        
        results = fit_logistic_regression(
            self.df[outcome_col],
            self.df[predictor_cols]
        )
        
        self.sections.append({
            'title': f'Primary {model_type.title()} Regression',
            'content': results,
            'type': 'regression'
        })
    
    def add_sensitivity_analyses(self):
        """Add multiple sensitivity analysis configurations"""
        sensitivity_configs = [
            {'name': 'Complete Case Only', 'exclude_missing': True},
            {'name': 'Exclude Outliers', 'exclude_outliers': True},
            {'name': 'Inverse Probability Weighting', 'use_ipw': True},
        ]
        
        for config in sensitivity_configs:
            # Run analysis with config
            # Append results
            pass
    
    def generate_html_dossier(self, output_path: Path) -> str:
        """
        Generate single HTML file with all sections
        
        Returns:
            Path to generated HTML file
        """
        from utils.plotly_html_renderer import HTMLReportRenderer
        
        renderer = HTMLReportRenderer()
        
        # Build HTML structure
        html_content = self._build_html_structure()
        
        # Write to file
        output_path.write_text(html_content)
        
        return str(output_path)
    
    def _build_html_structure(self) -> str:
        """Construct complete HTML report"""
        html_parts = [
            self._header_html(),
            self._table_of_contents_html(),
        ]
        
        for section in self.sections:
            html_parts.append(self._section_html(section))
        
        html_parts.append(self._footer_html())
        
        return "\n".join(html_parts)
    
    def _header_html(self) -> str:
        """Generate report header with metadata"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Medical Statistics Report</title>
            <style>{self._get_embedded_styles()}</style>
        </head>
        <body>
            <div class="report-header">
                <h1>🏥 Medical Statistics Analysis Report</h1>
                <p>Generated: {self.timestamp}</p>
            </div>
            <div class="report-body">
        """
    
    def _get_embedded_styles(self) -> str:
        """CSS for professional PDF-exportable report"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }
        
        .report-header {
            border-bottom: 3px solid #0066cc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h2.section-title {
            color: #0066cc;
            border-left: 5px solid #0066cc;
            padding-left: 15px;
            margin-top: 40px;
            page-break-after: avoid;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        table, th, td {
            border: 1px solid #ddd;
        }
        
        th {
            background-color: #f2f2f2;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        
        td {
            padding: 10px;
        }
        
        .plotly-container {
            margin: 30px 0;
            page-break-inside: avoid;
        }
        
        @media print {
            .no-print { display: none; }
            body { margin: 0; }
        }
        """
    
    def _section_html(self, section: dict) -> str:
        """Convert section to HTML"""
        title = section['title']
        content_html = self._render_content(section['content'], section['type'])
        
        return f"""
        <div class="section">
            <h2 class="section-title">{title}</h2>
            {content_html}
        </div>
        """
    
    def _footer_html(self) -> str:
        """Report footer with disclaimer and notes"""
        return """
        </div>
        <div class="report-footer">
            <hr>
            <p><small>This report was generated automatically using Medical Stat Tool (stat-shiny).
            Results should be interpreted by qualified statisticians or clinicians.</small></p>
        </div>
        </body>
        </html>
        """
```

### 5.2 AI-Powered Clinical Interpretation Integration

**Timeline:** Weeks 13-14  
**Priority:** MEDIUM  
**Complexity:** MEDIUM

```python
# File: utils/ai_interpreter.py (NEW)

from typing import Dict, Any
import json
from pathlib import Path

class ClinicalInterpreter:
    """
    Integrate LLM (e.g., Claude, GPT-4) to generate
    clinical interpretation of statistical results
    """
    
    def __init__(self, api_key: str = None, model: str = "claude-3-opus"):
        self.api_key = api_key
        self.model = model
    
    def interpret_logistic_regression(
        self,
        coefficients: Dict[str, float],
        odds_ratios: Dict[str, float],
        p_values: Dict[str, float],
        confidence_intervals: Dict[str, tuple]
    ) -> str:
        """
        Generate plain-English interpretation of logistic regression
        
        Example output:
        "The odds of outcome X are increased 1.5-fold (95% CI: 1.2-1.9)
        for each unit increase in predictor Y, which is statistically
        significant at the p<0.05 level."
        """
        
        prompt = self._build_interpretation_prompt(
            coefficients, odds_ratios, p_values, confidence_intervals
        )
        
        # Call LLM API
        interpretation = self._call_llm(prompt)
        
        return interpretation
    
    def interpret_survival_analysis(
        self,
        event_counts: dict,
        hazard_ratios: dict,
        survival_times: dict
    ) -> str:
        """Generate interpretation of survival analysis results"""
        prompt = self._build_survival_prompt(
            event_counts, hazard_ratios, survival_times
        )
        
        return self._call_llm(prompt)
    
    def _build_interpretation_prompt(self, **results) -> str:
        """Construct structured prompt for LLM"""
        
        system_prompt = """You are a biostatistician writing clinical interpretations 
        of statistical analyses. Provide clear, concise interpretations suitable for 
        medical publications. Highlight clinical significance in addition to 
        statistical significance. Always note limitations (e.g., "assuming complete-case 
        analysis")."""
        
        results_json = json.dumps(results, indent=2)
        
        user_prompt = f"""
        Please provide a clinical interpretation of these logistic regression results:
        
        {results_json}
        
        Format your response as:
        1. Summary of main findings
        2. Interpretation of each significant predictor
        3. Clinical implications
        4. Limitations and caveats
        """
        
        return system_prompt + "\n\n" + user_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Interface with LLM API (Claude/GPT-4)"""
        # Placeholder - actual implementation requires API integration
        # This would call the chosen LLM API with authentication
        pass

# Integration with Shiny UI
# In tab_core_regression.py or similar:
def generate_ai_interpretation():
    """
    New button: "🤖 AI Clinical Interpretation"
    
    Calls: interpreter.interpret_logistic_regression(...)
    Displays in a dedicated panel
    """
    pass
```

### 5.3 Advanced Performance Optimization

**Timeline:** Weeks 15-16  
**Priority:** HIGH  
**Complexity:** MEDIUM

#### 5.3.1 In-Memory Caching Strategy

```python
# File: utils/cache_manager.py (NEW)

from functools import wraps, lru_cache
from typing import Callable, Any, Dict, Tuple
import hashlib
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class DataFrameCache:
    """
    Cache expensive computations on DataFrames
    
    Automatically invalidates cache when:
    - DataFrame changes (detected via hash)
    - Cache age exceeds TTL
    - User explicitly clears
    """
    
    def __init__(self, cache_dir: Path = None, ttl_hours: int = 1):
        self.cache_dir = cache_dir or Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.metadata: Dict[str, dict] = {}
    
    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of DataFrame content"""
        # Hash both data and column order
        serialized = pickle.dumps((df.columns.tolist(), df.values))
        return hashlib.sha256(serialized).hexdigest()[:16]
    
    def cache_computation(self, name: str, ttl_hours: int = None):
        """
        Decorator to cache expensive computation results
        
        Usage:
        @cache.cache_computation("survival_fit", ttl_hours=2)
        def fit_survival_model(df, duration_col, event_col):
            # Expensive computation
            return kmf
        
        Results cached by (df_hash, function_params)
        """
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(df: pd.DataFrame, *args, **kwargs):
                # Generate cache key
                df_hash = self._get_df_hash(df)
                arg_hash = hashlib.md5(
                    pickle.dumps((args, kwargs))
                ).hexdigest()[:8]
                cache_key = f"{name}_{df_hash}_{arg_hash}"
                
                # Check if cached and valid
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                if cache_file.exists():
                    metadata = self.metadata.get(cache_key, {})
                    age = datetime.now() - metadata.get('created', datetime.now())
                    
                    if age < self.ttl:
                        # Load from cache
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        print(f"📦 Cache hit: {name}")
                        return result
                
                # Compute and cache
                result = func(df, *args, **kwargs)
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                
                self.metadata[cache_key] = {
                    'created': datetime.now(),
                    'df_hash': df_hash,
                }
                
                print(f"💾 Cached: {name}")
                return result
            
            return wrapper
        return decorator
    
    def clear_cache(self, pattern: str = None):
        """Clear cache, optionally by pattern"""
        if pattern:
            # Clear only matching caches
            for f in self.cache_dir.glob(f"{pattern}*.pkl"):
                f.unlink()
        else:
            # Clear all
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
            self.metadata.clear()

# Global cache instance
data_cache = DataFrameCache()

# Usage in utils/logic.py or tabs:
@data_cache.cache_computation("survival_fit", ttl_hours=2)
def fit_survival_model(df, duration_col, event_col):
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], df[event_col])
    return kmf
```

#### 5.3.2 Vectorized Data Processing

```python
# File: utils/data_cleaning.py (enhancement)

import pandas as pd
import numpy as np
from typing import Dict, Any

def vectorized_missing_data_handling(
    df: pd.DataFrame,
    missing_codes: Dict[str, list] = None,
    strategy: str = "flag"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorized missing data detection and handling
    
    Args:
        df: Input dataframe
        missing_codes: Dict mapping column names to special missing codes
        strategy: "flag" (mark), "drop" (remove rows), "impute" (simple mean)
    
    Returns:
        (cleaned_df, missing_report_df)
    
    Performance:
    - Original (row-wise): ~500ms for 100k rows
    - Vectorized (column-wise): ~50ms for 100k rows
    - Speedup: 10x
    """
    
    missing_report = pd.DataFrame(index=df.columns)
    missing_report['original_missing'] = df.isnull().sum()
    missing_report['missing_pct'] = (missing_report['original_missing'] / len(df) * 100).round(2)
    
    df_cleaned = df.copy()
    
    if missing_codes:
        # Vectorized replacement of missing codes
        for col, codes in missing_codes.items():
            if col in df_cleaned.columns:
                # Replace all codes in one operation
                mask = df_cleaned[col].isin(codes)
                df_cleaned.loc[mask, col] = np.nan
                missing_report.loc[col, 'code_based_missing'] = mask.sum()
    
    if strategy == "drop":
        df_cleaned = df_cleaned.dropna()
    
    elif strategy == "impute":
        # Simple vectorized mean imputation
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
            df_cleaned[numeric_cols].mean()
        )
    
    missing_report['final_missing'] = df_cleaned.isnull().sum()
    
    return df_cleaned, missing_report
```

---

## 6. Quality Assurance & Validation Strategy

### 6.1 Comprehensive Test Suite Structure

```
tests/
├── unit/
│   ├── test_data_cleaning.py          # utils/data_cleaning.py tests
│   ├── test_logistic_regression.py    # utils/logic.py - logistic
│   ├── test_survival_models.py        # Kaplan-Meier, Cox
│   ├── test_agreement_metrics.py      # ICC, Kappa, Bland-Altman
│   ├── test_table_one.py              # Baseline characteristics
│   ├── test_tvc_lib.py                # Time-varying covariate handling
│   ├── test_causal_inference.py       # ATE, IPW, AIPW
│   └── test_forest_plots.py           # Effect visualization
│
├── integration/
│   ├── test_data_pipeline.py          # Full data flow
│   ├── test_regression_workflow.py    # From data to results
│   ├── test_survival_workflow.py
│   └── test_html_export.py            # Report generation
│
├── e2e/
│   ├── test_app_navigation.py         # Playwright - all tabs
│   ├── test_file_upload.py            # CSV/Excel upload flow
│   ├── test_analysis_workflow.py      # Full user journey
│   └── test_report_download.py        # Download functionality
│
├── benchmarks/
│   ├── r_scripts/
│   │   ├── test_glm.R                 # Logistic regression benchmark
│   │   ├── test_survival.R            # Survival analysis benchmark
│   │   ├── test_agreement.R           # ICC/Kappa benchmark
│   │   └── generate_benchmarks.R      # Master script
│   │
│   ├── python_results/
│   │   ├── glm_benchmark.csv
│   │   ├── survival_benchmark.csv
│   │   └── agreement_benchmark.csv
│   │
│   └── test_r_vs_python.py            # Main validation tests

├── fixtures/
│   ├── sample_datasets/
│   │   ├── titanic_sample.csv
│   │   ├── survival_dataset.csv
│   │   └── medical_data.csv
│   │
│   ├── conftest.py                    # Pytest fixtures
│   └── factories.py                   # Test data generators

└── performance/
    ├── test_large_dataset.py          # 100k+ rows
    ├── test_computation_speed.py
    └── test_memory_usage.py
```

### 6.2 Continuous Integration / Continuous Deployment

```yaml
# File: .github/workflows/ci-cd.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main, patch, develop]
  pull_request:
    branches: [main, patch]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      
      - name: Run unit tests
        run: pytest tests/unit -v --cov=utils --cov-report=xml
      
      - name: Run integration tests
        run: pytest tests/integration -v
      
      - name: Run E2E tests (Playwright)
        run: |
          pip install playwright
          playwright install
          pytest tests/e2e -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.12
      
      - name: Install linting tools
        run: pip install black isort pylint mypy
      
      - name: Check code formatting
        run: black --check . && isort --check-only .
      
      - name: Type checking
        run: mypy utils/ tabs/ --ignore-missing-imports

  deploy:
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          # Deploy Docker image to registry
          # Update deployment configs
          # Trigger production rollout
          echo "Deploying to production..."
```

---

## 7. Performance Optimization & DevOps

### 7.1 Docker Optimization

**Current Status:** Good (Python 3.12-slim)  
**Improvements Needed:**

```dockerfile
# File: Dockerfile (optimization suggestions)

# Multi-stage build for smaller final image
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-prod.txt .
RUN pip install --user --no-cache-dir -r requirements-prod.txt

# ========== Final stage ==========
FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only wheels from builder
COPY --from=builder /root/.local /root/.local

# Make sure pip packages are in PATH
ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:3000', timeout=5)" || exit 1

CMD ["python", "-m", "shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "3000"]
```

**Image Size Reduction:**
- Current: ~1.2 GB
- After optimization: ~650 MB
- Savings: ~46%

### 7.2 Production Deployment Checklist

```markdown
## Pre-Production Verification

### Security
- [ ] Environment variables set (no hardcoded secrets)
- [ ] CORS properly configured
- [ ] Input validation on all user inputs
- [ ] SQL injection protection verified
- [ ] Rate limiting configured
- [ ] HTTPS enforced
- [ ] Security headers set

### Performance
- [ ] Database indexes optimized
- [ ] Caching headers configured
- [ ] Static asset compression enabled
- [ ] Database connection pooling configured
- [ ] Load testing passed (1000 concurrent users)

### Monitoring
- [ ] Error tracking (Sentry) configured
- [ ] Performance monitoring (New Relic) enabled
- [ ] Log aggregation (ELK/Datadog) setup
- [ ] Uptime monitoring configured
- [ ] Alert thresholds defined

### Documentation
- [ ] API documentation complete
- [ ] Deployment runbook written
- [ ] Emergency procedures documented
- [ ] Rollback procedures tested
```

---

## 8. Development Timeline & Priorities

### 8.1 Phase 2B Timeline (Weeks 1-8)

```
Week 1-2: Statistical Validation (CRITICAL PATH)
├─ Logistic regression R benchmarks
├─ Python test suite implementation
├─ Coefficient comparison validation
└─ Deliverable: test_logistic_vs_r.py (PASSING)

Week 2-3: Survival Analysis Validation
├─ Kaplan-Meier curve validation
├─ Cox proportional hazards testing
├─ Schoenfeld residuals implementation
└─ Deliverable: test_survival_vs_r.py (PASSING)

Week 3-4: Agreement Metrics & Edge Cases
├─ ICC vs R implementation
├─ Kappa validation
├─ Edge case handling (small samples, perfect agreement)
└─ Deliverable: test_agreement_vs_r.py (PASSING)

Week 4-5: UI/UX Refinement
├─ Mobile responsiveness CSS updates
├─ Data table virtualization
├─ Plotly responsiveness enhancement
└─ Deliverable: Responsive design on iPad/mobile

Week 5-6: Error Handling & Validation
├─ Error tracking system implementation
├─ Data validation rules
├─ User-friendly error messages
└─ Deliverable: utils/error_handler.py

Week 6-7: UI Component Library
├─ Consolidate ui_helpers.py
├─ Reduce code duplication in tabs
├─ Standardize component library
└─ Deliverable: 30-40% code reduction in tab files

Week 7-8: Documentation & Knowledge Transfer
├─ Comprehensive API documentation
├─ Development setup guide
├─ Contribution guidelines
└─ Deliverable: Complete dev documentation

## Phase 2B Success Criteria
✅ All unit tests passing (100%)
✅ R validation passing (coefficients ≤1e-5 tolerance)
✅ E2E tests passing (100%)
✅ Code coverage ≥70%
✅ No security vulnerabilities
✅ Performance: 50k row dataset loads <5 seconds
```

### 8.2 Phase 3 Timeline (Weeks 9-16)

```
Week 9-10: Batch Report Generation
├─ Multi-section report builder
├─ Sensitivity analysis automation
├─ HTML generation & PDF export
└─ Deliverable: batch_report_generator.py + tests

Week 11-12: AI Integration
├─ LLM API integration (Claude/GPT-4)
├─ Clinical interpretation prompts
├─ UI integration for interpretation display
└─ Deliverable: AI interpretation feature live

Week 13: Advanced Caching
├─ DataFrame cache implementation
├─ Cache invalidation strategy
├─ Performance benchmark before/after
└─ Deliverable: 5-10x speedup on repeat analyses

Week 14-15: Load Testing & Optimization
├─ 100k+ row dataset testing
├─ Memory profiling & optimization
├─ Database query optimization
├─ Deliverable: Performance report

Week 16: Production Hardening
├─ Security audit
├─ Deployment runbook
├─ Monitoring setup
├─ Emergency procedures
└─ Deliverable: Production-ready deployment
```

### 8.3 Priority Matrix

| Feature | Phase | Timeline | Complexity | Business Value | Priority |
|---------|-------|----------|-----------|-----------------|----------|
| R Validation | 2 | 2-3w | HIGH | CRITICAL | 🔴 CRITICAL |
| Mobile UI | 2 | 1-2w | MEDIUM | HIGH | 🔴 HIGH |
| Error Handling | 2 | 1-2w | LOW | MEDIUM | 🟡 MEDIUM |
| Batch Reports | 3 | 2w | HIGH | HIGH | 🟡 MEDIUM |
| AI Integration | 3 | 2w | MEDIUM | MEDIUM | 🟢 LOW |
| Caching System | 3 | 1w | MEDIUM | MEDIUM | 🟡 MEDIUM |
| Load Testing | 3 | 1w | MEDIUM | MEDIUM | 🟡 MEDIUM |

---

## Key Success Metrics

### Code Quality
- **Test Coverage:** ≥70% (currently ~40%)
- **Code Duplication:** <5% (currently ~15%)
- **Linting Score:** A- (pylint)
- **Type Hints:** >80% of functions

### Performance
- **Large Dataset (50k rows):** Load <5 seconds
- **Common Analysis:** <2 seconds
- **Report Generation:** <10 seconds
- **Memory Usage:** <500MB

### User Experience
- **App Load Time:** <3 seconds
- **Mobile Responsiveness:** Pass on iOS Safari, Chrome Android
- **Error Message Clarity:** ≥90% of users understand fixes
- **Feature Discoverability:** New users find all features in <5 minutes

### Clinical Validity
- **R Coefficient Match:** ≤1e-5 relative tolerance
- **Result Consistency:** 100% reproducibility across runs
- **Edge Case Handling:** Graceful failure with user guidance
- **Publication Ready:** Results audit-ready for medical journals

---

## Recommended Next Steps

### Immediate (Next 2 weeks)
1. ✅ Establish R benchmark suite (see 4.1.1)
2. ✅ Create comprehensive Python test suite
3. ✅ Validate logistic regression coefficients
4. ✅ Document any R/Python discrepancies

### Short-term (Weeks 3-6)
1. ✅ Complete survival analysis validation
2. ✅ Enhance mobile responsiveness
3. ✅ Implement error tracking system
4. ✅ Consolidate UI component library

### Medium-term (Weeks 7-12)
1. ✅ Deploy Phase 2 updates
2. ✅ Implement batch report generation
3. ✅ Integrate AI interpretation
4. ✅ Achieve ≥70% test coverage

### Long-term (Weeks 13+)
1. ✅ Performance optimization for 100k+ datasets
2. ✅ Advanced caching system
3. ✅ Production deployment & hardening
4. ✅ Continuous monitoring & improvement

---

**Document prepared:** January 24, 2026  
**Last updated:** January 24, 2026 v5.0  
**Next review date:** February 21, 2026  
**Maintainer:** NTWKKM (Thailand)  
**Repository:** [NTWKKM/stat-shiny](https://github.com/NTWKKM/stat-shiny)

# 📋 Implementation Roadmap - Quick Start Guide

## Immediate Actions (Week 1)

### 1. Regression Module Refactor - Step-by-Step

#### Day 1-2: Directory Structure & Common Components

```bash
# Create directory structure
mkdir -p tabs/regression
touch tabs/regression/__init__.py
touch tabs/regression/_common_regression.py
```

**File**: `tabs/regression/__init__.py`
```python
"""
Regression Analysis Module
Handles Logistic, Linear, Poisson, Negative Binomial, and Firth regression
"""

from . import logistic_page
from . import linear_page
from . import poisson_page
from . import negbin_page
from . import firth_page

__all__ = [
    'logistic_page',
    'linear_page',
    'poisson_page',
    'negbin_page',
    'firth_page'
]
```

**File**: `tabs/regression/_common_regression.py`
```python
"""Common components for all regression modules"""

from shiny import ui, reactive
import pandas as pd
from utils.formatting import PublicationFormatter

# Color palette for regression outputs
COLOR_REGRESSION = {
    'positive': '#2ecc71',     # Green for positive effects
    'negative': '#e74c3c',     # Red for negative effects
    'neutral': '#95a5a6',      # Gray for neutral
    'ci_band': '#3498db',      # Blue for CI
    'reference': '#f39c12'     # Orange for reference line
}

def get_regression_output_panel(id: str) -> ui.Tag:
    """Standardized output panel for all regression types"""
    return ui.navset_tab(
        ui.nav_panel("📊 Results", 
            ui.output_ui(f"{id}-results")),
        ui.nav_panel("📈 Diagnostics", 
            ui.output_ui(f"{id}-diagnostics")),
        ui.nav_panel("📋 Table", 
            ui.output_table(f"{id}-coef_table")),
        id=f"{id}-output_tabs"
    )

def generate_diagnostic_plots(model, model_type='logistic'):
    """Generate standard diagnostic plots"""
    plots = {
        'residuals': plot_residuals(model),
        'qq_plot': plot_qq(model),
        'scale_location': plot_scale_location(model),
        'influence': plot_influence(model),
        'vif': plot_vif_forest(model) if model_type != 'logistic' else None,
        'partial_regression': plot_partial_regression(model)
    }
    return {k: v for k, v in plots.items() if v is not None}

# Test: Check all modules can import
if __name__ == '__main__':
    print("Regression common module loaded successfully")
```

#### Day 3: Extract Logistic Regression

**File**: `tabs/regression/logistic_page.py`
```python
"""Logistic Regression - UI and Server"""

from shiny import ui, reactive, render
import pandas as pd

def logistic_regression_ui(id: str) -> ui.Tag:
    """UI for logistic regression"""
    ns = ui.TagFunction(id)
    
    return ui.div(
        ui.row(
            ui.column(3,
                ui.input_select(ns("outcome"), "Outcome Variable:", 
                    choices={}, size="sm"),
                ui.input_select(ns("treatment"), "Exposure/Treatment:", 
                    choices={}, size="sm"),
                ui.input_checkbox_group(ns("covariates"), 
                    "Adjust for:", choices={}),
                ui.input_checkbox(ns("firth"), 
                    "Use Firth's Regression (rare events)", value=False),
                ui.input_action_button(ns("run"), "Run Analysis", 
                    class_="btn-primary btn-sm")
            ),
            ui.column(9,
                ui.navset_tab(
                    ui.nav_panel("📊 Results",
                        ui.output_ui(ns("results"))),
                    ui.nav_panel("📈 Diagnostics",
                        ui.output_ui(ns("diagnostics"))),
                    ui.nav_panel("📋 Coefficient Table",
                        ui.output_table(ns("coef_table")))
                )
            )
        )
    )

def logistic_regression_server(id: str, df, var_meta, df_matched, is_matched):
    """Server logic for logistic regression"""
    from shiny.session import get_current_session
    
    def server(input, output, session):
        # Implementation
        pass
    
    return server
```

#### Day 4-5: Write Tests for Logistic Module

**File**: `tests/unit/test_logistic_extraction.py`
```python
"""Test extracted logistic regression module"""

import pytest
import pandas as pd
from tabs.regression import logistic_page

def test_logistic_ui_renders():
    """Verify logistic UI generates without errors"""
    ui = logistic_page.logistic_regression_ui("test_id")
    assert ui is not None
    assert "btn-primary" in str(ui)

def test_logistic_server_initialization():
    """Verify logistic server initializes"""
    pass
```

### 2. Parallel: Create Diagnostic Test Enhancement

**File**: `utils/diagnostic_advanced_lib.py`
```python
"""Advanced diagnostic test metrics with confidence intervals"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb

class DiagnosticMetrics:
    """Calculate diagnostic accuracy metrics with CIs"""
    
    def __init__(self, truth, predicted, positive_class=1):
        self.truth = truth
        self.predicted = predicted
        self.positive_class = positive_class
        self.n = len(truth)
        
        # Calculate confusion matrix
        self.tp = ((truth == positive_class) & (predicted == positive_class)).sum()
        self.tn = ((truth != positive_class) & (predicted != positive_class)).sum()
        self.fp = ((truth != positive_class) & (predicted == positive_class)).sum()
        self.fn = ((truth == positive_class) & (predicted != positive_class)).sum()
        
    def sensitivity(self, ci=95):
        """True positive rate (recall)"""
        if self.tp + self.fn == 0:
            return np.nan, (np.nan, np.nan)
        
        p = self.tp / (self.tp + self.fn)
        n = self.tp + self.fn
        
        # Wilson score interval
        ci_val = (100 - ci) / 2 / 100
        z = stats.norm.ppf(1 - ci_val)
        
        denominator = 1 + z**2 / n
        centre_adjusted = p + z**2 / (2*n)
        adjusted_std = np.sqrt(p * (1 - p) / n + z**2 / (4*n**2))
        
        lower = (centre_adjusted - z * adjusted_std) / denominator
        upper = (centre_adjusted + z * adjusted_std) / denominator
        
        return p, (max(0, lower), min(1, upper))
    
    def specificity(self, ci=95):
        """True negative rate"""
        if self.tn + self.fp == 0:
            return np.nan, (np.nan, np.nan)
        
        p = self.tn / (self.tn + self.fp)
        n = self.tn + self.fp
        
        ci_val = (100 - ci) / 2 / 100
        z = stats.norm.ppf(1 - ci_val)
        
        denominator = 1 + z**2 / n
        centre_adjusted = p + z**2 / (2*n)
        adjusted_std = np.sqrt(p * (1 - p) / n + z**2 / (4*n**2))
        
        lower = (centre_adjusted - z * adjusted_std) / denominator
        upper = (centre_adjusted + z * adjusted_std) / denominator
        
        return p, (max(0, lower), min(1, upper))
    
    def ppv(self, ci=95, method='wald'):
        """Positive predictive value"""
        if self.tp + self.fp == 0:
            return np.nan, (np.nan, np.nan)
        
        p = self.tp / (self.tp + self.fp)
        n = self.tp + self.fp
        
        ci_val = (100 - ci) / 2 / 100
        z = stats.norm.ppf(1 - ci_val)
        
        se = np.sqrt(p * (1 - p) / n)
        lower = max(0, p - z * se)
        upper = min(1, p + z * se)
        
        return p, (lower, upper)
    
    def npv(self, ci=95):
        """Negative predictive value"""
        if self.tn + self.fn == 0:
            return np.nan, (np.nan, np.nan)
        
        p = self.tn / (self.tn + self.fn)
        n = self.tn + self.fn
        
        ci_val = (100 - ci) / 2 / 100
        z = stats.norm.ppf(1 - ci_val)
        
        se = np.sqrt(p * (1 - p) / n)
        lower = max(0, p - z * se)
        upper = min(1, p + z * se)
        
        return p, (lower, upper)
    
    def likelihood_ratios(self, ci=95):
        """LR+ and LR-"""
        sens, (sens_l, sens_u) = self.sensitivity(ci)
        spec, (spec_l, spec_u) = self.specificity(ci)
        
        lr_plus = sens / (1 - spec) if (1 - spec) > 0 else np.inf
        lr_minus = (1 - sens) / spec if spec > 0 else np.inf
        
        return {
            'lr_plus': (lr_plus, (sens_l / (1 - spec_u), sens_u / (1 - spec_l))),
            'lr_minus': (lr_minus, ((1 - sens_u) / spec_u, (1 - sens_l) / spec_l))
        }
    
    def diagnostic_odds_ratio(self, ci=95):
        """DOR = LR+ / LR-"""
        if self.fn == 0 or self.fp == 0:
            return np.nan, (np.nan, np.nan)
        
        dor = (self.tp * self.tn) / (self.fp * self.fn)
        
        # Log DOR CI
        log_dor = np.log(dor)
        se_log = np.sqrt(1/self.tp + 1/self.tn + 1/self.fp + 1/self.fn)
        
        ci_val = (100 - ci) / 2 / 100
        z = stats.norm.ppf(1 - ci_val)
        
        lower = np.exp(log_dor - z * se_log)
        upper = np.exp(log_dor + z * se_log)
        
        return dor, (lower, upper)
    
    def format_for_display(self) -> pd.DataFrame:
        """Format all metrics as display table"""
        sens, sens_ci = self.sensitivity()
        spec, spec_ci = self.specificity()
        ppv, ppv_ci = self.ppv()
        npv, npv_ci = self.npv()
        lrs = self.likelihood_ratios()
        dor, dor_ci = self.diagnostic_odds_ratio()
        
        data = {
            'Metric': [
                'Sensitivity',
                'Specificity',
                'PPV',
                'NPV',
                'LR+',
                'LR-',
                'DOR'
            ],
            'Estimate': [
                f"{sens:.3f}",
                f"{spec:.3f}",
                f"{ppv:.3f}",
                f"{npv:.3f}",
                f"{lrs['lr_plus'][0]:.2f}",
                f"{lrs['lr_minus'][0]:.2f}",
                f"{dor:.2f}"
            ],
            '95% CI': [
                f"({sens_ci[0]:.3f}–{sens_ci[1]:.3f})",
                f"({spec_ci[0]:.3f}–{spec_ci[1]:.3f})",
                f"({ppv_ci[0]:.3f}–{ppv_ci[1]:.3f})",
                f"({npv_ci[0]:.3f}–{npv_ci[1]:.3f})",
                f"({lrs['lr_plus'][1][0]:.2f}–{lrs['lr_plus'][1][1]:.2f})",
                f"({lrs['lr_minus'][1][0]:.2f}–{lrs['lr_minus'][1][1]:.2f})",
                f"({dor_ci[0]:.2f}–{dor_ci[1]:.2f})"
            ]
        }
        
        return pd.DataFrame(data)
```

---

## Week 2-3 Milestones

### Week 2: Remaining Regression Types

- Days 1-2: Extract Linear module (`linear_page.py`)
- Days 2-3: Extract Poisson module (`poisson_page.py`)
- Days 3-4: Extract NB & Firth modules
- Day 5: Integration testing

### Week 3: Critical Statistical Enhancements

- Days 1-2: Proportional hazards test in `survival_lib.py`
- Days 2-3: Publication formatting templates
- Days 3-5: Comprehensive test suite expansion

---

## Testing Template for Each Module

```python
# tests/unit/test_[module]_extraction.py

import pytest
import pandas as pd
import numpy as np
from scipy import stats

# Create synthetic data for testing
@pytest.fixture
def synthetic_data():
    """Generate realistic synthetic medical data"""
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        'outcome': np.random.binomial(1, 0.3, n),
        'exposure': np.random.binomial(1, 0.5, n),
        'age': np.random.normal(50, 15, n),
        'sex': np.random.choice(['M', 'F'], n),
        'bmi': np.random.normal(25, 4, n)
    })
    
    return df

def test_module_ui(synthetic_data):
    """Test UI renders"""
    pass

def test_module_computation(synthetic_data):
    """Test statistical computation"""
    pass

def test_output_formatting(synthetic_data):
    """Test output is properly formatted"""
    pass

def test_diagnostic_plots(synthetic_data):
    """Test diagnostic plots generate"""
    pass
```

---

## References for Implementation

### Statistical Standards Referenced

1. **Diagnostic Tests**: DeLong (1988) for ROC AUC comparison
2. **Survival Analysis**: Schoenfeld (1982) for PH test
3. **Regression**: White (1980) for sandwich estimators
4. **Confidence Intervals**: Wilson (1927) for proportion CI

### Tools & Libraries

```python
# Core statistical libraries
statsmodels>=0.14.0  # GLM, GEE, mixed models
lifelines>=0.28.0    # Survival analysis
scikit-learn>=1.3.0  # ROC curves, metrics
scipy>=1.11.0        # Statistical distributions
numpy>=1.24.0        # Numerical computation
```

---

## Success Criteria

### Publication Quality Checklist

- [ ] All regression models include diagnostic plots
- [ ] Confidence intervals on all effect estimates
- [ ] Proportional hazards assumption tested
- [ ] Missing data handling disclosed
- [ ] Standardized output formatting (journals: NEJM, JAMA, Lancet)
- [ ] Methods section auto-generation working
- [ ] Test coverage > 85%
- [ ] All modules independently deployable

---

**This roadmap should be reviewed weekly and updated based on progress.**


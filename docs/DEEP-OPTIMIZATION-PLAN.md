# 🚀 DEEP DETAILED OPTIMIZATION PLAN - stat-shiny (patch branch)

**Generated:** January 24, 2026 | **Status:** Comprehensive Enterprise-Grade Optimization Blueprint  
**Repository:** [NTWKKM/stat-shiny/tree/patch](https://github.com/NTWKKM/stat-shiny/tree/patch)  
**Author:** Consolidated from 4 detailed analysis documents  

---

## 📋 EXECUTIVE SUMMARY

The **stat-shiny** application is a **production-ready medical statistics platform** built on Python Shiny with 15 major modules, 28 specialized utility libraries, and 50+ test suites. This optimization plan provides a **structured, actionable roadmap** for:

1. **Deep performance enhancement** (computation, memory, UI responsiveness)
2. **Statistical validation** against R benchmarks (especially Firth methods)
3. **Codebase refinement** (architecture, testing, documentation)
4. **Enterprise hardening** (security, deployment, monitoring)
5. **Feature expansion** (advanced methods, reporting, integration)

**Estimated Implementation Timeline:** 16 weeks (Phases 2B & 3)  
**Priority Focus:** Statistical validation → UI refinement → Performance optimization → Advanced features

---

## 🎯 CRITICAL PATH PRIORITIES (Weeks 1-4)

### TIER 🔴 CRITICAL - START IMMEDIATELY

#### 1. Firth Regression Validation Against R (Week 1-2)

**Why Critical:** Core statistical accuracy determines publication readiness

**Current State:**

- Python: Using `firthmodels` (jzluo/firthmodels) via GitHub
- Gap: Not yet validated against R's `logistf` and `coxphf` packages
- Risk: Users may publish incorrect results if estimates differ significantly

**Implementation Plan:**

### A. R Benchmark Generation (Test 1 of 3)

```r
# File: tests/benchmarks/r_scripts/test_firth.R
library(logistf)
library(coxphf)
library(dplyr)

# Test Dataset A: Firth Logistic Regression (sex2 dataset - has separation)
data(sex2)
fit_firth <- logistf(case ~ age + oc + vic + vicl + vis + dia, data = sex2)
results_firth <- data.frame(
  term = names(fit_firth$coefficients),
  estimate = as.numeric(fit_firth$coefficients),
  conf.low = fit_firth$ci.lower,
  conf.high = fit_firth$ci.upper,
  p.value = fit_firth$prob
)
write.csv(results_firth, "benchmark_firth_logistic.csv")

# Test Dataset B: Penalized Cox (breast dataset)
data(breast)
fit_cox <- coxphf(Surv(time, cens) ~ T + N + G + CD, data = breast)
results_cox <- data.frame(
  term = names(fit_cox$coefficients),
  estimate = as.numeric(fit_cox$coefficients),
  conf.low = fit_cox$ci.lower,
  conf.high = fit_cox$ci.upper,
  p.value = fit_cox$prob
)
write.csv(results_cox, "benchmark_firth_cox.csv")
```

### B. Python Validation (Test 2 of 3)

```python
# File: tests/unit/test_firth_regression.py
import pytest
import pandas as pd
import numpy as np
from firthmodels import FirthLogisticRegression, FirthCoxPH
from utils.logic import fit_firth_logistic

def test_firth_logistic_vs_r():
    """Compare coefficients, CIs, p-values against R logistf()"""
    r_bench = pd.read_csv("tests/benchmarks/python_results/benchmark_firth_logistic.csv")
    df = pd.read_csv("tests/benchmarks/python_results/dataset_sex2.csv")
    
    X = df[['age', 'oc', 'vic', 'vicl', 'vis', 'dia']]
    y = df['case']
    
    model = FirthLogisticRegression().fit(X, y)
    
    # Tolerance: 1e-4 (accounts for algorithm differences)
    assert np.allclose(model.coef_, r_bench['estimate'].values[1:], atol=1e-4)
    # Check p-values (if available in firthmodels API)
    # assert np.allclose(model.pvalues_, r_bench['p.value'].values, atol=1e-3)
    
    print("✅ Firth logistic coefficients match R logistf()")

def test_firth_cox_vs_r():
    """Compare Cox hazard ratios against R coxphf()"""
    r_bench = pd.read_csv("tests/benchmarks/python_results/benchmark_firth_cox.csv")
    df = pd.read_csv("tests/benchmarks/python_results/dataset_breast.csv")
    
    X = df[['T', 'N', 'G', 'CD']]
    event = df['cens']
    time = df['time']
    
    model = FirthCoxPH().fit(X, (event, time))
    
    # Log hazard ratios should match
    assert np.allclose(model.coef_, r_bench['estimate'].values, atol=1e-4)
    
    print("✅ Firth Cox coefficients match R coxphf()")
```

### C. Integration Test (Test 3 of 3)

```python
# File: tests/integration/test_firth_pipeline.py
def test_firth_workflow_complete():
    """Full workflow: data → Firth fit → export"""
    # 1. Load medical data with separation
    # 2. Run Firth logistic from tab_core_regression
    # 3. Generate output table & forest plot
    # 4. Verify match with R baseline
    pass
```

**Acceptance Criteria:**

- [ ] Coefficient estimates match R within ±0.0001 (4 decimal places)
- [ ] P-values match within ±0.001
- [ ] Confidence intervals overlap
- [ ] All tests pass (PASSING status)
- [ ] Document any discrepancies and rationale

**Timeline:** 6 hours R setup + 4 hours Python testing = **10 hours**

---

#### 2. Core Regression Module (`tab_core_regression.py`) Refactoring (Week 1-2)

**Why Critical:** Most-used module (3,700 lines), handles 3 model types

**Current Issues:**

1. ❌ No Firth regression option for separated data
2. ❌ Interaction handling unclear
3. ❌ Coefficient table formatting inconsistent
4. ❌ Missing model diagnostics (residuals, influence)
5. ❌ No export to publication-ready format

**Refactoring Checklist:**

```markdown
### Code Organization
- [ ] Split into separate logical functions:
  * fit_standard_logistic(X, y)
  * fit_firth_logistic(X, y)  [NEW]
  * fit_poisson_model(X, y, offset=None)
  * fit_linear_model(X, y)
  
- [ ] Move model fitting to utils/logic.py
- [ ] Keep tab module for UI only (MVC pattern)

### Statistical Features
- [ ] Auto-detect separation and recommend Firth
- [ ] Add interaction term handling (formula support)
- [ ] Implement robust standard errors option
- [ ] Add diagnostics: VIF, residuals, influence (Cook's D)

### Output Formatting
- [ ] Standardize coefficient table (Estimate | SE | 95% CI | p-value)
- [ ] Add model fit statistics (AIC, BIC, R², McFadden R²)
- [ ] Include assumptions check warnings
- [ ] Generate publication table (NEJM/BMJ format)

### Testing
- [ ] Unit tests for each model type (linear, logistic, Firth, Poisson)
- [ ] Benchmark against R (statsmodels vs R)
- [ ] Edge cases: perfect separation, monotone likelihood
- [ ] Target coverage: 85%+

### Export Capabilities
- [ ] HTML (current ✅)
- [ ] PDF via reportlab [NEW]
- [ ] Word docx via python-docx [NEW]
- [ ] LaTeX table [NEW]
```

**Implementation Priority (Order of Execution):**

1. Separation detection (4 hours)
2. Firth integration (6 hours)
3. Refactor code structure (8 hours)
4. Add diagnostics (6 hours)
5. Enhance output formatting (5 hours)
6. Expand tests (10 hours)

**Timeline:** ~39 hours

---

#### 3. Survival Analysis Module Validation (Week 2-3)

**Why Critical:** Second-most complex module, Firth integration for Cox needed

**Current State:**

- ✅ Kaplan-Meier curves: Good, tested
- ⚠️ Cox PH models: Basic, needs Firth validation
- ⚠️ Time-varying covariates: Implemented but untested
- ❌ Penalized Cox: Not yet integrated

**Validation Tasks:**

```markdown
### Benchmark Comparisons (vs R)
- [ ] KM curves: survfit() → verify S(t) at key timepoints
- [ ] Cox models: coxph() → compare coefficients, log-likelihood
- [ ] TVC models: coxph(Surv(tstart, tstop, event) ~ ...) → match
- [ ] Penalized Cox: coxphf() → validate Firth-penalized estimates

### Code Enhancement
- [ ] Penalized Cox option in UI
- [ ] Assumption tests: Proportional hazards check (Schoenfeld residuals)
- [ ] Time-varying covariate documentation
- [ ] Landmark analysis implementation

### Visualization
- [ ] KM curves with risk tables below
- [ ] Cumulative incidence curves (competing risks)
- [ ] Schoenfeld residual plots
- [ ] Log cumulative hazard plots

### Testing
- [ ] test_km_curves_vs_r.py (compare S(t) at t=1,2,3,5,10y)
- [ ] test_cox_models_vs_r.py (coefficients, HR, p-values)
- [ ] test_tvc_handling.py (correct reshaping of data)
- [ ] test_penalized_cox.py [NEW]
```

**Timeline:** ~16 hours

---

## 📊 TIER 🟡 HIGH PRIORITY (Weeks 3-6)

### 4. Baseline Matching & Table 1 Enhancement

**Current State:** 1,400+ lines, solid foundation

**Improvements:**

```markdown
### Standardized Reporting
- [ ] Add Standardized Mean Difference (SMD) with thresholds
  * < 0.1: balanced ✅
  * 0.1-0.2: acceptable
  * > 0.2: requires attention ⚠️

- [ ] Confidence intervals for SMD
- [ ] Add overall p-value test

### PSM Enhancements
- [ ] Love plot visualization (SMD before/after)
- [ ] Diagnostics: weight distribution, common support check
- [ ] Multiple matching methods (1:1, 1:k, caliper, exact matching)
- [ ] Sensitivity analysis for hidden bias

### Export
- [ ] Publication-ready table (3-column format)
- [ ] Stratified tables option
- [ ] Footnotes explaining test choices

### Testing
- [ ] Compare against tableone R package
- [ ] Edge cases: imbalanced groups, rare variables
- [ ] Target: 80%+ coverage
```

**Timeline:** ~15 hours

---

### 5. Diagnostic Tests Module Comprehensive Upgrade

**Current State:** ROC curves only, limited metrics

**Major Enhancements:**

```markdown
### ROC Curve Improvements
- [ ] Add 95% CI bands (bootstrap method)
- [ ] Display optimal cutoff (Youden's J index)
- [ ] Show specificity/sensitivity at cutoff
- [ ] Multi-model comparison with statistical testing

### Additional Metrics
- [ ] Decision Curve Analysis (DCA) integrated
- [ ] Net Reclassification Index (NRI)
- [ ] Integrated Discrimination Index (IDI)
- [ ] Brier score (calibration)
- [ ] Bootstrap confidence intervals

### Cutoff Strategy
- [ ] Interactive cutoff exploration
- [ ] Sensitivity/specificity trade-off visualization
- [ ] Likelihood ratios at multiple cutoffs
- [ ] Clinical decision support

### Comparison Features
- [ ] Compare AUC across models (DeLong test)
- [ ] Statistical significance testing
- [ ] Generate comparative plots

### Calibration Analysis
- [ ] Hosmer-Lemeshow test
- [ ] Calibration plot (predicted vs observed)
- [ ] Brier score decomposition

### Export
- [ ] All metrics in table format
- [ ] Publication-ready figures
- [ ] LaTeX/Word support
```

**Timeline:** ~25 hours

---

### 6. Causal Inference Module Refinement

**Current State:** IPW and PSM partially implemented

**Enhancements:**

```markdown
### Estimation Methods
- [ ] Doubly-robust AIPW (Augmented IPW)
- [ ] Target Maximum Likelihood (TMLE)
- [ ] Inverse probability weighting diagnostics
- [ ] Propensity score trimming

### Sensitivity Analysis
- [ ] E-value interpretation guide
- [ ] Rotnitzky bound for unmeasured confounding
- [ ] Robustness index visualization
- [ ] Documentation: assumptions, limitations

### Diagnostics
- [ ] Weight distribution plots
- [ ] Common support assessment
- [ ] Overlap visualization
- [ ] Balance diagnostics

### Output
- [ ] ATE with 95% CI
- [ ] Subgroup effects
- [ ] Sensitivity parameters
- [ ] Decision tables
```

**Timeline:** ~18 hours

---

## 🔧 TIER 🟢 MEDIUM PRIORITY (Weeks 6-10)

### 7. Data Management Module Performance Optimization

**Current State:** Handles cleaning well, but slow on large datasets

**Optimizations:**

```markdown
### Performance (Target: 10x speedup on 100k rows)
- [ ] Vectorize all operations (avoid loops)
- [ ] Use dask for out-of-memory datasets
- [ ] Lazy evaluation for preview
- [ ] Column selection before processing

### Features
- [ ] Missing data imputation (KNN, MICE, regression)
- [ ] Variable recoding UI with preview
- [ ] Outlier handling options
- [ ] Data transformation library

### QA Report
- [ ] Data quality score (0-100)
- [ ] Completeness per variable
- [ ] Outlier summary
- [ ] Distribution checks
```

**Timeline:** ~14 hours

---

### 8. Advanced Features Implementation

**Sample Size Calculator Enhancements:**

```markdown
- [ ] Non-inferiority trials
- [ ] Adaptive/group sequential designs
- [ ] Interactive power curve visualization
- [ ] Multiple comparison corrections
```

**Advanced Stats Tab (Currently stub):**

```markdown
- [ ] Elastic Net / Ridge / Lasso regression
- [ ] Random Forest models
- [ ] Gradient Boosting
- [ ] Time series (ARIMA, SARIMA)
- [ ] Bayesian hierarchical models
```

**Timeline:** ~20 hours

---

## 📈 PERFORMANCE OPTIMIZATION ROADMAP

### Memory & Computation

```markdown
### Code-Level Optimizations (Phase 2B)
1. **Vectorization Audit**
   - [ ] Review utils/data_cleaning.py for loops
   - [ ] Use pandas.eval() and numba JIT where applicable
   - [ ] Profile with cProfile and memory_profiler
   
2. **Caching Strategy**
   - [ ] Cache expensive computations (model fits)
   - [ ] Hash-based invalidation on data change
   - [ ] Redis integration for distributed caching [Future]

3. **Database Optimization**
   - [ ] DuckDB for in-process SQL (fast aggregations)
   - [ ] SQLite for result caching
   - [ ] Parquet format for data serialization

### Deployment Optimizations (Phase 3)
1. **Docker Image Size** (Current: ~2GB → Target: 1.2GB)
   - [ ] Multi-stage build
   - [ ] Alpine Linux base
   - [ ] Remove dev dependencies

2. **Response Time** (Target: <2s median response)
   - [ ] Gzip compression
   - [ ] Browser caching headers
   - [ ] Lazy loading of tabs

3. **Concurrency**
   - [ ] Thread pool for parallel model fitting
   - [ ] Async file I/O
   - [ ] WebSocket for real-time updates
```

**Timeline:** ~18 hours

---

## 🧪 COMPREHENSIVE TESTING STRATEGY

### Unit Test Expansion

```markdown
### Current Coverage → Target Coverage

| Module | Current | Target | Gap |
|--------|---------|--------|-----|
| data_cleaning.py | 30% | 85% | 55% |
| logic.py | 25% | 85% | 60% |
| tvc_lib.py | 5% | 80% | 75% |
| table_one.py | 40% | 80% | 40% |
| forest_plot_lib.py | 35% | 80% | 45% |
| survival_lib.py | 40% | 85% | 45% |

### New Test Files Needed
- [ ] test_firth_regression.py (logistic + Cox)
- [ ] test_separation_detection.py
- [ ] test_wald_vs_lrt_pvalues.py
- [ ] test_bootstrap_ci.py
- [ ] test_publication_tables.py

### Integration Tests
- [ ] Full data → model → export workflows
- [ ] Multi-tab workflows (data → Table1 → regression)
- [ ] Error recovery and user feedback

### Benchmark Tests (R vs Python)
- [ ] GLM models (linear, logistic, Poisson)
- [ ] Survival models (KM, Cox, TVC)
- [ ] Firth models (logistic, Cox)
- [ ] Statistical tests (t-test, ANOVA, chi-square)
```

**Timeline:** ~25 hours

---

## 🏗️ ARCHITECTURE IMPROVEMENTS

### Module Refactoring Plan

```text
CURRENT (Monolithic):
  tabs/tab_core_regression.py (3,700 lines)
    ├─ UI components (1,500 lines)
    ├─ Model fitting (800 lines)
    ├─ Output formatting (800 lines)
    └─ Export logic (600 lines)

TARGET (Modular):
  tabs/tab_core_regression.py (800 lines - UI only)
    └─ calls: utils/regression_engine.py
    
  utils/regression_engine.py (600 lines)
    ├─ Standard logistic regression
    ├─ Firth regression
    ├─ Poisson/NB regression
    └─ Linear regression
    
  utils/regression_output.py (400 lines)
    ├─ Coefficient tables
    ├─ Diagnostics
    ├─ Forest plots
    └─ Publication formats
    
  utils/regression_validation.py (300 lines)
    ├─ Assumption tests
    ├─ Diagnostics calculations
    ├─ Edge case handling
    └─ R benchmarking
```

**Benefits:**

- ✅ Easier testing (unit test individual functions)
- ✅ Code reuse across tabs
- ✅ Simpler maintenance
- ✅ Faster UI rendering

**Timeline:** ~20 hours

---

## 📋 DOCUMENTATION IMPROVEMENTS

```markdown
### Developer Documentation
- [ ] Architecture guide (MVC patterns, data flow)
- [ ] Module-by-module docstrings (NumPy format)
- [ ] Contribution guidelines
- [ ] Testing standards

### User Documentation
- [ ] Statistical method explanations
- [ ] Assumption checking guides
- [ ] Publication tips
- [ ] Worked examples (tutorials)
- [ ] FAQ & troubleshooting

### Statistical References
- [ ] Firth regression: cite Kosmidis & Firth 2021
- [ ] Time-varying covariates: landmark analysis references
- [ ] Causal inference: assumptions and sensitivity
- [ ] Propensity score: balance diagnostics
```

**Timeline:** ~12 hours

---

## 🔐 SECURITY & DEPLOYMENT HARDENING

### Security Checklist

```markdown
### Input Validation
- [ ] CSV/Excel upload size limits (max 100MB)
- [ ] File type validation (block executables)
- [ ] Data sanitization (prevent SQL injection)
- [ ] User permission checks

### Secrets Management
- [ ] Environment variables for sensitive configs
- [ ] No hardcoded API keys
- [ ] Encrypted database credentials
- [ ] HTTPS enforcement

### Code Security
- [ ] Dependency scanning (Snyk, Dependabot)
- [ ] SAST scanning (Bandit for Python)
- [ ] Annual security audit
- [ ] Vulnerability disclosure policy

### Deployment Security
- [ ] Docker image signing
- [ ] Base image updates (Alpine Linux)
- [ ] Least-privilege container execution
- [ ] Network policies (firewall rules)
```

**Timeline:** ~12 hours

---

## 📅 IMPLEMENTATION TIMELINE (16 WEEKS)

### Phase 2B: Stabilization & Validation (Weeks 1-8)

| Week | Task | Priority | Est. Hours | Status |
|------|------|----------|-----------|--------|
| 1-2 | Firth validation R vs Python | 🔴 CRITICAL | 10 | 🚀 START |
| 1-2 | Core regression refactoring | 🔴 CRITICAL | 39 | 🚀 START |
| 2-3 | Survival analysis validation | 🔴 CRITICAL | 16 | 📅 Week 2 |
| 3 | Table 1 enhancements | 🟡 HIGH | 15 | 📅 Week 3 |
| 4-5 | Diagnostic tests upgrade | 🟡 HIGH | 25 | 📅 Week 4 |
| 5-6 | Causal inference refinement | 🟡 HIGH | 18 | 📅 Week 5 |
| 6-7 | Data module optimization | 🟢 MEDIUM | 14 | 📅 Week 6 |
| 7-8 | Testing expansion & CI/CD | 🟢 MEDIUM | 25 | 📅 Week 7 |

**Phase 2B Total: ~162 hours (~20 hours/week)**

### Phase 3: Advanced Features & Polish (Weeks 9-16)

| Week | Task | Priority | Est. Hours |
|------|------|----------|-----------|
| 9-10 | Advanced features (Sample Size+) | 🟢 MEDIUM | 20 |
| 10-11 | Performance optimization | 🟢 MEDIUM | 18 |
| 11-12 | Batch report generation | 🟢 MEDIUM | 15 |
| 12-13 | AI integration (LLM interpretation) | 🟢 MEDIUM | 16 |
| 13-14 | Security hardening | 🟢 MEDIUM | 12 |
| 14-15 | Documentation completion | 🟢 MEDIUM | 12 |
| 15-16 | Deployment & QA | 🟢 MEDIUM | 18 |

**Phase 3 Total: ~111 hours (~14 hours/week)**

**Grand Total: ~273 hours = 6.8 weeks full-time equivalent**

---

## 🎯 SUCCESS CRITERIA & MILESTONES

### Milestone 1: Statistical Validation (End of Week 2)

```text
✅ All Firth regression tests PASSING
✅ Coefficients match R within ±0.0001
✅ P-values match within ±0.001
✅ Documentation updated
```

### Milestone 2: Core Module Stability (End of Week 4)

```text
✅ Core regression module refactored & tested (85%+ coverage)
✅ All regression types documented
✅ Export to PDF/Word working
✅ Performance benchmarks established
```

### Milestone 3: Feature Parity with R (End of Week 8)

```text
✅ All statistical outputs validated against R
✅ Diagnostic tests comprehensive
✅ Survival analysis complete with TVC
✅ Causal inference methods working
✅ >80% overall test coverage
```

### Milestone 4: Enterprise Ready (End of Week 16)

```text
✅ Advanced features fully implemented
✅ Performance: <2s median response time
✅ Security audit passed
✅ Comprehensive documentation
✅ Docker image optimized (<1.3GB)
✅ Production deployment checklist complete
```

---

## 📊 RISK ASSESSMENT & MITIGATION

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Firth implementation differs from R | HIGH | MEDIUM | Early validation (Weeks 1-2), fallback to statsmodels.GLM |
| Large dataset performance | MEDIUM | MEDIUM | Profiling & optimization, dask integration |
| Dependency version conflicts | MEDIUM | LOW | CI/CD testing, version pinning |
| Complex refactoring breaks UI | HIGH | LOW | Comprehensive test coverage, gradual rollout |

### Mitigation Actions

1. **Early validation** - Start Firth testing immediately
2. **Incremental deployment** - Feature branches, staged rollout
3. **Comprehensive testing** - Unit + integration + E2E tests
4. **Documentation** - Keep docs updated as code evolves
5. **Feedback loops** - Regular user testing, metrics monitoring

---

## 💡 KEY RECOMMENDATIONS

### Immediate Actions (This Week)

1. **Pull R benchmark script** into `tests/benchmarks/r_scripts/test_firth.R`
2. **Run R benchmarks** locally to generate CSV files
3. **Create Python tests** that load R benchmarks and validate
4. **Create GitHub issue** for each critical task with checklists
5. **Set up CI/CD** to run benchmarks on every commit

### Process Improvements

1. **Use issue checklists** for tracking task progress
2. **Daily standup notes** (5-min updates in dedicated channel)
3. **Weekly code reviews** for merged PRs
4. **Monthly demo sessions** with stakeholders
5. **Quarterly roadmap updates** based on user feedback

### Development Standards

- **Code style:** Black (Python), line length 100
- **Docstrings:** NumPy format (required)
- **Test coverage:** >80% target
- **Benchmark suite:** Automatic comparison against R
- **Documentation:** Every feature must be documented

---

## 📚 REFERENCE MATERIALS

### Statistical References

- Kosmidis & Firth (2021): Bias reduction & finite estimates
- Heinze & Schemper (2002): Firth for logistic regression
- Heinze & Schemper (2001): Firth for Cox regression
- Konis (2007): Separation detection via linear programming

### Technical References

- Python Shiny: <https://shiny.posit.co/py/>
- firthmodels: <https://github.com/jzluo/firthmodels>
- lifelines: <https://lifelines.readthedocs.io/>
- statsmodels: <https://www.statsmodels.org/>

### Publication Standards

- NEJM: <https://www.nejm.org/authors/manuscript-submission>
- Lancet: <https://www.thelancet.com/authors>
- BMJ: <https://www.bmj.com/about-bmj/resources-authors>
- JAMA: <https://jamanetwork.com/journals/jama>

---

## 📞 SUPPORT & CONTACT

**Repository:** [NTWKKM/stat-shiny](https://github.com/NTWKKM/stat-shiny)  
**Branch:** patch  
**Maintainer:** NTWKKM (Thailand)  
**Last Updated:** January 24, 2026

---

## APPENDIX A: Code Examples

### Example 1: Firth Regression Integration

```python
from firthmodels import FirthLogisticRegression
from utils.data_quality import detect_separation

def fit_best_logistic_model(X, y):
    """Auto-select standard vs Firth based on separation detection"""
    
    # Check for separation
    from firthmodels import detect_separation
    result = detect_separation(X, y)
    
    if result.separation:
        print("⚠️  Separation detected! Using Firth regression.")
        model = FirthLogisticRegression(backend='auto')
    else:
        print("✅ No separation. Standard logistic regression.")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    return model.fit(X, y)
```

### Example 2: Publication Table Generation

```python
def generate_publication_table(model_results):
    """Format regression output for journal submission"""
    
    table = pd.DataFrame({
        'Predictor': model_results['term'],
        'Estimate': model_results['estimate'].apply(lambda x: f"{x:.3f}"),
        'SE': model_results['se'].apply(lambda x: f"{x:.3f}"),
        '95% CI': [f"{l:.3f}–{u:.3f}" 
                   for l, u in zip(model_results['conf.low'], 
                                    model_results['conf.high'])],
        'P-value': [format_pvalue(p) for p in model_results['p.value']]
    })
    
    return table

def format_pvalue(p):
    """Format p-value with asterisks (***p<0.001, etc.)"""
    if p < 0.001:
        return "<0.001***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"
```

### Example 3: Benchmark Testing

```python
import pandas as pd
import numpy as np

def validate_against_r_benchmark(python_results, r_results_csv):
    """Compare Python estimates against R gold standard"""
    
    r_bench = pd.read_csv(r_results_csv)
    
    # Extract coefficients (skip intercept if needed)
    python_est = python_results['estimate'].values
    r_est = r_bench['estimate'].values
    
    # Calculate maximum absolute difference
    max_diff = np.max(np.abs(python_est - r_est))
    
    print(f"Max coefficient difference: {max_diff:.6f}")
    
    if max_diff < 1e-4:
        print("✅ VALIDATION PASSED: Excellent agreement with R")
    elif max_diff < 1e-3:
        print("⚠️  VALIDATION WARNING: Acceptable, minor differences noted")
    else:
        print("❌ VALIDATION FAILED: Significant difference from R")
        print("\nDetailed comparison:")
        print(pd.DataFrame({
            'Term': r_bench['term'],
            'R Estimate': r_bench['estimate'],
            'Python Estimate': python_results['estimate'],
            'Difference': python_est - r_est
        }))
    
    return max_diff < 1e-3
```

---

**END OF OPTIMIZATION PLAN**

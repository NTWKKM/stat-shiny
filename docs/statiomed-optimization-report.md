# 📊 Medical Stat Tool - Master Optimization & Development Plan

**Document Version:** 4.3 (7-Tab Standard Edition)
**Date:** January 24, 2026
**Status:** 🚀 Production-Hardening (Validation & Optimization Phase)
**Target:** Enterprise-Grade / Medical Publication Standard

---

## 📑 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture & Health](#2-system-architecture--health)
3. [Module Optimization Roadmap (Updated)](#3-module-optimization-roadmap-updated)
4. [Deep Dive: Key Technical Implementations](#4-deep-dive-key-technical-implementations)
5. [Quality Assurance & R-Validation](#5-quality-assurance--r-validation)
6. [Deployment & DevOps](#6-deployment--devops)

---

## 1. Executive Summary

The Medical Stat Tool (stat-shiny) has transitioned into a **High Stability Phase**. The current architecture successfully decouples **Statistical Logic** from the **User Interface (UI)**, strictly adhering to the **MVC (Model-View-Controller) Pattern**.

### 🎯 Strategic Focus (v4.3)

* **Logic Isolation:** Complex statistical logic (e.g., Logistic Regression in `utils/logic.py`) is completely isolated from UI files to enhance testability and maintainability.
* **7-Tab Navigation Standard:** The UI has been standardized into **7 core navigation tabs** to prioritize Table 1 access:
    1. **🏠 Home** (Dashboard & Quick Start)
    2. **📋 Table 1** (Baseline Characteristics & Matching)
    3. **📁 Data** (Management & Health)
    4. **📊 General** (Descriptive & Diagnostic)
    5. **🔬 Advanced** (Regression & Survival)
    6. **🏥 Clinical** (Sample Size & Causal)
    7. **⚙️ Settings** (Configuration)
* **Data Integrity:** The `utils/data_cleaning.py` module acts as the primary **Gatekeeper**, enforcing strict Missing Data handling and Type Casting rules.
* **HTML-First Export:** Every module is required to generate self-contained **Single-File HTML Reports** capable of embedding interactive Plotly graphs.

---

## 2. System Architecture & Health

### 2.1 Hybrid Architecture (Shiny + Pure Python)

The system has evolved from a Monolithic Shiny App into a fully **Modular Architecture**:

```mermaid
graph TD
    User[User / Client] --> App[app.py / Navbar Shell (7 Tabs)]
    App --> Tabs[UI Components (tabs/*)]
    Tabs --> Helper[UI Helpers (utils/ui_helpers.py)]
    Tabs --> DataPipe[Data Cleaning Pipeline (utils/data_cleaning.py)]
    Tabs --> StatEngine[Pure Python Logic (utils/logic.py, diag_test.py)]
    StatEngine --> Libs[Statsmodels / Scikit-learn / Lifelines]
    StatEngine --> Renderer[Plotly HTML Renderer (utils/plotly_html_renderer.py)]

```

### 2.2 Critical Components Status

| Component | File Source | Status | Improvement Needed |
| --- | --- | --- | --- |
| **Home Dashboard** | `tabs/tab_home.py` | 🟢 **Ready** | Landing page with project overview. |
| **Table 1 & Matching** | `tabs/tab_baseline_matching.py` | 🟢 **Promoted** | Now a top-level tab for instant access to Table 1 generation. |
| **Data Pipeline** | `utils/data_cleaning.py` | 🟢 **Excellent** | Vectorized cleaning & quality reports. |
| **Core Regression** | `utils/logic.py` | 🟢 **Good** | Logic isolated; supports Firth/Logit. |
| **Diagnostic UI** | `tabs/tab_diag.py` | 🟢 **Feature-Rich** | ROC, DCA, Chi-Square with HTML export. |
| **UI Structure** | `tabs/*` | 🟡 **Transitioning** | Refactoring navigation to match the **7-Tab Standard**. |

---

## 3. Module Optimization Roadmap (Updated)

### 🟢 PHASE 1: Architecture & Core Logic (Completed/Refining)

* **Objective:** Decouple Business Logic from UI and establish a robust Data Pipeline.
* **Achievements:**
* ✅ **Data Cleaning:** `utils/data_cleaning.py` now supports vectorized handling of Missing Values and Outliers.
* ✅ **Regression Logic:** `utils/logic.py` implements Logistic Regression via MVC, enabling calculation of OR/AOR and Interaction Terms.
* ✅ **Diagnostic Tool:** `tabs/tab_diag.py` successfully executes ROC/DCA analysis and exports standalone HTML reports.

### 🟡 PHASE 2: UI Standardization & Clinical Validation (Current Focus)

* **Objective:** Standardize the UI into the **7-Tab structure** and validate statistical accuracy against R.
* **Action Items:**

#### A. UI Refactoring (Big 7 Restructure)

* [ ] **Navigation Update:** Update `app.py` to display the 7 distinct tabs.
* [ ] **Promote Table 1:** Ensure `tab_baseline_matching.py` is directly accessible via the "Table 1" tab.
* [ ] **Consolidate Advanced:** Group `tab_core_regression.py` and `tab_survival.py` under the "🔬 Advanced" NavMenu.
* [ ] **Code Reduction:** Utilize `utils/ui_helpers.py` to eliminate redundant code in UI files.

#### B. Statistical Validation (Hardening)

* [ ] **Regression:** Implement Unit Tests to verify OR/CI values in `utils/logic.py` against R benchmarks (`glm`).
* [ ] **Survival:** Validate Assumption Checks (Schoenfeld residuals) within `tab_survival.py`.
* [ ] **Table 1:** Ensure decimal rounding standardization aligns with medical journal requirements (e.g., "Table 1" tab output).

### 🔴 PHASE 3: Advanced Features & Reporting (Next Steps)

* **Objective:** Enhance reporting capabilities and implement advanced analytical features.
* **Action Items:**

1. **Batch Report Generation:** Develop a "Generate All Reports" feature to aggregate analysis from multiple modules into a single HTML dossier.
2. **AI Integration:** Design Prompt Templates to feed statistical outputs into LLMs for automated clinical interpretation.
3. **Performance Optimization:** Implement Caching (`@functools.lru_cache` or Shiny caching) to support datasets exceeding 50k rows.

---

## 4. Deep Dive: Key Technical Implementations

### 4.1 The Statistical Engine (`utils/logic.py`)

The core calculation engine is now completely separated from the UI, supporting a "Pure Python" workflow that facilitates rigorous testing.

```python
# Production Code Structure Example
def run_binary_logit(y, X, method="default", ci_method="wald"):
    """
    Core function returning raw params, conf_int, and pvalues.
    Zero dependency on Shiny UI elements.
    """
    # 1. Validation (via validate_logit_data)
    # 2. Method Selection (Firth vs Logit)
    # 3. Model Fitting (statsmodels)
    # 4. Return Dictionary/TypedDict

```

### 4.2 Robust Data Cleaning (`utils/data_cleaning.py`)

A data cleaning system specifically architected for Medical Data:

* **Smart Numeric Conversion:** Automatically handles special characters common in clinical data (e.g., `"<5"`, `">100"`, `1,200`).
* **Missing Data Strategy:** Supports both `complete-case` analysis and user-defined `missing_codes` (e.g., -99, 999).
* **Audit Trail:** Every cleaning step is logged, allowing for the generation of impact reports (Data Loss analysis).

### 4.3 Embedded HTML Reports (`utils/plotly_html_renderer.py`)

Technique for embedding Plotly JS and CSS into a single file, ensuring reports are **Offline-Ready**:

* **CDN Injection:** Utilizes CDN links for Bootstrap/MathJax when online.
* **Base64 Encoding:** Embeds static images directly into the HTML to prevent broken links.
* **Responsive Design:** Fully optimized for viewing on iPads and Tablets.

---

## 5. Quality Assurance & R-Validation

### 5.1 E2E Testing Strategy (`tests/e2e/test_app_flow.py`)

Currently utilizing **Playwright** for User Flow validation:

* ✅ App Loading & Title Verification
* ✅ Tab Navigation (Verification of all **7 main categories**)
* ✅ File Upload Interaction
* ✅ Error Handling (Console Log Monitoring)

### 5.2 Statistical Unit Tests Needed

A comprehensive Test Suite is required to benchmark results against R:

```python
# Future Test Plan Example
def test_logistic_vs_r_results():
    # Load Benchmark Dataset (e.g., Titanic)
    py_res = run_binary_logit(y, X)
    r_res = load_r_benchmark("logistic_benchmark.csv")
    
    # Assert Coefficient match within tolerance 1e-5
    assert np.allclose(py_res['coef'], r_res['coef'], atol=1e-5)

```

---

## 6. Deployment & DevOps

### 6.1 Containerization

* **Docker:** Deployed using an optimized `Dockerfile` (Python 3.12-slim).
* **Environment Strategy:** Separation of `requirements.txt` (Development) and `requirements-prod.txt` (Production) to minimize Image size.

### 6.2 Maintenance Protocol

1. **CSS Sync:** Direct editing of `static/styles.css` is strictly prohibited. Changes must be made in `tabs/_styling.py`, followed by execution of `utils/update_css.py`.
2. **Repo Structure:** Maintain a clean root directory. No Python files other than `app.py` and `config.py` should exist at the root level.

---

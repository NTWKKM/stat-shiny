# 🏗️ System Architecture & AI Context

This document provides a deep technical overview of the **Medical Statistical Tool** for developers and AI agents (Antigravity/Bots).

---

## 📱 App Shell & Navigation

The application uses a fluid, multi-menu navbar designed for responsiveness and accessibility.

### Fluid UI & Layout

- **Container**: Built using `ui.page_fluid` for full-width flexibility across devices.
- **Navigation**: Utilizes nested `ui.page_navbar` and `ui.nav_menu` to organize 10+ statistical modules into logical groups (General Stats, Inference, etc.).
- **Accessibility**: Includes skip-links, ARIA landmarks, and standardized focus states.

### ⚡ Modular Structure

The application is built on a modular architecture where each tab is a self-contained component. While modules follow a common structure for consistency, they are fully initialized during the application startup to ensure immediate availability and a smooth user experience.

## 1. Overall Application Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Shiny UI
    participant Server as Shiny Server
    participant DataMgmt as Data Management
    participant Analysis as Core Analysis
    participant Viz as Visualization
    participant Report as Report Generator
    
    User->>UI: Upload data / Select variables
    UI->>Server: Trigger data load
    Server->>DataMgmt: Load & validate data
    DataMgmt-->>Server: Return validated data + metadata
    Server-->>UI: Update status & preview
    
    User->>UI: Configure analysis parameters
    UI->>Server: User selections
    Server->>Analysis: Prepare & run analysis
    Analysis->>Analysis: Execute statistical tests
    Analysis-->>Server: Return results
    
    Server->>Viz: Generate visualizations
    Viz-->>Server: Return plots (Plotly/Matplotlib)
    Server-->>UI: Display results + plots
    
    User->>UI: Request report download
    UI->>Server: Generate report
    Server->>Report: Compile HTML/PDF report
    Report-->>Server: Return report file
    Server-->>UI: Download report
    UI-->>User: Report downloaded
```

## 2. Advanced Data Cleaning Workflow

```mermaid
sequenceDiagram
    participant User
    participant UI as Data Tab UI
    participant Server as Data Server
    participant Cleaning as data_cleaning.py
    participant Quality as data_quality.py
    participant Viz as visualizations.py
    
    User->>UI: Navigate to Cleaning & Imputation
    UI->>Server: Load numeric columns
    Server->>Quality: Get missing summary
    Quality-->>Server: Return missing stats
    Server->>Viz: Generate missing pattern
    Viz-->>Server: Return heatmap + bar chart
    Server-->>UI: Display missing data viz
    
    User->>UI: Select imputation method (KNN/MICE)
    User->>UI: Click "Run Imputation"
    UI->>Server: Trigger imputation
    Server->>Cleaning: impute_missing_data(df, cols, method)
    Cleaning->>Cleaning: Validate inputs
    Cleaning->>Cleaning: Apply imputation (KNN/MICE/mean/median)
    Cleaning-->>Server: Return imputed DataFrame
    Server-->>UI: Show success notification
    
    User->>UI: Select transformation (log/sqrt/zscore)
    UI->>Server: Request transformation
    Server->>Cleaning: transform_variable(series, method)
    Cleaning->>Cleaning: Apply transformation
    Cleaning-->>Server: Return transformed series
    Server->>Cleaning: check_assumptions(series)
    Cleaning-->>Server: Return normality test results
    Server-->>UI: Display transformation preview + tests
    
    User->>UI: Configure outlier handling
    UI->>Server: Trigger outlier detection
    Server->>Cleaning: handle_outliers(df, cols, method, action)
    Cleaning->>Cleaning: Detect outliers (IQR/Z-score)
    Cleaning->>Cleaning: Apply action (winsorize/cap/remove)
    Cleaning-->>Server: Return cleaned DataFrame
    Server-->>UI: Show outlier handling results
```

## 2.5. Multiple Imputation Workflow (NEW)

```mermaid
sequenceDiagram
    participant User
    participant UI as MI Tab UI
    participant Server as Data Server
    participant MICE as MICEImputer
    participant Rubin as pool_estimates()
    participant Analysis as Regression Server
    
    User->>UI: Navigate to Multiple Imputation tab
    UI->>Server: Load columns with missing data
    Server-->>UI: Display missing summary
    
    User->>UI: Select columns & set m (imputations)
    User->>UI: Click "Run Multiple Imputation"
    UI->>Server: Trigger MI
    Server->>MICE: fit_transform(df, columns)
    MICE->>MICE: Run m iterations of chained equations
    MICE-->>Server: Return MICEResult (m datasets)
    Server-->>UI: Show success + diagnostics
    
    Note over User,Analysis: Later: Analysis Tab
    
    User->>Analysis: Run regression analysis
    Analysis->>Analysis: Check for MI datasets
    alt MI datasets available
        loop For each imputed dataset
            Analysis->>Analysis: Fit model
        end
        Analysis->>Rubin: Pool coefficients (Rubin's Rules)
        Rubin->>Rubin: Calculate pooled SE, FMI
        Rubin-->>Analysis: Return PooledResult
        Analysis-->>User: Display pooled estimates + FMI
    else No MI
        Analysis->>Analysis: Fit model on original data
        Analysis-->>User: Display standard results
    end
```

## 2.6. Shared MI State Architecture

```mermaid
flowchart TD
    A[app.py: mi_imputed_datasets] -->|shared| B[tab_data.py]
    A -->|shared| C[tab_core_regression.py]
    A -->|shared| D[tab_survival.py]
    A -->|shared| E[tab_advanced_inference.py]
    
    B -->|sets| A
    C -->|reads & pools| A
    D -->|reads & pools| A
    E -->|reads & pools| A
```

## 3. Logistic Regression Subgroup Analysis Workflow

```mermaid
sequenceDiagram
    participant User
    participant UI as Core Regression UI
    participant Server as Regression Server
    participant Analysis as SubgroupAnalysisLogit
    participant Forest as Forest Plot Generator
    participant Report as HTML Report
    
    User->>UI: Navigate to Subgroup Analysis tab
    UI->>Server: Load available variables
    Server-->>UI: Populate dropdowns
    
    User->>UI: Select outcome, treatment, subgroup
    User->>UI: Select adjustment variables (optional)
    User->>UI: Configure minimum counts
    User->>UI: Click "Run Subgroup Analysis"
    
    UI->>Server: Trigger _run_sg_logit()
    Server->>Server: Validate inputs (outcome, treatment, subgroup)
    Server->>Server: Check minimum counts per stratum
    
    Server->>Analysis: SubgroupAnalysisLogit.run()
    Analysis->>Analysis: Split data by subgroup levels
    loop For each subgroup level
        Analysis->>Analysis: Fit logistic regression
        Analysis->>Analysis: Extract OR, CI, p-value
    end
    Analysis->>Analysis: Perform interaction test (Likelihood Ratio Test)
    Analysis-->>Server: Return subgroup results + interaction p
    
    Server->>Forest: Generate forest plot
    Forest-->>Server: Return base64-encoded plot (annotated with P-interaction)

    Server-->>UI: Display forest plot + results table
    
    User->>UI: Click "Download Report"
    UI->>Server: Generate HTML report
    Server->>Report: Compile results + plot + interaction test
    Report-->>Server: Return HTML string
    Server-->>UI: Download HTML file
```

## 4. Report Generation & Download Workflow

```mermaid
sequenceDiagram
    participant User
    participant UI as Analysis Tab UI
    participant Server as Tab Server
    participant Results as Results Storage
    participant HTML as HTML Builder
    participant Download as Download Handler
    
    User->>UI: Complete analysis
    UI->>Server: Analysis request
    Server->>Server: Run statistical analysis
    Server->>Results: Store results (reactive.Value)
    Results-->>Server: Confirm storage
    Server-->>UI: Display results
    
    User->>UI: Click "Download Report"
    UI->>Server: Trigger download handler
    Server->>Results: Retrieve stored results
    Results-->>Server: Return results dict
    
    Server->>HTML: Build HTML structure
    HTML->>HTML: Add header & metadata
    HTML->>HTML: Add results tables
    HTML->>HTML: Embed visualizations (base64)
    HTML->>HTML: Add interpretation text
    HTML->>HTML: Add footer with timestamp
    HTML-->>Server: Return complete HTML string
    
    Server->>Download: Prepare download response
    Download->>Download: Set filename (with timestamp)
    Download->>Download: Set content-type: text/html
    Download-->>Server: Return download object
    Server-->>UI: Initiate browser download
    UI-->>User: File downloaded
```

## 5. Data Quality Check & Validation Workflow

```mermaid
sequenceDiagram
    participant User
    participant UI as Data Tab
    participant Server as Data Server
    participant Quality as data_quality.py
    participant Viz as Missing Pattern Viz
    participant Meta as Variable Metadata
    
    User->>UI: Upload CSV/Excel file
    UI->>Server: File upload event
    Server->>Server: Read file into DataFrame
    
    Server->>Quality: run_data_quality_checks(df, var_meta)
    Quality->>Quality: Check for missing data
    Quality->>Quality: Check for numeric in text columns
    Quality->>Quality: Check for rare categories
    Quality->>Quality: Check for strict NaN violations
    Quality-->>Server: Return error_cells dict + warnings
    
    Server->>Meta: Parse variable metadata
    Meta->>Meta: Detect variable types
    Meta->>Meta: Parse missing value codes
    Meta-->>Server: Return var_meta dict
    
    Server->>Viz: plot_missing_pattern(df)
    Viz->>Viz: Calculate missing percentages
    Viz->>Viz: Generate bar chart + heatmap
    Viz->>Viz: Apply Golden Ratio sampling
    Viz-->>Server: Return Plotly figure
    
    Server-->>UI: Display data preview with highlighted errors
    Server-->>UI: Show quality warnings accordion
    Server-->>UI: Display missing pattern visualization
    
    User->>UI: Review quality issues
    User->>UI: Navigate to Cleaning tab
    Note over User,Server: Proceed to cleaning workflow
```

---

## 🎨 UI Styling System

The application uses a centralized styling system to ensure visual consistency across all modules.

### Core Components & Sync Workflow

| File | Role | Description |
| :--- | :--- | :--- |
| `tabs/_common.py` | **Source of Truth** | Defines the central `get_color_palette()` and common UI constants. |
| `tabs/_styling.py` | **CSS Generator** | Main injector that generates CSS using the palette from `_common.py`. |
| `static/styles.css` | **Compiled CSS** | Auto-generated output. **DO NOT EDIT DIRECTLY.** |
| `utils/update_css.py` | **Sync Utility** | Script to update `static/styles.css` whenever `_styling.py` changes. |

> [!TIP]
**To update styles**: Edit `tabs/_styling.py` then run:

- Unix/Linux/macOS: `.venv/bin/python utils/update_css.py`
- Windows: `.venv\Scripts\python.exe utils/update_css.py`

### Visual Consistency (Plotly & HTML)

- `utils/plotly_html_renderer.py`: Standardizes Plotly figure rendering (Inter font, theme-sync).
- `utils/formatting.py`: Handles P-value styling, logic-driven badges, and statistical report HTML structure (syncing with `config.py`).
  - **PublicationFormatter**: Provides per-journal templates (**NEJM, JAMA, Lancet, BMJ**) and automated "Methods" text generation.
  - **MissingDataStatement**: Generates standardized reporting on missingness handling.
- `utils/table_one_advanced.py`: **Advanced Table 1 Generator** (OOP). Handles intelligent variable classification and **Odds Ratio (OR) calculation**.
  - **Categorical OR**: Uses **2x2 Contingency Tables** comparing each level (or Target) against the **First Level (Reference)**. Applies **Haldane-Anscombe correction** (+0.5) for zero cells. Supports "All Levels" and "Simple" (Binary Target vs Ref) styles.
  - **Continuous OR**: Uses **Univariate Logistic Regression**.
- `utils/diagnostic_advanced_lib.py`: **Advanced Diagnostic Engine** (OOP) providing robust ROC analysis, **DeLong's Test**, and Wilson Score confidence intervals.
- `utils/agreement_lib.py`: **Agreement Analysis Engine** providing Cohen's/Fleiss' Kappa, advanced Bland-Altman (CI bands), and ICC.
- `utils/psm_lib.py`: **Propensity Score Engine**.
  - `PropensityScoreDiagnostics`: Handles **Common Support** assessment (distribution overlap stats) and **Love Plot** generation.
  - Implements **Inverse Probability Weighting (IPW)** with optional **Weight Truncation** (1st/99th percentiles).
- `utils/multiple_imputation.py`: **Multiple Imputation Engine** (MICE with Rubin's Rules).
  - `MICEImputer`: Generates m imputed datasets using `IterativeImputer` with `sample_posterior=True`.
  - `pool_estimates()`: Implements **Rubin's Rules** for pooling point estimates and standard errors.
  - **Auto-Pooling**: `tab_core_regression.py` automatically detects MI datasets and pools logistic regression results, reporting **Fraction of Missing Information (FMI)**.
  - Diagnostic plots: Density comparisons and imputation traces.
- `utils/effect_sizes.py`: **Effect Size Engine**. Calculates **Cohen's d, Hedges' g, η², ω²** with interpretation badges.
- `utils/sensitivity_lib.py`: **Sensitivity Analysis**. Implements **Bootstrap CI**, **Jackknife**, and **LOO-CV**.
- `utils/statistical_assumptions.py`: **Assumption Testing**. Centralized normality and homogeneity variance tests.

### Dynamic UI Enhancements (Animations)

- **Fade-in Entry**: All statistical results utilize a smooth fade-in animation to improve user experience.
- **Logic**: Defined in `tabs/_styling.py` as the `.fade-in-entry` class, providing a 0.4s ease-out entry with a subtle upward translate.
- **Standardized Wrapper**: The `create_results_container` utility in `utils/ui_helpers.py` has been updated to support these animations via the `class_` parameter.

---

## 📊 Analysis Modules

The application covers a wide range of medical statistical needs, organized into functional clusters:

| Category | Modules | Key Features |
| :--- | :--- | :--- |
| **Standard** | `tab_corr`, `tab_diag`, `tab_agreement` | Multi-method Correlation (**Kendall/Spearman/Pearson**), **ROC/AUC** (Youden/F1/Calibration), **Paired DeLong Test**, **Sens/Spec vs Threshold**, **Agreement** (Cohen's/Fleiss' Kappa, Bland-Altman with CI bands, ICC with interpretation). |
| **Inference** | `tab_core_regression`, `tab_advanced_inference` | Linear/Logistic/Cox Regressions (**Firth/Deep Diagnostics**), **Subgroup analysis** (Logistic/Cox), Forest Plots. |
| **Causal** | `tab_causal_inference`, `tab_baseline_matching` | EconML Integration, Propensity Score Matching (PSM), Covariate Balance (Love Plots: Green <0.1, Yellow 0.1–0.2 (Red: >0.2, not rendered)), Common Support Visualization. |
| **Specialized** | `tab_survival`, `tab_advanced_stats`, `tab_sample_size` | Kaplan-Meier, **Extended Diagnostics** (Schoenfeld/Martingale/Deviance), Time-Varying Cox (with Interaction Check), G-Computation, Power Analysis. |

---

## 🔄 Data Processing & Statistical Pipeline

The data flow is standardized to ensure consistent handling of missing values and variable metadata.

### 1. Ingestion & Quality Check (`tab_data.py` & `utils/data_quality.py`)

- Users upload files (CSV/Excel) or load example data.
- **Immediate Data Health Report**: Uses `check_data_quality()` and the `DataQualityReport` class to perform deep validation:
  - **Numeric Validation**: Detects non-standard values like `"<5"`, `"10%"`, or symbols (`<`, `>`, `,`, `%`, `$`, `€`, `£`), among others.
  - **Categorical Validation**: Identifies numbers accidentally placed in categorical columns and flags rare categories.
  - **Data Quality Scorecard**: A structured report with 4 dimension scores (Completeness, Consistency, Uniqueness, Validity) displayed via Value Boxes.
  - **Detailed Diagnostic Log**: Provides a **complete list** of row indices for issues, serving as the full "Error Log".
- **Configuration**: Individual variable type casting and missing value strategy selection based on the health report.

### 2. **Configuration & Cleaning**

- **Interactive Setup**: Users interactively cast variable types and choose missing value strategies.
- **Advanced Cleaning**: Users can apply Imputation (KNN/MICE), handle Outliers (Winsorize/Cap), and Transform variables (Log/Sqrt) directly within the UI.
- **UI Standardization**: Recent updates have unified the "Variable Selection" UI across modules (e.g., Table 1, Survival Analysis, Regression), utilizing full-width Selectize inputs with "remove button" plugins for better usability. Action buttons are also standardized to full-width (`w-100`) for consistent click targets.

### 3. Multiple Imputation (`utils/multiple_imputation.py`) [NEW]

- **Purpose**: Handles missing data properly using **Multiple Imputation by Chained Equations (MICE)** with **Rubin's Rules** for pooling.
- **Auto-Integration**: When MI is enabled in the Data Management tab, imputed datasets are automatically used in regression analyses.
- **Key Components**:
  - **MICEImputer**: Generates m imputed datasets (default m=5).
  - **Rubin's Rules**: Combines within-imputation and between-imputation variance for valid inference.
  - **Fraction of Missing Information (FMI)**: Reported to indicate missingness impact on each estimate.

### 4. Central Preparation (`utils/data_cleaning.py`)

- **Before Analysis**: Data is passed through `prepare_data_for_analysis()` which handles exclusion logic and logging.
- **Outlier Logic**: `handle_outliers()` supports 'iqr' and 'zscore' detection with 'remove', 'winsorize', or 'cap' actions.

### 5. Integrated Reporting (`utils/formatting.py`)

- **Missing Data Statistics**: Automatically analyzed and included in the final report.
  - **Smart Visualization Sync**: The report is architected to be the "detailed companion" to the visualization, covering "blind spots" if the heatmap is subsampled.
  - **Diagnostic Metrics**: Multi-metric reports follow a "Table 2" publication-grade layout (Metric, Value, 95% CI, Interpretation).
  - **Logistic Regression**: Deep diagnostics (AUC/C-stat, Hosmer-Lemeshow, AIC/BIC)
  - **Subgroup Analysis**: Formal Interaction Tests (Likelihood Ratio Test) and Forest Plots annotated with P-values.
  - **Evidence-Based Badges**: Logic-driven badges (Landis-Koch, Cicchetti, EBM standards for LR) and **STROBE/TRIPOD alignment text** provide immediate clinical and reporting context.
  - **Outcome**: A standardized "Missing Data Summary" and localized interpretation guides are automatically included.

---

## 🧪 Testing & Quality Assurance

### Continuous Integration (CI)

- `.github/workflows/quality_check.yml`: Runs the full suite of Unit, Integration, and E2E (Playwright) tests.
- `.github/workflows/ui-styling.yml`: Verifies CSS/JS/Python styling consistency.
- `.github/workflows/docker-publish.yml`: Automates production image builds.

### Test Suites

- **Unit**: Core utilities, statistics math, color palette.
- **Integration**: PSM pipelines, data cleaning flow.
- **E2E**: Multi-tab navigation, smart variable defaults, loading states.

---

> [!NOTE]
> **AI AGENT GUIDANCE**: When adding new features:
>
> 1. Verify UI components use `get_color_palette()`.
> 2. Ensure all data-heavy modules call `prepare_data_for_analysis()`.
> 3. Add tests to the corresponding suite in `tests/`.

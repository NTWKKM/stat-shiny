---
title: shinystat
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

<!-- AI-AGENT-GUIDE-START -->
> [!IMPORTANT]
> **🤖 FOR AI AGENTS & BOTS (Antigravity, Cursor, etc.)**:
>
> 1. **Architecture First**: ALWAYS Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand the **Styling System** and **Data Processing Pipeline** before editing.
> 2. **Environment**: This project requires **Python 3.12+**. Versions < 3.12 will cause SyntaxErrors due to the use of **PEP 695** (Type Parameter Syntax/Type Aliases).
> 3. **Context**: See [docs/deep-ai-bot-optimization.md](docs/deep-ai-bot-optimization.md) for detailed implementation rules.
>
> | Requirement | Version    |
> | ----------- | ---------- |
> | Python      | **≥ 3.12** |
> | Environment | `.venv`    |
<!-- AI-AGENT-GUIDE-END -->

**Available StatioMed**:  
👉 [HuggingFace Space](https://ntwkkm-shinystat.hf.space)  
👉 [Posit Cloud](https://ntwkkm-shinymedstat.share.connect.posit.cloud/)  

[--- REPOSITORY-TREE-START ---]

📂 Repository Contents (File Structure)

This content reflects the repository structure (updated by GitHub Actions):

```text
|-- AI_CONTEXT.md
|-- ARCHITECTURE.md
|-- DESIGN.md
|-- Dockerfile
|-- app.py
|-- asgi.py
|-- config.py
|-- docker-compose.yml
|-- docs /
|   |-- .DS_Store
|   `-- assets /
|       |-- .DS_Store
|       `-- navigation_sequence.png
|-- logger.py
|-- pyproject.toml
|-- pytest.ini
|-- python_results /
|   |-- benchmark_firth_cox.csv
|   |-- benchmark_firth_logistic.csv
|   |-- dataset_breast.csv
|   `-- dataset_sex2.csv
|-- requirements-causal.txt
|-- requirements-prod.txt
|-- requirements.txt
|-- static /
|   |-- js /
|   |   |-- custom_handlers.js
|   |   `-- interactions.js
|   |-- styles.css
|   `-- styles.min.css
|-- tabs /
|   |-- __init__.py
|   |-- _common.py
|   |-- _dataset_mixin.py
|   |-- _styling.py
|   |-- _tvc_components.py
|   |-- tab_advanced_inference.py
|   |-- tab_advanced_stats.py
|   |-- tab_agreement.py
|   |-- tab_baseline_matching.py
|   |-- tab_causal_inference.py
|   |-- tab_core_regression.py
|   |-- tab_corr.py
|   |-- tab_data.py
|   |-- tab_diag.py
|   |-- tab_home.py
|   |-- tab_meta_analysis.py
|   |-- tab_sample_size.py
|   |-- tab_settings.py
|   `-- tab_survival.py
|-- tests /
|   |-- benchmarks /
|   |   |-- python_results /
|   |   |   |-- benchmark_firth_cox.csv
|   |   |   |-- benchmark_firth_logistic.csv
|   |   |   |-- dataset_breast.csv
|   |   |   `-- dataset_sex2.csv
|   |   `-- r_scripts /
|   |       `-- test_firth.R
|   |-- conftest.py
|   |-- e2e /
|   |   |-- test_app_flow.py
|   |   |-- test_load_example_data.py
|   |   `-- test_smart_defaults.py
|   |-- integration /
|   |   |-- test_advanced_features.py
|   |   |-- test_corr_cleaning.py
|   |   |-- test_corr_pipeline.py
|   |   |-- test_data_cleaning_pipeline.py
|   |   |-- test_diag_cleaning.py
|   |   |-- test_diag_pipeline.py
|   |   |-- test_forest_plot_lib.py
|   |   |-- test_interaction_pipeline.py
|   |   |-- test_logic_pipeline.py
|   |   |-- test_poisson_cleaning.py
|   |   |-- test_poisson_pipeline.py
|   |   |-- test_psm_pipeline.py
|   |   |-- test_robustness_check.py
|   |   |-- test_subgroup_pipeline.py
|   |   |-- test_survival_cleaning.py
|   |   |-- test_survival_landmark.py
|   |   |-- test_survival_pipeline.py
|   |   `-- test_table_one_pipeline.py
|   `-- unit /
|       |-- output /
|       |   `-- test_output_2.html
|       |-- test_advanced_stats.py
|       |-- test_bland_altman.py
|       |-- test_calibration_ici.py
|       |-- test_causal.py
|       |-- test_chi_html.py
|       |-- test_collinearity.py
|       |-- test_color_palette.py
|       |-- test_correlation_returns.py
|       |-- test_data_cleaning.py
|       |-- test_data_cleaning_advanced.py
|       |-- test_data_cleaning_workflow.py
|       |-- test_data_quality.py
|       |-- test_data_quality_report.py
|       |-- test_dca.py
|       |-- test_diag_returns.py
|       |-- test_diagnostic_advanced.py
|       |-- test_download_helpers.py
|       |-- test_effect_sizes.py
|       |-- test_fagan_nomogram.py
|       |-- test_firth_regression.py
|       |-- test_formatting.py
|       |-- test_formatting_styles.py
|       |-- test_glm.py
|       |-- test_heterogeneity.py
|       |-- test_linear_lib.py
|       |-- test_mediation.py
|       |-- test_medical_edge_cases.py
|       |-- test_meta_analysis.py
|       |-- test_mi_reporting.py
|       |-- test_missing_data.py
|       |-- test_model_diagnostics.py
|       |-- test_model_diagnostics_plots.py
|       |-- test_multiple_imputation.py
|       |-- test_pdf_helpers.py
|       |-- test_phase3_features.py
|       |-- test_plotly_html_rendering.py
|       |-- test_poisson_lib.py
|       |-- test_publication_renderer.py
|       |-- test_regression_publication.py
|       |-- test_repeated_measures.py
|       |-- test_reporting_checklists.py
|       |-- test_sample_size.py
|       |-- test_sensitivity.py
|       |-- test_sensitivity_fixes.py
|       |-- test_state_machine.py
|       |-- test_statistical_assumptions.py
|       |-- test_statistics.py
|       |-- test_survival_assumptions.py
|       |-- test_survival_lib_patch.py
|       |-- test_tab_diag_html_logic.py
|       |-- test_tabs_common.py
|       |-- test_tvc_lib.py
|       |-- test_ui_ux_styles.py
|       `-- verify_table_one.py
|-- utils /
|   |-- .DS_Store
|   |-- __init__.py
|   |-- advanced_stats_lib.py
|   |-- agreement_lib.py
|   |-- calibration_lib.py
|   |-- collinearity_lib.py
|   |-- correlation.py
|   |-- data_cleaning.py
|   |-- data_quality.py
|   |-- decision_curve_lib.py
|   |-- diag_test.py
|   |-- diagnostic_advanced_lib.py
|   |-- download_helpers.py
|   |-- effect_sizes.py
|   |-- fagan_nomogram_lib.py
|   |-- forest_plot_lib.py
|   |-- formatting.py
|   |-- heterogeneity_lib.py
|   |-- interaction_lib.py
|   |-- linear_lib.py
|   |-- logic.py
|   |-- mediation_lib.py
|   |-- meta_analysis_lib.py
|   |-- mi_helpers.py
|   |-- model_diagnostics_lib.py
|   |-- multiple_imputation.py
|   |-- pdf_helpers.py
|   |-- plotly_html_renderer.py
|   |-- poisson_lib.py
|   |-- psm_lib.py
|   |-- publication_renderer.py
|   |-- rcs_lib.py
|   |-- repeated_measures_lib.py
|   |-- reporting_checklists.py
|   |-- sample_size_lib.py
|   |-- sensitivity_lib.py
|   |-- state_machine.py
|   |-- statistical_assumptions.py
|   |-- stratified_lib.py
|   |-- subgroup_analysis_module.py
|   |-- survival_lib.py
|   |-- table_one.py
|   |-- table_one_advanced.py
|   |-- tvc_lib.py
|   |-- ui_helpers.py
|   |-- update_css.py
|   `-- visualizations.py
`-- uv.lock
```


[--- REPOSITORY-TREE-END ---]

## 🏥 Medical Statistical Tool (Shiny for Python)

A comprehensive, interactive web application for medical statistical analysis, built with [Shiny for Python](https://shiny.posit.co/py/). This tool simplifies the process of data management, cohort matching, and advanced statistical modeling for medical researchers.

## 🚀 Key Features

This application is a complete statistical workbench organized into modular tabs:

### 📁 Data Management

- **Comprehensive Data Control**: Upload CSV/Excel or load example datasets.
- **Data Health Report**: Automated deep checks via `utils/data_quality.py` for:
  - **Quality Scorecard**: Instant rating of Completeness, Consistency, Uniqueness, and Validity.
  - **Missing Data**: Detailed reporting of missing values with row positions.
  - **Non-standard Numeric**: Smart detection of medical strings like `"<5"`, `">10"`, or currency.
  - **Categorical Integrity**: Identifies numeric values in categorical text and flags rare categories (< 5 occurrences).
- **Variable Configuration**: Interactive type casting and missing value handling.
- **Advanced Cleaning**:
  - **Imputation**: Support for Mean, Median, KNN, and MICE strategies.
  - **Multiple Imputation (NEW)**: Full MICE workflow with **Auto-Pooled Regression** (Rubin's Rules, FMI reporting).
  - **Transformation**: Log, Sqrt, and Z-Score standardization with normality assumption checks (Shapiro-Wilk/K-S).
  - **Outlier Handling**: Detection (IQR/Z-Score) and treatment (Winsorize, Cap, Remove).

### 📋 Baseline & Matching

- **Table 1 Generation**: Publication-ready baseline tables with **Intelligent Variable Classification**, automated statistical testing, and **Odds Ratios with 95% CI** (Configurable: "All Levels" vs Reference or "Simple" Target vs Ref).
- **Propensity Score Matching (PSM)**: Advanced matching with customizable calipers and variable selection.
- **Balance Diagnostics**: Enhanced **Love Plots** with Green/Yellow zones (<0.1/<0.2 SMD) for assessing covariate balance.
- **Common Support**: Propensity Score overlap assessment with **automated distribution plots** and exclusion recommendations.
- **Weight Truncation**: Option to handling extreme weights (1%/99% trimming) for stable IPW estimates.
- **Matched Data Export**: seamless integration of matched cohorts into other analyses.

### 🔢 Sample Size & Power

- **Calculators**: Power and sample size estimation for:
  - **Means** (T-test)
  - **Proportions** (Chi-Square)
  - **Survival** (Log-Rank based on HR or Median)
  - **Correlation** (Pearson)

### 📈 Core Regression Models

- **GLM Framework**:
  - **Logistic Regression**: Standard MLE, BFGS, **Firth's Penalized Logistic** (for rare events and separation via `firthmodels 0.8.2`), and **Subgroup Analysis** (Forest Plots with **Interaction P-value** annotation and **ICEMAN credibility**).
  - **Firth Enhancements (`0.8.2`)**: Penalized Likelihood Ratio Tests (LRT) for P-values, Profile Likelihood Confidence Intervals (PL-CI), Konis LP separation detection with collinearity differentiation, and **explicit UI/report annotations whenever Wald P-value or Wald CI fallback is triggered**.
  - **Count Models**: Poisson and Negative Binomial regression (handling overdispersion) with offset support.
  - **Linear Regression**: OLS with robust standard error options (HC0, HC1, HC2, HC3 / White's).
- **Multiple Testing Correction (MCC)**: Family-wise error and false discovery rate control via Bonferroni, Benjamini-Hochberg (FDR), Holm, Hochberg, Hommel, and BY.
- **NEJM/Lancet Publication Standards**:
  - **Model Diagnostics**: C-statistic / AUC with DeLong 95% CI, Brier Score, Calibration Slope, Hosmer-Lemeshow goodness-of-fit test, Calibration plots with LOWESS smoothing.
  - **Absolute Measures**: ARD (Absolute Risk Difference) with Newcombe CI, NNT/NNH with Altman method.
  - **Sensitivity Analysis**: E-value for unmeasured confounding assessment (point estimate and CI limit).
  - **Reporting Alignment**: STROBE checklist auto-population and TRIPOD guideline alignment.
- **Effect Sizes**: Cohen's d, Hedges' g, **η² (Eta-squared)**, **ω² (Omega-squared)** with evidence-based interpretation badges.
- **MI Auto-Pooling**: When Multiple Imputation (MICE) is active, regression automatically pools results using **Rubin's Rules** with **FMI (Fraction Missing Information)** reporting across Logistic, Linear, Cox PH, and Mediation models.

### ⏳ Survival Analysis

- **Visualizations**: Interactive Kaplan-Meier curves (with Number-at-Risk tables, median survival times, and log-rank tests) and Nelson-Aalen cumulative hazard plots.
- **Cox Modeling**: Cox Proportional Hazards regression with forest plots and **Firth's Penalized Cox PH (`firthmodels 0.8.2`)** for rare events or small samples:
  - **Penalized LRT P-values**: Penalized Likelihood Ratio Tests for superior inference in rare events (with automated Wald fallback and clear UI table annotations).
  - **Profile Likelihood CIs**: 95% PL-CIs for exact confidence bounds.
  - **Baseline Hazard & Survival**: Breslow-type cumulative hazard and survival predictions.
- **Advanced Techniques**:
  - **Time-Varying Cox (TVC)**: Episodic data splitting to handle covariates that change over time, including time-interaction ($t \times X$) checks.
  - **Landmark Analysis**: Pre-specified landmark cutoff to mitigate immortal time bias.
  - **Subgroup Analysis**: Treatment effect heterogeneity with **Formal Interaction Tests (LRT)** and annotated Forest Plots.
- **Model Diagnostics**:
  - **Assumption Checks**: Automated Schoenfeld residuals with scatter/trend plots and actionable clinical remedies.
  - **Extended Plots**: Martingale residuals (non-linearity assessment) and Deviance residuals (outlier detection).

### 🌐 Meta-Analysis

- **Pooling Models**: Fixed-Effect (Inverse Variance, Mantel-Haenszel) and Random-Effects (DerSimonian-Laird).
- **Effect Size Metrics**: Odds Ratio (OR), Risk Ratio (RR), Risk Difference (RD), Standardized Mean Difference (SMD: Hedges' g, Cohen's d), and Mean Difference (MD).
- **Forest Plots**: Publication-grade interactive Forest Plots showing study weights, point estimates, confidence intervals, and pooled summary diamonds.
- **Heterogeneity Assessment**: Cochran's Q test, $I^2$ index, and $\tau^2$ (between-study variance) with Higgins clinical interpretation standards.
- **Publication Bias Diagnostics**: Funnel plots with pseudo 95% confidence limits, **Egger's linear regression test**, and **Begg-Mazumdar rank correlation test**.
- **Subgroup Meta-Analysis**: Stratified study pooling with between-subgroup heterogeneity test ($Q_{\text{between}}$).

### 🧬 Advanced Statistical Modeling & Repeated Measures

- **G-Computation (Parametric G-Formula)**: Standardized marginal effect estimation for causal inference under non-linear settings.
- **Generalized Estimating Equations (GEE)**: Population-averaged models for clustered longitudinal data with customizable working correlation structures (Exchangeable, AR(1), Independent).
- **Linear Mixed Models (LMM / MixedLM)**: Random intercept and random slope models for hierarchical/repeated measures data.

### 🎯 Causal Inference

- **Propensity Methods**: Inverse Probability Weighting (IPW) with stabilized weights and optional 1st/99th percentile weight truncation, and Propensity Score Matching (PSM).
- **Stratified Analysis**: Mantel-Haenszel odds ratios and Breslow-Day homogeneity tests.
- **Sensitivity Analysis**: **E-Value** calculation for unmeasured confounding.
- **Subgroup Credibility**: **ICEMAN framework** (8 core clinical criteria) for assessing heterogeneity claims with Bonferroni adjustment.
- **Diagnostics**: Comprehensive **Propensity Score Diagnostics** including overlapping density plots (Common Support) and summary statistics.

### 🧪 Diagnostic Tests & Agreement

- **Diagnostic Accuracy**: Sensitivity, Specificity, PPV, NPV, Positive Likelihood Ratio (LR+), Negative Likelihood Ratio (LR-), Diagnostic Odds Ratio (DOR), and Overall Accuracy with Wilson score CIs.
- **ROC / AUC Analysis**: Area Under Curve evaluation with paired **DeLong's Test** for comparing diagnostic models.
- **Optimal Cutpoints**: Automated threshold identification via Youden's Index ($J$), F1-Score, and Sensitivity/Specificity vs Threshold curves.
- **Fagan's Nomogram**: Interactive Bayesian pre-test to post-test probability mapping for positive and negative findings.
- **Decision Curve Analysis (DCA)**: Clinical Net Benefit curves across threshold probabilities compared to treat-all and treat-none strategies.
- **Agreement Statistics**:
  - **Cohen's Kappa**: Unweighted and quadratic weighted with Landis-Koch interpretation badges.
  - **Fleiss' Kappa**: Inter-rater reliability for multi-rater setups (> 2 raters).
  - **Bland-Altman**: Mean difference and Limits of Agreement (LoA) with 95% confidence bands and proportional bias test.
  - **Intraclass Correlation (ICC)**: Pingouin-powered ICC1, ICC2, ICC3 (single and average measures) with Cicchetti interpretation badges.
- **Contingency Analysis**: Chi-Square test of independence, Fisher's Exact Test, Risk Ratios, and Odds Ratios.

### 🧩 Advanced Inference

- **Mediation Analysis**: Counterfactual and Baron-Kenny causal mediation decomposition into Average Direct Effect (ADE), Average Causal Mediation Effect (ACME), Total Effect, and Proportion Mediated (with Quasi-Bayesian Monte Carlo / Bootstrap CIs).
- **Model Diagnostics**: Residual vs Fitted plots, Q-Q plots, Cook's distance for influential observations, and Breusch-Pagan heteroscedasticity tests.
- **Multicollinearity**: Variance Inflation Factor (VIF) analysis with clinical alert levels.
- **Model Robustness Validation**: **Bootstrap CI**, **Jackknife resampling**, and **Leave-One-Out Cross-Validation (LOO-CV)**.

### 🔗 Correlation & Reliability

- **Correlation Matrix**: Multi-method pairwise correlation (**Pearson, Spearman, Kendall's Tau**) with interactive heatmaps.
- **Scatter Matrix & Regression**: Bivariate scatter plots with linear trendlines and residual distributions.

### 📄 Publication Reports & Download Safety

- **Dual-Format Publication Export**:
  - **HTML Reports**: Self-contained, styled publication reports with embedded CSS and Plotly charts.
  - **PDF Reports**: Publication-grade PDF compilation via Playwright headless Chromium (`safe_download_pdf`).
- **Download Safety Layer (`download_helpers.py` & `pdf_helpers.py`)**: Guaranteed valid document output or styled error page, with user notifications (✅ Success, ⚠️ Incomplete, ❌ Error).
- **Reporting Styles**: Journal-specific presets (**NEJM, JAMA, Lancet, BMJ**).
- **Automated Checklists**: **STROBE** (Observational) and **CONSORT** (RCTs) checklist auto-fill and export.
- **Auto-Methods**: Automated generation of standardized "Methods" and "Missing Data" statements.
- **Figure Legends**: Automated publication-ready figure legends.
- **Theme & Design System**: Minimal Slate-Monochrome design token system (`_common.py`, `_styling.py`, `DESIGN.md`), 100% WCAG AA contrast compliant.

## 🏗️ System Architecture

The application is built with a modular architecture that separates styling, data processing, and statistical logic to ensure scalability and consistency.

### 🎨 UI Styling System

The application uses a centralized styling system to ensure visual consistency across all modules.

| File | Role | Description |
| :--- | :--- | :--- |
| `tabs/_common.py` | **Source of Truth** | Defines the central `get_color_palette()` and common UI constants. |
| `tabs/_styling.py` | **CSS Generator** | Main injector that generates CSS using the palette from `_common.py`. |
| `static/styles.css` | **Compiled CSS** | Auto-generated output from `_styling.py`. Used for performance and deployment. |
| `utils/update_css.py` | **Sync Utility** | Script to update `static/styles.css` whenever `_styling.py` changes. |
| `static/js/custom_handlers.js` | **JS Hooks** | Shiny custom message handlers for dynamic client-side styling. |

**Visual Consistency & Formatting:**

- **Plotly Integration**: `utils/plotly_html_renderer.py` and `forest_plot_lib.py` sync interactive charts with the central palette and "Inter" typography.
- **Reporting Labels**: `utils/formatting.py` standardizes P-value styling and badge generation across all statistical outputs.

### 🔄 Data Processing & Statistical Pipeline

Every statistical analysis follows a rigorous, standardized data flow to ensure reliable results:

1. **Ingestion & Quality Check (`tab_data.py`)**: Immediate identification of missingness and data types upon upload or example loading.
2. **Configuration & Cleaning**:
   - **Interactive Setup**: Users interactively cast variable types and choose missing value strategies.
   - **Advanced Cleaning**: Users can apply Imputation (KNN/MICE), handle Outliers (Winsorize/Cap), and Transform variables (Log/Sqrt) directly within the UI.
3. **Central Preparation (`utils/data_cleaning.py`)**: Before analysis, data is passed through `prepare_data_for_analysis()` which handles exclusion logic and logging.
4. **Download Safety (`utils/download_helpers.py` & `utils/pdf_helpers.py`)**: All `@render.download` handlers use `safe_download_html()` / `safe_report_generation()` for HTML, or `safe_download_pdf()` / `safe_pdf_report_generation()` for PDF output. Both layers guarantee valid output (or a styled error page) and show user-facing notifications (✅ success, ⚠️ no results, ❌ generation error). PDF conversion uses Playwright (requires `playwright install chromium`).
5. **Integrated Reporting (`utils/formatting.py`)**: Missing data statistics are automatically analyzed and included in the final report for every module.

### 🧪 Testing & Quality Assurance

- **Automated CI**: [ui-styling.yml](.github/workflows/ui-styling.yml) runs on every push to verify:
  - **Palette Integrity**: Colors in `_common.py` match branding.
  - **System Sync**: Cross-file consistency between Python, CSS, JS, and Plotly layers via `tests/unit/test_ui_ux_styles.py`.

## 🛠️ Installation & Usage

### Option 1: Run Locally (Python)

Ensure you have **Python 3.12+** installed (required for **PEP 695** type parameter syntax).

1. **Clone the repository:**

   ```bash
   git clone https://huggingface.co/spaces/ntwkkm/shinystat
   cd shinystat
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   shiny run app.py --port 7860
   ```

   The app will be available at `http://localhost:7860`.

### Option 2: Running Tests

To run the test suite, ensure you use the `pytest` from your virtual environment (assuming venv is named `.venv`):

```bash
# Run all tests
.venv/bin/pytest
# Or if using a different venv name/activation:
# python -m pytest

# Run specific test
.venv/bin/pytest tests/unit/test_statistics.py
```

### Option 3: Run with Docker

This project is containerized for easy deployment and local development. An automated image is published to **Docker Hub** on every update to the `main` branch.

1. **Pull and run from Docker Hub (Easiest):**

   ```bash
   docker run -p 7860:7860 ntwkkm/statiomed:latest
   ```

2. **Local Development with Docker Compose (Live Reload):**
   This method mounts your local code into the container, allowing for real-time updates as you edit files.

   ```bash
   docker compose up --build
   ```

3. **Standard Local Build:**

   ```bash
   # Build the image
   docker build -t medical-stat-tool .
   # Run the container
   docker run -p 7860:7860 medical-stat-tool
   ```

Access the app at `http://localhost:7860`.

### Option 4: VS Code Dev Containers

If you use VS Code, you can open the project in a pre-configured [Dev Container](https://code.visualstudio.com/docs/devcontainers/containers):

1. Ensure the **Dev Containers** extension is installed in VS Code.
2. Select **"Reopen in Container"** when prompted, or use the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and search for `Dev Containers: Reopen in Container`.
3. The environment will be automatically set up with all dependencies, Python 3.12, and the recommended extensions.

## 💻 Tech Stack

- **Framework**: [Shiny for Python](https://shiny.posit.co/py/)
- **Data Processing**: Pandas, NumPy, OpenPyXL
- **Statistics**:
  - **Core**: SciPy, Statsmodels (OLS, GLM, GEE, MixedLM)
  - **Machine Learning**: Scikit-learn
  - **Survival**: Lifelines (KM, CoxPH)
  - **Causal Inference**: EconML, PsmPy
  - **Advanced & Penalized**: FirthModels 0.8.2 (Penalized Logistic & Cox PH with LRT / Profile Likelihood CI)
- **Visualization**: Plotly (Interactive), Matplotlib, Seaborn
- **Quality, Export & Testing**: Pytest, Playwright (E2E & PDF Engine), Ruff
- **Deployment**: Docker, Gunicorn/Uvicorn

## ✅ Deployment Features

This application is designed for enterprise and restricted environments:

- **Hybrid Deployment**: Optimized for both **Hugging Face Spaces** (Docker) and **Posit Connect** (Python).
- **Dependency Management**: Split requirements for Production (`requirements-prod.txt`) vs Development (`requirements.txt`).
- **Network Friendly**: Uses Plotly CDN strategies or local serving considerations (configurable).
- **Containerized**: Full Docker support with non-root user security practices (standard in HF Spaces).

## 📝 License

This project is intended for educational and research purposes. Please ensure data privacy compliance when using with patient data.

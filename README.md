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
|-- Dockerfile
|-- app.py
|-- asgi.py
|-- config.py
|-- docker-compose.yml
|-- docs /
|   |-- ARCHITECTURE.md
|   |-- DEEP-OPTIMIZATION-PLAN.md
|   |-- IMPLEMENTATION_ROADMAP.md
|   |-- OPTIMIZATION_REPORT.md
|   |-- assets /
|   |   `-- navigation_sequence.png
|   |-- deep-ai-bot-optimization.md
|   |-- firthmodels.md
|   |-- rabbitai-report.md
|   |-- statiomed-detailed-roadmap.md
|   `-- updated-R-benchmark-script.md
|-- logger.py
|-- pyproject.toml
|-- pytest.ini
|-- requirements-prod.txt
|-- requirements.txt
|-- static /
|   |-- js /
|   |   `-- custom_handlers.js
|   `-- styles.css
|-- tabs /
|   |-- __init__.py
|   |-- _common.py
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
|   |   |-- test_survival_pipeline.py
|   |   `-- test_table_one_pipeline.py
|   `-- unit /
|       |-- test_advanced_stats.py
|       |-- test_bland_altman.py
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
|       |-- test_firth_regression.py
|       |-- test_formatting.py
|       |-- test_glm.py
|       |-- test_heterogeneity.py
|       |-- test_linear_lib.py
|       |-- test_mediation.py
|       |-- test_missing_data.py
|       |-- test_model_diagnostics.py
|       |-- test_model_diagnostics_plots.py
|       |-- test_plotly_html_rendering.py
|       |-- test_poisson_lib.py
|       |-- test_regression_publication.py
|       |-- test_repeated_measures.py
|       |-- test_sample_size.py
|       |-- test_sensitivity.py
|       |-- test_statistics.py
|       |-- test_tab_diag_html_logic.py
|       |-- test_tvc_lib.py
|       |-- test_ui_ux_styles.py
|       `-- verify_table_one.py
`-- utils /
    |-- __init__.py
    |-- advanced_stats_lib.py
    |-- collinearity_lib.py
    |-- correlation.py
    |-- data_cleaning.py
    |-- data_quality.py
    |-- decision_curve_lib.py
    |-- diag_test.py
    |-- forest_plot_lib.py
    |-- formatting.py
    |-- heterogeneity_lib.py
    |-- interaction_lib.py
    |-- linear_lib.py
    |-- logic.py
    |-- mediation_lib.py
    |-- model_diagnostics_lib.py
    |-- plotly_html_renderer.py
    |-- poisson_lib.py
    |-- psm_lib.py
    |-- repeated_measures_lib.py
    |-- sample_size_lib.py
    |-- sensitivity_lib.py
    |-- stratified_lib.py
    |-- subgroup_analysis_module.py
    |-- survival_lib.py
    |-- table_one.py
    |-- tvc_lib.py
    |-- ui_helpers.py
    |-- update_css.py
    `-- visualizations.py
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
  - **Transformation**: Log, Sqrt, and Z-Score standardization with normality assumption checks (Shapiro-Wilk/K-S).
  - **Outlier Handling**: Detection (IQR/Z-Score) and treatment (Winsorize, Cap, Remove).

### 📋 Baseline & Matching

- **Table 1 Generation**: Publication-ready baseline tables with **Intelligent Variable Classification** (detects Normal/Skewed/Categorical) and automated statistical testing (T-test/MWU/Chi2/Fisher).
- **Propensity Score Matching (PSM)**: Advanced matching with customizable calipers and variable selection.
- **Balance Diagnostics**: Enhanced Love plots (Green/Yellow zones) and Standardized Mean Differences (SMD).
- **Common Support**: Visual inspection of propensity score overlap distributions.
- **Matched Data Export**: seamless integration of matched cohorts into other analyses.

### 🔢 Sample Size & Power

- **Calculators**: Power and sample size estimation for:
  - **Means** (T-test)
  - **Proportions** (Chi-Square)
  - **Survival** (Log-Rank based on HR or Median)
  - **Correlation** (Pearson)

### 📈 Core Regression Models

- **GLM Framework**:
  - **Logistic Regression**: Standard, Auto, **Firth's Regression** (rare events), and **Subgroup Analysis** (Forest Plots).
  - **Count Models**: Poisson and Negative Binomial regression.
  - **Linear Regression**: OLS with options for robust standard errors.
- **Repeated Measures**: Generalized Estimating Equations (GEE) and Linear Mixed Models (LMM).

### ⏳ Survival Analysis

- **Visualizations**: Kaplan-Meier curves and Nelson-Aalen cumulative hazard plots.
- **Cox Modeling**: Cox Proportional Hazards regression with forest plots.
- **Advanced Techniques**:
  - **Time-Varying Cox**: Handle covariates that change over time.
  - **Landmark Analysis**: Address immortal time bias.
  - **Subgroup Analysis**: Explore treatment effect heterogeneity.

### 🎯 Causal Inference

- **Propensity Methods**: IPW (Inverse Probability Weighting) and PSM integration.
- **Stratified Analysis**: Mantel-Haenszel odds ratios and Breslow-Day homogeneity tests.
- **Sensitivity Analysis**: **E-Value** calculation for unmeasured confounding.
- **Diagnostics**: Detailed covariate balance assessment.

### 🧪 Diagnostic Tests & Agreement

- **Diagnostic Accuracy**: ROC Curves, AUC comparisons, and detailed metrics (Sens/Spec/PPV/NPV).
- **Decision Curve Analysis (DCA)**: Assess clinical net benefit.
- **Agreement Statistics**: Cohen's Kappa, Bland-Altman plots, and concordance metrics.
- **Contingency Analysis**: Chi-Square, Fisher's Exact Test, Risk Ratios, and Odds Ratios.

### 🧩 Advanced Inference

- **Mediation Analysis**: Decomposition into Direct (ADE) and Indirect (ACME) effects.
- **Model Diagnostics**: Residual plots, Q-Q plots, Cook's distance for influence, and heteroscedasticity tests.
- **Multicollinearity**: Variance Inflation Factor (VIF) analysis.
- **Heterogeneity**: Statistics for meta-analysis contexts.

### 🔗 Correlation & Reliability

- **Correlation**: Pairwise Pearson/Spearman matrices with heatmap visualizations.
- **Intraclass Correlation (ICC)**: Assess reliability and consistency.

### ⚙️ Settings & Performance

- **Customization**: NEJM-style p-value formatting, theme switching (Light/Dark), and plot sizing.
- **Logging**: Configurable logging levels and file output.
- **Performance**: Caching and multi-threading options for large datasets.

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
4. **Integrated Reporting (`utils/formatting.py`)**: Missing data statistics are automatically analyzed and included in the final report for every module.

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
  - **Advanced**: FirthModels (Penalized Logistic)
- **Visualization**: Plotly (Interactive), Matplotlib, Seaborn
- **Quality & Testing**: Pytest, Playwright, Ruff
- **Deployment**: Docker, Gunicorn/Uvicorn

## ✅ Deployment Features

This application is designed for enterprise and restricted environments:

- **Hybrid Deployment**: Optimized for both **Hugging Face Spaces** (Docker) and **Posit Connect** (Python).
- **Dependency Management**: Split requirements for Production (`requirements-prod.txt`) vs Development (`requirements.txt`).
- **Network Friendly**: Uses Plotly CDN strategies or local serving considerations (configurable).
- **Containerized**: Full Docker support with non-root user security practices (standard in HF Spaces).

## 📝 License

This project is intended for educational and research purposes. Please ensure data privacy compliance when using with patient data.

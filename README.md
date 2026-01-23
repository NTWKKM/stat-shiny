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
> **FOR AI AGENTS & BOTS (e.g., Antigravity)**:
>
> 1. **Architecture First**: Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand the Styling System and Data Processing Pipeline before editing.
> 2. **Environment**: This project requires **Python 3.12+**. Versions < 3.12 will cause SyntaxErrors.
>
> | Requirement | Version    |
> | ----------- | ---------- |
> | Python      | **≥ 3.12** |
> | pip         | Latest     |
<!-- AI-AGENT-GUIDE-END -->

[Hugging Face Space](https://ntwkkm-shinystat.hf.space)

[--- REPOSITORY-TREE-START ---]

📂 Repository Contents (File Structure)

This content reflects the repository structure (updated by GitHub Actions):

```text
|-- Dockerfile
|-- app.py
|-- asgi.py
|-- config.py
|-- firthmodels.md
|-- logger.py
|-- md /
|   `-- 20.1.26_stat-shiny-analysis v2
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
|   |-- tab_baseline_matching.py
|   |-- tab_causal_inference.py
|   |-- tab_core_regression.py
|   |-- tab_corr.py
|   |-- tab_data.py
|   |-- tab_diag.py
|   |-- tab_sample_size.py
|   |-- tab_settings.py
|   `-- tab_survival.py
|-- tests /
|   |-- .DS_Store
|   |-- conftest.py
|   |-- e2e /
|   |   `-- test_app_flow.py
|   |-- integration /
|   |   |-- test_corr_pipeline.py
|   |   |-- test_diag_pipeline.py
|   |   |-- test_forest_plot_lib.py
|   |   |-- test_interaction_pipeline.py
|   |   |-- test_logic_pipeline.py
|   |   |-- test_poisson_pipeline.py
|   |   |-- test_psm_pipeline.py
|   |   |-- test_subgroup_pipeline.py
|   |   |-- test_survival_pipeline.py
|   |   `-- test_table_one_pipeline.py
|   |-- test_bland_altman.py
|   |-- test_color_palette.py
|   |-- test_dca.py
|   |-- test_diag_returns.py
|   |-- test_glm.py
|   |-- test_plotly_html_rendering.py
|   |-- test_repeated_measures.py
|   |-- test_sample_size.py
|   `-- unit /
|       |-- test_advanced_stats.py
|       |-- test_data_cleaning.py
|       |-- test_data_cleaning_workflow.py
|       |-- test_formatting.py
|       |-- test_linear_lib.py
|       |-- test_missing_data.py
|       |-- test_statistics.py
|       |-- test_tvc_lib.py
|       `-- verify_table_one.py
|-- utils /
|   |-- __init__.py
|   |-- advanced_stats_lib.py
|   |-- correlation.py
|   |-- data_cleaning.py
|   |-- decision_curve_lib.py
|   |-- diag_test.py
|   |-- forest_plot_lib.py
|   |-- formatting.py
|   |-- interaction_lib.py
|   |-- linear_lib.py
|   |-- logic.py
|   |-- plotly_html_renderer.py
|   |-- poisson_lib.py
|   |-- psm_lib.py
|   |-- repeated_measures_lib.py
|   |-- sample_size_lib.py
|   |-- subgroup_analysis_module.py
|   |-- survival_lib.py
|   |-- table_one.py
|   |-- tvc_lib.py
|   `-- update_css.py
```

[--- REPOSITORY-TREE-END ---]

# 🏥 Medical Statistical Tool (Shiny for Python)

A comprehensive, interactive web application for medical statistical analysis, built with [Shiny for Python](https://shiny.posit.co/py/). This tool simplifies the process of data management, cohort matching, and advanced statistical modeling for medical researchers.

## 🚀 Key Features

This application is a complete statistical workbench organized into modular tabs:

### 📁 Data Management

- **Comprehensive Data Control**: Upload CSV/Excel or load example datasets.
- **Data Health Report**: Automated checks for missing values and data quality.
- **Variable Configuration**: Interactive type casting and missing value handling.

### 📋 Baseline & Matching

- **Table 1 Generation**: Publication-ready baseline tables with p-values and standardized formatting.
- **Propensity Score Matching (PSM)**: Advanced matching with customizable calipers and variable selection.
- **Balance Diagnostics**: Love plots and Standardized Mean Differences (SMD) to verify matching quality.
- **Matched Data Export**: seamless integration of matched cohorts into other analyses.

### 🔢 Sample Size & Power

- **Calculators**: Power and sample size estimation for:
  - **Means** (T-test)
  - **Proportions** (Chi-Square)
  - **Survival** (Log-Rank based on HR or Median)
  - **Correlation** (Pearson)

### 📈 Core Regression Models

- **GLM Framework**:
  - **Logistic Regression**: Standard, Auto, and **Firth's Regression** (for rare events).
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

The application uses a centralized styling system to ensures visual consistency across all modules.

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
2. **Configuration**: Users interactively cast variable types and choose missing value strategies (e.g., complete-case).
3. **Central Preparation (`utils/data_cleaning.py`)**: Before analysis, data is passed through `prepare_data_for_analysis()` which handles exclusion logic and logging.
4. **Integrated Reporting (`utils/formatting.py`)**: Missing data statistics are automatically analyzed and included in the final report for every module.

### 🧪 Testing & Quality Assurance

- **Automated CI**: [ui-styling.yml](.github/workflows/ui-styling.yml) runs on every push to verify:
  - **Palette Integrity**: Colors in `_common.py` match branding.
  - **System Sync**: Cross-file consistency between Python, CSS, JS, and Plotly layers via `tests/unit/test_ui_ux_styles.py`.

## 🛠️ Installation & Usage

### Option 1: Run Locally (Python)

Ensure you have **Python 3.12+** installed (required for modern f-string syntax).

1. **Clone the repository:**

   ```bash
   git clone https://huggingface.co/spaces/ntwkkm/shinystat
   cd shinystat
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   shiny run app.py
   ```

   The app will be available at `http://localhost:8000`.

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
  - **Causal Inference**: EconML, PsmPy (implied)
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

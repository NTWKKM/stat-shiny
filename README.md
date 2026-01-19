---
title: shinystat
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

[Hugging Face Space](https://huggingface.co/spaces/ntwkkm/shinystat)

[--- REPOSITORY-TREE-START ---]

📂 Repository Contents (File Structure)

This content reflects the repository structure (updated by GitHub Actions):

```text
|-- Dockerfile
|-- app.py
|-- asgi.py
|-- config.py
|-- correlation.py
|-- diag_test.py
|-- firthmodels.md
|-- forest_plot_lib.py
|-- interaction_lib.py
|-- logger.py
|-- logic.py
|-- md /
|   |-- 260110 Multiple Comparison Corrections.md
|   |-- 260110executive-summary.md
|   |-- 260110implementation-guide.md
|   |-- 260110stat-shiny-review.md
|   |-- CACHE_INTEGRATION_GUIDE.md
|   |-- MIGRATION_NOTES.md
|   |-- Migration from shinywidgets to HTML-Plotly.md
|   |-- Missing Data Integration Walkthrough.md
|   |-- OPTIMIZATION.md
|   |-- fix_table_one.md
|   `-- missing_data_implementation_plan.md
|-- poisson_lib.py
|-- psm_lib.py
|-- pytest.ini
|-- requirements.txt
|-- static /
|   `-- styles.css
|-- subgroup_analysis_module.py
|-- survival_lib.py
|-- table_one.py
|-- tabs /
|   |-- __init__.py
|   |-- _common.py
|   |-- _styling.py
|   |-- tab_advanced_stats.py
|   |-- tab_baseline_matching.py
|   |-- tab_corr.py
|   |-- tab_data.py
|   |-- tab_diag.py
|   |-- tab_logit.py
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
|   |-- test_color_palette.py
|   |-- test_plotly_html_rendering.py
|   `-- unit /
|       |-- test_advanced_stats.py
|       |-- test_data_cleaning.py
|       |-- test_data_cleaning_workflow.py
|       |-- test_missing_data.py
|       |-- test_statistics.py
|       `-- verify_table_one.py
|-- utils /
|   |-- __init__.py
|   |-- advanced_stats_lib.py
|   |-- data_cleaning.py
|   |-- formatting.py
|   |-- plotly_html_renderer.py
|   `-- update_css.py
`-- wakeup coderabbitai
```

[--- REPOSITORY-TREE-END ---]

# 🏥 Medical Statistical Tool (Shiny for Python)

A comprehensive, interactive web application for medical statistical analysis, built with [Shiny for Python](https://shiny.posit.co/py/). This tool simplifies the process of data management, cohort matching, and advanced statistical modeling for medical researchers.

## 🚀 Key Features

The application is organized into modular tabs for different analytical workflows.

### 📁 Data Management

Upload CSV/Excel datasets, preview data, and check variable types.

### 📋 Table 1 & Matching

- Generate standard "Table 1" baseline characteristics.
- Perform **Propensity Score Matching (PSM)** to create balanced cohorts.

### 🧪 Diagnostic Tests

Calculate sensitivity, specificity, PPV, NPV, and visualize ROC curves.

### 📊 Risk Factors (Logistic Regression)

- Run Univariable and Multivariable Logistic Regression.
- Visualize results with Forest Plots.
- Supports **Firth's Regression** for rare events (requires `firthmodels` package; see installation).

### 📈 Correlation & ICC

Analyze Pearson/Spearman correlations and Intraclass Correlation Coefficients.

### ⏳ Survival Analysis

- Kaplan-Meier survival curves.
- Cox Proportional Hazards modeling.

### ⚙️ Settings

Configure analysis parameters (e.g., p-value thresholds, methods) and UI themes.

### 📝 Logging & Diagnostics

- **Smart Logging**: Default log level set to `WARNING` for production to reduce noise.
- **Performance Tracking**: Optional performance logging for optimization.

### 🧩 Missing Data Integration (Enhanced)

A professional-grade missing data handling system that surpasses standard tools:

- **Transparency**: Detailed "Missing Data Report" for every analysis (initial N, excluded N, missing breakdown).
- **Consistency**: Unified "Complete-Case Analysis" strategy enforced across all 8 modules.
- **Flexibility**: Define custom missing value codes (e.g., -99, 999) globally.
- **Robustness**: Validation checks prevent crashes on insufficient data.
- **Quality Assurance**: Protected by automated regression tests in CI/CD.

## 🛠️ Installation & Usage

### Option 1: Run Locally (Python)

Ensure you have Python 3.10+ installed.

1. **Clone the repository:**

```bash
git clone https://huggingface.co/spaces/ntwkkm/shinystat
cd shinystat
```

1. **Install dependencies:**
It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Optional: Install for Firth Regression support
pip install firthmodels


```

1. **Run the app:**

```bash
shiny run app.py

```

The app will be available at `http://localhost:8000`.

### Option 2: Run with Docker

This project is containerized for easy deployment.

1. **Build the image:**

```bash
docker build -t medical-stat-tool .
```

1. **Run the container:**

```bash
docker run -p 7860:7860 medical-stat-tool
```

Access the app at `http://localhost:7860`.

## 💻 Tech Stack

- **Framework**: [Shiny for Python](https://shiny.posit.co/py/)
- **Data Processing**: Pandas, NumPy
- **Statistics**: SciPy, Statsmodels, Scikit-learn, Lifelines
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deployment**: Docker / Hugging Face Spaces

## ✅ Deployment Features

This application is designed for enterprise and restricted environments:

- **Compatible with restricted platforms** - Hugging Face Private Spaces, corporate networks
- **Uses Plotly CDN for interactive visualization** - No local widget servers required

## 📝 License

This project is intended for educational and research purposes. Please ensure data privacy compliance when using with patient data.

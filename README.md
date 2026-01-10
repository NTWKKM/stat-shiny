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
|   |-- 260110executive-summary.md
|   |-- 260110implementation-guide.md
|   |-- 260110stat-shiny-review.md
|   |-- CACHE_INTEGRATION_GUIDE.md
|   `-- OPTIMIZATION.md
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
|   `-- unit /
|       `-- test_statistics.py
|-- utils /
|   |-- __init__.py
|   |-- data_cleaning.py
|   |-- formatting.py
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
- Supports **Firth's Regression** for rare events (if dependencies are met).
- **NEW**: Automatic **Collinearity Diagnostics** (VIF) to detect multicollinearity.
  - Variance Inflation Factor (VIF) calculated for all predictors
  - Plain-English interpretation and actionable recommendations
  - Automatic categorical variable encoding
- **NEW**: **Multiple Comparison Corrections** for interaction terms.
  - Bonferroni, Holm, and Benjamini-Hochberg methods
  - HTML reports with method comparisons

### 📈 Correlation & ICC

Analyze Pearson/Spearman correlations and Intraclass Correlation Coefficients.

### ⏳ Survival Analysis

- Kaplan-Meier survival curves.
- Cox Proportional Hazards modeling.
- **NEW**: Multiple **Landmark Time** analysis with automatic correction.
  - Compare survival at multiple time points
  - Bonferroni, Holm, or Benjamini-Hochberg corrections applied automatically
  - Statistical significance thresholds adjusted for multiple testing

### ⚙️ Settings

Configure analysis parameters (e.g., p-value thresholds, methods) and UI themes.

## 🛠️ Installation & Usage

### Option 1: Run Locally (Python)

Ensure you have Python 3.9+ installed.

1. **Clone the repository:**

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

### Option 2: Run with Docker

This project is containerized for easy deployment.

1. **Build the image:**
```bash
docker build -t medical-stat-tool .

```


2. **Run the container:**
```bash
docker run -p 7860:7860 medical-stat-tool

```


Access the app at `http://localhost:7860`.

## 💻 Tech Stack

* **Framework**: [Shiny for Python](https://shiny.posit.co/py/)
* **Data Processing**: Pandas, NumPy
* **Statistics**: SciPy, Statsmodels, Scikit-learn, Lifelines
* **Visualization**: Matplotlib, Seaborn, Plotly
* **Deployment**: Docker / Hugging Face Spaces

## 📝 License

This project is intended for educational and research purposes. Please ensure data privacy compliance when using with patient data.

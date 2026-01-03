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
[--- REPOSITORY-TREE-END ---]

# 🏥 Medical Statistical Tool (Shiny for Python)

A comprehensive, interactive web application for medical statistical analysis, built with [Shiny for Python](https://shiny.posit.co/py/). This tool simplifies the process of data management, cohort matching, and advanced statistical modeling for medical researchers.

## 🚀 Key Features

The application is organized into modular tabs for different analytical workflows:

* **📁 Data Management**: Upload CSV/Excel datasets, preview data, and check variable types.
* **📋 Table 1 & Matching**:
* Generate standard "Table 1" baseline characteristics.
* Perform **Propensity Score Matching (PSM)** to create balanced cohorts.


* **🧪 Diagnostic Tests**: Calculate sensitivity, specificity, PPV, NPV, and visualize ROC curves.
* **📊 Risk Factors (Logistic Regression)**:
* Run Univariable and Multivariable Logistic Regression.
* Visualize results with Forest Plots.
* Supports **Firth's Regression** for rare events (if dependencies are met).


* **📈 Correlation & ICC**: Analyze Pearson/Spearman correlations and Intraclass Correlation Coefficients.
* **⏳ Survival Analysis**:
* Kaplan-Meier survival curves.
* Cox Proportional Hazards modeling.


* **⚙️ Settings**: Configure analysis parameters (e.g., p-value thresholds, methods) and UI themes.

## 🛠️ Installation & Usage

### Option 1: Run Locally (Python)

Ensure you have Python 3.9+ installed.

1. **Clone the repository:**
```bash
git clone [https://github.com/NTWKKM/stat-shiny.git](https://github.com/your-username/stat-shiny.git)
cd stat-shiny

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

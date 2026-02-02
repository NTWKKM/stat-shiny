from __future__ import annotations

import gc
import html
import json
import numbers
from itertools import combinations, islice

# Use built-in list/dict/tuple for Python 3.9+ and typing for complex types
from typing import Any

import numpy as np
import pandas as pd
from shiny import module, reactive, render, req, ui

from config import CONFIG
from logger import get_logger
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from utils.forest_plot_lib import create_forest_plot
from utils.formatting import (
    PublicationFormatter,
    create_missing_data_report_html,
    format_p_value,
)
from utils.linear_lib import (
    analyze_linear_outcome,
    bootstrap_ols,
    format_bootstrap_results,
    format_stepwise_history,
    stepwise_selection,
)
from utils.logic import analyze_outcome, run_glm
from utils.multiple_imputation import pool_estimates
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.poisson_lib import analyze_poisson_outcome
from utils.repeated_measures_lib import (
    create_trajectory_plot,
    extract_model_results,
    run_gee,
    run_lmm,
)
from utils.subgroup_analysis_module import SubgroupAnalysisLogit, SubgroupResult
from utils.ui_helpers import (
    create_empty_state_ui,
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
    create_skeleton_loader_ui,
    create_tooltip_label,
)

# Import internal modules


logger = get_logger(__name__)
COLORS = get_color_palette()


def _has_mi_datasets(
    mi_imputed_datasets: reactive.Value[list[pd.DataFrame]] | None,
) -> bool:
    """Check if MI datasets are available for pooled analysis."""
    if mi_imputed_datasets is None:
        return False
    datasets = mi_imputed_datasets.get()
    return isinstance(datasets, list) and len(datasets) > 0


def _get_mi_datasets(
    mi_imputed_datasets: reactive.Value[list[pd.DataFrame]] | None,
) -> list[pd.DataFrame]:
    """Safely get MI datasets."""
    if mi_imputed_datasets is None:
        return []
    datasets = mi_imputed_datasets.get()
    if isinstance(datasets, list):
        return datasets
    return []


# ==============================================================================
# Helper Functions (Pure Logic)
# ==============================================================================
def check_perfect_separation(df: pd.DataFrame, target_col: str) -> list[str]:
    """Identify columns causing perfect separation."""
    risky_vars: list[str] = []
    try:
        y = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if y.nunique() < 2:
            return []
    except (KeyError, TypeError, ValueError):
        return []

    for col in df.columns:
        if col == target_col:
            continue

        if df[col].nunique() < 10:
            try:
                # Use crosstab to find cells with 0 count (perfect separation indicator)
                tab = pd.crosstab(df[col], y)
                if (tab == 0).any().any():
                    risky_vars.append(col)
            except (ValueError, TypeError):
                pass
    return risky_vars


# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def core_regression_ui() -> ui.TagChild:
    """
    Builds the main user interface for the core regression module.

    Provides the dataset selector and info header plus a tabbed interface with controls, actions, and result panels for:
    Binary Outcomes (logistic), Subgroup Analysis (logit), Count & Special (Poisson, Negative Binomial, GLM), Continuous Outcomes (linear), Repeated Measures (GEE/LMM), and a Reference guide.

    Returns:
        ui.TagChild: A UI fragment containing the dataset selector/info and the tabbed analysis panels with inputs, run/download controls, and result containers.
    """
    return ui.div(
        # Title + Data Summary inline
        ui.output_ui("ui_title_with_summary"),
        # Dataset Info Box
        ui.output_ui("ui_matched_info"),
        ui.br(),
        # Dataset Selector
        ui.output_ui("ui_dataset_selector"),
        ui.br(),
        # Main Analysis Tabs
        ui.navset_tab(
            # =====================================================================
            # TAB 1: Binary Outcomes (formerly Binary Logistic)
            # =====================================================================
            ui.nav_panel(
                "📈 Binary Outcomes",
                # Control section (top)
                ui.card(
                    ui.card_header("📈 Analysis Options"),
                    ui.layout_columns(
                        create_input_group(
                            "Variable Selection",
                            ui.input_select(
                                "sel_outcome",
                                create_tooltip_label(
                                    "Select Outcome (Y)",
                                    "Must be binary (0/1 or Yes/No).",
                                ),
                                choices=[],
                            ),
                            ui.output_ui("ui_separation_warning"),
                            type="required",
                        ),
                        create_input_group(
                            "Method & Settings",
                            ui.input_radio_buttons(
                                "radio_method",
                                "Regression Method:",
                                {
                                    "auto": "Auto (Recommended)",
                                    "bfgs": "Standard (MLE)",
                                    "firth": "Firth's (Penalized)",
                                },
                            ),
                            type="required",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_logit_validation"),
                    ui.div(
                        create_input_group(
                            "Advanced Adjustments",
                            create_tooltip_label(
                                "Exclude Variables",
                                "Remove specific variables from the model.",
                            ),
                            ui.input_selectize(
                                "sel_exclude",
                                label=None,
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={"plugins": ["remove_button"]},
                            ),
                            # Interaction Pairs selector
                            ui.h6("🔗 Interaction Pairs:"),
                            ui.input_selectize(
                                "sel_interactions",
                                label=None,
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={
                                    "placeholder": "Select variable pairs to test interactions...",
                                    "plugins": ["remove_button"],
                                },
                            ),
                            type="optional",
                        )
                    ),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_logit",
                            "🚀 Run Regression",
                            class_="btn-primary btn-sm w-100",
                        ),
                        ui.download_button(
                            "btn_dl_report",
                            "📥 Download Report",
                            class_="btn-secondary btn-sm w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                # Content section (bottom)
                ui.output_ui("out_logit_status"),
                create_results_container(
                    "Analysis Results", ui.output_ui("ui_logit_results_area")
                ),
            ),
            # =====================================================================
            # TAB 1.5: Subgroup Analysis (Logit)
            # =====================================================================
            ui.nav_panel(
                "🔛 Subgroup Analysis",
                ui.card(
                    ui.card_header("Binary Logistic Subgroup Analysis - Heterogeneity"),
                    ui.layout_columns(
                        create_input_group(
                            "Variables",
                            ui.input_select(
                                "sg_logit_outcome",
                                create_tooltip_label(
                                    "Outcome (Y)", "Must be binary (0/1 or Yes/No)."
                                ),
                                choices=["Select..."],
                            ),
                            ui.input_select(
                                "sg_logit_treatment",
                                create_tooltip_label(
                                    "Treatment/Exposure",
                                    "Primary variable of interest.",
                                ),
                                choices=["Select..."],
                            ),
                            create_input_group(
                                "Stratification & Adjustment",
                                ui.input_select(
                                    "sg_logit_subgroup",
                                    create_tooltip_label(
                                        "Stratify By",
                                        "Categorical variable defining subgroups.",
                                    ),
                                    choices=["Select..."],
                                ),
                                ui.input_selectize(
                                    "sg_logit_adjust",
                                    create_tooltip_label(
                                        "Adjustment Variables",
                                        "Covariates to adjust for within subgroups.",
                                    ),
                                    choices=[],
                                    multiple=True,
                                    width="100%",
                                    options={"plugins": ["remove_button"]},
                                ),
                                type="required",
                            ),
                            type="required",
                        ),
                        col_widths=[12],
                    ),
                    ui.accordion(
                        ui.accordion_panel(
                            "⚠️ Advanced Settings",
                            create_input_group(
                                "Minimum Counts",
                                ui.input_numeric(
                                    "sg_logit_min_n",
                                    "Min N per subgroup:",
                                    value=10,
                                    min=5,
                                    max=100,
                                ),
                                type="advanced",
                            ),
                        ),
                        open=False,
                    ),
                    ui.output_ui("out_sg_logit_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_sg_logit",
                            "🚀 Run Subgroup Analysis",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_sg_logit",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                ui.output_ui("out_sg_logit_status"),
                create_results_container(
                    "Subgroup Analysis Results", ui.output_ui("out_sg_logit_result")
                ),
            ),
            # =====================================================================
            # TAB 3: Count & Special (formerly Poisson & GLM)
            # =====================================================================
            # =====================================================================
            # TAB 3: Count & Special (formerly Poisson & GLM)
            # =====================================================================
            ui.nav_panel(
                "🔢 Count & Special",
                ui.navset_card_tab(
                    ui.nav_panel(
                        "📊 Poisson Regression",
                        # Control section (top)
                        ui.card(
                            ui.card_header("📊 Poisson Analysis Options"),
                            ui.layout_columns(
                                create_input_group(
                                    "Variable Selection",
                                    ui.input_select(
                                        "poisson_outcome",
                                        create_tooltip_label(
                                            "Select Count Outcome (Y)",
                                            "Outcome must be positive integers.",
                                        ),
                                        choices=[],
                                    ),
                                    ui.input_select(
                                        "poisson_offset",
                                        create_tooltip_label(
                                            "Offset Column",
                                            "Use for rate calculations (e.g., person-years).",
                                        ),
                                        choices=["None"],
                                    ),
                                    type="required",
                                ),
                                create_input_group(
                                    "Advanced Settings",
                                    create_tooltip_label(
                                        "Exclude Variables",
                                        "Remove specific variables from the model.",
                                    ),
                                    ui.input_selectize(
                                        "poisson_exclude",
                                        label=None,
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={"plugins": ["remove_button"]},
                                    ),
                                    type="advanced",
                                ),
                                col_widths=[6, 6],
                            ),
                            # Interaction Pairs selector
                            ui.div(
                                create_input_group(
                                    "Model Refinement",
                                    ui.h6("🔗 Interaction Pairs:"),
                                    ui.input_selectize(
                                        "poisson_interactions",
                                        label=None,
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={
                                            "placeholder": "Select variable pairs to test interactions...",
                                            "plugins": ["remove_button"],
                                        },
                                    ),
                                    type="optional",
                                )
                            ),
                            ui.output_ui("out_poisson_validation"),
                            ui.hr(),
                            ui.layout_columns(
                                ui.input_action_button(
                                    "btn_run_poisson",
                                    "🚀 Run Poisson Regression",
                                    class_="btn-primary btn-sm w-100",
                                ),
                                ui.download_button(
                                    "btn_dl_poisson_report",
                                    "📥 Download Report",
                                    class_="btn-secondary btn-sm w-100",
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        # Content section (bottom)
                        ui.output_ui("out_poisson_status"),
                        create_results_container(
                            "Poisson Results", ui.output_ui("ui_poisson_results_area")
                        ),
                    ),
                    ui.nav_panel(
                        "📉 Negative Binomial",
                        ui.card(
                            ui.card_header("📉 Negative Binomial Analysis Options"),
                            ui.layout_columns(
                                create_input_group(
                                    "Variable Selection",
                                    ui.input_select(
                                        "nb_outcome",
                                        create_tooltip_label(
                                            "Select Count Outcome (Y)",
                                            "Use when data is overdispersed (variance > mean).",
                                        ),
                                        choices=[],
                                    ),
                                    ui.input_select(
                                        "nb_offset",
                                        create_tooltip_label(
                                            "Offset Column",
                                            "Use for rate calculations.",
                                        ),
                                        choices=["None"],
                                    ),
                                    type="required",
                                ),
                                create_input_group(
                                    "Advanced Settings",
                                    create_tooltip_label(
                                        "Exclude Variables",
                                        "Remove specific variables.",
                                    ),
                                    ui.input_selectize(
                                        "nb_exclude",
                                        label=None,
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={"plugins": ["remove_button"]},
                                    ),
                                    type="advanced",
                                ),
                                col_widths=[6, 6],
                            ),
                            ui.div(
                                create_input_group(
                                    "Model Refinement",
                                    ui.h6("🔗 Interaction Pairs:"),
                                    ui.input_selectize(
                                        "nb_interactions",
                                        label=None,
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={
                                            "placeholder": "Select variable pairs to test interactions...",
                                            "plugins": ["remove_button"],
                                        },
                                    ),
                                    type="optional",
                                )
                            ),
                            ui.output_ui("out_nb_validation"),
                            ui.hr(),
                            ui.layout_columns(
                                ui.input_action_button(
                                    "btn_run_nb",
                                    "🚀 Run Negative Binomial",
                                    class_="btn-primary btn-sm w-100",
                                ),
                                ui.download_button(
                                    "btn_dl_nb_report",
                                    "📥 Download Report",
                                    class_="btn-secondary btn-sm w-100",
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        ui.output_ui("out_nb_status"),
                        create_results_container(
                            "Negative Binomial Results",
                            ui.output_ui("ui_nb_results_area"),
                        ),
                    ),
                    ui.nav_panel(
                        "📈 Generalized Linear Model",
                        ui.card(
                            ui.card_header("📈 GLM Options"),
                            ui.layout_columns(
                                create_input_group(
                                    "Variable Selection",
                                    ui.input_select(
                                        "glm_outcome",
                                        create_tooltip_label(
                                            "Outcome (Y)",
                                            "Dependent variable for the model.",
                                        ),
                                        choices=[],
                                    ),
                                    ui.h6("Distribution & Link:"),
                                    ui.input_select(
                                        "glm_family",
                                        "Family:",
                                        {
                                            "Gaussian": "Gaussian (Continuous)",
                                            "Binomial": "Binomial (Binary 0/1)",
                                            "Poisson": "Poisson (Count)",
                                            "Gamma": "Gamma (Continuous +)",
                                            "InverseGaussian": "Inverse Gaussian",
                                        },
                                    ),
                                    ui.input_select(
                                        "glm_link",
                                        "Link Function:",
                                        {
                                            "identity": "Identity",
                                            "log": "Log",
                                            "logit": "Logit",
                                            "probit": "Probit",
                                            "cloglog": "Cloglog",
                                            "inverse_power": "Inverse",
                                            "sqrt": "Sqrt",
                                        },
                                    ),
                                    type="required",
                                ),
                                create_input_group(
                                    "Predictors",
                                    ui.input_selectize(
                                        "glm_predictors",
                                        create_tooltip_label(
                                            "Select Predictors (X)",
                                            "Independent variables.",
                                        ),
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={"plugins": ["remove_button"]},
                                    ),
                                    ui.input_selectize(
                                        "glm_interactions",
                                        "Interactions:",
                                        choices=[],
                                        multiple=True,
                                        width="100%",
                                        options={
                                            "placeholder": "Select variable pairs...",
                                            "plugins": ["remove_button"],
                                        },
                                    ),
                                    type="required",
                                ),
                                col_widths=[6, 6],
                            ),
                            ui.output_ui("out_glm_validation"),
                            ui.hr(),
                            ui.layout_columns(
                                ui.input_action_button(
                                    "btn_run_glm",
                                    "🚀 Run GLM",
                                    class_="btn-primary btn-sm w-100",
                                ),
                                ui.download_button(
                                    "btn_dl_glm_report",
                                    "📥 Download Report",
                                    class_="btn-secondary btn-sm w-100",
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        create_results_container(
                            "GLM Results", ui.output_ui("ui_glm_results_area")
                        ),
                    ),
                ),
            ),
            # =====================================================================
            # TAB 2: Continuous Outcomes (formerly Linear & Diagnostics)
            # =====================================================================
            ui.nav_panel(
                "📉 Continuous Outcomes",
                # Control section (top)
                ui.card(
                    ui.card_header("📐 Linear Regression Options"),
                    ui.layout_columns(
                        create_input_group(
                            "Variable Selection",
                            ui.input_select(
                                "linear_outcome",
                                create_tooltip_label(
                                    "Continuous Outcome (Y)",
                                    "Must be numeric/continuous.",
                                ),
                                choices=[],
                            ),
                            ui.input_selectize(
                                "linear_predictors",
                                create_tooltip_label(
                                    "Predictors (X)", "Independent variables."
                                ),
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={
                                    "placeholder": "Select predictors or leave empty for auto-selection...",
                                    "plugins": ["remove_button"],
                                },
                            ),
                            ui.p(
                                "💡 Leave predictors empty to auto-include all numeric variables",
                                style="font-size: 0.8em; color: #666; margin-top: 4px;",
                            ),
                            type="required",
                        ),
                        create_input_group(
                            "Method & Settings",
                            ui.input_radio_buttons(
                                "linear_method",
                                "Regression Method:",
                                {"ols": "Standard OLS", "robust": "Robust (Huber)"},
                                selected="ols",
                            ),
                            ui.input_checkbox(
                                "linear_robust_se",
                                "Use Robust Standard Errors (HC3)",
                                value=False,
                            ),
                            type="required",
                        ),
                        col_widths=[6, 6],
                    ),
                    # Advanced Options Accordion
                    ui.accordion(
                        ui.accordion_panel(
                            "🔧 Advanced Options",
                            ui.layout_columns(
                                create_input_group(
                                    "Stepwise Selection",
                                    ui.input_checkbox(
                                        "linear_stepwise_enable",
                                        "Enable Stepwise Selection",
                                        value=False,
                                    ),
                                    ui.input_radio_buttons(
                                        "linear_stepwise_dir",
                                        "Direction:",
                                        {
                                            "both": "Both",
                                            "forward": "Forward",
                                            "backward": "Backward",
                                        },
                                        selected="both",
                                        inline=True,
                                    ),
                                    ui.input_radio_buttons(
                                        "linear_stepwise_crit",
                                        "Criterion:",
                                        {
                                            "aic": "AIC",
                                            "bic": "BIC",
                                            "pvalue": "P-value",
                                        },
                                        selected="aic",
                                        inline=True,
                                    ),
                                    type="advanced",
                                ),
                                create_input_group(
                                    "Bootstrap CI",
                                    ui.input_checkbox(
                                        "linear_bootstrap_enable",
                                        "Enable Bootstrap CIs",
                                        value=False,
                                    ),
                                    ui.input_numeric(
                                        "linear_bootstrap_n",
                                        "Bootstrap Samples:",
                                        value=1000,
                                        min=100,
                                        max=10000,
                                    ),
                                    ui.input_radio_buttons(
                                        "linear_bootstrap_method",
                                        "CI Method:",
                                        {"percentile": "Percentile", "bca": "BCa"},
                                        selected="percentile",
                                        inline=True,
                                    ),
                                    type="advanced",
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        open=False,
                    ),
                    ui.div(
                        create_input_group(
                            "Ad Hoc Exclusions",
                            create_tooltip_label(
                                "Exclude Variables", "Remove specific variables."
                            ),
                            ui.input_selectize(
                                "linear_exclude",
                                label=None,
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={"plugins": ["remove_button"]},
                            ),
                            type="optional",
                        )
                    ),
                    ui.output_ui("out_linear_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_linear",
                            "🚀 Run Linear Regression",
                            class_="btn-primary btn-sm w-100",
                        ),
                        ui.download_button(
                            "btn_dl_linear_report",
                            "📥 Download Report",
                            class_="btn-secondary btn-sm w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                # Content section (bottom)
                ui.output_ui("ui_linear_results_area"),
            ),
            # (Subgroup Analysis Removed - Moved to Causal Inference Tab)
            # =====================================================================
            # TAB 5: Repeated Measures
            # =====================================================================
            ui.nav_panel(
                "🔄 Repeated Measures",
                ui.card(
                    ui.card_header("🔄 GEE & LMM Analysis"),
                    ui.layout_columns(
                        create_input_group(
                            "Variable Selection",
                            ui.input_select(
                                "rep_outcome",
                                create_tooltip_label(
                                    "Outcome (Y)", "Response variable."
                                ),
                                choices=[],
                            ),
                            ui.input_select(
                                "rep_treatment",
                                create_tooltip_label(
                                    "Group/Treatment", "Main group of interest."
                                ),
                                choices=[],
                            ),
                            ui.input_select(
                                "rep_time",
                                create_tooltip_label(
                                    "Time Variable", "Timepoint or sequence."
                                ),
                                choices=[],
                            ),
                            ui.input_select(
                                "rep_subject",
                                create_tooltip_label(
                                    "Subject ID", "Unique identifier."
                                ),
                                choices=[],
                            ),
                            type="required",
                        ),
                        create_input_group(
                            "Model Settings",
                            ui.input_radio_buttons(
                                "rep_model_type",
                                "Model Type:",
                                {
                                    "gee": "GEE (Generalized Estimating Equations)",
                                    "lmm": "LMM (Linear Mixed Model)",
                                },
                            ),
                            ui.panel_conditional(
                                "input.rep_model_type === 'gee'",
                                ui.input_select(
                                    "rep_family",
                                    "Family:",
                                    {
                                        "gaussian": "Gaussian (Continuous)",
                                        "binomial": "Binomial (Binary)",
                                        "poisson": "Poisson (Count)",
                                        "gamma": "Gamma",
                                    },
                                ),
                                ui.input_select(
                                    "rep_cov_struct",
                                    "Correlation:",
                                    {
                                        "exchangeable": "Exchangeable",
                                        "independence": "Independence",
                                        "ar1": "AR(1)",
                                    },
                                ),
                            ),
                            ui.panel_conditional(
                                "input.rep_model_type === 'lmm'",
                                ui.input_checkbox(
                                    "rep_random_slope",
                                    "Random Slope for Time",
                                    value=False,
                                ),
                            ),
                            create_tooltip_label(
                                "Adjustments (Covariates)", "Control variables."
                            ),
                            ui.input_selectize(
                                "rep_covariates",
                                label=None,
                                choices=[],
                                multiple=True,
                                width="100%",
                                options={"plugins": ["remove_button"]},
                            ),
                            type="required",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_repeated_validation"),
                    ui.hr(),
                    ui.input_action_button(
                        "btn_run_repeated",
                        "🚀 Run Analysis",
                        class_="btn-primary btn-sm w-100",
                    ),
                ),
                ui.output_ui("ui_repeated_results_area"),
            ),
            # =====================================================================
            # TAB 6: Reference
            # =====================================================================
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                ## 📚 Core Regression Reference Guide

                ### 1. 📈 Binary Outcomes (Logistic Regression)
                **Use For:** Predicting Yes/No outcomes (e.g., Disease vs Healthy, Died vs Survived).
                
                **Interpretation:**
                *   **Odds Ratio (OR):**
                    *   **OR > 1:** Risk factor (Increases likelihood of event).
                    *   **OR < 1:** Protective factor (Decreases likelihood).
                    *   **OR = 1:** No association.
                
                **Methods:**
                *   **Standard (MLE):** Best for large datasets. Fails with "Perfect Separation".
                *   **Firth's Penalized:** Use for small samples or rare events. Fixes perfect separation.
                *   **Auto:** Automatically switches to Firth if separation is detected.

                ---

                ### 2. 📉 Continuous Outcomes (Linear Regression)
                **Use For:** Predicting numeric values (e.g., Blood Pressure, Length of Stay, Cost).
                
                **Interpretation:**
                *   **Beta Coefficient (β):**
                    *   **β > 0:** Positive relationship (As X increases, Y increases).
                    *   **β < 0:** Negative relationship (As X increases, Y decreases).
                *   **R-squared (R²):** Percentage of variance explained by the model (>0.7 is usually strong).

                **Assumptions Checking:**
                *   **Linearity:** Residuals vs Fitted plot should be flat.
                *   **Normality:** Q-Q plot points should follow the diagonal line.
                *   **Homoscedasticity:** Scale-Location plot should have constant spread.

                ---

                ### 3. 🔢 Count Outcomes (Poisson / Neg. Binomial)
                **Use For:** Count data (e.g., Number of exacerbations, Days in hospital).
                
                **Model Choice:**
                *   **Poisson:** Variance = Mean. Good for simple counts.
                *   **Negative Binomial:** Variance > Mean (Overdispersion). Use if Poisson fails.
                *   **Zero-Inflated:** If there are excess zeros (e.g., many patients with 0 visits).

                **Interpretation:**
                *   **Incidence Rate Ratio (IRR):** Similar to OR. 
                    *   **IRR = 1.5:** Count increases by 50% for every 1-unit increase in X.

                ---

                ### 4. 🔄 Repeated Measures (GEE / LMM)
                **Use For:** Clustered data (e.g., Multiple visits per patient, Eyes per patient).

                **Model Choice:**
                *   **GEE (Generalized Estimating Equations):** Population-averaged effects. Robust to correlation structure errors. Best for binary/count outcomes.
                *   **LMM (Linear Mixed Models):** Subject-specific effects. Handles missing data better. Best for continuous outcomes.

                **Correlation Structures:**
                *   **Exchangeable:** All time points equally correlated.
                *   **AR(1):** Correlation decays over time.
                *   **Unstructured:** No assumption (requires more data).

                ---

                ### 5. 🔛 Subgroup Analysis
                **Use For:** Checking if treatment effect differs across groups (Heterogeneity).
                
                **Interpretation:**
                *   **P-interaction < 0.05:** Significant difference in effect. Report results separately for each group.
                *   **P-interaction ≥ 0.05:** Consistent effect. Report the overall main effect.
                """),
            ),
        ),
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def core_regression_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
    mi_imputed_datasets: reactive.Value[list[pd.DataFrame]] | None = None,  # NEW: MI
) -> None:
    # --- State Management ---
    # Store main logit results: {'html': str, 'fig_adj': FigureWidget, 'fig_crude': FigureWidget}
    """
    Initialize server-side logic and reactive UI handlers for the logistic/poisson/subgroup analysis module.

    Sets up reactive state, input-driven effects, UI renderers, and download endpoints to:
    - manage dataset selection (original vs matched) and dynamic input updates,
    - run binary logistic regression and generate HTML report + forest plots,
    - run Poisson regression and generate HTML report + forest plots,
    - run subgroup analyses and produce forest plot, summary values, and exportable results,
    - surface separation warnings, progress feedback, notifications, and memory cleanup after analyses.

    Parameters:
        input: object providing UI input accessors (e.g., selected values, buttons).
        output: UI output registry (unused directly but provided by framework).
        session: current UI session object.
        df: reactive.Value containing the primary pandas DataFrame or None.
        var_meta: reactive.Value containing variable metadata dictionary.
        df_matched: reactive.Value containing an optional matched pandas DataFrame.
        is_matched: reactive.Value[bool] indicating whether a matched dataset is available/selected.
    """
    logit_res = reactive.Value(None)
    logit_is_running = reactive.Value(False)  # Track running state
    logit_sg_res = reactive.Value(None)
    logit_sg_is_running = reactive.Value(False)  # Track subgroup running state
    # Store Poisson results: {'html': str, 'fig_adj': FigureWidget, 'fig_crude': FigureWidget}
    poisson_res = reactive.Value(None)
    poisson_is_running = reactive.Value(False)
    # Store Negative Binomial results
    nb_res = reactive.Value(None)
    nb_is_running = reactive.Value(False)
    # Store Linear Regression results: {'html_fragment': str, 'html_full': str, 'plots': dict, 'results': dict}
    linear_res = reactive.Value(None)
    linear_is_running = reactive.Value(False)
    # Store subgroup results: SubgroupResult
    subgroup_res: reactive.Value[SubgroupResult | None] = reactive.Value(None)
    # Store analyzer instance: SubgroupAnalysisLogit
    subgroup_analyzer: reactive.Value[SubgroupAnalysisLogit | None] = reactive.Value(
        None
    )
    # Store Repeated Measures results: {'results': DataFrame, 'plot': Figure, 'model_type': str}
    repeated_res = reactive.Value(None)
    repeated_is_running = reactive.Value(False)

    # Store GLM results
    glm_res = reactive.Value(None)
    glm_processing = reactive.Value(False)

    # --- Cache Clearing on Tab Change ---
    @reactive.Effect
    @reactive.event(
        input.btn_run_logit,
        input.btn_run_poisson,
        input.btn_run_nb,
        input.btn_run_subgroup,
        input.btn_run_subgroup,
        input.btn_run_linear,
    )
    def _cleanup_after_analysis():
        """
        OPTIMIZATION: Clear cache after completing analysis.
        This prevents memory buildup from heavy computations.
        """
        try:
            gc.collect()  # Force garbage collection
            logger.debug("Post-analysis cache cleared")
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df() -> pd.DataFrame | None:
        if is_matched.get() and input.radio_logit_source() == "matched":
            return df_matched.get()
        return df.get()

    @reactive.Calc
    def has_mi() -> bool:
        """Check if MI datasets are available for auto-pooling."""
        return _has_mi_datasets(mi_imputed_datasets)

    @reactive.Calc
    def mi_datasets_list() -> list[pd.DataFrame]:
        """Get MI datasets for pooled analysis."""
        return _get_mi_datasets(mi_imputed_datasets)

    @render.ui
    def ui_title_with_summary():
        """Display title with dataset summary and MI status."""
        d = current_df()
        mi_active = has_mi()
        mi_count = len(mi_datasets_list()) if mi_active else 0

        if d is not None:
            mi_badge = ""
            if mi_active:
                mi_badge = ui.span(
                    f"🔄 MI Active (m={mi_count})",
                    class_="badge bg-success ms-2",
                    style="font-size: 0.7em; vertical-align: middle;",
                )
            return ui.div(
                ui.h3("📈 Regression Models", mi_badge),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3",
                ),
            )
        return ui.h3("📈 Regression Models")

    @render.ui
    def ui_matched_info():
        """Display matched dataset availability info."""
        if is_matched.get():
            return ui.div(
                ui.tags.div(
                    "✅ **Matched Dataset Available** - You can select it below for analysis",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_dataset_selector():
        """Render dataset selector radio buttons."""
        if is_matched.get():
            original = df.get()
            matched = df_matched.get()
            original_len = len(original) if original is not None else 0
            matched_len = len(matched) if matched is not None else 0
            return ui.input_radio_buttons(
                "radio_logit_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original ({original_len:,} rows)",
                    "matched": f"✅ Matched ({matched_len:,} rows)",
                },
                selected="matched",
                inline=True,
            )
        return None

    # --- Dynamic Input Updates ---
    @reactive.Effect
    def _update_inputs():
        """
        Update all regression module input widgets to reflect the currently active dataframe.

        Inspects the active dataframe and refreshes choices and sensible defaults across tabs (Binary Logit, Logit Subgroup, Poisson, Negative Binomial, Linear, GLM, and Repeated Measures). If no dataframe is available or it is empty, no updates are performed.

        Detailed behavior:
        - Detects binary, count (non-negative integer), numeric, and candidate subgroup columns to build choice lists.
        - Chooses preferred default variables by keyword heuristics for outcomes, treatments, subgroups, offsets, time/subject identifiers, and predictors.
        - Updates select and selectize widgets for outcomes, predictors, exclusions, interactions, offsets, subgroup adjustment, repeated-measures fields, and GLM inputs.
        """
        d = current_df()
        if d is None or d.empty:
            return

        cols = d.columns.tolist()

        # Identify binary columns for outcomes
        binary_cols = [c for c in cols if d[c].nunique() == 2]

        # Identify potential subgroups (2-10 levels)
        sg_cols = [c for c in cols if 2 <= d[c].nunique() <= 10]

        # Update Tab 1 (Binary Logit) Inputs
        # Update Tab 1 (Binary Logit) Inputs
        default_logit_y = select_variable_by_keyword(
            binary_cols, ["outcome", "cured", "death", "status", "event"]
        )

        ui.update_select("sel_outcome", choices=binary_cols, selected=default_logit_y)
        ui.update_selectize("sel_exclude", choices=cols)

        # Generate interaction pair choices for Logit
        interaction_choices = list(
            islice((f"{a} × {b}" for a, b in combinations(cols, 2)), 50)
        )
        ui.update_selectize("sel_interactions", choices=interaction_choices)

        # Update Tab 2 (Poisson) Inputs
        # Identify count columns (non-negative integers)
        count_cols = [
            c
            for c in cols
            if pd.api.types.is_numeric_dtype(d[c])
            and (d[c].dropna() >= 0).all()
            and (d[c].dropna() % 1 == 0).all()
        ]

        # Prefer "Count_" or "Visits" or "Falls"
        default_poisson_y = select_variable_by_keyword(
            count_cols, ["count", "visit", "fall", "event"]
        )
        if not default_poisson_y and count_cols:
            default_poisson_y = count_cols[0]

        ui.update_select(
            "poisson_outcome",
            choices=count_cols if count_cols else cols,
            selected=default_poisson_y,
        )
        ui.update_select("poisson_offset", choices=["None"] + cols)
        ui.update_selectize("poisson_exclude", choices=cols)
        ui.update_selectize("poisson_interactions", choices=interaction_choices[:50])

        # Update Tab 2.5 (Negative Binomial) Inputs
        ui.update_select(
            "nb_outcome",
            choices=count_cols if count_cols else cols,
            selected=default_poisson_y,
        )
        ui.update_select("nb_offset", choices=["None"] + cols)
        ui.update_selectize("nb_exclude", choices=cols)
        ui.update_selectize("nb_interactions", choices=interaction_choices[:50])

        # Update Tab 3 (Linear Regression) Inputs
        # Identify continuous numeric columns for outcome
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]

        # Prefer "Lab_", "Cost", "Score"
        default_linear_y = select_variable_by_keyword(
            numeric_cols, ["lab_", "cost", "score", "chol", "hba1c"]
        )

        linear_outcome_choices = numeric_cols
        ui.update_select(
            "linear_outcome", choices=linear_outcome_choices, selected=default_linear_y
        )

        # Default predictors: exclude ID and outcome, pick numeric/categorical meaningful ones
        default_linear_x = [
            c
            for c in cols
            if c != default_linear_y
            and c not in ["ID", "id_tvc"]
            and not c.startswith("Time_")
        ][:5]  # limit to 5
        ui.update_selectize(
            "linear_predictors", choices=numeric_cols, selected=default_linear_x
        )
        ui.update_selectize("linear_exclude", choices=cols)

        # Update Tab 1.5 (Logit Subgroup) Inputs
        ui.update_select(
            "sg_logit_outcome", choices=binary_cols, selected=default_logit_y
        )

        def_sg_treat = select_variable_by_keyword(
            cols, ["treatment", "group"], default_to_first=True
        )
        ui.update_select("sg_logit_treatment", choices=cols, selected=def_sg_treat)

        def_sg_sub = select_variable_by_keyword(
            sg_cols, ["group", "subgroup"], default_to_first=True
        )
        ui.update_select("sg_logit_subgroup", choices=sg_cols, selected=def_sg_sub)

        ui.update_selectize("sg_logit_adjust", choices=cols)

        # Update Tab 5 (Repeated Measures) Inputs
        # Default: Outcome_Cured, Treatment_Group, Time_Months, ID
        default_rep_y = select_variable_by_keyword(
            cols, ["outcome_cured", "outcome", "cured"], default_to_first=True
        )
        default_rep_treat = select_variable_by_keyword(
            cols, ["treatment_group", "treatment", "group"], default_to_first=True
        )
        default_rep_time = select_variable_by_keyword(
            cols, ["time_months", "time", "month"], default_to_first=True
        )
        default_rep_subj = select_variable_by_keyword(
            cols, ["id", "subject", "subjid"], default_to_first=True
        )

        ui.update_select("rep_outcome", choices=cols, selected=default_rep_y)
        ui.update_select("rep_treatment", choices=cols, selected=default_rep_treat)
        ui.update_select("rep_time", choices=cols, selected=default_rep_time)
        ui.update_select("rep_subject", choices=cols, selected=default_rep_subj)
        ui.update_selectize("rep_covariates", choices=cols)

        # Update Tab 2.5 (GLM) Inputs
        # Similar logic to linear but broader
        ui.update_select("glm_outcome", choices=cols, selected=default_linear_y)
        ui.update_selectize("glm_predictors", choices=cols, selected=default_linear_x)
        ui.update_selectize("glm_interactions", choices=interaction_choices[:50])

    # --- Separation Warning Logic ---
    @render.ui
    def ui_separation_warning():
        d = current_df()
        target = input.sel_outcome()
        if d is None or d.empty or not target:
            return None

        risky = check_perfect_separation(d, target)
        if risky:
            return ui.div(
                ui.h6("⚠️ Perfect Separation Risk", class_="text-warning"),
                ui.p(f"Variables: {', '.join(risky)}"),
                ui.p(
                    "Result: Standard logistic regression may fail (infinite coefficients).",
                    style="font-size: 0.9em;",
                ),
                ui.p(
                    "Recommendation: Select 'Firth's (Penalized)' method or use 'Auto'.",
                    style="font-weight: bold; font-size: 0.9em;",
                ),
            )
        return None

    # --- Validation Logic ---
    @render.ui
    def out_logit_validation():
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude() if input.sel_exclude() else []

        if d is None or d.empty:
            return None

        alerts = []

        # Check 1: Target selected?
        if not target:
            return (
                None  # Already handled by dropdown placeholder or validation elsewhere
            )

        # Check 2: Target is binary?
        if target and target in d.columns:
            if d[target].nunique() != 2:
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' is not binary (has {d[target].nunique()} unique values). Please select a binary variable.",
                        title="Invalid Outcome",
                    )
                )

        # Check 3: Target in Exclude list? (Redundant but possible)
        if target in exclude:
            alerts.append(
                create_error_alert(
                    f"Outcome '{target}' is currently excluded from the analysis options.",
                    title="Configuration Error",
                )
            )

        # Check 4: Predictors available?
        # Get all potential predictors (columns - target - exclude)
        potential_predictors = [
            c for c in d.columns if c != target and c not in exclude
        ]
        if not potential_predictors:
            alerts.append(
                create_error_alert(
                    "All variables have been excluded. Please allow at least one predictor.",
                    title="No Predictors",
                )
            )

        if alerts:
            return ui.div(*alerts)

        return None

    @render.ui
    def out_poisson_validation():
        d = current_df()
        target = input.poisson_outcome()
        if d is None or d.empty:
            return None
        alerts = []
        if not target:
            return None
        # Check non-negative integers
        if target in d.columns:
            if not pd.api.types.is_numeric_dtype(d[target]):
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' must be numeric.", title="Invalid Outcome"
                    )
                )
            elif (d[target].dropna() < 0).any():
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' contains negative values.",
                        title="Invalid Outcome",
                    )
                )
        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_nb_validation():
        d = current_df()
        target = input.nb_outcome()
        if d is None or d.empty:
            return None
        alerts = []
        if not target:
            return None
        if target in d.columns:
            if not pd.api.types.is_numeric_dtype(d[target]):
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' must be numeric.", title="Invalid Outcome"
                    )
                )
            elif (d[target].dropna() < 0).any():
                alerts.append(
                    create_error_alert(
                        f"Outcome '{target}' contains negative values.",
                        title="Invalid Outcome",
                    )
                )
        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_glm_validation():
        d = current_df()
        target = input.glm_outcome()
        preds = input.glm_predictors()
        if d is None or d.empty:
            return None
        alerts = []
        if not target:
            return None
        if preds and target in preds:
            alerts.append(
                create_error_alert(
                    "Outcome variable cannot be a predictor.",
                    title="Configuration Error",
                )
            )
        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_linear_validation():
        d = current_df()
        target = input.linear_outcome()
        preds = input.linear_predictors()
        if d is None or d.empty:
            return None
        alerts = []
        if not target:
            return None
        if preds and target in preds:
            alerts.append(
                create_error_alert(
                    "Outcome variable cannot be a predictor.",
                    title="Configuration Error",
                )
            )
        if target in d.columns and not pd.api.types.is_numeric_dtype(d[target]):
            alerts.append(
                create_error_alert(
                    f"Outcome '{target}' must be continuous/numeric.",
                    title="Invalid Outcome",
                )
            )
        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_repeated_validation():
        d = current_df()
        target = input.rep_outcome()
        subject = input.rep_subject()
        time_var = input.rep_time()
        covs = input.rep_covariates()

        if d is None or d.empty:
            return None
        alerts = []

        if not target or not subject or not time_var:
            return None  # Wait for selection

        if len({target, subject, time_var}) < 3:
            alerts.append(
                create_error_alert(
                    "Outcome, Subject ID, and Time Variable must be different variables.",
                    title="Configuration Error",
                )
            )

        if covs:
            if target in covs or subject in covs or time_var in covs:
                alerts.append(
                    create_error_alert(
                        "Main variables (Outcome, Subject, Time) cannot be used as covariates.",
                        title="Configuration Error",
                    )
                )

        if alerts:
            return ui.div(*alerts)
        return None

    # --- Results Area Logic (Dynamic Loading) ---
    @render.ui
    def ui_logit_results_area():
        # Check if running
        if logit_is_running.get():
            return ui.div(
                create_loading_state("Running Logistic Regression..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        # Check if results exist
        res = logit_res.get()
        if res:
            if "error" in res:
                return ui.div(create_error_alert(res["error"]), class_="fade-in-entry")

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel("🌳 Forest Plots", ui.output_ui("ui_forest_tabs")),
                    ui.nav_panel("📋 Detailed Report", ui.HTML(res["html_fragment"])),
                    ui.nav_panel(
                        "✅ Assumptions",
                        ui.markdown("""
                            ### 🧐 Model Assumptions Checklist
                            
                            1.  **Multicollinearity:** Check standard errors in the report. Extremely large SEs often indicate high correlation between predictors (VIF > 5-10).
                            2.  **Linearity:** Log-odds should be linearly related to continuous predictors (Box-Tidwell test).
                            3.  **Independence:** Observations should be independent. If you have repeated measures per patient, use the **Repeated Measures** tab.
                            4.  **Separation:** If standard errors are huge (e.g., > 1000), you may have "Perfect Separation". Consider using **Firth's Method**.
                            """),
                    ),
                ),
                class_="fade-in-entry",
            )

        # Default Placeholder
        return create_empty_state_ui(
            message="No Logistic Regression Results",
            sub_message="Select an outcome and click '🚀 Run Logistic Regression' to start.",
            icon="📈",
        )

    # ==========================================================================
    # LOGIC: Main Logistic Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_logit)
    def _run_logit():
        """
        Run the logistic regression analysis for the currently selected outcome and publish results for the UI and download.

        Prepares the analysis dataframe (applying exclusions), parses any interaction pairs, and invokes the analysis backend. If available, builds adjusted and crude forest plots and appends them to an HTML fragment used for in-UI display. Also wraps the fragment into a complete HTML document for download. On success, stores the following keys in `logit_res`: `"html_fragment"`, `"html_full"`, `"fig_adj"`, and `"fig_crude"`, and shows a completion notification. On error, logs the exception and shows an error notification.
        """
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude()
        method = input.radio_method()
        interactions_raw = input.sel_interactions()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select an outcome variable", type="error")
            return

        # Prepare data
        final_df = d.drop(columns=exclude, errors="ignore")

        # Parse interaction pairs from "var1 × var2" format
        interaction_pairs: list[tuple[str, str]] | None = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"Logit: Using {len(interaction_pairs)} interaction pairs")

            logger.info(f"Logit: Using {len(interaction_pairs)} interaction pairs")

        # Start Loading State
        logit_is_running.set(True)
        logit_res.set(None)  # Clear previous results

        # Use reactive flush to ensure UI updates before heavy computation
        # (Note: In standard Shiny, this might still block if not async, but we set state first)

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Logistic Regression...", detail="Calculating...")

            try:
                # Check if MI datasets are available for pooled analysis
                mi_active = has_mi()
                mi_dfs = mi_datasets_list() if mi_active else []

                if mi_active and len(mi_dfs) > 0:
                    # ====== MI POOLED ANALYSIS ======
                    p.set(
                        message="Running MI Pooled Logistic Regression...",
                        detail=f"Analyzing {len(mi_dfs)} imputed datasets...",
                    )
                    logger.info(
                        f"Logit: Running pooled analysis on {len(mi_dfs)} MI datasets"
                    )

                    # Run analysis on each imputed dataset
                    all_results = []
                    for i, mi_df in enumerate(mi_dfs):
                        # Apply exclusions to MI dataset
                        mi_df_clean = mi_df.drop(columns=exclude, errors="ignore")

                        _, or_res_i, aor_res_i, _ = analyze_outcome(
                            target,
                            mi_df_clean,
                            var_meta=var_meta.get(),
                            method=method,
                            interaction_pairs=interaction_pairs,
                            adv_stats=CONFIG,
                        )
                        all_results.append(
                            {
                                "or_res": or_res_i,
                                "aor_res": aor_res_i,
                            }
                        )

                    # Pool results using Rubin's rules
                    pooled_or = {}
                    pooled_aor = {}

                    # Pool crude ORs
                    if all_results[0].get("or_res"):
                        for var_key in all_results[0]["or_res"].keys():
                            estimates = []
                            variances = []
                            for res in all_results:
                                if res.get("or_res") and var_key in res["or_res"]:
                                    v = res["or_res"][var_key]
                                    # Extract estimate on log scale
                                    or_val = v.get("or", 1.0)
                                    se = v.get("se", 0.1)
                                    if or_val > 0 and se > 0:
                                        estimates.append(np.log(or_val))
                                        variances.append(se**2)

                            if len(estimates) == len(mi_dfs):
                                pooled = pool_estimates(
                                    estimates, variances, n_obs=len(final_df)
                                )
                                pooled_or[var_key] = {
                                    "or": np.exp(pooled.estimate),
                                    "ci_low": np.exp(pooled.ci_lower),
                                    "ci_high": np.exp(pooled.ci_upper),
                                    "p_value": pooled.p_value,
                                    "fmi": pooled.fmi,
                                    "label": all_results[0]["or_res"][var_key].get(
                                        "label", var_key
                                    ),
                                }

                    # Pool adjusted ORs
                    if all_results[0].get("aor_res"):
                        for var_key in all_results[0]["aor_res"].keys():
                            estimates = []
                            variances = []
                            for res in all_results:
                                if res.get("aor_res") and var_key in res["aor_res"]:
                                    v = res["aor_res"][var_key]
                                    aor_val = v.get("aor", 1.0)
                                    se = v.get("se", 0.1)
                                    if aor_val > 0 and se > 0:
                                        estimates.append(np.log(aor_val))
                                        variances.append(se**2)

                            if len(estimates) == len(mi_dfs):
                                pooled = pool_estimates(
                                    estimates, variances, n_obs=len(final_df)
                                )
                                pooled_aor[var_key] = {
                                    "aor": np.exp(pooled.estimate),
                                    "ci_low": np.exp(pooled.ci_lower),
                                    "ci_high": np.exp(pooled.ci_upper),
                                    "p_value": pooled.p_value,
                                    "fmi": pooled.fmi,
                                    "label": all_results[0]["aor_res"][var_key].get(
                                        "label", var_key
                                    ),
                                }

                    # Build HTML report with pooled results
                    html_rep = f"""
                    <div class="alert alert-success mb-3">
                        <strong>🔄 Multiple Imputation Analysis</strong><br>
                        Results pooled from {len(mi_dfs)} imputed datasets using Rubin's Rules.
                    </div>
                    """

                    # Generate pooled results table
                    if pooled_aor:
                        html_rep += "<h4>Pooled Adjusted Odds Ratios</h4>"
                        html_rep += "<table class='table table-striped'><thead><tr>"
                        html_rep += "<th>Variable</th><th>AOR</th><th>95% CI</th><th>P-value</th><th>FMI</th></tr></thead><tbody>"
                        for k, v in pooled_aor.items():
                            p_fmt = format_p_value(v["p_value"])
                            fmi_pct = (
                                f"{v['fmi'] * 100:.1f}%" if v.get("fmi") else "N/A"
                            )
                            html_rep += f"<tr><td>{html.escape(v.get('label', k))}</td>"
                            html_rep += f"<td>{v['aor']:.2f}</td>"
                            html_rep += (
                                f"<td>{v['ci_low']:.2f} - {v['ci_high']:.2f}</td>"
                            )
                            html_rep += f"<td>{p_fmt}</td><td>{fmi_pct}</td></tr>"
                        html_rep += "</tbody></table>"

                    or_res = pooled_or
                    aor_res = pooled_aor
                    interaction_res = None

                else:
                    # ====== STANDARD ANALYSIS ======
                    # Run Logic from logic.py
                    html_rep, or_res, aor_res, interaction_res = analyze_outcome(
                        target,
                        final_df,
                        var_meta=var_meta.get(),
                        method=method,
                        interaction_pairs=interaction_pairs,
                        adv_stats=CONFIG,
                    )
            except Exception as e:
                err_msg = f"Error running logistic regression: {e!s}"
                logit_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Logistic regression error")
                return

            # Generate Forest Plots using library (for interactive widgets)
            fig_adj = None
            fig_crude = None

            if aor_res:
                df_adj = pd.DataFrame(
                    [{"variable": v.get("label", k), **v} for k, v in aor_res.items()]
                )
                if not df_adj.empty:
                    try:
                        fig_adj = create_forest_plot(
                            df_adj,
                            "aor",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Multivariable: Adjusted OR</b>",
                            x_label="Adjusted OR",
                        )
                    except ValueError as e:
                        logger.warning(
                            "Logit Adjusted Forest Plot creation failed: %s", e
                        )

            if or_res:
                df_crude = pd.DataFrame(
                    [{"variable": v.get("label", k), **v} for k, v in or_res.items()]
                )
                if not df_crude.empty:
                    try:
                        fig_crude = create_forest_plot(
                            df_crude,
                            "or",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Univariable: Crude OR</b>",
                            x_label="Crude OR",
                        )
                    except ValueError as e:
                        logger.warning("Logit Crude Forest Plot creation failed: %s", e)

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Table + Plots) ---
            # 1. Create Fragment for UI (Table + Plots)
            logit_fragment_html = html_rep

            # Note: The "Method Used" banner is now handled inside logic.py -> analyze_outcome
            # to correctly reflect "Auto" decisions.

            # Append Adjusted Plot if available
            if fig_adj:
                plot_html = plotly_figure_to_html(fig_adj, include_plotlyjs="cdn")
                logit_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Adjusted Forest Plot</h3>{plot_html}</div>"

            # Append Crude Plot if available
            if fig_crude:
                plot_html = plotly_figure_to_html(fig_crude, include_plotlyjs="cdn")
                logit_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Crude Forest Plot</h3>{plot_html}</div>"

            # 2. Create Full HTML for Download (Wrapped)
            full_logit_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Logistic Regression Report: {html.escape(target)}</title>
            </head>
            <body>
                <div class="report-container">
                    {logit_fragment_html}
                </div>
            </body>
            </html>
            """

            # Store Results
            logit_res.set(
                {
                    "html_fragment": logit_fragment_html,  # For UI
                    "html_full": full_logit_html,  # For Download
                    "fig_adj": fig_adj,
                    "fig_crude": fig_crude,
                }
            )

            ui.notification_show("✅ Analysis Complete!", type="message")

        # End Loading State
        logit_is_running.set(False)

    # --- Render Main Results ---
    @render.ui
    def out_logit_status():
        res = logit_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Regression Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
            )
        return None

        res = logit_res.get()
        if res:
            return ui.card(
                ui.card_header("📋 Detailed Report"), ui.HTML(res["html_fragment"])
            )
        return None

    @render.ui
    def ui_forest_tabs():
        """
        Render tabbed forest plot panels for the most recent logistic regression results.

        Returns:
            ui.Component: A UI element containing "Crude OR" and/or "Adjusted OR" tabs when corresponding forest figures are present.
            If analysis exists but no forest figures are available, returns a muted message indicating no plots are available.
        """
        res = logit_res.get()
        if not res:
            return None  # Should be handled by parent container logic

        tabs = []
        if res["fig_crude"]:
            tabs.append(ui.nav_panel("Crude OR", ui.output_ui("out_forest_crude")))
        if res["fig_adj"]:
            tabs.append(ui.nav_panel("Adjusted OR", ui.output_ui("out_forest_adj")))

        if not tabs:
            return ui.div(
                "No forest plots generated from the model.", class_="text-muted p-3"
            )
        return ui.navset_card_tab(*tabs)

    @render.ui
    def out_forest_adj():
        """
        Render the adjusted forest plot panel for logistic regression results.

        Returns:
            ui.Component: A UI HTML component containing the adjusted forest plot when available; otherwise a centered placeholder message indicating results are pending.
        """
        res = logit_res.get()
        if res is None or not res.get("fig_adj"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["fig_adj"],
            div_id="plot_logit_forest_adj",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.ui
    def out_forest_crude():
        """
        Render the crude (unadjusted) forest plot UI for the current logistic regression results.

        Returns:
            ui_component: A UI element containing the plot HTML when a crude figure is available; otherwise a centered placeholder message indicating results are pending.
        """
        res = logit_res.get()
        if res is None or not res.get("fig_crude"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["fig_crude"],
            div_id="plot_logit_forest_crude",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.download(filename="logit_report.html")
    def btn_dl_report():
        """
        Yield the complete HTML report for download when a logit analysis result is available.

        Yields the standalone HTML document stored in the current logistic regression result under the key "html_full".

        Returns:
            str: An HTML string containing the full, download-ready report, yielded if present.
        """
        res = logit_res.get()
        if res:
            yield res["html_full"]

    # ==========================================================================
    # LOGIC: Poisson Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_poisson)
    def _run_poisson():
        d = current_df()
        target = input.poisson_outcome()
        exclude = input.poisson_exclude()
        offset_col = input.poisson_offset()
        interactions_raw = input.poisson_interactions()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select a count outcome variable", type="error")
            return

        # Prepare data
        final_df = d.drop(columns=exclude, errors="ignore")
        offset = offset_col if offset_col != "None" else None

        # Parse interaction pairs
        interaction_pairs: list[tuple[str, str]] | None = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"Poisson: Using {len(interaction_pairs)} interaction pairs")

        # Start Loading State
        poisson_is_running.set(True)
        poisson_res.set(None)

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Poisson Regression...", detail="Calculating...")

            try:
                # Run Poisson Logic
                # Expecting 4 values from the updated poisson_lib.py
                html_rep, irr_res, airr_res, interaction_res, _ = (
                    analyze_poisson_outcome(
                        target,
                        final_df,
                        var_meta=var_meta.get(),
                        offset_col=offset,
                        interaction_pairs=interaction_pairs,
                    )
                )
            except Exception as e:
                err_msg = f"Error running Poisson regression: {e!s}"
                poisson_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Poisson regression error")
                return

            # Generate Forest Plots for IRR
            fig_adj = None
            fig_crude = None

            if airr_res:
                df_adj = pd.DataFrame(
                    [{"variable": k, **v} for k, v in airr_res.items()]
                )
                if not df_adj.empty:
                    try:
                        fig_adj = create_forest_plot(
                            df_adj,
                            "airr",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Multivariable: Adjusted IRR</b>",
                            x_label="Adjusted IRR",
                        )
                    except ValueError as e:
                        logger.warning(
                            "Poisson Adjusted Forest Plot creation failed: %s", e
                        )

            if irr_res:
                df_crude = pd.DataFrame(
                    [{"variable": k, **v} for k, v in irr_res.items()]
                )
                if not df_crude.empty:
                    try:
                        fig_crude = create_forest_plot(
                            df_crude,
                            "irr",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Univariable: Crude IRR</b>",
                            x_label="Crude IRR",
                        )
                    except ValueError as e:
                        logger.warning(
                            "Poisson Crude Forest Plot creation failed: %s", e
                        )

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Combined Table + Plot) ---
            # Unlike logic.py, poisson_lib might return just the table HTML.
            # We inject CSS and append the Forest Plot HTML here to match the requested format.

            # Keep a fragment for in-app rendering
            poisson_fragment_html = html_rep

            # Append Adjusted Plot if available, else Crude
            plot_html = ""
            if fig_adj:
                plot_html = fig_adj.to_html(full_html=False, include_plotlyjs="cdn")
                poisson_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Adjusted Forest Plot</h3>{plot_html}</div>"
            elif fig_crude:
                plot_html = fig_crude.to_html(full_html=False, include_plotlyjs="cdn")
                poisson_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Crude Forest Plot</h3>{plot_html}</div>"

            # Wrap in standard HTML structure for standalone download correctness
            wrapped_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Poisson Regression Report: {html.escape(target)}</title>
            </head>
            <body>
                <div class="report-container">
                    {poisson_fragment_html}
                </div>
            </body>
            </html>
            """
            full_poisson_html = wrapped_html

            # Store Results
            poisson_res.set(
                {
                    "html_fragment": poisson_fragment_html,  # For UI rendering
                    "html_full": full_poisson_html,  # For downloads
                    "fig_adj": fig_adj,
                    "fig_crude": fig_crude,
                }
            )

            ui.notification_show("✅ Poisson Analysis Complete!", type="message")

        # End Loading State
        poisson_is_running.set(False)

    # --- Render Poisson Results ---
    @render.ui
    def out_poisson_status():
        res = poisson_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Poisson Regression Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
            )
        return None

    @render.ui
    def ui_poisson_results_area():
        if poisson_is_running.get():
            return ui.div(
                create_loading_state("Running Poisson Regression..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = poisson_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel(
                        "🌳 Forest Plots",
                        ui.output_ui("ui_poisson_forest_tabs"),
                    ),
                    ui.nav_panel(
                        "📋 Detailed Report",
                        ui.HTML(res["html_fragment"]),
                    ),
                    ui.nav_panel(
                        "📚 Reference",
                        ui.markdown("""
                        ### Poisson Regression Reference
                        
                        **When to Use:**
                        * Count outcomes (e.g., number of events, visits, infections)
                        * Rate data with exposure offset (e.g., events per person-year)
                        
                        **Interpretation:**
                        * **IRR > 1**: Higher incidence rate (Risk factor) 🔴
                        * **IRR < 1**: Lower incidence rate (Protective) 🟢
                        * **IRR = 1**: No effect on rate
                        
                        **Overdispersion:**
                        If variance >> mean, consider Negative Binomial regression.
                        """),
                    ),
                ),
                class_="fade-in-entry",
            )

        # Default Placeholder
        return create_empty_state_ui(
            message="No Poisson Regression Results",
            sub_message="Select count outcome and predictors, then click '🚀 Run Random Forest'.. oops wait 'Run Poisson'.",
            icon="🔢",
        )

    @render.ui
    def ui_poisson_forest_tabs():
        """
        Render a tabbed UI containing Poisson regression forest plots when results are available.

        Returns:
            ui.Component: A UI fragment showing one or both tabs ("Crude IRR", "Adjusted IRR") with embedded outputs, or an informational div if results or plots are not available.
        """
        res = poisson_res.get()
        if not res:
            return ui.div(
                "Run analysis to see forest plots.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;",
            )

        tabs = []
        if res["fig_crude"]:
            tabs.append(
                ui.nav_panel("Crude IRR", ui.output_ui("out_poisson_forest_crude"))
            )
        if res["fig_adj"]:
            tabs.append(
                ui.nav_panel("Adjusted IRR", ui.output_ui("out_poisson_forest_adj"))
            )

        if not tabs:
            return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render.ui
    def out_poisson_forest_adj():
        """
        Render the adjusted Poisson regression forest plot or a waiting placeholder if results are not ready.

        Returns:
            A UI component containing the plot HTML when an adjusted figure is available; otherwise a centered placeholder div with a "Waiting for results..." message.
        """
        res = poisson_res.get()
        if res is None or not res.get("fig_adj"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["fig_adj"],
            div_id="plot_poisson_forest_adj",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.ui
    def out_poisson_forest_crude():
        """
        Render the crude Poisson forest plot as a UI HTML component, or a waiting placeholder when results are not available.

        Returns:
            ui.Element: An HTML-wrapped Plotly figure for the crude Poisson forest plot if present; otherwise a centered div containing a "Waiting for results..." message.
        """
        res = poisson_res.get()
        if res is None or not res.get("fig_crude"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["fig_crude"],
            div_id="plot_poisson_forest_crude",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.download(filename="poisson_report.html")
    def btn_dl_poisson_report():
        """
        Provide the complete Poisson regression report as a standalone HTML document for download.

        Returns:
            str: Full HTML document containing the Poisson analysis report, or nothing if no results are available.
        """
        res = poisson_res.get()
        if res:
            yield res["html_full"]

    # ==========================================================================
    # LOGIC: Negative Binomial Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_nb)
    def _run_nb():
        d = current_df()
        target = input.nb_outcome()
        exclude = input.nb_exclude()
        offset_col = input.nb_offset()
        interactions_raw = input.nb_interactions()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select a count outcome variable", type="error")
            return

        # Prepare data
        final_df = d.drop(columns=exclude, errors="ignore")
        offset = offset_col if offset_col != "None" else None

        # Parse interaction pairs
        interaction_pairs: list[tuple[str, str]] | None = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"NB: Using {len(interaction_pairs)} interaction pairs")

        # Start Loading State
        nb_is_running.set(True)
        nb_res.set(None)

        with ui.Progress(min=0, max=1) as p:
            p.set(
                message="Running Negative Binomial Regression...",
                detail="Calculating...",
            )

            try:
                # Run NB Logic (via refactored poisson_lib)
                html_rep, irr_res, airr_res, interaction_res, _ = (
                    analyze_poisson_outcome(
                        target,
                        final_df,
                        var_meta=var_meta.get(),
                        offset_col=offset,
                        interaction_pairs=interaction_pairs,
                        model_type="negative_binomial",
                    )
                )
            except Exception as e:
                err_msg = f"Error running Negative Binomial regression: {e!s}"
                nb_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Negative Binomial regression error")
                return

            # Generate Forest Plots for IRR
            fig_adj = None
            fig_crude = None

            if airr_res:
                df_adj = pd.DataFrame(
                    [{"variable": k, **v} for k, v in airr_res.items()]
                )
                if not df_adj.empty:
                    try:
                        fig_adj = create_forest_plot(
                            df_adj,
                            "airr",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Multivariable: Adjusted IRR</b>",
                            x_label="Adjusted IRR",
                        )
                    except ValueError as e:
                        logger.warning("NB Adjusted Forest Plot creation failed: %s", e)

            if irr_res:
                df_crude = pd.DataFrame(
                    [{"variable": k, **v} for k, v in irr_res.items()]
                )
                if not df_crude.empty:
                    try:
                        fig_crude = create_forest_plot(
                            df_crude,
                            "irr",
                            "ci_low",
                            "ci_high",
                            "variable",
                            pval_col="p_value",
                            title="<b>Univariable: Crude IRR</b>",
                            x_label="Crude IRR",
                        )
                    except ValueError as e:
                        logger.warning("NB Crude Forest Plot creation failed: %s", e)

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Combined Table + Plot) ---
            nb_fragment_html = html_rep

            # Append Adjusted Plot if available, else Crude
            plot_html = ""
            if fig_adj:
                plot_html = fig_adj.to_html(full_html=False, include_plotlyjs="cdn")
                nb_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Adjusted Forest Plot</h3>{plot_html}</div>"
            elif fig_crude:
                plot_html = fig_crude.to_html(full_html=False, include_plotlyjs="cdn")
                nb_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Crude Forest Plot</h3>{plot_html}</div>"

            # Wrap in standard HTML structure for standalone download correctness
            wrapped_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Negative Binomial Regression Report: {html.escape(target)}</title>
            </head>
            <body>
                <div class="report-container">
                    {nb_fragment_html}
                </div>
            </body>
            </html>
            """
            full_nb_html = wrapped_html

            # Store Results
            nb_res.set(
                {
                    "html_fragment": nb_fragment_html,
                    "html_full": full_nb_html,
                    "fig_adj": fig_adj,
                    "fig_crude": fig_crude,
                }
            )

            ui.notification_show("✅ NB Analysis Complete!", type="message")

        # End Loading State
        nb_is_running.set(False)

    # --- Render NB Results ---
    @render.ui
    def out_nb_status():
        res = nb_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Negative Binomial Regression Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
            )
        return None

    @render.ui
    def ui_nb_results_area():
        if nb_is_running.get():
            return ui.div(
                create_loading_state("Running Negative Binomial Regression..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = nb_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel(
                        "🌳 Forest Plots",
                        ui.output_ui("ui_nb_forest_tabs"),
                    ),
                    ui.nav_panel(
                        "📋 Detailed Report",
                        ui.HTML(res["html_fragment"]),
                    ),
                    ui.nav_panel(
                        "📚 Reference",
                        ui.markdown("""
                        ### Negative Binomial Regression Reference
                        
                        **When to Use:**
                        * Overdispersed count data (Variance > Mean)
                        * When Poisson model shows lack of fit due to overdispersion
                        
                        **Interpretation:**
                        * Similar to Poisson (IRR)
                        * **Alpha**: Dispersion parameter estimated by the model
                        * **IRR**: Incidence Rate Ratio
                        """),
                    ),
                ),
                class_="fade-in-entry",
            )

        return create_empty_state_ui(
            message="No Negative Binomial Regression Results",
            sub_message="Select count outcome and predictors, then click 'Run Negative Binomial'.",
            icon="📉",
        )

    @render.ui
    def ui_nb_forest_tabs():
        res = nb_res.get()
        if not res:
            return ui.div(
                "Run analysis to see forest plots.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;",
            )

        tabs = []
        if res["fig_crude"]:
            tabs.append(ui.nav_panel("Crude IRR", ui.output_ui("out_nb_forest_crude")))
        if res["fig_adj"]:
            tabs.append(ui.nav_panel("Adjusted IRR", ui.output_ui("out_nb_forest_adj")))

        if not tabs:
            return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render.ui
    def out_nb_forest_adj():
        res = nb_res.get()
        if res is None or not res.get("fig_adj"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["fig_adj"],
            div_id="plot_nb_forest_adj",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.ui
    def out_nb_forest_crude():
        res = nb_res.get()
        if res is None or not res.get("fig_crude"):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["fig_crude"],
            div_id="plot_nb_forest_crude",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.download(filename="nb_report.html")
    def btn_dl_nb_report():
        res = nb_res.get()
        if res:
            yield res["html_full"]

    # ==========================================================================
    # LOGIC: Linear Regression (OLS)
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_linear)
    def _run_linear():
        """Run linear regression analysis for continuous outcome."""
        d = current_df()
        target = input.linear_outcome()
        predictors = (
            list(input.linear_predictors()) if input.linear_predictors() else None
        )
        exclude = list(input.linear_exclude()) if input.linear_exclude() else []
        method = input.linear_method()
        robust_se = input.linear_robust_se()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show(
                "Please select a continuous outcome variable", type="error"
            )
            return

        # Start Loading State
        linear_is_running.set(True)
        linear_res.set(None)

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Linear Regression...", detail="Preparing data...")

            try:
                # Run analysis from linear_lib
                html_report, results, plots, missing_info = analyze_linear_outcome(
                    outcome_name=target,
                    df=d,
                    predictor_cols=predictors,
                    var_meta=var_meta.get(),
                    exclude_cols=exclude,
                    regression_type=method,
                    robust_se=robust_se,
                )
            except Exception as e:
                err_msg = f"Error running Linear Regression: {e!s}"
                linear_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Linear regression error")
                return

            # Create full HTML for download
            target_escaped = html.escape(target)
            full_linear_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Linear Regression Report: {target_escaped}</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
                    .table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                    .table th, .table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
                    .table th {{ background: #f5f5f5; }}
                    .table-striped tbody tr:nth-child(odd) {{ background: #fafafa; }}
                </style>
            </head>
            <body>
                <div class="report-container">
                    {html_report}
                </div>
            </body>
            </html>
            """

            # Store Results
            linear_res.set(
                {
                    "html_fragment": html_report,
                    "html_full": full_linear_html,
                    "plots": plots,
                    "results": results,
                    "missing_info": missing_info,
                }
            )

            ui.notification_show("✅ Linear Regression Complete!", type="message")

        # End Loading State
        linear_is_running.set(False)

    # --- Render Linear Regression Results ---
    @render.ui
    def ui_linear_results_area():
        if linear_is_running.get():
            return ui.div(
                create_loading_state("Running Linear Regression..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = linear_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            r2 = res["results"].get("r_squared", 0)
            n_obs = res["results"].get("n_obs", 0)
            r2_text = f"R² = {r2:.4f}" if np.isfinite(r2) else "R² = N/A"

            return create_results_container(
                "Regression Results",
                ui.navset_tab(
                    ui.nav_panel(
                        "📋 Regression Results",
                        ui.div(
                            ui.div(
                                ui.h5(
                                    f"✅ Linear Regression Complete ({r2_text}, n = {n_obs:,})"
                                ),
                                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
                            ),
                            ui.output_ui("out_linear_html_report"),
                        ),
                    ),
                    ui.nav_panel(
                        "📈 Diagnostic Plots",
                        ui.output_ui("out_linear_diagnostic_plots"),
                    ),
                    ui.nav_panel(
                        "🔍 Variable Selection", ui.output_ui("out_linear_stepwise")
                    ),
                    ui.nav_panel(
                        "🎲 Bootstrap CI", ui.output_ui("out_linear_bootstrap")
                    ),
                    ui.nav_panel(
                        "📚 Reference",
                        ui.markdown("""
                            ### Linear Regression Reference
                            
                            **When to Use:**
                            * Continuous outcomes (blood pressure, glucose, length of stay)
                            * Understanding effect size of predictors (β coefficients)
                            * Analyzing relationships between continuous variables
                            
                            **Interpretation:**
                            * **β > 0**: Positive relationship (Y increases with X)
                            * **β < 0**: Negative relationship (Y decreases with X)
                            * **p < 0.05**: Statistically significant effect
                            * **CI not crossing 0**: Significant effect
                            
                            **Model Fit:**
                            * **R² > 0.7**: Strong explanatory power
                            * **R² 0.4-0.7**: Moderate explanatory power
                            * **R² < 0.4**: Weak explanatory power
                            
                            **Assumptions:**
                            1. **Linearity**: Check Residuals vs Fitted plot
                            2. **Normality**: Check Q-Q plot
                            3. **Homoscedasticity**: Check Scale-Location plot
                            4. **Independence**: Check Durbin-Watson statistic
                            5. **No Multicollinearity**: Check VIF values
                            
                            **Stepwise Selection:**
                            Automatically selects the best subset of variables using AIC, BIC, or p-value criteria.
                            - Forward: Start empty, add significant variables
                            - Backward: Start full, remove non-significant variables
                            - Both: Stepwise forward and backward
                            
                            **Bootstrap CI:**
                            Non-parametric confidence intervals via resampling.
                            - Percentile: Simple quantile-based CIs
                            - BCa: Bias-corrected and accelerated (more accurate)
                            """),
                    ),
                ),
                class_="fade-in-entry",
            )

        return create_empty_state_ui(
            message="No Linear Regression Results",
            sub_message="Select an outcome and predictors, then click 'Run Linear Regression'.",
            icon="📉",
        )

    @render.ui
    def out_linear_html_report():
        """Render the Linear Regression detailed report."""
        res = linear_res.get()
        if res:
            return ui.card(
                ui.card_header("📋 Linear Regression Report"),
                ui.HTML(res["html_fragment"]),
            )
        return ui.card(
            ui.card_header("📋 Linear Regression Report"),
            ui.div(
                "Run analysis to see detailed report.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;",
            ),
        )

    @render.ui
    def out_linear_diagnostic_plots():
        """Render diagnostic plots for linear regression."""
        res = linear_res.get()
        if not res or not res.get("plots"):
            return ui.div(
                "Run analysis to see diagnostic plots.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;",
            )

        plots = res["plots"]
        plot_sections = []

        # Plot descriptions
        plot_info = [
            (
                "residuals_vs_fitted",
                "📊 Residuals vs Fitted",
                "✅ Random scatter = good (linearity, homoscedasticity) | ❌ Pattern = potential issues",
            ),
            (
                "qq_plot",
                "📈 Normal Q-Q Plot",
                "✅ Points on line = normal residuals | ❌ Deviation = non-normality",
            ),
            (
                "scale_location",
                "📉 Scale-Location Plot",
                "✅ Horizontal trend = constant variance | ❌ Slope = heteroscedasticity",
            ),
            (
                "residuals_vs_leverage",
                "🔍 Residuals vs Leverage",
                "✅ Blue points = normal | 🔴 Red points = influential observations (high Cook's D)",
            ),
        ]

        for plot_key, plot_title, plot_desc in plot_info:
            if plot_key in plots and plots[plot_key] is not None:
                plot_html = plotly_figure_to_html(
                    plots[plot_key],
                    div_id=f"linear_diag_{plot_key}",
                    include_plotlyjs="cdn",
                    responsive=True,
                )
                plot_sections.append(
                    ui.card(
                        ui.card_header(plot_title),
                        ui.HTML(plot_html),
                        ui.p(
                            plot_desc,
                            style="font-size: 0.85em; color: #666; margin-top: 10px;",
                        ),
                    )
                )

        if not plot_sections:
            return ui.div("No diagnostic plots available.", class_="text-muted")

        return ui.div(*plot_sections)

    @render.download(filename="linear_regression_report.html")
    def btn_dl_linear_report():
        """Download the complete Linear Regression report as HTML."""
        res = linear_res.get()
        if res:
            yield res["html_full"]

    # --- Stepwise Selection Results ---
    @render.ui
    def out_linear_stepwise():
        """Render stepwise variable selection results."""

        # Check if stepwise is enabled and data available
        if not input.linear_stepwise_enable():
            return ui.card(
                ui.card_header("🔍 Variable Selection"),
                ui.div(
                    "Enable stepwise selection in Advanced Options to use this feature.",
                    style="color: gray; font-style: italic; padding: 20px; text-align: center;",
                ),
            )

        d = current_df()
        target = input.linear_outcome()
        predictors = (
            list(input.linear_predictors()) if input.linear_predictors() else None
        )
        exclude = list(input.linear_exclude()) if input.linear_exclude() else []

        if d is None or d.empty or not target:
            return ui.card(
                ui.card_header("🔍 Variable Selection"),
                ui.div(
                    "Load data and select outcome to run stepwise selection.",
                    style="color: gray; font-style: italic; padding: 20px; text-align: center;",
                ),
            )

        # Determine candidate columns
        if predictors:
            candidates = [c for c in predictors if c not in exclude and c != target]
        else:
            numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
            candidates = [c for c in numeric_cols if c != target and c not in exclude]

        if len(candidates) < 2:
            return ui.card(
                ui.card_header("🔍 Variable Selection"),
                ui.div(
                    "Need at least 2 candidate variables for stepwise selection.",
                    style="color: gray; font-style: italic; padding: 20px; text-align: center;",
                ),
            )

        # Run stepwise selection
        try:
            step_result = stepwise_selection(
                df=d.dropna(subset=[target] + candidates),
                outcome_col=target,
                candidate_cols=candidates,
                direction=input.linear_stepwise_dir(),
                criterion=input.linear_stepwise_crit(),
            )
        except Exception as e:
            return ui.card(
                ui.card_header("🔍 Variable Selection"),
                ui.div(f"Error: {e}", style="color: red; padding: 20px;"),
            )

        # Format history
        history_df = format_stepwise_history(step_result.get("history", []))

        # Sanitize all columns (stepwise doesn't have styled p-values in this tab's helper)
        # Wait, format_stepwise_history might have p-values.
        # Let's check headers.
        df_safe = history_df.copy()
        for col in df_safe.columns:
            df_safe[col] = df_safe[col].astype(str).map(html.escape)

        history_html = df_safe.to_html(
            index=False, escape=False, classes="table table-sm", border=0
        )

        selected = step_result.get("selected_vars", [])
        criterion_val = step_result.get("final_criterion", 0)

        return ui.card(
            ui.card_header(
                f"🔍 Stepwise Selection ({input.linear_stepwise_dir().title()}, {input.linear_stepwise_crit().upper()})"
            ),
            ui.div(
                ui.h5(f"✅ Selected Variables ({len(selected)}):"),
                (
                    ui.tags.ul([ui.tags.li(v) for v in selected])
                    if selected
                    else ui.p("No variables selected")
                ),
                ui.p(
                    f"Final {input.linear_stepwise_crit().upper()}: {criterion_val:.2f}"
                ),
                ui.hr(),
                ui.h5("Selection History:"),
                ui.HTML(history_html),
                style="padding: 15px;",
            ),
        )

    # --- Bootstrap CI Results ---
    @render.ui
    def out_linear_bootstrap():
        """Render bootstrap confidence interval results."""
        if not input.linear_bootstrap_enable():
            return ui.card(
                ui.card_header("🎲 Bootstrap Confidence Intervals"),
                ui.div(
                    "Enable Bootstrap CIs in Advanced Options to use this feature.",
                    style="color: gray; font-style: italic; padding: 20px; text-align: center;",
                ),
            )

        d = current_df()
        target = input.linear_outcome()
        predictors = (
            list(input.linear_predictors()) if input.linear_predictors() else None
        )
        exclude = list(input.linear_exclude()) if input.linear_exclude() else []

        if d is None or d.empty or not target:
            return ui.card(
                ui.card_header("🎲 Bootstrap Confidence Intervals"),
                ui.div(
                    "Load data and select outcome to compute bootstrap CIs.",
                    style="color: gray; font-style: italic; padding: 20px; text-align: center;",
                ),
            )

        # Determine predictor columns
        if predictors:
            use_predictors = [c for c in predictors if c not in exclude and c != target]
        else:
            numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
            use_predictors = [
                c for c in numeric_cols if c != target and c not in exclude
            ]

        if not use_predictors:
            return ui.card(
                ui.card_header("🎲 Bootstrap Confidence Intervals"),
                ui.div(
                    "No predictor variables available.",
                    style="color: gray; padding: 20px;",
                ),
            )

        # Run bootstrap
        n_boot = input.linear_bootstrap_n()
        ci_method = input.linear_bootstrap_method()

        with ui.Progress(min=0, max=1) as p:
            p.set(
                message=f"Running {n_boot} bootstrap samples...",
                detail="This may take a moment...",
            )

            try:
                boot_result = bootstrap_ols(
                    df=d.dropna(subset=[target] + use_predictors),
                    outcome_col=target,
                    predictor_cols=use_predictors,
                    n_bootstrap=n_boot,
                    random_state=42,
                )
            except Exception as e:
                return ui.card(
                    ui.card_header("🎲 Bootstrap Confidence Intervals"),
                    ui.div(f"Error: {e}", style="color: red; padding: 20px;"),
                )

        if "error" in boot_result:
            return ui.card(
                ui.card_header("🎲 Bootstrap Confidence Intervals"),
                ui.div(
                    f"Error: {boot_result['error']}", style="color: red; padding: 20px;"
                ),
            )

        # Format results
        formatted = format_bootstrap_results(boot_result, ci_method=ci_method)

        # Sanitize all columns (bootstrap doesn't have styled p-values usually, but let's be safe)
        df_safe = formatted.copy()
        for col in df_safe.columns:
            df_safe[col] = df_safe[col].astype(str).map(html.escape)

        result_html = df_safe.to_html(
            index=False, escape=False, classes="table table-striped", border=0
        )

        return ui.card(
            ui.card_header(
                f"🎲 Bootstrap CIs (n={boot_result['n_bootstrap']}, {ci_method.upper()})"
            ),
            ui.div(
                ui.HTML(result_html),
                ui.p(
                    f"Failed samples: {boot_result['failed_samples']} | "
                    f"CI Level: {int(boot_result['ci_level'] * 100)}%",
                    style="font-size: 0.85em; color: #666; margin-top: 10px;",
                ),
                style="padding: 15px;",
            ),
        )

    # ==========================================================================
    # LOGIC: Subgroup Analysis
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_subgroup)
    def _run_subgroup():
        d = current_df()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if (
            not input.sg_outcome()
            or not input.sg_treatment()
            or not input.sg_subgroup()
        ):
            ui.notification_show("Please fill all required fields", type="error")
            return

        analyzer = SubgroupAnalysisLogit(d)

        with ui.Progress(min=0, max=1) as p:
            p.set(
                message="Running Subgroup Analysis...", detail="Testing interactions..."
            )

            try:
                results = analyzer.analyze(
                    outcome_col=input.sg_outcome(),
                    treatment_col=input.sg_treatment(),
                    subgroup_col=input.sg_subgroup(),
                    adjustment_cols=list(input.sg_adjust()),
                    min_subgroup_n=input.sg_min_n(),
                    var_meta=var_meta.get(),
                )

                subgroup_res.set(results)
                subgroup_analyzer.set(analyzer)
                ui.notification_show("✅ Subgroup Analysis Complete!", type="message")

            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Subgroup analysis error")

    # --- Render Subgroup Results ---
    @render.ui
    def out_subgroup_status():
        """
        Render a completion banner when subgroup analysis results are available.

        Returns:
            ui.div: A success banner UI element indicating subgroup analysis is complete when results exist, `None` otherwise.
        """
        res = subgroup_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Subgroup Analysis Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
            )
        return None

    @render.ui
    def out_sg_forest_plot():
        """
        Render the subgroup analysis forest plot as an HTML UI component.

        Generates a Plotly-based forest plot using the current SubgroupAnalysisLogit analyzer and returns it wrapped as a ui.HTML component for embedding in the UI. If the analyzer is not yet available, if plot creation raises a ValueError, or if no figure is produced, returns a styled placeholder ui.div with a short status or warning message instead.

        Returns:
            A UI component: `ui.HTML` containing the plotly-generated HTML when a figure is available, or a `ui.div` placeholder message when results are waiting, missing, or plot creation fails.
        """
        analyzer = subgroup_analyzer.get()
        if analyzer is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        # Use txt_edit_forest_title if provided, fallback to sg_title
        title = input.txt_edit_forest_title() or input.sg_title() or None
        try:
            fig = analyzer.create_forest_plot(title=title)
        except ValueError as e:
            logger.warning("Forest plot creation failed: %s", e)
            return ui.div(
                ui.markdown("⚠️ *Run analysis first to generate forest plot.*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        if fig is None:
            return ui.div(
                ui.markdown("⏳ *No forest plot available...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            fig, div_id="plot_logit_subgroup", include_plotlyjs="cdn", responsive=True
        )
        return ui.HTML(html_str)

    @reactive.Effect
    @reactive.event(input.btn_update_plot_title)
    def _update_sg_title():
        # Invalidate to trigger re-render of the forest plot widget
        subgroup_analyzer.set(subgroup_analyzer.get())

    @render.text
    def val_overall_or():
        res = subgroup_res.get()
        if res:
            overall = res.get("overall", {})
            or_val = overall.get("or")
            return f"{or_val:.3f}" if or_val is not None else "N/A"
        return "-"

    @render.text
    def val_overall_p():
        res = subgroup_res.get()
        if res:
            return format_p_value(res["overall"]["p_value"], use_style=False)
        return "-"

    @render.text
    def val_interaction_p():
        res = subgroup_res.get()
        if res:
            p_int = res["interaction"]["p_value"]
            return (
                format_p_value(p_int, use_style=False) if p_int is not None else "N/A"
            )
        return "-"

    @render.ui
    def out_interpretation_box():
        res = subgroup_res.get()
        analyzer = subgroup_analyzer.get()
        if res and analyzer:
            interp = analyzer.get_interpretation()
            is_het = res["interaction"]["significant"]
            color = "alert-warning" if is_het else "alert-success"
            icon = "⚠️" if is_het else "✅"
            return ui.div(f"{icon} {interp}", class_=f"alert {color}")
        return None

    @render.data_frame
    def out_sg_table():
        res = subgroup_res.get()
        if res:
            df_res = res["results_df"].copy()
            # Simple formatting for display
            cols = ["group", "n", "events", "or", "ci_low", "ci_high", "p_value"]
            available_cols = [c for c in cols if c in df_res.columns]
            return render.DataGrid(df_res[available_cols].round(4))
        return None

    @render.ui
    def out_sg_missing_report() -> ui.TagChild | None:
        res = subgroup_res.get()
        if res and "missing_data_info" in res:
            return ui.HTML(
                create_missing_data_report_html(
                    res["missing_data_info"], var_meta.get() or {}
                )
            )
        return None

    # --- Subgroup Downloads ---
    @render.download(filename=lambda: f"subgroup_plot_{input.sg_subgroup()}.html")
    def dl_sg_html():
        analyzer = subgroup_analyzer.get()
        if analyzer and analyzer.figure:
            yield analyzer.figure.to_html(include_plotlyjs="cdn")

    @render.download(filename=lambda: f"subgroup_res_{input.sg_subgroup()}.csv")
    def dl_sg_csv():
        res = subgroup_res.get()
        if res:
            yield res["results_df"].to_csv(index=False)

    @render.download(filename=lambda: f"subgroup_data_{input.sg_subgroup()}.json")
    def dl_sg_json():
        """
        Produce a JSON-formatted representation of the latest subgroup analysis results.

        Yields:
            str: A JSON-formatted string of the subgroup results (indent=2). Non-JSON-native types (e.g., NumPy scalars/arrays) are converted to strings to ensure serializability.
        """
        res = subgroup_res.get()
        if res:
            # Need to handle numpy types for JSON serialization
            yield json.dumps(res, indent=2, default=str)

    # =========================================================================
    # LOGISTIC SUBGROUP SERVER LOGIC
    # =========================================================================

    @render.ui
    def out_sg_logit_status():
        """
        Render a loading indicator while the logistic subgroup analysis is running.

        Returns:
            ui.TagChild | None: A loading UI element when the subgroup analysis is in progress, otherwise None.
        """
        if logit_sg_is_running.get():
            return create_loading_state("Running Subgroup Analysis...")
        return None

    @reactive.Effect
    @reactive.event(input.btn_run_sg_logit)
    def _run_sg_logit():
        """
        Execute a logistic subgroup analysis using the current dataset and UI selections.

        Validates that outcome, treatment, and subgroup are selected, shows progress notifications, and sets the running state while the analysis executes. On success stores the analysis results (including a generated `forest_plot`) in `logit_sg_res`; on error stores an error in `logit_sg_res` and displays an error notification. Does not return a value.
        """
        d = current_df()
        y = input.sg_logit_outcome()
        treat = input.sg_logit_treatment()
        subgroup = input.sg_logit_subgroup()
        adjust = input.sg_logit_adjust()
        min_n = input.sg_logit_min_n()

        if d is None:
            return

        if not all([y, treat, subgroup]) or any(
            x == "Select..." for x in [y, treat, subgroup]
        ):
            ui.notification_show("Please select all required variables", type="warning")
            return

        logit_sg_is_running.set(True)
        logit_sg_res.set(None)
        ui.notification_show(
            "Running Subgroup Analysis...", duration=None, id="run_sg_logit"
        )

        try:
            analyzer = SubgroupAnalysisLogit(d)
            result = analyzer.analyze(
                outcome_col=y,
                treatment_col=treat,
                subgroup_col=subgroup,
                adjustment_cols=list(adjust) if adjust else None,
                min_subgroup_n=min_n,
                var_meta=var_meta.get(),
            )

            if "error" in result:
                logit_sg_res.set({"error": result["error"]})
                ui.notification_show(result["error"], type="error")
                ui.notification_remove("run_sg_logit")
                return

            # Generate forest plot
            forest_fig = analyzer.create_forest_plot()
            result["forest_plot"] = forest_fig

            logit_sg_res.set(result)
            ui.notification_remove("run_sg_logit")
            ui.notification_show("✅ Analysis Complete", type="message")

        except Exception as e:
            ui.notification_remove("run_sg_logit")
            ui.notification_show(f"Analysis failed: {e}", type="error")
            logger.exception("Logit Subgroup Analysis Error")
        finally:
            logit_sg_is_running.set(False)

    @render.ui
    def out_sg_logit_result():
        """
        Render the logistic subgroup analysis results UI.

        If no results are available, returns a placeholder prompting the user to run the analysis.
        If the results contain an error, returns an error alert. Otherwise, returns a composed UI
        containing a forest plot card, a detailed results table card, and an interaction test card
        displaying the interaction p-value and a heterogeneity message.

        Returns:
            ui.TagChild: A UI element representing the subgroup analysis output (placeholder, error alert,
            or cards with forest plot, results table, and interaction test).
        """
        res = logit_sg_res.get()
        if res is None:
            return create_placeholder_state("Run analysis to see results", "🔛")
        if "error" in res:
            return create_error_alert(res["error"])

        # Create summary table
        summary_df = pd.DataFrame(res["results_df"])
        # Format P-values
        if "p_value" in summary_df.columns:
            summary_df["p_value"] = summary_df["p_value"].apply(
                lambda x: format_p_value(x) if isinstance(x, numbers.Real) else x
            )

        # Sanitize all non-p-value columns to prevent XSS
        df_safe = summary_df.copy()
        for col in df_safe.columns:
            if col != "p_value":
                df_safe[col] = df_safe[col].astype(str).map(html.escape)

        table_html = df_safe.to_html(
            classes="table table-striped table-hover", index=False, escape=False
        )

        # Plot
        fig = res.get("forest_plot")
        plot_html = plotly_figure_to_html(fig) if fig else ""

        return ui.div(
            ui.card(
                ui.card_header("🌳 Forest Plot (Treatment Effect by Subgroup)"),
                ui.HTML(plot_html),
            ),
            ui.br(),
            ui.card(
                ui.card_header("📊 Detailed Results"),
                ui.HTML(table_html),
            ),
            ui.br(),
            ui.card(
                ui.card_header("Interaction Test"),
                ui.div(
                    ui.p(
                        f"P-interaction: {format_p_value(res['interaction']['p_value'])}"
                        if res["interaction"]["p_value"] is not None
                        else "N/A"
                    ),
                    ui.p(
                        "Significant Heterogeneity detected."
                        if res["interaction"]["significant"]
                        else "No significant heterogeneity detected."
                    ),
                    class_="alert alert-info"
                    if not res["interaction"]["significant"]
                    else "alert alert-warning",
                ),
            ),
        )

    @render.download(filename="logit_subgroup_report.html")
    def btn_dl_sg_logit():
        """
        Produce an HTML report for the completed logistic subgroup analysis.

        Yields a single HTML string containing a forest plot and a detailed results table when analysis results exist; yields the message "No results available." if no results are present.

        Returns:
            generator (str): Yields the HTML report string or an availability message.
        """
        res = logit_sg_res.get()
        if not res:
            yield "No results available."
            return

        # Simplified Report generation for now
        html_content = f"""
        <html>
        <head><title>Subgroup Analysis Report</title></head>
        <body>
            <h1>Subgroup Analysis (Logistic Regression)</h1>
            <h2>Forest Plot</h2>
            {plotly_figure_to_html(res.get("forest_plot"))}
            <h2>Detailed Results</h2>
            {pd.DataFrame(res["results_df"]).to_html()}
        </body>
        </html>
        """
        yield html_content

    # ==========================================================================
    # LOGIC: Repeated Measures
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_repeated)
    def _run_repeated():
        d = current_df()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return

        outcome = input.rep_outcome()
        treatment = input.rep_treatment()
        time_var = input.rep_time()
        subject = input.rep_subject()

        if not all([outcome, treatment, time_var, subject]):
            ui.notification_show(
                "Please select all required variables (Outcome, Treatment, Time, Subject)",
                type="error",
            )
            return

        model_type = input.rep_model_type()
        covariates = list(input.rep_covariates()) if input.rep_covariates() else []

        # Exclude rows with missing data in selected columns
        cols_needed = [outcome, treatment, time_var, subject] + covariates
        d.dropna(subset=cols_needed)

        # Start Loading State
        repeated_is_running.set(True)
        repeated_res.set(None)

        with ui.Progress(min=0, max=1) as p:
            p.set(message=f"Running {model_type.upper()}...", detail="Analyzing...")

            try:
                if model_type == "gee":
                    results, missing_info = run_gee(
                        d,  # Pass original d, cleaning handled inside lib
                        outcome_col=outcome,
                        treatment_col=treatment,
                        time_col=time_var,
                        subject_col=subject,
                        covariates=covariates,
                        cov_struct=input.rep_cov_struct(),
                        family_str=input.rep_family(),
                        var_meta=var_meta.get() or {},
                    )
                else:  # lmm
                    results, missing_info = run_lmm(
                        d,
                        outcome_col=outcome,
                        treatment_col=treatment,
                        time_col=time_var,
                        subject_col=subject,
                        covariates=covariates,
                        random_slope=input.rep_random_slope(),
                        var_meta=var_meta.get() or {},
                    )

                # Use the indices from missing_info to get the cleaned df for plotting
                # (since the original code used df_clean for create_trajectory_plot)
                df_clean_subset = (
                    d.loc[missing_info.get("analyzed_indices", [])]
                    if "analyzed_indices" in missing_info
                    else d
                )

                # Check for error string
                if isinstance(results, str):
                    repeated_res.set({"error": results})
                    ui.notification_show("Analysis failed", type="error")
                    return

                # Extract Results
                df_res = extract_model_results(results, model_type)

                # Create Plot
                fig = create_trajectory_plot(
                    df_clean_subset,
                    outcome_col=outcome,
                    time_col=time_var,
                    group_col=treatment,
                    subject_col=subject,
                )

                repeated_res.set(
                    {
                        "results": df_res,
                        "plot": fig,
                        "model_type": model_type,
                        "missing_data_info": missing_info,
                    }
                )

                ui.notification_show(
                    f"✅ {model_type.upper()} Analysis Complete!", type="message"
                )

            except Exception as e:
                err_msg = f"Error running Repeated Measures: {e!s}"
                repeated_res.set({"error": err_msg})
                ui.notification_show("Analysis failed", type="error")
                logger.exception("Repeated measures error")
            finally:
                repeated_is_running.set(False)

    @render.ui
    def ui_repeated_results_area():
        if repeated_is_running.get():
            return ui.div(
                create_loading_state("Running Repeated Measures Analysis..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = repeated_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            return create_results_container(
                "Analysis Results",
                ui.div(
                    ui.navset_tab(
                        ui.nav_panel(
                            "📋 Model Results",
                            ui.div(
                                ui.div(
                                    ui.h5(
                                        f"✅ {res['model_type'].upper()} Analysis Complete"
                                    ),
                                    style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
                                ),
                                ui.output_data_frame("out_rep_results"),
                            ),
                        ),
                        ui.nav_panel(
                            "📈 Trajectory Plot", ui.output_ui("out_rep_plot")
                        ),
                    ),
                    ui.hr(),
                    ui.HTML(
                        create_missing_data_report_html(
                            res.get("missing_data_info", {}), var_meta.get() or {}
                        )
                    ),
                ),
                class_="fade-in-entry",
            )

        # Default Placeholder
        return create_empty_state_ui(
            message="No Repeated Measures Results",
            sub_message="Configure Outcome, Subject ID, and Time, then click '🚀 Run analysis'.",
            icon="🔄",
        )

    @render.data_frame
    def out_rep_results():
        res = repeated_res.get()
        if res:
            return render.DataGrid(res["results"])
        return None

    @render.ui
    def out_rep_plot():
        res = repeated_res.get()
        if res and res["plot"]:
            return ui.HTML(plotly_figure_to_html(res["plot"], include_plotlyjs="cdn"))
        return ui.div(
            "Run analysis to see trajectory plot.",
            style="color: gray; font-style: italic; padding: 20px; text-align: center;",
        )

    # --- GLM Logic (Tab 2.5) ---
    @render.ui
    def ui_glm_results_area():
        if glm_processing.get():
            return ui.div(
                create_loading_state("Running Generalized Linear Model..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = glm_res.get()
        if res:
            if "error" in res:
                return create_error_alert(res["error"])

            metrics = res["fit_metrics"]
            # We can put the status banner inside the report or as a separate div if needed.
            # But create_results_container is mostly content.
            # Let's include the status banner inside the report panel or forest plot panel?
            # Or just render it on top of the generic content.
            # But ui_glm_results_area is inside create_results_container.

            return ui.div(
                ui.navset_tab(
                    ui.nav_panel(
                        "📋 Model Results",
                        ui.div(
                            ui.div(
                                ui.h5(
                                    f"✅ Analysis Complete (AIC: {metrics.get('aic', 'N/A'):.2f}, Deviance: {metrics.get('deviance', 'N/A'):.2f})"
                                ),
                                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
                            ),
                            ui.HTML(res["html_report"]),
                        ),
                    ),
                    ui.nav_panel(
                        "🌳 Forest Plot",
                        (
                            ui.HTML(
                                plotly_figure_to_html(
                                    res["forest_plot"], include_plotlyjs="cdn"
                                )
                            )
                            if res.get("forest_plot")
                            else ui.div(
                                "No forest plot available", class_="text-muted p-3"
                            )
                        ),
                    ),
                ),
                class_="fade-in-entry",
            )

        return create_placeholder_state(
            "Select an outcome and predictors, then click 'Run GLM'.", icon="📈"
        )

    @reactive.Effect
    @reactive.event(input.btn_run_glm)
    def _run_glm():
        d = current_df()
        req(d is not None, input.glm_outcome(), input.glm_predictors())

        glm_processing.set(True)
        glm_res.set(None)

        try:
            # Prepare Data
            outcome = input.glm_outcome()
            predictors = list(input.glm_predictors())
            interactions = list(input.glm_interactions())

            # Simple interaction handling (create dummy cols in a copy)
            X = d[predictors].copy()
            y = pd.to_numeric(d[outcome], errors="coerce")

            # Create interactions if any
            if interactions:
                for pair_str in interactions:
                    if " × " in pair_str:
                        v1, v2 = pair_str.split(" × ")
                        if v1 in d.columns and v2 in d.columns:
                            # Convert to numeric for interaction
                            p1 = pd.to_numeric(d[v1], errors="coerce")
                            p2 = pd.to_numeric(d[v2], errors="coerce")
                            col_name = f"{v1}:{v2}"
                            X[col_name] = p1 * p2

            # Drop Data with NaNs
            valid_idx = y.notna() & X.notna().all(axis=1)
            y = y[valid_idx]
            X = X[valid_idx]

            # Run GLM
            params, conf_int, pvalues, status, metrics = run_glm(
                y, X, family_name=input.glm_family(), link_name=input.glm_link()
            )

            if status != "OK":
                ui.notification_show(f"GLM Failed: {status}", type="error")
                return

            # Format Results for Forest Plot & Table
            res_df = pd.DataFrame(
                {
                    "var": params.index,
                    "coef": params.values,
                    "ci_low": conf_int[0].values,
                    "ci_high": conf_int[1].values,
                    "p_value": pvalues.values,
                }
            )

            # Exclude constant from forest plot usually
            plot_df = res_df[res_df["var"] != "const"]

            # Generate Forest Plot
            forest_data = []
            link = input.glm_link()
            is_ratio = link in ["log", "logit", "cloglog"]

            for _, row in plot_df.iterrows():
                val = np.exp(row["coef"]) if is_ratio else row["coef"]
                low = np.exp(row["ci_low"]) if is_ratio else row["ci_low"]
                high = np.exp(row["ci_high"]) if is_ratio else row["ci_high"]

                forest_data.append(
                    {
                        "label": row["var"],
                        "mean": val,
                        "lower": low,
                        "upper": high,
                        "p_value": row["p_value"],
                        "is_ratio": is_ratio,
                    }
                )

            try:
                forest_df = pd.DataFrame(forest_data)
                fig = create_forest_plot(
                    forest_df,
                    estimate_col="mean",
                    ci_low_col="lower",
                    ci_high_col="upper",
                    label_col="label",
                    pval_col="p_value",
                    title=f"GLM ({input.glm_family()}/{input.glm_link()}) Results",
                    x_label="Exp(Coef) [OR/RR]" if is_ratio else "Coefficient",
                )
            except ValueError as e:
                logger.warning("GLM Forest Plot creation failed: %s", e)
                fig = None

            # Generate HTML Report
            html_parts = [
                f"<h4>GLM Results: {outcome}</h4>",
                f"<p><b>Family:</b> {input.glm_family()} | <b>Link:</b> {input.glm_link()}</p>",
                f"<p><b>AIC:</b> {metrics.get('aic', 'N/A'):.2f} | <b>Deviance:</b> {metrics.get('deviance', 'N/A'):.2f}</p>",
                "<table class='table table-striped table-sm'>",
                "<thead><tr><th>Variable</th><th>Coef</th><th>Exp(Coef)</th><th>95% CI</th><th>P-value</th></tr></thead>",
                "<tbody>",
            ]

            for _, row in res_df.iterrows():
                coef = row["coef"]
                exp_coef = np.exp(coef)  # Always show exp coef for reference
                ci_l = row["ci_low"]
                ci_h = row["ci_high"]
                p = row["p_value"]

                p_fmt = format_p_value(p)
                p_style = "color:red; font-weight:bold;" if p < 0.05 else ""

                # CI Display based on link
                if is_ratio:
                    ci_disp = PublicationFormatter.format_ci(np.exp(ci_l), np.exp(ci_h))
                else:
                    ci_disp = PublicationFormatter.format_ci(ci_l, ci_h)

                html_parts.append(
                    f"<tr>"
                    f"<td>{row['var']}</td>"
                    f"<td>{coef:.3f}</td>"
                    f"<td>{exp_coef:.3f}</td>"
                    f"<td>{ci_disp}</td>"
                    f"<td style='{p_style}'>{p_fmt}</td>"
                    f"</tr>"
                )
            html_parts.append("</tbody></table>")

            glm_res.set(
                {
                    "fit_metrics": metrics,
                    "params": params,
                    "forest_plot": fig,
                    "html_report": "".join(html_parts),
                }
            )

        except Exception as e:
            err_msg = f"Error running GLM: {e!s}"
            glm_res.set({"error": err_msg})
            ui.notification_show("GLM Failed", type="error")
            logger.exception("GLM Fatal Error")

        finally:
            glm_processing.set(False)

    @render.download(filename="glm_report.html")
    def btn_dl_glm_report():
        res = glm_res.get()
        if res and "html_report" in res:
            yield res["html_report"]

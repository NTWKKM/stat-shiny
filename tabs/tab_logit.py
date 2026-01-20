from __future__ import annotations

import gc
import html
import json
from itertools import combinations, islice

# Use built-in list/dict/tuple for Python 3.9+ and typing for complex types
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from htmltools import HTML, div
from shiny import module, reactive, render, req, ui

from config import CONFIG
from logger import get_logger
from tabs._common import get_color_palette
from utils.forest_plot_lib import create_forest_plot
from utils.formatting import create_missing_data_report_html
from utils.linear_lib import (
    analyze_linear_outcome,
    bootstrap_ols,
    compare_models,
    format_bootstrap_results,
    format_model_comparison,
    format_stepwise_history,
    stepwise_selection,
)
from utils.logic import analyze_outcome, run_glm
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.poisson_lib import analyze_poisson_outcome
from utils.repeated_measures_lib import (
    create_trajectory_plot,
    extract_model_results,
    run_gee,
    run_lmm,
)
from utils.subgroup_analysis_module import SubgroupAnalysisLogit, SubgroupResult

# Import internal modules


logger = get_logger(__name__)
COLORS = get_color_palette()


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
def logit_ui() -> ui.TagChild:
    """
    Constructs the main UI for the regression module, providing controls and result panels for logistic, Poisson, and subgroup analyses.

    Returns:
        ui.TagChild: A UI fragment containing dataset selectors and info, tabbed panels for
        Binary Logistic Regression, Poisson Regression, Subgroup Analysis, and Reference,
        each with controls (variable selection, method/settings, exclusions, interactions),
        run/download actions, and result panels for forest plots and detailed reports.
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
            # TAB 1: Binary Logistic Regression
            # =====================================================================
            ui.nav_panel(
                "📈 Binary Logistic Regression",
                # Control section (top)
                ui.card(
                    ui.card_header("📈 Analysis Options"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Variable Selection:"),
                            ui.input_select(
                                "sel_outcome", "Select Outcome (Y):", choices=[]
                            ),
                            ui.output_ui("ui_separation_warning"),
                        ),
                        ui.card(
                            ui.card_header("Method & Settings:"),
                            ui.input_radio_buttons(
                                "radio_method",
                                "Regression Method:",
                                {
                                    "auto": "Auto (Recommended)",
                                    "bfgs": "Standard (MLE)",
                                    "firth": "Firth's (Penalized)",
                                },
                            ),
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.h6("Exclude Variables (Optional):"),
                    ui.input_selectize(
                        "sel_exclude", label=None, choices=[], multiple=True
                    ),
                    # Interaction Pairs selector
                    ui.h6("🔗 Interaction Pairs (Optional):"),
                    ui.input_selectize(
                        "sel_interactions",
                        label=None,
                        choices=[],
                        multiple=True,
                        options={
                            "placeholder": "Select variable pairs to test interactions..."
                        },
                    ),
                    ui.p(
                        "💡 Select pairs of variables to test for interaction effects (e.g., 'age × sex')",
                        style="font-size: 0.8em; color: #666; margin-top: 4px;",
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
                ui.navset_tab(
                    ui.nav_panel("🌳 Forest Plots", ui.output_ui("ui_forest_tabs")),
                    ui.nav_panel("📋 Detailed Report", ui.output_ui("out_html_report")),
                ),
            ),
            # =====================================================================
            # TAB 2: Poisson Regression
            # =====================================================================
            ui.nav_panel(
                "📊 Poisson Regression",
                # Control section (top)
                ui.card(
                    ui.card_header("📊 Poisson Analysis Options"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Variable Selection:"),
                            ui.input_select(
                                "poisson_outcome",
                                "Select Count Outcome (Y):",
                                choices=[],
                            ),
                            ui.input_select(
                                "poisson_offset",
                                "Offset Column (Optional):",
                                choices=["None"],
                            ),
                            ui.p(
                                "💡 Offset: Use for rate calculations (e.g., person-years, population)",
                                style="font-size: 0.8em; color: #666; margin-top: 4px;",
                            ),
                        ),
                        ui.card(
                            ui.card_header("Advanced Settings:"),
                            ui.h6("Exclude Variables (Optional):"),
                            ui.input_selectize(
                                "poisson_exclude", label=None, choices=[], multiple=True
                            ),
                        ),
                        col_widths=[6, 6],
                    ),
                    # Interaction Pairs selector
                    ui.h6("🔗 Interaction Pairs (Optional):"),
                    ui.input_selectize(
                        "poisson_interactions",
                        label=None,
                        choices=[],
                        multiple=True,
                        options={
                            "placeholder": "Select variable pairs to test interactions..."
                        },
                    ),
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
                ui.navset_tab(
                    ui.nav_panel(
                        "🌳 Forest Plots", ui.output_ui("ui_poisson_forest_tabs")
                    ),
                    ui.nav_panel(
                        "📋 Detailed Report", ui.output_ui("out_poisson_html_report")
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
            ),
            # =====================================================================
            # TAB 2.5: Generalized Linear Model (GLM)
            # =====================================================================
            ui.nav_panel(
                "📈 Generalized Linear Model",
                ui.card(
                    ui.card_header("📈 GLM Options"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Variable Selection:"),
                            ui.input_select("glm_outcome", "Outcome (Y):", choices=[]),
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
                        ),
                        ui.card(
                            ui.card_header("Predictors:"),
                            ui.input_selectize(
                                "glm_predictors",
                                "Select Predictors (X):",
                                choices=[],
                                multiple=True,
                            ),
                            ui.input_selectize(
                                "glm_interactions",
                                "Interactions (Optional):",
                                choices=[],
                                multiple=True,
                                options={"placeholder": "Select variable pairs..."},
                            ),
                        ),
                        col_widths=[6, 6],
                    ),
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
                ui.output_ui("out_glm_status"),
                ui.navset_tab(
                    ui.nav_panel("📋 Model Results", ui.output_ui("out_glm_report")),
                    ui.nav_panel("🌳 Forest Plot", ui.output_ui("out_glm_forest")),
                ),
            ),
            # =====================================================================
            # TAB 3: Linear Regression (OLS)
            # =====================================================================
            ui.nav_panel(
                "📐 Linear Regression",
                # Control section (top)
                ui.card(
                    ui.card_header("📐 Linear Regression Options"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Variable Selection:"),
                            ui.input_select(
                                "linear_outcome",
                                "Select Continuous Outcome (Y):",
                                choices=[],
                            ),
                            ui.input_selectize(
                                "linear_predictors",
                                "Select Predictors (X):",
                                choices=[],
                                multiple=True,
                                options={
                                    "placeholder": "Select predictors or leave empty for auto-selection..."
                                },
                            ),
                            ui.p(
                                "💡 Leave predictors empty to auto-include all numeric variables",
                                style="font-size: 0.8em; color: #666; margin-top: 4px;",
                            ),
                        ),
                        ui.card(
                            ui.card_header("Method & Settings:"),
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
                        ),
                        col_widths=[6, 6],
                    ),
                    # Advanced Options Accordion
                    ui.accordion(
                        ui.accordion_panel(
                            "🔧 Advanced Options",
                            ui.layout_columns(
                                ui.card(
                                    ui.card_header("Variable Selection:"),
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
                                ),
                                ui.card(
                                    ui.card_header("Bootstrap CI:"),
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
                                ),
                                col_widths=[6, 6],
                            ),
                        ),
                        open=False,
                    ),
                    ui.h6("Exclude Variables (Optional):"),
                    ui.input_selectize(
                        "linear_exclude", label=None, choices=[], multiple=True
                    ),
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
                ui.output_ui("out_linear_status"),
                ui.navset_tab(
                    ui.nav_panel(
                        "📋 Regression Results", ui.output_ui("out_linear_html_report")
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
            ),
            # =====================================================================
            # TAB 4: Subgroup Analysis
            # =====================================================================
            ui.nav_panel(
                "🗣️ Subgroup Analysis",
                # Control section (top)
                ui.card(
                    ui.card_header("🗣️ Subgroup Settings"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Core Variables:"),
                            ui.input_select(
                                "sg_outcome", "Outcome (Binary):", choices=[]
                            ),
                            ui.input_select(
                                "sg_treatment", "Treatment/Exposure:", choices=[]
                            ),
                            ui.input_select("sg_subgroup", "Stratify By:", choices=[]),
                        ),
                        ui.card(
                            ui.card_header("Adjustment & Advanced:"),
                            ui.input_selectize(
                                "sg_adjust",
                                "Adjustment Covariates:",
                                choices=[],
                                multiple=True,
                            ),
                            ui.input_numeric(
                                "sg_min_n", "Min N per subgroup:", value=5, min=2
                            ),
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.accordion(
                        ui.accordion_panel(
                            "✏️ Custom Settings",
                            ui.input_text(
                                "sg_title",
                                "Custom Title:",
                                placeholder="Subgroup Analysis...",
                            ),
                        ),
                        open=False,
                    ),
                    ui.hr(),
                    ui.input_action_button(
                        "btn_run_subgroup",
                        "🚀 Run Subgroup Analysis",
                        class_="btn-primary btn-sm w-100",
                    ),
                ),
                # Content section (bottom)
                ui.output_ui("out_subgroup_status"),
                ui.navset_tab(
                    ui.nav_panel(
                        "🌳 Forest Plot",
                        ui.output_ui("out_sg_forest_plot"),
                        ui.hr(),
                        ui.input_text(
                            "txt_edit_forest_title",
                            "Edit Plot Title:",
                            placeholder="Enter new title...",
                        ),
                        ui.input_action_button(
                            "btn_update_plot_title", "Update Title", class_="btn-sm"
                        ),
                    ),
                    ui.nav_panel(
                        "📂 Summary & Interpretation",
                        ui.layout_columns(
                            ui.value_box(
                                "Overall OR", ui.output_text("val_overall_or")
                            ),
                            ui.value_box(
                                "Overall P-value", ui.output_text("val_overall_p")
                            ),
                            ui.value_box(
                                "Interaction P-value",
                                ui.output_text("val_interaction_p"),
                            ),
                            col_widths=[4, 4, 4],
                        ),
                        ui.hr(),
                        ui.output_ui("out_interpretation_box"),
                        ui.h5("Detailed Results"),
                        ui.output_data_frame("out_sg_table"),
                        ui.output_ui("out_sg_missing_report"),
                    ),
                    ui.nav_panel(
                        "💾 Exports",
                        ui.h5("Download Results"),
                        ui.layout_columns(
                            ui.download_button(
                                "dl_sg_html", "💿 HTML Plot", class_="btn-sm w-100"
                            ),
                            ui.download_button(
                                "dl_sg_csv", "📋 CSV Results", class_="btn-sm w-100"
                            ),
                            ui.download_button(
                                "dl_sg_json", "📁 JSON Data", class_="btn-sm w-100"
                            ),
                            col_widths=[4, 4, 4],
                        ),
                    ),
                ),
            ),
            # =====================================================================
            # TAB 5: Repeated Measures
            # =====================================================================
            ui.nav_panel(
                "🔄 Repeated Measures",
                ui.card(
                    ui.card_header("🔄 GEE & LMM Analysis"),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Variable Selection:"),
                            ui.input_select("rep_outcome", "Outcome (Y):", choices=[]),
                            ui.input_select(
                                "rep_treatment", "Group/Treatment:", choices=[]
                            ),
                            ui.input_select("rep_time", "Time Variable:", choices=[]),
                            ui.input_select("rep_subject", "Subject ID:", choices=[]),
                        ),
                        ui.card(
                            ui.card_header("Model Settings:"),
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
                            ui.h6("Adjustments (Covariates):"),
                            ui.input_selectize(
                                "rep_covariates", label=None, choices=[], multiple=True
                            ),
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.hr(),
                    ui.input_action_button(
                        "btn_run_repeated",
                        "🚀 Run Analysis",
                        class_="btn-primary btn-sm w-100",
                    ),
                ),
                ui.output_ui("out_rep_status"),
                ui.navset_tab(
                    ui.nav_panel(
                        "📋 Model Results", ui.output_data_frame("out_rep_results")
                    ),
                    ui.nav_panel("📈 Trajectory Plot", ui.output_ui("out_rep_plot")),
                ),
            ),
            # =====================================================================
            # TAB 6: Reference
            # =====================================================================
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                ## 📚 Logistic Regression Reference
                
                ### When to Use:
                * Predicting binary outcomes (Disease/No Disease)
                * Understanding risk/protective factors (Odds Ratios)
                * Adjustment for confounders in observational studies
                
                ### Interpretation:
                
                **Odds Ratios (OR):**
                * **OR > 1**: Risk Factor (Increased odds) 🔴
                * **OR < 1**: Protective Factor (Decreased odds) 🟢
                * **OR = 1**: No Effect
                * **CI crosses 1**: Not statistically significant
                
                **Example:**
                * OR = 2.5 (CI 1.2-5.0): Exposure increases odds of outcome by 2.5× (Range: 1.2× to 5×)
                
                ### Regression Methods:
                
                **Standard (MLE)** - Most common
                * Uses Maximum Likelihood Estimation
                * Fast and reliable for most datasets
                * Issues: Perfect separation causes failure
                
                **Firth's (Penalized)** - For separation issues
                * Reduces bias using penalized likelihood
                * Better for rare outcomes or small samples
                * Handles perfect separation well
                
                **Auto** - Recommended
                * Automatically detects separation
                * Uses Firth if needed, Standard otherwise
                
                ### Perfect Separation:
                Occurs when a predictor perfectly predicts the outcome (e.g., all smokers died).
                * **Solution:** Use **Auto** or **Firth's** method, or exclude the variable.
                
                ### Subgroup Analysis:
                * Tests if treatment effect varies by group (Interaction test)
                * **P-interaction < 0.05**: Significant heterogeneity → Report subgroups separately
                * **P-interaction ≥ 0.05**: Homogeneous effect → Report overall effect
                """),
            ),
        ),
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def logit_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[Optional[pd.DataFrame]],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[Optional[pd.DataFrame]],
    is_matched: reactive.Value[bool],
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
    # Store Poisson results: {'html': str, 'fig_adj': FigureWidget, 'fig_crude': FigureWidget}
    poisson_res = reactive.Value(None)
    # Store Linear Regression results: {'html_fragment': str, 'html_full': str, 'plots': dict, 'results': dict}
    linear_res = reactive.Value(None)
    # Store subgroup results: SubgroupResult
    subgroup_res: reactive.Value[Optional[SubgroupResult]] = reactive.Value(None)
    # Store analyzer instance: SubgroupAnalysisLogit
    subgroup_analyzer: reactive.Value[Optional[SubgroupAnalysisLogit]] = reactive.Value(
        None
    )
    # Store Repeated Measures results: {'results': DataFrame, 'plot': Figure, 'model_type': str}
    repeated_res = reactive.Value(None)

    # Store GLM results
    glm_res = reactive.Value(None)
    glm_processing = reactive.Value(False)

    # --- Cache Clearing on Tab Change ---
    @reactive.Effect
    @reactive.event(
        input.btn_run_logit,
        input.btn_run_poisson,
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
    def current_df() -> Optional[pd.DataFrame]:
        if is_matched.get() and input.radio_logit_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_title_with_summary():
        """Display title with dataset summary."""
        d = current_df()
        if d is not None:
            return ui.div(
                ui.h3("📈 Regression Models"),
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
        d = current_df()
        if d is None or d.empty:
            return

        cols = d.columns.tolist()

        # Identify binary columns for outcomes
        binary_cols = [c for c in cols if d[c].nunique() == 2]

        # Identify potential subgroups (2-10 levels)
        sg_cols = [c for c in cols if 2 <= d[c].nunique() <= 10]

        # Update Tab 1 (Binary Logit) Inputs
        ui.update_select("sel_outcome", choices=binary_cols)
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
        ui.update_select("poisson_outcome", choices=count_cols if count_cols else cols)
        ui.update_select("poisson_offset", choices=["None"] + cols)
        ui.update_selectize("poisson_exclude", choices=cols)
        ui.update_selectize("poisson_interactions", choices=interaction_choices[:50])

        # Update Tab 3 (Linear Regression) Inputs
        # Identify continuous numeric columns for outcome
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]
        continuous_cols = [
            c
            for c in numeric_cols
            if d[c].nunique() > 10  # More than 10 unique values suggests continuous
        ]
        # Fall back to all numeric if no continuous found
        linear_outcome_choices = continuous_cols if continuous_cols else numeric_cols
        ui.update_select("linear_outcome", choices=linear_outcome_choices)
        ui.update_selectize("linear_predictors", choices=numeric_cols)
        ui.update_selectize("linear_exclude", choices=cols)

        # Update Tab 4 (Subgroup) Inputs
        ui.update_select("sg_outcome", choices=binary_cols)
        ui.update_select("sg_treatment", choices=cols)
        ui.update_select("sg_subgroup", choices=sg_cols)
        ui.update_selectize("sg_adjust", choices=cols)

        # Update Tab 5 (Repeated Measures) Inputs
        ui.update_select(
            "rep_outcome", choices=numeric_cols
        )  # LMM/GEE(Gaussian) usually numeric outcome
        ui.update_select("rep_treatment", choices=cols)
        ui.update_select("rep_time", choices=numeric_cols)  # Time usually numeric
        ui.update_select("rep_subject", choices=cols)
        ui.update_selectize("rep_covariates", choices=cols)

        # Update Tab 2.5 (GLM) Inputs
        ui.update_select("glm_outcome", choices=cols)
        ui.update_selectize("glm_predictors", choices=cols)
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
                    "Recommendation: Use 'Auto' method or exclude variables.",
                    style="font-size: 0.8em;",
                ),
            )
        return None

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
        interaction_pairs: Optional[list[tuple[str, str]]] = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"Logit: Using {len(interaction_pairs)} interaction pairs")

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Logistic Regression...", detail="Calculating...")

            try:
                # Run Logic from logic.py
                # Note: updated logic.py returns 4 values and html_rep typically includes the plot + css
                html_rep, or_res, aor_res, interaction_res = analyze_outcome(
                    target,
                    final_df,
                    var_meta=var_meta.get(),
                    method=method,
                    interaction_pairs=interaction_pairs,
                    adv_stats=CONFIG,
                )
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
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

            if or_res:
                df_crude = pd.DataFrame(
                    [{"variable": v.get("label", k), **v} for k, v in or_res.items()]
                )
                if not df_crude.empty:
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

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Table + Plots) ---
            # 1. Create Fragment for UI (Table + Plots)
            logit_fragment_html = html_rep

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

    @render.ui
    def out_html_report():
        """
        Render the "Detailed Report" card showing the latest logistic regression HTML fragment or a placeholder.

        Returns:
            ui.card: A UI card containing the report HTML fragment when results are available; otherwise a card with a centered placeholder message prompting the user to run the analysis.
        """
        res = logit_res.get()
        if res:
            return ui.card(
                ui.card_header("📋 Detailed Report"), ui.HTML(res["html_fragment"])
            )
        return ui.card(
            ui.card_header("📋 Detailed Report"),
            ui.div(
                "Run analysis to see detailed report.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;",
            ),
        )

    @render.ui
    def ui_forest_tabs():
        """
        Render tabbed forest plot panels for the most recent logistic regression results.

        Returns:
            ui.Component: A UI element containing "Crude OR" and/or "Adjusted OR" tabs when corresponding forest figures are present.
            If no analysis has been run, returns a centered placeholder prompting the user to run the analysis.
            If analysis exists but no forest figures are available, returns a muted message indicating no plots are available.
        """
        res = logit_res.get()
        if not res:
            return ui.div(
                "Run analysis to see forest plots.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;",
            )

        tabs = []
        if res["fig_crude"]:
            tabs.append(ui.nav_panel("Crude OR", ui.output_ui("out_forest_crude")))
        if res["fig_adj"]:
            tabs.append(ui.nav_panel("Adjusted OR", ui.output_ui("out_forest_adj")))

        if not tabs:
            return ui.div("No forest plots available.", class_="text-muted")
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
        interaction_pairs: Optional[list[tuple[str, str]]] = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"Poisson: Using {len(interaction_pairs)} interaction pairs")

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Poisson Regression...", detail="Calculating...")

            try:
                # Run Poisson Logic
                # Expecting 4 values from the updated poisson_lib.py
                html_rep, irr_res, airr_res, interaction_res = analyze_poisson_outcome(
                    target,
                    final_df,
                    var_meta=var_meta.get(),
                    offset_col=offset,
                    interaction_pairs=interaction_pairs,
                )
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
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

            if irr_res:
                df_crude = pd.DataFrame(
                    [{"variable": k, **v} for k, v in irr_res.items()]
                )
                if not df_crude.empty:
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
    def out_poisson_html_report():
        res = poisson_res.get()
        if res:
            return ui.card(
                ui.card_header("📋 Poisson Regression Report"),
                ui.HTML(res["html_fragment"]),
            )
        return ui.card(
            ui.card_header("📋 Poisson Regression Report"),
            ui.div(
                "Run analysis to see detailed report.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;",
            ),
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
                ui.notification_show(f"Error: {e!s}", type="error")
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

    # --- Render Linear Regression Results ---
    @render.ui
    def out_linear_status():
        res = linear_res.get()
        if res:
            r2 = res["results"].get("r_squared", 0)
            n_obs = res["results"].get("n_obs", 0)
            r2_text = f"R² = {r2:.4f}" if np.isfinite(r2) else "R² = N/A"
            return ui.div(
                ui.h5(f"✅ Linear Regression Complete ({r2_text}, n = {n_obs:,})"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
            )
        return None

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
        history_html = history_df.to_html(
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
        result_html = formatted.to_html(
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
            return f"{res['overall']['p_value']:.4f}"
        return "-"

    @render.text
    def val_interaction_p():
        res = subgroup_res.get()
        if res:
            p_int = res["interaction"]["p_value"]
            return f"{p_int:.4f}" if p_int is not None else "N/A"
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
        res = subgroup_res.get()
        if res:
            # Need to handle numpy types for JSON serialization
            yield json.dumps(res, indent=2, default=str)

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
        df_clean = d.dropna(subset=cols_needed).copy()

        with ui.Progress(min=0, max=1) as p:
            p.set(message=f"Running {model_type.upper()}...", detail="Analyzing...")

            try:
                if model_type == "gee":
                    results = run_gee(
                        df_clean,
                        outcome_col=outcome,
                        treatment_col=treatment,
                        time_col=time_var,
                        subject_col=subject,
                        covariates=covariates,
                        cov_struct=input.rep_cov_struct(),
                        family_str=input.rep_family(),
                    )
                else:  # lmm
                    results = run_lmm(
                        df_clean,
                        outcome_col=outcome,
                        treatment_col=treatment,
                        time_col=time_var,
                        subject_col=subject,
                        covariates=covariates,
                        random_slope=input.rep_random_slope(),
                    )

                # Check for error string
                if isinstance(results, str):
                    ui.notification_show(results, type="error")
                    return

                # Extract Results
                df_res = extract_model_results(results, model_type)

                # Create Plot
                fig = create_trajectory_plot(
                    df_clean,
                    outcome_col=outcome,
                    time_col=time_var,
                    group_col=treatment,
                    subject_col=subject,
                )

                repeated_res.set(
                    {"results": df_res, "plot": fig, "model_type": model_type}
                )

                ui.notification_show(
                    f"✅ {model_type.upper()} Analysis Complete!", type="message"
                )

            except Exception as e:
                ui.notification_show(f"Error: {str(e)}", type="error")
                logger.exception("Repeated measures error")

    @render.ui
    def out_rep_status():
        res = repeated_res.get()
        if res:
            return ui.div(
                ui.h5(f"✅ {res['model_type'].upper()} Analysis Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
            )
        return None

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
    def out_glm_status():
        """Show processing status or success message."""
        if glm_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "Running Generalized Linear Model... Please wait",
                    class_="alert alert-info",
                )
            )
        res = glm_res.get()
        if res and isinstance(res, dict) and "fit_metrics" in res:
            metrics = res["fit_metrics"]
            return ui.div(
                ui.h5(
                    f"✅ Analysis Complete (AIC: {metrics.get('aic', 'N/A'):.2f}, Deviance: {metrics.get('deviance', 'N/A'):.2f})"
                ),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;",
            )
        return None

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

            fig = create_forest_plot(
                forest_data,
                title=f"GLM ({input.glm_family()}/{input.glm_link()}) Results",
                x_label="Exp(Coef) [OR/RR]" if is_ratio else "Coefficient",
            )

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

                p_fmt = f"{p:.4f}" if p >= 0.001 else "<0.001"
                p_style = "color:red; font-weight:bold;" if p < 0.05 else ""

                # CI Display based on link
                if is_ratio:
                    ci_disp = f"{np.exp(ci_l):.3f} - {np.exp(ci_h):.3f}"
                else:
                    ci_disp = f"{ci_l:.3f} - {ci_h:.3f}"

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
            ui.notification_show(f"Error: {str(e)}", type="error")
            logger.exception("GLM Fatal Error")

        finally:
            glm_processing.set(False)

    @render.ui
    def out_glm_report():
        res = glm_res.get()
        if res and "html_report" in res:
            return ui.HTML(res["html_report"])
        return ui.div("Run GLM to see results.", class_="text-secondary p-3")

    @render.ui
    def out_glm_forest():
        res = glm_res.get()
        if res and "forest_plot" in res:
            return ui.HTML(
                plotly_figure_to_html(res["forest_plot"], include_plotlyjs="cdn")
            )
        return ui.div("Run GLM to see plots.", class_="text-secondary p-3")

    @render.download(filename="glm_report.html")
    def btn_dl_glm_report():
        res = glm_res.get()
        if res and "html_report" in res:
            yield res["html_report"]

from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget  # type: ignore
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from htmltools import HTML, div
import gc
from typing import Optional, List, Dict, Any, Tuple, Union, cast

# Import internal modules
from logic import analyze_outcome
from poisson_lib import analyze_poisson_outcome
from forest_plot_lib import create_forest_plot
from subgroup_analysis_module import SubgroupAnalysisLogit, SubgroupResult
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()

# ==============================================================================
# Helper Functions (Pure Logic)
# ==============================================================================
def check_perfect_separation(df: pd.DataFrame, target_col: str) -> List[str]:
    """Identify columns causing perfect separation."""
    risky_vars: List[str] = []
    try:
        y = pd.to_numeric(df[target_col], errors='coerce').dropna()
        if y.nunique() < 2: 
            return []
    except (KeyError, TypeError, ValueError): 
        return []

    for col in df.columns:
        if col == target_col: continue

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
                            ui.input_select("sel_outcome", "Select Outcome (Y):", choices=[]),
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
                                    "firth": "Firth's (Penalized)"
                                }
                            ),
                        ),
                        
                        col_widths=[6, 6]
                    ),
                    
                    ui.h6("Exclude Variables (Optional):"),
                    ui.input_selectize("sel_exclude", label=None, choices=[], multiple=True),
                    
                    # Interaction Pairs selector
                    ui.h6("🔗 Interaction Pairs (Optional):"),
                    ui.input_selectize(
                        "sel_interactions", 
                        label=None, 
                        choices=[], 
                        multiple=True,
                        options={"placeholder": "Select variable pairs to test interactions..."}
                    ),
                    ui.p(
                        "💡 Select pairs of variables to test for interaction effects (e.g., 'age × sex')",
                        style="font-size: 0.8em; color: #666; margin-top: 4px;"
                    ),

                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_logit",
                            "🚀 Run Regression",
                            class_="btn-primary btn-sm w-100"
                        ),
                        ui.download_button(
                            "btn_dl_report",
                            "📥 Download Report",
                            class_="btn-secondary btn-sm w-100"
                        ),
                        col_widths=[6, 6]
                    ),
                ),

                # Content section (bottom)
                ui.output_ui("out_logit_status"),
                ui.navset_card_underline(
                    ui.nav_panel(
                        "🌳 Forest Plots",
                        ui.output_ui("ui_forest_tabs")
                    ),
                    ui.nav_panel(
                        "📋 Detailed Report",
                        ui.output_ui("out_html_report")
                    )
                )
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
                            ui.input_select("poisson_outcome", "Select Count Outcome (Y):", choices=[]),
                            ui.input_select("poisson_offset", "Offset Column (Optional):", choices=["None"]),
                            ui.p(
                                "💡 Offset: Use for rate calculations (e.g., person-years, population)",
                                style="font-size: 0.8em; color: #666; margin-top: 4px;"
                            ),
                        ),

                        ui.card(
                            ui.card_header("Advanced Settings:"),
                            ui.h6("Exclude Variables (Optional):"),
                            ui.input_selectize("poisson_exclude", label=None, choices=[], multiple=True),
                        ),
                        
                        col_widths=[6, 6]
                    ),
                    
                    # Interaction Pairs selector
                    ui.h6("🔗 Interaction Pairs (Optional):"),
                    ui.input_selectize(
                        "poisson_interactions", 
                        label=None, 
                        choices=[], 
                        multiple=True,
                        options={"placeholder": "Select variable pairs to test interactions..."}
                    ),

                    ui.hr(),

                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_poisson",
                            "🚀 Run Poisson Regression",
                            class_="btn-primary btn-sm w-100"
                        ),
                        ui.download_button(
                            "btn_dl_poisson_report",
                            "📥 Download Report",
                            class_="btn-secondary btn-sm w-100"
                        ),
                        col_widths=[6, 6]
                    ),
                ),

                # Content section (bottom)
                ui.output_ui("out_poisson_status"),
                ui.navset_card_underline(
                    ui.nav_panel(
                        "🌳 Forest Plots",
                        ui.output_ui("ui_poisson_forest_tabs")
                    ),
                    ui.nav_panel(
                        "📋 Detailed Report",
                        ui.output_ui("out_poisson_html_report")
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
                        """)
                    )
                )
            ),

            # =====================================================================
            # TAB 3: Subgroup Analysis
            # =====================================================================
            ui.nav_panel(
                "🗣️ Subgroup Analysis",
                
                # Control section (top)
                ui.card(
                    ui.card_header("🗣️ Subgroup Settings"),
                    
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Core Variables:"),
                            ui.input_select("sg_outcome", "Outcome (Binary):", choices=[]),
                            ui.input_select("sg_treatment", "Treatment/Exposure:", choices=[]),
                            ui.input_select("sg_subgroup", "Stratify By:", choices=[]),
                        ),
                        
                        ui.card(
                            ui.card_header("Adjustment & Advanced:"),
                            ui.input_selectize("sg_adjust", "Adjustment Covariates:", choices=[], multiple=True),
                            ui.input_numeric("sg_min_n", "Min N per subgroup:", value=5, min=2),
                        ),
                        
                        col_widths=[6, 6]
                    ),
                    
                    ui.accordion(
                        ui.accordion_panel(
                            "✏️ Custom Settings",
                            ui.input_text("sg_title", "Custom Title:", placeholder="Subgroup Analysis..."),
                        ),
                        open=False
                    ),
                    
                    ui.hr(),
                    
                    ui.input_action_button(
                        "btn_run_subgroup",
                        "🚀 Run Subgroup Analysis",
                        class_="btn-primary btn-sm w-100"
                    ),
                ),
                
                # Content section (bottom)
                ui.output_ui("out_subgroup_status"),
                ui.navset_card_underline(
                    ui.nav_panel(
                        "🌳 Forest Plot",
                        output_widget("out_sg_forest_plot"),
                        ui.hr(),
                        ui.input_text("txt_edit_forest_title", "Edit Plot Title:", placeholder="Enter new title..."),
                        ui.input_action_button("btn_update_plot_title", "Update Title", class_="btn-sm"),
                    ),
                    ui.nav_panel(
                        "📂 Summary & Interpretation",
                        ui.layout_columns(
                            ui.value_box("Overall OR", ui.output_text("val_overall_or")),
                            ui.value_box("Overall P-value", ui.output_text("val_overall_p")),
                            ui.value_box("Interaction P-value", ui.output_text("val_interaction_p")),
                            col_widths=[4, 4, 4]
                        ),
                        ui.hr(),
                        ui.output_ui("out_interpretation_box"),
                        ui.h5("Detailed Results"),
                        ui.output_data_frame("out_sg_table")
                    ),
                    ui.nav_panel(
                        "💾 Exports",
                        ui.h5("Download Results"),
                        ui.layout_columns(
                            ui.download_button("dl_sg_html", "📐 HTML Plot", class_="btn-sm w-100"),
                            ui.download_button("dl_sg_csv", "📋 CSV Results", class_="btn-sm w-100"),
                            ui.download_button("dl_sg_json", "📁 JSON Data", class_="btn-sm w-100"),
                            col_widths=[4, 4, 4]
                        )
                    )
                )
            ),

            # =====================================================================
            # TAB 4: Reference
            # =====================================================================
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                ## 📚 Logistic Regression Reference
                
                ### When to Use:
                * Predicting binary outcomes (Disease/No Disease)
                * Understanding risk/protective factors
                * Calculating probabilities for classifications
                
                ### Interpretation:
                
                **Odds Ratio (OR):**
                * **OR > 1**: Increased odds of outcome (Risk factor) 🔴
                * **OR < 1**: Decreased odds of outcome (Protective) 🟢
                * **OR = 1**: No association
                * **95% CI**: If CI excludes 1.0, result is statistically significant at p<0.05
                
                **P-value:**
                * **p < 0.05**: Statistically significant association
                * **p ≥ 0.05**: Not statistically significant
                
                ### Model Performance:
                * **Sensitivity**: Ability to correctly identify cases (Disease+)
                * **Specificity**: Ability to correctly identify non-cases (Disease-)
                * **AUC**: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
                * **AIC/BIC**: Model comparison (lower is better)
                
                ### Key Assumptions:
                * Binary outcome variable
                * Linear relationship between log-odds and predictors
                * No perfect separation
                * Independence of observations
                
                ### Methods:
                * **Standard (MLE)**: Standard Maximum Likelihood Estimation
                * **Firth's**: Penalized MLE (handles separation better)
                """)
            ),
        ),
    )

# ==============================================================================
# Server Logic (Placeholder - Implementation depends on your logic modules)
# ==============================================================================
@module.server
def logit_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[Optional[pd.DataFrame]],
    var_meta: reactive.Value[Dict[str, Any]],
    df_matched: reactive.Value[Optional[pd.DataFrame]],
    is_matched: reactive.Value[bool],
) -> None:
    pass

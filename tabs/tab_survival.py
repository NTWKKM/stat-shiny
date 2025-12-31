"""
⛳ Survival Analysis Module (Shiny)

Provides UI and server logic for:
- Kaplan-Meier curves with log-rank tests
- Nelson-Aalen cumulative hazard curves
- Landmark analysis for late endpoints
- Cox proportional hazards regression
- Subgroup analysis for treatment heterogeneity
"""

from shiny import ui, reactive, render, Session
import pandas as pd
import numpy as np
import survival_lib
from subgroup_analysis_module import SubgroupAnalysisCox
from logger import get_logger
import json

logger = get_logger(__name__)


def _get_dataset_for_survival(df: pd.DataFrame, df_matched: reactive.Value, is_matched: reactive.Value) -> tuple[pd.DataFrame, str]:
    """
    Choose between original and matched datasets for survival analysis.
    
    Args:
        df: Original dataset
        df_matched: Reactive value containing matched data
        is_matched: Reactive value indicating if matching has been done
        
    Returns:
        Tuple of (selected_dataframe, label_string)
    """
    if is_matched.get() and df_matched.get() is not None:
        return df_matched.get().copy(), f"✅ Matched Data ({len(df_matched.get())} rows)"
    else:
        return df, f"📊 Original Data ({len(df)} rows)"


def surv_ui(namespace: str) -> ui.TagChild:
    """
    Create the UI for survival analysis tab.
    
    Args:
        namespace: Shiny module namespace ID
        
    Returns:
        UI elements for the survival tab
    """
    return ui.navset_tab(
        # TAB 1: Survival Curves (KM & Nelson-Aalen)
        ui.nav_panel(
            "📈 Survival Curves",
            ui.card(
                ui.card_header("Kaplan-Meier & Nelson-Aalen Curves"),
                
                ui.layout_columns(
                    ui.input_select(
                        f"{namespace}_surv_time",
                        "⛳ Time Variable:",
                        choices=["Select..."]
                    ),
                    ui.input_select(
                        f"{namespace}_surv_event",
                        "🗣️ Event Variable (1=Event):",
                        choices=["Select..."]
                    ),
                    ui.input_select(
                        f"{namespace}_surv_group",
                        "Compare Groups (Optional):",
                        choices=["None"]
                    ),
                    col_widths=[4, 4, 4]
                ),
                
                ui.input_radio_buttons(
                    f"{namespace}_plot_type",
                    "Select Plot Type:",
                    choices={
                        "km": "Kaplan-Meier (Survival Function)",
                        "na": "Nelson-Aalen (Cumulative Hazard)"
                    },
                    selected="km",
                    inline=True
                ),
                
                ui.layout_columns(
                    ui.input_action_button(
                        f"{namespace}_btn_run_curves",
                        "🚀 Generate Curve",
                        class_="btn-primary",
                        width="100%"
                    ),
                    ui.input_action_button(
                        f"{namespace}_btn_dl_curves",
                        "📥 Download Report",
                        class_="btn-secondary",
                        width="100%"
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.output_ui(f"{namespace}_out_curves_result"),
                
                full_screen=True
            )
        ),
        
        # TAB 2: Landmark Analysis
        ui.nav_panel(
            "📊 Landmark Analysis",
            ui.card(
                ui.card_header("Landmark Analysis for Late Endpoints"),
                
                ui.markdown("**Principle:** Exclude patients with event/censoring before landmark time."),
                
                ui.input_slider(
                    f"{namespace}_landmark_t",
                    "Landmark Time (t):",
                    min=0, max=100, value=10, step=1
                ),
                
                ui.input_select(
                    f"{namespace}_landmark_group",
                    "Compare Group:",
                    choices=["Select..."]
                ),
                
                ui.layout_columns(
                    ui.input_action_button(
                        f"{namespace}_btn_run_landmark",
                        "🚀 Run Landmark Analysis",
                        class_="btn-primary",
                        width="100%"
                    ),
                    ui.input_action_button(
                        f"{namespace}_btn_dl_landmark",
                        "📥 Download Report",
                        class_="btn-secondary",
                        width="100%"
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.output_ui(f"{namespace}_out_landmark_result"),
                
                full_screen=True
            )
        ),
        
        # TAB 3: Cox Regression
        ui.nav_panel(
            "📈 Cox Regression",
            ui.card(
                ui.card_header("Cox Proportional Hazards Regression"),
                
                ui.input_checkbox_group(
                    f"{namespace}_cox_covariates",
                    "Select Covariates (Predictors):",
                    choices=[],
                    selected=[]
                ),
                
                ui.layout_columns(
                    ui.input_action_button(
                        f"{namespace}_btn_run_cox",
                        "🚀 Run Cox Model",
                        class_="btn-primary",
                        width="100%"
                    ),
                    ui.input_action_button(
                        f"{namespace}_btn_dl_cox",
                        "📥 Download Report",
                        class_="btn-secondary",
                        width="100%"
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.output_ui(f"{namespace}_out_cox_result"),
                
                full_screen=True
            )
        ),
        
        # TAB 4: Subgroup Analysis
        ui.nav_panel(
            "📛 Subgroup Analysis",
            ui.card(
                ui.card_header("Cox Subgroup Analysis - Treatment Heterogeneity"),
                
                ui.layout_columns(
                    ui.input_select(
                        f"{namespace}_sg_time",
                        "Follow-up Time:",
                        choices=["Select..."]
                    ),
                    ui.input_select(
                        f"{namespace}_sg_event",
                        "Event Indicator (Binary):",
                        choices=["Select..."]
                    ),
                    ui.input_select(
                        f"{namespace}_sg_treatment",
                        "Treatment/Exposure:",
                        choices=["Select..."]
                    ),
                    col_widths=[4, 4, 4]
                ),
                
                ui.layout_columns(
                    ui.input_select(
                        f"{namespace}_sg_subgroup",
                        "📌 Stratify By:",
                        choices=["Select..."]
                    ),
                    ui.input_checkbox_group(
                        f"{namespace}_sg_adjust",
                        "Adjustment Variables:",
                        choices=[]
                    ),
                    col_widths=[4, 8]
                ),
                
                # FIXED: Changed ui.details to ui.tags.details
                ui.tags.details(
                    ui.tags.summary("⚠️ Advanced Settings"),
                    ui.input_numeric(
                        f"{namespace}_sg_min_n",
                        "Min N per subgroup:",
                        value=5, min=2, max=50
                    ),
                    ui.input_numeric(
                        f"{namespace}_sg_min_events",
                        "Min events per subgroup:",
                        value=2, min=1, max=50
                    )
                ),
                
                ui.input_action_button(
                    f"{namespace}_btn_run_sg",
                    "🚀 Run Subgroup Analysis",
                    class_="btn-primary",
                    width="100%"
                ),
                
                ui.output_ui(f"{namespace}_out_sg_result"),
                
                full_screen=True
            )
        ),
        
        # TAB 5: Reference & Interpretation
        ui.nav_panel(
            "ℹ️ Reference",
            ui.card(
                ui.card_header("📚 Quick Reference: Survival Analysis"),
                
                ui.markdown("""
                ### 🎲 When to Use What:
                
                | Method | Purpose | Output |
                |--------|---------|--------|
                | **KM Curves** | Visualize time-to-event by group | Survival %, median, p-value |
                | **Nelson-Aalen** | Cumulative hazard over time | H(t) curve, risk accumulation |
                | **Landmark** | Late/surrogate endpoints | Filtered KM, immortal time removed |
                | **Cox** | Multiple predictors of survival | HR, CI, p-value per variable + forest plot |
                | **Subgroup Analysis** | Treatment effect heterogeneity | HR by subgroup, interaction test |
                """)
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Kaplan-Meier (KM) Curves"),
                    ui.markdown("""
                    **When to Use:**
                    - Time-to-event analysis (survival, recurrence)
                    - Comparing survival between groups
                    - Estimating survival at fixed times
                    
                    **Interpretation:**
                    - Y-axis = % surviving
                    - X-axis = time
                    - Step down = event occurred
                    - Median survival = 50% point
                    
                    **Log-Rank Test:**
                    - p < 0.05: Curves differ ✅
                    - p ≥ 0.05: No difference ⚠️
                    """)
                ),
                ui.card(
                    ui.card_header("Nelson-Aalen Curves"),
                    ui.markdown("""
                    **When to Use:**
                    - Estimate cumulative hazard function H(t)
                    - Compare risk accumulation between groups
                    - Assess hazard rate over time
                    
                    **Interpretation:**
                    - Y-axis = cumulative hazard
                    - X-axis = time
                    - Steeper slope = higher hazard
                    - Useful for checking PH assumption
                    
                    **vs KM:**
                    - NA is more direct from hazard
                    - KM shows 1 - S(t)
                    """)
                ),
                col_widths=[6, 6]
            )
        )
    )


def surv_server(namespace: str, df: reactive.Value, var_meta: reactive.Value,
               df_matched: reactive.Value, is_matched: reactive.Value):
    """
    Server logic for survival analysis module.
    
    Args:
        namespace: Module namespace ID
        df: Reactive value with original dataframe
        var_meta: Reactive value with variable metadata
        df_matched: Reactive value with matched dataframe
        is_matched: Reactive value indicating if matching completed
    """
    
    # ==================== REACTIVE STATES ====================
    
    surv_df_current = reactive.Value(None)  # Current dataset (original or matched)
    curves_result = reactive.Value(None)    # KM/NA analysis result
    landmark_result = reactive.Value(None)  # Landmark analysis result
    cox_result = reactive.Value(None)       # Cox regression result
    
    # ==================== UPDATE DATASETS ====================
    
    @reactive.Effect
    def _update_current_dataset():
        """Update current dataset when data or matching status changes."""
        data = df.get()
        if data is not None:
            selected_df, _ = _get_dataset_for_survival(data, df_matched, is_matched)
            surv_df_current.set(selected_df)
            
            # Update variable selects
            cols = data.columns.tolist()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            ui.update_select(f"{namespace}_surv_time", choices=numeric_cols)
            ui.update_select(f"{namespace}_surv_event", choices=cols)
            ui.update_select(f"{namespace}_surv_group", choices=["None"] + cols)
            ui.update_select(f"{namespace}_sg_time", choices=numeric_cols)
            ui.update_select(f"{namespace}_sg_event", choices=cols)
            ui.update_select(f"{namespace}_sg_treatment", choices=cols)
            ui.update_select(f"{namespace}_sg_subgroup", choices=cols)
            ui.update_checkbox_group(f"{namespace}_sg_adjust", choices=cols)
            ui.update_checkbox_group(f"{namespace}_cox_covariates", choices=cols)
            ui.update_select(f"{namespace}_landmark_group", choices=cols)
    
    # ==================== CURVES (KM/NA) ====================
    
    @render.ui
    def out_curves_result():
        """Display curves analysis results."""
        result = curves_result.get()
        if result is None:
            return ui.markdown("*Results will appear here after clicking 'Generate Curve'*")
        
        return ui.card(
            ui.card_header("📈 Results"),
            ui.output_plot(f"{namespace}_out_curves_plot"),
            ui.card_header("📄 Statistics"),
            ui.output_data_frame(f"{namespace}_out_curves_table")
        )
    
    @render.plot
    def out_curves_plot():
        """Render curves plot."""
        result = curves_result.get()
        if result is None or result['fig'] is None:
            return None
        return result['fig']
    
    @render.data_frame
    def out_curves_table():
        """Render curves statistics table."""
        result = curves_result.get()
        if result is None or result['stats'] is None:
            return None
        return render.DataGrid(result['stats'])
    
    # ==================== LANDMARK ANALYSIS ====================
    
    @render.ui
    def out_landmark_result():
        """Display landmark analysis results."""
        result = landmark_result.get()
        if result is None:
            return ui.markdown("*Results will appear here after clicking 'Run Landmark Analysis'*")
        
        return ui.card(
            ui.card_header("📈 Landmark Results"),
            ui.markdown(f"""
            **Total N before filter:** {result['n_pre']}
            **N Included (Survived >= {result['t']:.2f}):** {result['n_post']}
            **N Excluded:** {result['n_pre'] - result['n_post']}
            """),
            ui.output_plot(f"{namespace}_out_landmark_plot"),
            ui.card_header("📄 Log-Rank Test"),
            ui.output_data_frame(f"{namespace}_out_landmark_table")
        )
    
    @render.plot
    def out_landmark_plot():
        """Render landmark analysis plot."""
        result = landmark_result.get()
        if result is None or result['fig'] is None:
            return None
        return result['fig']
    
    @render.data_frame
    def out_landmark_table():
        """Render landmark statistics table."""
        result = landmark_result.get()
        if result is None or result['stats'] is None:
            return None
        return render.DataGrid(result['stats'])
    
    # ==================== COX REGRESSION ====================
    
    @render.ui
    def out_cox_result():
        """Display cox regression results."""
        result = cox_result.get()
        if result is None:
            return ui.markdown("*Results will appear here after clicking 'Run Cox Model'*")
        
        return ui.card(
            ui.card_header("📄 Cox Model Results"),
            ui.output_data_frame(f"{namespace}_out_cox_table"),
            ui.card_header("🌳 Forest Plot (Hazard Ratios)"),
            ui.output_plot(f"{namespace}_out_cox_forest"),
            ui.card_header("🔍 PH Assumption Check"),
            ui.output_ui(f"{namespace}_out_cox_assumptions")
        )
    
    @render.data_frame
    def out_cox_table():
        """Render cox results table."""
        result = cox_result.get()
        if result is None or result['results_df'] is None:
            return None
        return render.DataGrid(result['results_df'])
    
    @render.plot
    def out_cox_forest():
        """Render forest plot."""
        result = cox_result.get()
        if result is None or result['forest_fig'] is None:
            return None
        return result['forest_fig']
    
    @render.ui
    def out_cox_assumptions():
        """Display assumptions checking results."""
        result = cox_result.get()
        if result is None:
            return None
        
        ui_elements = []
        
        if result.get('assumptions_text'):
            ui_elements.append(
                # FIXED: Changed ui.details to ui.tags.details
                ui.tags.details(
                    ui.tags.summary("View Assumption Advice (Text)"),
                    ui.markdown(f"```\n{result['assumptions_text']}\n```")
                )
            )
        
        if result.get('assumptions_plots'):
            ui_elements.append(ui.markdown("**Schoenfeld Residuals Plots:**"))
            for fig in result['assumptions_plots']:
                ui_elements.append(ui.output_plot(f"{namespace}_out_assumption_plot_{id(fig)}"))
        
        return ui.card(*ui_elements) if ui_elements else None
    
    # ==================== SUBGROUP ANALYSIS ====================
    
    @render.ui
    def out_sg_result():
        """Display subgroup analysis results."""
        # This will be rendered after analysis completes
        return ui.card(
            ui.markdown("*Subgroup analysis results will appear here*")
        )


# ==================== MODULE EXPORT ====================

def survival_ui(namespace: str) -> ui.TagChild:
    """Wrapper for compatibility."""
    return surv_ui(namespace)


def survival_server(namespace: str, df: reactive.Value, var_meta: reactive.Value,
                   df_matched: reactive.Value, is_matched: reactive.Value):
    """Wrapper for compatibility."""
    return surv_server(namespace, df, var_meta, df_matched, is_matched)

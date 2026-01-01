"""
⛳ Survival Analysis Module (Shiny) - FIXED & OPTIMIZED

Provides UI and server logic for:
- Kaplan-Meier curves with log-rank tests
- Nelson-Aalen cumulative hazard curves
- Landmark analysis for late endpoints
- Cox proportional hazards regression
- Subgroup analysis for treatment heterogeneity
"""

from shiny import ui, reactive, render, req
from shinywidgets import output_widget, render_widget  # ✅ IMPORT REQUIRED
import pandas as pd
import numpy as np
import survival_lib
# ตรวจสอบว่ามี module นี้จริงหรือไม่ ถ้าไม่มีให้ comment ออก
try:
    from subgroup_analysis_module import SubgroupAnalysisCox
except ImportError:
    SubgroupAnalysisCox = None
    
from logger import get_logger

logger = get_logger(__name__)

# ==============================================================================
# Helper Function
# ==============================================================================
def _get_dataset_for_survival(df: pd.DataFrame, df_matched: reactive.Value, is_matched: reactive.Value) -> tuple[pd.DataFrame, str]:
    if is_matched.get() and df_matched.get() is not None:
        return df_matched.get().copy(), f"✅ Matched Data ({len(df_matched.get())} rows)"
    else:
        return df, f"📊 Original Data ({len(df)} rows)"

# ==============================================================================
# UI Definition
# ==============================================================================
def surv_ui(namespace: str) -> ui.TagChild:
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
                        class_="btn-primary w-100",
                    ),
                    ui.input_action_button(
                        f"{namespace}_btn_dl_curves",
                        "📥 Download Report",
                        class_="btn-secondary w-100",
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
                        class_="btn-primary w-100",
                    ),
                    ui.input_action_button(
                        f"{namespace}_btn_dl_landmark",
                        "📥 Download Report",
                        class_="btn-secondary w-100",
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
                        class_="btn-primary w-100",
                    ),
                    ui.input_action_button(
                        f"{namespace}_btn_dl_cox",
                        "📥 Download Report",
                        class_="btn-secondary w-100",
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
                        f"{namespace}_sg_time", "Follow-up Time:", choices=["Select..."]
                    ),
                    ui.input_select(
                        f"{namespace}_sg_event", "Event Indicator (Binary):", choices=["Select..."]
                    ),
                    ui.input_select(
                        f"{namespace}_sg_treatment", "Treatment/Exposure:", choices=["Select..."]
                    ),
                    col_widths=[4, 4, 4]
                ),
                
                ui.layout_columns(
                    ui.input_select(
                        f"{namespace}_sg_subgroup", "📌 Stratify By:", choices=["Select..."]
                    ),
                    ui.input_checkbox_group(
                        f"{namespace}_sg_adjust", "Adjustment Variables:", choices=[]
                    ),
                    col_widths=[4, 8]
                ),
                
                ui.accordion(
                    ui.accordion_panel(
                        "⚠️ Advanced Settings",
                        ui.input_numeric(
                            f"{namespace}_sg_min_n", "Min N per subgroup:", value=5, min=2, max=50
                        ),
                        ui.input_numeric(
                            f"{namespace}_sg_min_events", "Min events per subgroup:", value=2, min=1, max=50
                        )
                    ),
                    open=False
                ),
                
                ui.input_action_button(
                    f"{namespace}_btn_run_sg",
                    "🚀 Run Subgroup Analysis",
                    class_="btn-primary w-100"
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
            )
        )
    )

# ==============================================================================
# Server Logic
# ==============================================================================
def surv_server(namespace: str, df: reactive.Value, var_meta: reactive.Value,
                df_matched: reactive.Value, is_matched: reactive.Value):
    
    # ==================== REACTIVE VALUES ====================
    surv_df_current = reactive.Value(None)
    curves_result = reactive.Value(None)
    landmark_result = reactive.Value(None)
    cox_result = reactive.Value(None)
    sg_result = reactive.Value(None)
    
    # ==================== DATASET UPDATES ====================
    @reactive.Effect
    def _update_current_dataset():
        data = df.get()
        if data is not None:
            selected_df, _ = _get_dataset_for_survival(data, df_matched, is_matched)
            surv_df_current.set(selected_df)
            
            # Update Dropdowns
            cols = data.columns.tolist()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # KM
            ui.update_select(f"{namespace}_surv_time", choices=numeric_cols)
            ui.update_select(f"{namespace}_surv_event", choices=cols)
            ui.update_select(f"{namespace}_surv_group", choices=["None"] + cols)
            
            # Landmark
            ui.update_select(f"{namespace}_landmark_group", choices=cols)
            
            # Cox
            ui.update_checkbox_group(f"{namespace}_cox_covariates", choices=cols)
            
            # Subgroup
            ui.update_select(f"{namespace}_sg_time", choices=numeric_cols)
            ui.update_select(f"{namespace}_sg_event", choices=cols)
            ui.update_select(f"{namespace}_sg_treatment", choices=cols)
            ui.update_select(f"{namespace}_sg_subgroup", choices=cols)
            ui.update_checkbox_group(f"{namespace}_sg_adjust", choices=cols)

    # ==================== 1. CURVES LOGIC (KM/NA) ====================
    @reactive.Effect
    @reactive.event(input[f"{namespace}_btn_run_curves"])
    def _run_curves():
        data = surv_df_current.get()
        time_col = input[f"{namespace}_surv_time"]()
        event_col = input[f"{namespace}_surv_event"]()
        group_col = input[f"{namespace}_surv_group"]()
        plot_type = input[f"{namespace}_plot_type"]()
        
        if data is None or time_col == "Select..." or event_col == "Select...":
            ui.notification_show("Please select Time and Event variables", type="warning")
            return
            
        if group_col == "None": group_col = None
        
        try:
            ui.notification_show("Generating curves...", duration=None, id="run_curves")
            
            if plot_type == "km":
                fig, stats = survival_lib.fit_km_logrank(data, time_col, event_col, group_col)
                # Compute Medians
                medians = survival_lib.calculate_median_survival(data, time_col, event_col, group_col)
                stats = pd.concat([stats, medians], axis=1) # Combine for display
                stats = stats.loc[:,~stats.columns.duplicated()]
            else:
                fig, stats = survival_lib.fit_nelson_aalen(data, time_col, event_col, group_col)
            
            curves_result.set({'fig': fig, 'stats': stats})
            ui.notification_remove("run_curves")
            
        except Exception as e:
            ui.notification_remove("run_curves")
            ui.notification_show(f"Error: {e}", type="error")
            logger.error(f"Curve error: {e}")

    @render.ui
    def out_curves_result():
        res = curves_result.get()
        if res is None:
            return ui.markdown("*Results will appear here...*")
        
        return ui.card(
            ui.card_header("📈 Plot"),
            output_widget(f"{namespace}_out_curves_plot"), # ✅ Changed to output_widget
            ui.card_header("📄 Statistics"),
            ui.output_data_frame(f"{namespace}_out_curves_table")
        )

    @render_widget # ✅ Changed to render_widget
    def out_curves_plot():
        res = curves_result.get()
        return res['fig'] if res else None

    @render.data_frame
    def out_curves_table():
        res = curves_result.get()
        return render.DataGrid(res['stats']) if res else None
    
    # Download Report (Curves)
    @render.download(filename="survival_report.html")
    def _btn_dl_curves():
        res = curves_result.get()
        if res:
            elements = [
                {'type': 'header', 'data': 'Survival Analysis Report'},
                {'type': 'plot', 'data': res['fig']},
                {'type': 'header', 'data': 'Statistics'},
                {'type': 'table', 'data': res['stats']}
            ]
            yield survival_lib.generate_report_survival("Survival Analysis", elements)

    # ==================== 2. LANDMARK LOGIC ====================
    @reactive.Effect
    @reactive.event(input[f"{namespace}_btn_run_landmark"])
    def _run_landmark():
        data = surv_df_current.get()
        time_col = input[f"{namespace}_surv_time"]()
        event_col = input[f"{namespace}_surv_event"]()
        group_col = input[f"{namespace}_landmark_group"]()
        t = input[f"{namespace}_landmark_t"]()
        
        if data is None or group_col == "Select...":
            ui.notification_show("Please configure variables properly", type="warning")
            return

        try:
            ui.notification_show("Running Landmark Analysis...", duration=None, id="run_landmark")
            fig, stats, n_pre, n_post, err = survival_lib.fit_km_landmark(data, time_col, event_col, group_col, t)
            
            if err:
                ui.notification_show(err, type="error")
            else:
                landmark_result.set({'fig': fig, 'stats': stats, 'n_pre': n_pre, 'n_post': n_post, 't': t})
            
            ui.notification_remove("run_landmark")
        except Exception as e:
            ui.notification_remove("run_landmark")
            logger.error(f"Landmark error: {e}")

    @render.ui
    def out_landmark_result():
        res = landmark_result.get()
        if res is None: return None
        return ui.card(
            ui.card_header("📈 Landmark Plot"),
            ui.markdown(f"**Total N:** {res['n_pre']} | **Included (Survived > {res['t']}):** {res['n_post']}"),
            output_widget(f"{namespace}_out_landmark_plot"), # ✅ Changed to output_widget
            ui.output_data_frame(f"{namespace}_out_landmark_table")
        )

    @render_widget
    def out_landmark_plot():
        res = landmark_result.get()
        return res['fig'] if res else None

    @render.data_frame
    def out_landmark_table():
        res = landmark_result.get()
        return render.DataGrid(res['stats']) if res else None

    # ==================== 3. COX REGRESSION LOGIC ====================
    @reactive.Effect
    @reactive.event(input[f"{namespace}_btn_run_cox"])
    def _run_cox():
        data = surv_df_current.get()
        time_col = input[f"{namespace}_surv_time"]()
        event_col = input[f"{namespace}_surv_event"]()
        covars = input[f"{namespace}_cox_covariates"]()
        
        if not covars:
            ui.notification_show("Select at least one covariate", type="warning")
            return
            
        try:
            ui.notification_show("Fitting Cox Model...", duration=None, id="run_cox")
            
            # 1. Fit Model
            cph, res_df, clean_data, err = survival_lib.fit_cox_ph(data, time_col, event_col, list(covars))
            
            if err:
                ui.notification_show(err, type="error")
                ui.notification_remove("run_cox")
                return
            
            # 2. Forest Plot
            forest_fig = survival_lib.create_forest_plot_cox(res_df)
            
            # 3. Check Assumptions (Schoenfeld)
            assump_text, assump_plots = survival_lib.check_cph_assumptions(cph, clean_data)
            
            cox_result.set({
                'results_df': res_df,
                'forest_fig': forest_fig,
                'assumptions_text': assump_text,
                'assumptions_plots': assump_plots
            })
            
            ui.notification_remove("run_cox")
            
        except Exception as e:
            ui.notification_remove("run_cox")
            ui.notification_show(f"Cox error: {e}", type="error")
            logger.exception("Cox Error")

    @render.ui
    def out_cox_result():
        res = cox_result.get()
        if res is None: return None
        
        return ui.card(
            ui.card_header("📄 Cox Results"),
            ui.output_data_frame(f"{namespace}_out_cox_table"),
            
            ui.card_header("🌳 Forest Plot"),
            output_widget(f"{namespace}_out_cox_forest"), # ✅ Changed to output_widget
            
            ui.card_header("🔍 PH Assumption (Schoenfeld Residuals)"),
            ui.output_ui(f"{namespace}_out_cox_assumptions_ui")
        )

    @render.data_frame
    def out_cox_table():
        res = cox_result.get()
        return render.DataGrid(res['results_df']) if res else None

    @render_widget
    def out_cox_forest():
        res = cox_result.get()
        return res['forest_fig'] if res else None

    @render.ui
    def out_cox_assumptions_ui():
        res = cox_result.get()
        if not res: return None
        
        # Display Text Report
        elements = [ui.markdown(f"**Interpretation:**\n\n{res['assumptions_text']}")]
        
        # Display Plots (Embed as HTML because dynamic widgets list is hard in Shiny)
        if res['assumptions_plots']:
            html_plots = ""
            for fig in res['assumptions_plots']:
                # Convert Plotly fig to HTML div string
                html_plots += fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            elements.append(ui.HTML(html_plots))
            
        return ui.div(*elements)

    # ==================== 4. SUBGROUP LOGIC ====================
    @reactive.Effect
    @reactive.event(input[f"{namespace}_btn_run_sg"])
    def _run_sg():
        if SubgroupAnalysisCox is None:
            ui.notification_show("Subgroup module not found", type="error")
            return
            
        data = surv_df_current.get()
        time = input[f"{namespace}_sg_time"]()
        event = input[f"{namespace}_sg_event"]()
        treat = input[f"{namespace}_sg_treatment"]()
        subgroup = input[f"{namespace}_sg_subgroup"]()
        adjust = input[f"{namespace}_sg_adjust"]()
        
        if any(x == "Select..." for x in [time, event, treat, subgroup]):
            ui.notification_show("Please select all required variables", type="warning")
            return
            
        try:
            ui.notification_show("Running Subgroup Analysis...", duration=None, id="run_sg")
            
            analyzer = SubgroupAnalysisCox(data, time, event, treat, list(adjust))
            res = analyzer.run_analysis(subgroup)
            sg_result.set(res)
            
            ui.notification_remove("run_sg")
        except Exception as e:
            ui.notification_remove("run_sg")
            ui.notification_show(f"Error: {e}", type="error")

    @render.ui
    def out_sg_result():
        res = sg_result.get()
        if res is None: return None
        
        # Assuming result contains a Forest Plot figure and a Table
        # If SubgroupAnalysisCox returns a specific structure, adapt here.
        # This is a generic display based on typical output:
        
        elements = []
        if 'forest_plot' in res:
            # We need to render this plot. Since it's dynamic, we might need a dedicated output_widget or use HTML
            # Here we use a dedicated widget for the subgroup forest plot
            elements.append(output_widget(f"{namespace}_out_sg_forest"))
            
        if 'interaction_table' in res:
            elements.append(ui.output_data_frame(f"{namespace}_out_sg_table"))
            
        return ui.card(*elements)

    @render_widget
    def out_sg_forest():
        res = sg_result.get()
        return res.get('forest_plot') if res else None
        
    @render.data_frame
    def out_sg_table():
        res = sg_result.get()
        return render.DataGrid(res.get('interaction_table')) if res else None

# ==================== WRAPPERS ====================
def survival_ui(namespace: str) -> ui.TagChild:
    return surv_ui(namespace)

def survival_server(namespace: str, df: reactive.Value, var_meta: reactive.Value,
                    df_matched: reactive.Value, is_matched: reactive.Value):
    return surv_server(namespace, df, var_meta, df_matched, is_matched)

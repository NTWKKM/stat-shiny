"""
⛳ Survival Analysis Module (Shiny) - MODERN MODULE PATTERN

Provides UI and server logic for:
- Kaplan-Meier curves with log-rank tests
- Nelson-Aalen cumulative hazard curves
- Landmark analysis for late endpoints
- Cox proportional hazards regression
- Subgroup analysis for treatment heterogeneity

Uses Modern Shiny Module Pattern (@module.ui, @module.server decorators)
for automatic namespace scoping and cleaner code organization.
"""

from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import survival_lib
try:
    from subgroup_analysis_module import SubgroupAnalysisCox
except ImportError:
    SubgroupAnalysisCox = None
    
from logger import get_logger
# --- นำเข้าสีจาก _common.py ---
from tabs._common import get_color_palette

logger = get_logger(__name__)
# --- กำหนดค่า COLORS สำหรับใช้งานในโมดูลนี้ ---
COLORS = get_color_palette()

# ==============================================================================
# Helper Function
# ==============================================================================
def _get_dataset_for_survival(df: pd.DataFrame, df_matched: reactive.Value, is_matched: reactive.Value) -> tuple[pd.DataFrame, str]:
    """Select between original and matched dataset based on user preference."""
    if is_matched.get() and df_matched.get() is not None:
        return df_matched.get().copy(), f"✅ Matched Data ({len(df_matched.get())} rows)"
    else:
        return df.copy(), f"📊 Original Data ({len(df)} rows)"

# ==============================================================================
# UI Definition - Modern Module Pattern
# ==============================================================================
@module.ui
def survival_ui():
    """Modern Shiny UI module - no namespace argument needed."""
    return ui.navset_tab(
        # TAB 1: Survival Curves (KM & Nelson-Aalen)
        ui.nav_panel(
            "📈 Survival Curves",
            ui.card(
                ui.card_header("Kaplan-Meier & Nelson-Aalen Curves"),
                
                ui.layout_columns(
                    ui.input_select(
                        "surv_time",
                        "⛳ Time Variable:",
                        choices=["Select..."]
                    ),
                    ui.input_select(
                        "surv_event",
                        "🗣️ Event Variable (1=Event):",
                        choices=["Select..."]
                    ),
                    ui.input_select(
                        "surv_group",
                        "Compare Groups (Optional):",
                        choices=["None"]
                    ),
                    col_widths=[4, 4, 4]
                ),
                
                ui.input_radio_buttons(
                    "plot_type",
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
                        "btn_run_curves",
                        "🚀 Generate Curve",
                        class_="btn-primary w-100",
                    ),
                    ui.input_action_button(
                        "btn_dl_curves",
                        "📥 Download Report",
                        class_="btn-secondary w-100",
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.output_ui("out_curves_result"),
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
                    "landmark_t",
                    "Landmark Time (t):",
                    min=0, max=100, value=10, step=1
                ),
                
                ui.input_select(
                    "landmark_group",
                    "Compare Group:",
                    choices=["Select..."]
                ),
                
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run_landmark",
                        "🚀 Run Landmark Analysis",
                        class_="btn-primary w-100",
                    ),
                    ui.input_action_button(
                        "btn_dl_landmark",
                        "📥 Download Report",
                        class_="btn-secondary w-100",
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.output_ui("out_landmark_result"),
                full_screen=True
            )
        ),
        
        # TAB 3: Cox Regression
        ui.nav_panel(
            "📈 Cox Regression",
            ui.card(
                ui.card_header("Cox Proportional Hazards Regression"),
                
                ui.input_checkbox_group(
                    "cox_covariates",
                    "Select Covariates (Predictors):",
                    choices=[],
                    selected=[]
                ),
                
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run_cox",
                        "🚀 Run Cox Model",
                        class_="btn-primary w-100",
                    ),
                    ui.input_action_button(
                        "btn_dl_cox",
                        "📥 Download Report",
                        class_="btn-secondary w-100",
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.output_ui("out_cox_result"),
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
                        "sg_time", "Follow-up Time:", choices=["Select..."]
                    ),
                    ui.input_select(
                        "sg_event", "Event Indicator (Binary):", choices=["Select..."]
                    ),
                    ui.input_select(
                        "sg_treatment", "Treatment/Exposure:", choices=["Select..."]
                    ),
                    col_widths=[4, 4, 4]
                ),
                
                ui.layout_columns(
                    ui.input_select(
                        "sg_subgroup", "📌 Stratify By:", choices=["Select..."]
                    ),
                    ui.input_checkbox_group(
                        "sg_adjust", "Adjustment Variables:", choices=[]
                    ),
                    col_widths=[4, 8]
                ),
                
                ui.accordion(
                    ui.accordion_panel(
                        "⚠️ Advanced Settings",
                        ui.input_numeric(
                            "sg_min_n", "Min N per subgroup:", value=5, min=2, max=50
                        ),
                        ui.input_numeric(
                            "sg_min_events", "Min events per subgroup:", value=2, min=1, max=50
                        )
                    ),
                    open=False
                ),
                
                ui.input_action_button(
                    "btn_run_sg",
                    "🚀 Run Subgroup Analysis",
                    class_="btn-primary w-100"
                ),
                
                ui.output_ui("out_sg_result"),
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
# Server Logic - Modern Module Pattern
# ==============================================================================
@module.server
def survival_server(input, output, session, df: reactive.Value, var_meta: reactive.Value,
                    df_matched: reactive.Value, is_matched: reactive.Value):
    """Modern Shiny server module - automatic namespace scoping via decorator."""
    
    # ==================== REACTIVE VALUES ====================
    surv_df_current = reactive.Value(None)
    curves_result = reactive.Value(None)
    landmark_result = reactive.Value(None)
    cox_result = reactive.Value(None)
    sg_result = reactive.Value(None)
    
    # ==================== LABEL MAPPING LOGIC ====================
    @reactive.Calc
    def label_map():
        """Build a dictionary mapping raw column names to user-friendly labels from var_meta."""
        meta = var_meta.get() if hasattr(var_meta, 'get') else var_meta()
        if not meta:
            return {}
        # Assuming meta is a list of dicts with 'name' and 'label' or similar structure
        # Fallback to empty dict if structure is different
        try:
            return {item.get('name', k): item.get('label', k) for k, item in meta.items()}
        except (AttributeError, TypeError, KeyError):
            return {}

    def get_label(col_name: str) -> str:
        """Helper to get label for a column, falling back to column name."""
        return label_map().get(col_name, col_name)

    # ==================== DATASET UPDATES ====================
    @reactive.Effect
    def _update_current_dataset():
        """Update current dataset and refresh all input choices."""
        data = df.get()
        if data is not None:
            selected_df, _ = _get_dataset_for_survival(data, df_matched, is_matched)
            surv_df_current.set(selected_df)
            
            # Update Dropdowns
            cols = data.columns.tolist()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            # --- AUTO-DETECTION LOGIC WITH PRIORITY ---
            
            # 1. Detect Time Variables
            time_keywords = ['time', 'day', 'month', 'year', 'range', 'followup', 'fu']
            default_time = "Select..."
            
            for kw in time_keywords:
                matched = [c for c in numeric_cols if kw in c.lower()]
                if matched:
                    default_time = matched[0]
                    break
            
            if default_time == "Select..." and numeric_cols:
                default_time = numeric_cols[0]

            # 2. Detect Event Variables
            event_keywords = ['status', 'event', 'death', 'cure', 'survive', 'died', 'outcome']
            default_event = "Select..."
            
            for kw in event_keywords:
                matched = [c for c in cols if kw in c.lower()]
                if matched:
                    default_event = matched[0]
                    break
            
            if default_event == "Select..." and cols:
                default_event = cols[0]
                
            # 3. Detect compare Variables
            compare_keywords = ['treatment', 'group', 'comorbid', 'comorb', 'dz', 'lab', 'diag', 'sex', 'age']
            default_compare = "Select..."
            
            for kw in compare_keywords:
                matched = [c for c in cols if kw in c.lower()]
                if matched:
                    default_compare = matched[0]
                    break
            
            if default_compare == "Select..." and cols:
                default_compare = cols[0]

            # 4. Detect treatment Variables
            tx_keywords = ['treatment', 'tx', 'group', 'drug', 'give']
            default_tx = "Select..."
            
            for kw in tx_keywords:
                matched = [c for c in cols if kw in c.lower()]
                if matched:
                    default_tx = matched[0]
                    break
            
            if default_tx == "Select..." and cols:
                default_tx = cols[0]

            # 5. Detect subgroup Variables
            subgr_keywords = ['comorbid', 'comorb', 'group', 'control', 'contr', 'ctr', 'dz', 'lab', 'diag', 'sex', 'age']
            default_subgr = "Select..."
            
            for kw in subgr_keywords:
                matched = [c for c in cols if kw in c.lower()]
                if matched:
                    default_subgr = matched[0]
                    break
            
            if default_subgr == "Select..." and cols:
                default_subgr = cols[0]
                
            # Update UI choices with labels if available
            choices_with_labels = {c: get_label(c) for c in cols}
            num_choices_with_labels = {c: get_label(c) for c in numeric_cols}

            # KM Curves
            ui.update_select("surv_time", choices=num_choices_with_labels, selected=default_time)
            ui.update_select("surv_event", choices=choices_with_labels, selected=default_event)
            ui.update_select("surv_group", choices={"None": "None", **choices_with_labels})
            
            # Landmark Analysis
            ui.update_select("landmark_group", choices=choices_with_labels, selected=default_compare)
            
            # Cox Regression
            ui.update_checkbox_group("cox_covariates", choices=choices_with_labels)
            
            # Subgroup Analysis
            ui.update_select("sg_time", choices=num_choices_with_labels, selected=default_time)
            ui.update_select("sg_event", choices=choices_with_labels, selected=default_event)
            ui.update_select("sg_treatment", choices=choices_with_labels, selected=default_tx)
            ui.update_select("sg_subgroup", choices=choices_with_labels, selected=default_subgr)
            ui.update_checkbox_group("sg_adjust", choices=choices_with_labels)

    # ==================== 1. CURVES LOGIC (KM/NA) ====================
    @reactive.Effect
    @reactive.event(input.btn_run_curves)
    def _run_curves():
        """Run Kaplan-Meier or Nelson-Aalen curves."""
        data = surv_df_current.get()
        time_col = input.surv_time()
        event_col = input.surv_event()
        group_col = input.surv_group()
        plot_type = input.plot_type()
        
        if data is None or time_col == "Select..." or event_col == "Select...":
            ui.notification_show("Please select Time and Event variables", type="warning")
            return
            
        if group_col == "None": 
            group_col = None
        
        try:
            ui.notification_show("Generating curves...", duration=None, id="run_curves")
            
            if plot_type == "km":
                fig, stats = survival_lib.fit_km_logrank(data, time_col, event_col, group_col)
                medians = survival_lib.calculate_median_survival(data, time_col, event_col, group_col)
                if 'Group' in stats.columns and 'Group' in medians.columns:
                    stats = stats.merge(medians, on='Group', how='outer')
                else:
                    # Avoid column name collisions by suffixing
                    overlapping = set(stats.columns) & set(medians.columns)
                    if overlapping:
                        medians = medians.rename(columns={c: f"{c}_median" for c in overlapping})
                    stats = pd.concat([stats, medians], axis=1)
            else:
                fig, stats = survival_lib.fit_nelson_aalen(data, time_col, event_col, group_col)
            
            curves_result.set({'fig': fig, 'stats': stats})
            ui.notification_remove("run_curves")
            
        except Exception as e:
            ui.notification_remove("run_curves")
            ui.notification_show(f"Error: {e}", type="error")
            logger.exception("Curve error")

    @render.ui
    def out_curves_result():
        """Render curves results."""
        res = curves_result.get()
        if res is None:
            return ui.div(
                ui.markdown("*Results will appear here...*"),
                style=f"color: {COLORS['text_secondary']}; text-align: center; padding: 20px;"
            )
        
        return ui.card(
            ui.card_header("📈 Plot"),
            output_widget("out_curves_plot"),
            ui.card_header("📄 Statistics"),
            ui.output_data_frame("out_curves_table")
        )

    @render_widget
    def out_curves_plot():
        """Render Kaplan-Meier or Nelson-Aalen plot."""
        res = curves_result.get()
        return res['fig'] if res else None

    @render.data_frame
    def out_curves_table():
        """Render curves statistics table."""
        res = curves_result.get()
        return render.DataGrid(res['stats']) if res else None
    
    @render.download(filename="survival_report.html")
    def btn_dl_curves():
        """Download survival curves report."""
        res = curves_result.get()
        if not res:
            ui.notification_show("No results to download. Run analysis first.", type="warning")
            return
        else:
            elements = [
                {'type': 'header', 'data': 'Survival Analysis Report'},
                {'type': 'plot', 'data': res['fig']},
                {'type': 'header', 'data': 'Statistics'},
                {'type': 'table', 'data': res['stats']}
            ]
            yield survival_lib.generate_report_survival("Survival Analysis", elements)

    # ==================== 2. LANDMARK LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_run_landmark)
    def _run_landmark():
        """Run landmark analysis."""
        data = surv_df_current.get()
        time_col = input.surv_time()
        event_col = input.surv_event()
        group_col = input.landmark_group()
        t = input.landmark_t()
        
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
            logger.exception("Landmark error")
            ui.notification_show(f"Landmark analysis error: {e}", type="error")

    @render.ui
    def out_landmark_result():
        """Render landmark analysis results."""
        res = landmark_result.get()
        if res is None: 
            return None
        return ui.card(
            ui.card_header("📈 Landmark Plot"),
            ui.div(
                ui.markdown(f"**Total N:** {res['n_pre']} | **Included (Survived > {res['t']}):** {res['n_post']}"),
                style=f"padding: 10px; border-radius: 5px; background-color: {COLORS['info']}15; margin-bottom: 15px; border-left: 4px solid {COLORS['info']};"
            ),
            output_widget("out_landmark_plot"),
            ui.output_data_frame("out_landmark_table")
        )

    @render_widget
    def out_landmark_plot():
        """Render landmark analysis plot."""
        res = landmark_result.get()
        return res['fig'] if res else None

    @render.data_frame
    def out_landmark_table():
        """Render landmark analysis table."""
        res = landmark_result.get()
        return render.DataGrid(res['stats']) if res else None

    @render.download(filename="landmark_report.html")
    def btn_dl_landmark():
        """Download landmark analysis report."""
        res = landmark_result.get()
        if res:
            elements = [
                {'type': 'header', 'data': f'Landmark Analysis (t={res["t"]})'},
                {'type': 'plot', 'data': res['fig']},
                {'type': 'header', 'data': 'Statistics'},
                {'type': 'table', 'data': res['stats']}
            ]
            yield survival_lib.generate_report_survival("Landmark Analysis", elements)

    # ==================== 3. COX REGRESSION LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_run_cox)
    def _run_cox():
        """Run Cox proportional hazards regression."""
        data = surv_df_current.get()
        time_col = input.surv_time()
        event_col = input.surv_event()
        covars = input.cox_covariates()
        
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
            
            # Update results with labels for forest plot if possible
            # Note: res_df column 'Variable' could be mapped to labels here if forest plot lib supports it
            
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
        """Render Cox regression results."""
        res = cox_result.get()
        if res is None: 
            return None
        
        return ui.card(
            ui.card_header("📄 Cox Results"),
            ui.output_data_frame("out_cox_table"),
            
            ui.card_header("🌳 Forest Plot"),
            output_widget("out_cox_forest"),
            
            ui.card_header("🔍 PH Assumption (Schoenfeld Residuals)"),
            ui.output_ui("out_cox_assumptions_ui")
        )

    @render.data_frame
    def out_cox_table():
        """Render Cox results table."""
        res = cox_result.get()
        return render.DataGrid(res['results_df']) if res else None

    @render_widget
    def out_cox_forest():
        """Render Cox forest plot."""
        res = cox_result.get()
        return res['forest_fig'] if res else None

    @render.ui
    def out_cox_assumptions_ui():
        """Render Cox assumptions diagnostics."""
        res = cox_result.get()
        if not res: 
            return None
        
        # Display Text Report
        elements = [
            ui.div(
                ui.markdown(f"**Interpretation:**\n\n{res['assumptions_text']}"),
                style=f"padding: 15px; border-radius: 5px; background-color: {COLORS['primary']}10; border-left: 5px solid {COLORS['primary']};"
            )
        ]
        
        # Display Plots
        if res['assumptions_plots']:
            html_plots = ""
            for i, fig in enumerate(res['assumptions_plots']):
                # Include Plotly.js only for the first plot
                include_js = 'cdn' if i == 0 else False
                html_plots += fig.to_html(full_html=False, include_plotlyjs=include_js)
            
            elements.append(ui.HTML(html_plots))
            
        return ui.div(*elements)

    @render.download(filename="cox_report.html")
    def btn_dl_cox():
        """Download Cox regression report."""
        res = cox_result.get()
        if res:
            elements = [
                {'type': 'header', 'data': 'Cox Proportional Hazards Regression'},
                {'type': 'table', 'data': res['results_df']},
                {'type': 'plot', 'data': res['forest_fig']},
                {'type': 'header', 'data': 'PH Assumptions'},
                {'type': 'text', 'data': res['assumptions_text']}
            ]
            yield survival_lib.generate_report_survival("Cox Regression", elements)

    # ==================== 4. SUBGROUP LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_run_sg)
    def _run_sg():
        """Run subgroup analysis."""
        if SubgroupAnalysisCox is None:
            ui.notification_show("Subgroup module not found", type="error")
            return
            
        data = surv_df_current.get()
        time = input.sg_time()
        event = input.sg_event()
        treat = input.sg_treatment()
        subgroup = input.sg_subgroup()
        adjust = input.sg_adjust()
        
        if any(x == "Select..." for x in [time, event, treat, subgroup]):
            ui.notification_show("Please select all required variables", type="warning")
            return
            
        try:
            ui.notification_show("Running Subgroup Analysis...", duration=None, id="run_sg")
            
            analyzer = SubgroupAnalysisCox(data)
            result, _, error = analyzer.analyze(
                duration_col=time,
                event_col=event,
                treatment_col=treat,
                subgroup_col=subgroup,
                adjustment_cols=list(adjust) if adjust else None,
                min_subgroup_n=input.sg_min_n(),
                min_events=input.sg_min_events()
            )
            if error:
                ui.notification_show(error, type="error")
                ui.notification_remove("run_sg")
                return
            
            # Generate forest plot
            forest_fig = analyzer.create_forest_plot()
            result['forest_plot'] = forest_fig
            result['interaction_table'] = analyzer.results
            sg_result.set(result)
            
            ui.notification_remove("run_sg")
        except Exception as e:
            ui.notification_remove("run_sg")
            ui.notification_show(f"Error: {e}", type="error")
            logger.exception("Subgroup analysis error")

    @render.ui
    def out_sg_result():
        """Render subgroup analysis results."""
        res = sg_result.get()
        if res is None: 
            return None
        
        elements = []
        if 'forest_plot' in res:
            elements.append(ui.card_header("🌳 Subgroup Forest Plot"))
            elements.append(output_widget("out_sg_forest"))
            
        if 'interaction_table' in res:
            elements.append(ui.card_header("📄 Interaction Analysis"))
            elements.append(ui.output_data_frame("out_sg_table"))
            
        return ui.card(*elements)

    @render_widget
    def out_sg_forest():
        """Render subgroup forest plot."""
        res = sg_result.get()
        return res.get('forest_plot') if res else None
        
    @render.data_frame
    def out_sg_table():
        """Render subgroup interaction table."""
        res = sg_result.get()
        return render.DataGrid(res.get('interaction_table')) if res else None

    @render.download(filename="subgroup_report.html")
    def btn_dl_sg():
        """Download subgroup analysis report."""
        res = sg_result.get()
        if res:
            # สร้างรายการ elements เริ่มต้น
            elements = [
                {'type': 'header', 'data': 'Cox Subgroup Analysis'},
                {'type': 'plot', 'data': res.get('forest_plot')},
                {'type': 'header', 'data': 'Results'},
                {'type': 'table', 'data': res.get('interaction_table')}
            ]
            
            # ปรับปรุง: กรองเอาเฉพาะ element ที่ data ไม่เป็น None เพื่อป้องกัน "None" ปรากฏใน report
            # และป้องกัน Error ในกรณีที่ plot หรือ table ไม่ถูกสร้างขึ้น
            elements = [e for e in elements if e.get('data') is not None]
            
            yield survival_lib.generate_report_survival("Subgroup Analysis", elements)

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
    """
    Choose the active dataset for survival analyses based on the matching flag.
    
    Parameters:
        df (pd.DataFrame): Original dataset.
        df_matched (reactive.Value): Reactive container holding the matched dataset or None.
        is_matched (reactive.Value): Reactive boolean indicating whether to use the matched dataset.
    
    Returns:
        tuple[pd.DataFrame, str]: (selected_df, status_label) where `selected_df` is the DataFrame chosen for analysis
        (a copy of the matched DataFrame when matched is selected), and `status_label` is a human-readable string
        indicating whether matched or original data was selected along with row count (e.g., "✅ Matched Data (N rows)").
    """
    if is_matched.get() and df_matched.get() is not None:
        return df_matched.get().copy(), f"✅ Matched Data ({len(df_matched.get())} rows)"
    else:
        return df.copy(), f"📊 Original Data ({len(df)} rows)"

# ==============================================================================
# UI Definition - Modern Module Pattern
# ==============================================================================
@module.ui
def survival_ui():
    """
    Create the survival analysis module UI containing tabs for survival curves, landmark analysis, Cox regression, subgroup analysis, and a reference panel.
    
    The UI provides inputs and outputs for:
    - Kaplan–Meier and Nelson–Aalen curve generation (time, event, optional group, plot type, run/download).
    - Landmark analysis (landmark time, group selection, run/download).
    - Cox proportional hazards regression (covariate selection, run/download, results/assumptions).
    - Subgroup Cox analysis (time, event, treatment, subgroup, adjustments, advanced thresholds, run/download).
    - A reference tab summarizing methods and typical outputs.
    
    Returns:
        ui_element: A Shiny `navset_tab` UI element representing the survival analysis module.
    """
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
    """
                    Register server-side reactive logic and outputs for the Survival Analysis module, including Kaplan–Meier / Nelson–Aalen curves, landmark analysis, Cox regression, and subgroup analysis.
                    
                    Parameters:
                        input: Shiny input bindings for module controls.
                        output: Shiny output bindings for module results.
                        session: Shiny session object scoped to the module.
                        df (reactive.Value): Reactive holding the primary DataFrame used for analyses.
                        var_meta (reactive.Value): Reactive metadata about variables (used to populate and validate inputs).
                        df_matched (reactive.Value): Reactive holding an alternative (matched) DataFrame; selected when matching is enabled.
                        is_matched (reactive.Value): Reactive boolean that indicates whether to use `df_matched` instead of `df`.
                    """
    
    # ==================== REACTIVE VALUES ====================
    surv_df_current = reactive.Value(None)
    curves_result = reactive.Value(None)
    landmark_result = reactive.Value(None)
    cox_result = reactive.Value(None)
    sg_result = reactive.Value(None)
    
    # ==================== DATASET UPDATES ====================
    @reactive.Effect
    def _update_current_dataset():
        """
        Selects the active survival dataset and refreshes related UI input choices.
        
        Chooses between the original and matched DataFrame, sets the selected DataFrame into
        surv_df_current, and updates UI selectors and checkbox groups for survival curves,
        landmark analysis, Cox regression, and subgroup analysis. Time selectors use numeric
        columns; other selectors use all columns from the active DataFrame.
        """
        data = df.get()
        if data is not None:
            selected_df, _ = _get_dataset_for_survival(data, df_matched, is_matched)
            surv_df_current.set(selected_df)
            
            # Update Dropdowns
            cols = data.columns.tolist()
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # KM Curves
            ui.update_select("surv_time", choices=numeric_cols)
            ui.update_select("surv_event", choices=cols)
            ui.update_select("surv_group", choices=["None"] + cols)
            
            # Landmark Analysis
            ui.update_select("landmark_group", choices=cols)
            
            # Cox Regression
            ui.update_checkbox_group("cox_covariates", choices=cols)
            
            # Subgroup Analysis
            ui.update_select("sg_time", choices=numeric_cols)
            ui.update_select("sg_event", choices=cols)
            ui.update_select("sg_treatment", choices=cols)
            ui.update_select("sg_subgroup", choices=cols)
            ui.update_checkbox_group("sg_adjust", choices=cols)

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
                    stats = pd.concat([stats, medians], axis=1)
                    stats = stats.loc[:, ~stats.columns.duplicated()]
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
        """
        Return the current survival curve figure for Kaplan–Meier or Nelson–Aalen plotting.
        
        Returns:
            Figure or None: The plot figure to render (survival curve) or `None` if no result is available.
        """
        res = curves_result.get()
        return res['fig'] if res else None

    @render.data_frame
    def out_curves_table():
        """
        Render the statistics table for survival curves.
        
        Returns:
            DataGrid or None: A DataGrid widget containing the curves statistics when results are available; otherwise None.
        """
        res = curves_result.get()
        return render.DataGrid(res['stats']) if res else None
    
    @render.download(filename="survival_report.html")
    def btn_dl_curves():
        """
        Generate a downloadable survival curves report.
        
        Returns:
            str: HTML content of the generated survival analysis report when curve results are available.
        """
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
    @reactive.event(input.btn_run_landmark)
    def _run_landmark():
        """
        Perform a landmark survival analysis using the currently selected dataset and input controls.
        
        Validates that a dataset is available and a landmark group is selected; shows a running notification while processing. On success stores a dictionary in `landmark_result` with keys `fig`, `stats`, `n_pre`, `n_post`, and `t`. On validation failure or analysis error, displays a warning or error notification and logs exceptions as appropriate.
        """
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
        """
        Create a UI card that displays the landmark analysis plot, a brief summary of included/excluded counts, and the landmark statistics table.
        
        Returns:
            ui component or None: A Shiny UI card containing the landmark plot, summary counts, and statistics table when landmark results are present; `None` if no results are available.
        """
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
        """
        Provide the landmark analysis figure for rendering.
        
        Returns:
            fig: The landmark analysis plotting object, or `None` if no results are available.
        """
        res = landmark_result.get()
        return res['fig'] if res else None

    @render.data_frame
    def out_landmark_table():
        """
        Render the landmark analysis statistics table for the UI.
        
        Returns:
            DataGrid or None: A DataGrid widget containing the landmark analysis `stats` table when results are available, or `None` if no results exist.
        """
        res = landmark_result.get()
        return render.DataGrid(res['stats']) if res else None

    @render.download(filename="landmark_report.html")
    def btn_dl_landmark():
        """
        Generate the landmark analysis report for download.
        
        @returns Yields the generated report content (HTML) for download.
        """
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
        """
        Fit a Cox proportional hazards model on the currently selected dataset and store results.
        
        Validates that at least one covariate is selected; shows user notifications while fitting. On success, stores a dictionary in the module's `cox_result` reactive with keys:
        - `results_df`: model results table,
        - `forest_fig`: forest plot figure,
        - `assumptions_text`: interpretation of proportional hazards checks,
        - `assumptions_plots`: any diagnostic plots.
        
        On failure, removes the running notification, shows an error notification, and logs the exception.
        """
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
        """
        Render the Cox regression results card for the UI.
        
        Returns:
            ui_component (Optional[UI]): A card containing the Cox results table, forest plot, and PH-assumptions UI, or `None` if no results are available.
        """
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
        """
        Render the Cox regression results as a data grid.
        
        Returns:
            DataGrid: A data grid widget containing the Cox results table (`results_df`) if results are available, `None` otherwise.
        """
        res = cox_result.get()
        return render.DataGrid(res['results_df']) if res else None

    @render_widget
    def out_cox_forest():
        """
        Return the forest plot figure for the fitted Cox regression model.
        
        Returns:
            Figure: The forest plot figure object, or None if Cox results are not available.
        """
        res = cox_result.get()
        return res['forest_fig'] if res else None

    @render.ui
    def out_cox_assumptions_ui():
        """
        Render UI showing the Cox proportional hazards assumptions interpretation and any diagnostic plots.
        
        Displays an interpretation text block and, if present, embeds HTML versions of diagnostic plots produced by the Cox assumptions check.
        
        Returns:
            ui_element (shiny.ui.tag|None): A container with the interpretation and plots, or `None` when no Cox results are available.
        """
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
            for fig in res['assumptions_plots']:
                html_plots += fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            elements.append(ui.HTML(html_plots))
            
        return ui.div(*elements)

    @render.download(filename="cox_report.html")
    def btn_dl_cox():
        """
        Generate the Cox regression report for download.
        
        Yields:
            The generated HTML report content as a string when Cox results are available.
        """
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
        """
        Execute a subgroup Cox analysis on the currently selected dataset and store the results in `sg_result`.
        
        Validates that required variables (follow-up time, event indicator, treatment, and subgroup) are selected; shows a persistent "Running Subgroup Analysis..." notification while processing. On success, attaches a `forest_plot` and `interaction_table` to the result and sets `sg_result`. On validation failure or analysis error, shows an appropriate notification and logs the exception.
        """
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
        """
        Render the subgroup analysis result card containing available forest plot and interaction table.
        
        Returns:
            ui.card: Card UI that includes the subgroup forest plot and/or interaction analysis table when present, or `None` if no subgroup results are available.
        """
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
        """
        Provide the subgroup analysis forest plot for display.
        
        Returns:
            The forest plot figure/widget if available, otherwise None.
        """
        res = sg_result.get()
        return res.get('forest_plot') if res else None
        
    @render.data_frame
    def out_sg_table():
        """
        Render the subgroup interaction results as a data grid.
        
        Returns:
            DataGrid: A data grid constructed from the stored subgroup `interaction_table`, or `None` if no results are available.
        """
        res = sg_result.get()
        return render.DataGrid(res.get('interaction_table')) if res else None

    @render.download(filename="subgroup_report.html")
    def btn_dl_sg():
        """
        Produce a downloadable subgroup analysis report using the latest subgroup results.
        
        Returns:
            generator: Yields the generated report as an HTML string when subgroup results are available.
        """
        res = sg_result.get()
        if res:
            elements = [
                {'type': 'header', 'data': 'Cox Subgroup Analysis'},
                {'type': 'plot', 'data': res.get('forest_plot')},
                {'type': 'header', 'data': 'Results'},
                {'type': 'table', 'data': res.get('interaction_table')}
            ]
            yield survival_lib.generate_report_survival("Subgroup Analysis", elements)
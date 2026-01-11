"""
⛳ Survival Analysis Module (Shiny) - MODERN MODULE PATTERN (UPDATED)

Provides UI and server logic for:
- Kaplan-Meier curves with log-rank tests (Enhanced)
- Nelson-Aalen cumulative hazard curves
- Survival Probabilities at specific time points (New)
- Landmark analysis for late endpoints
- Cox proportional hazards regression (with Model Stats)
- Subgroup analysis for treatment heterogeneity

Uses Modern Shiny Module Pattern (@module.ui, @module.server decorators)
"""

from shiny import ui, module, reactive, render, req
from utils.plotly_html_renderer import plotly_figure_to_html
import pandas as pd
import numpy as np
import survival_lib
from typing import Optional, List, Dict, Any, Tuple, Union, cast

try:
    from subgroup_analysis_module import SubgroupAnalysisCox
except ImportError:
    SubgroupAnalysisCox = None  # type: ignore
    
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()

# ==============================================================================
# UI Definition - Modern Module Pattern
# ==============================================================================
@module.ui
def survival_ui() -> ui.TagChild:
    """Modern Shiny UI module - no namespace argument needed."""
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
                    
                    # ✅ NEW: Input for specific time points
                    ui.layout_columns(
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
                        ui.input_text(
                            "surv_time_points", 
                            "🕰️ Survival Probability at (time units, comma separated):", 
                            placeholder="e.g. 12, 36, 60 (Non-negative numbers only)"
                        ),
                        col_widths=[6, 6]
                    ),
                    
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_curves",
                            "🚀 Generate Curve",
                            class_="btn-primary w-100",
                        ),
                        # ✅ FIXED: Use download_button
                        ui.download_button(
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
                        # ✅ FIXED: Use download_button
                        ui.download_button(
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
                        # ✅ FIXED: Use download_button
                        ui.download_button(
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
                "🔛 Subgroup Analysis",
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
                    
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_sg",
                            "🚀 Run Subgroup Analysis",
                            class_="btn-primary w-100"
                        ),
                        # ✅ FIXED: Use download_button
                        ui.download_button(
                            "btn_dl_sg",
                            "📥 Download Report",
                            class_="btn-secondary w-100"
                        ),
                        col_widths=[6, 6]
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
    )

# ==============================================================================
# Server Logic - Modern Module Pattern
# ==============================================================================
@module.server
def survival_server(
    input: Any, 
    output: Any, 
    session: Any, 
    df: reactive.Value[Optional[pd.DataFrame]], 
    var_meta: reactive.Value[Dict[str, Any]],
    df_matched: reactive.Value[Optional[pd.DataFrame]], 
    is_matched: reactive.Value[bool]
) -> None:
    """
    Initialize server-side logic for the Survival Analysis Shiny module, wiring reactive state, dataset selection, input-choice auto-detection, and handlers for curves, landmark, Cox, and subgroup analyses.
    
    Parameters:
        input: Shiny input bindings for UI controls used by the module.
        output: Shiny output bindings for rendering UI elements from the module.
        session: Shiny session object scoped to this module instance.
        df (reactive.Value[Optional[pd.DataFrame]]): Reactive reference to the primary dataset.
        var_meta (reactive.Value[Dict[str, Any]]): Reactive mapping of variable metadata used to derive user-friendly labels.
        df_matched (reactive.Value[Optional[pd.DataFrame]]): Reactive reference to an optional matched dataset.
        is_matched (reactive.Value[bool]): Reactive flag indicating whether matched data may be selected as the active dataset.
    
    Behavior:
        - Maintains reactive result stores for survival curves, landmark analysis, Cox regression, and subgroup analysis.
        - Exposes reactive utilities for selecting the current active dataset and mapping column names to display labels.
        - Auto-detects sensible default columns for time, event, group, and subgroup inputs and updates input choices when the dataset changes.
        - Implements handlers to run analyses (KM / Nelson–Aalen curves, landmark, Cox PH, subgroup) with input validation, notifications, and result storage.
        - Renders UI outputs and generates downloadable HTML reports for each analysis type when results are available.
    
    Side effects:
        - Updates Shiny inputs and outputs, shows/hides notifications, and writes reactive result state used by renderers and download handlers.
    """
    
    # ==================== REACTIVE VALUES ====================
    curves_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    landmark_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    cox_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    sg_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    
    # ==================== DATASET SELECTION LOGIC ====================
    @reactive.Calc
    def current_df() -> Optional[pd.DataFrame]:
        """Select between original and matched dataset based on user preference."""
        if is_matched.get() and input.radio_survival_source() == "matched":
            return df_matched.get()
        return df.get()
    
    @render.ui
    def ui_title_with_summary():
        """Display title with dataset summary."""
        d = current_df()
        if d is not None:
            return ui.div(
                ui.h3("⛳ Survival Analysis"),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3"
                )
            )
        return ui.h3("⛳ Survival Analysis")
    
    @render.ui
    def ui_matched_info():
        """Display matched dataset availability info."""
        if is_matched.get():
            return ui.div(
                ui.tags.div(
                    "✅ **Matched Dataset Available** - You can select it below for analysis",
                    class_="alert alert-info"
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
                "radio_survival_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original Data ({original_len:,} rows)",
                    "matched": f"✅ Matched Data ({matched_len:,} rows)"
                },
                selected="matched",
                inline=True
            )
        return None
    
    # ==================== LABEL MAPPING LOGIC ====================
    @reactive.Calc
    def label_map() -> Dict[str, str]:
        """Build a dictionary mapping raw column names to user-friendly labels from var_meta."""
        meta = var_meta.get()
        if not meta:
            return {}
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
        data = current_df()
        if data is None:
            return
            
        # Update Dropdowns
        cols = data.columns.tolist()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # --- AUTO-DETECTION LOGIC ---
        # (Same detection logic as before)
        time_keywords = ['time', 'day', 'month', 'year', 'range', 'followup', 'fu']
        default_time = "Select..."
        for kw in time_keywords:
            matched = [c for c in numeric_cols if kw in c.lower()]
            if matched:
                default_time = matched[0]
                break
        if default_time == "Select..." and numeric_cols:
            default_time = numeric_cols[0]

        event_keywords = ['status', 'event', 'death', 'cure', 'survive', 'died', 'outcome']
        default_event = "Select..."
        for kw in event_keywords:
            matched = [c for c in cols if kw in c.lower()]
            if matched:
                default_event = matched[0]
                break
        if default_event == "Select..." and cols:
            default_event = cols[0]
            
        # Detect groups
        compare_keywords = ['treatment', 'group', 'comorbid', 'comorb', 'dz', 'lab', 'diag', 'sex', 'age']
        default_compare = "Select..."
        for kw in compare_keywords:
            matched = [c for c in cols if kw in c.lower()]
            if matched:
                default_compare = matched[0]
                break
        
        # Detect Subgroup
        subgr_keywords = ['comorbid', 'comorb', 'group', 'control', 'contr', 'ctr', 'dz', 'lab', 'diag', 'sex', 'age']
        default_subgr = "Select..."
        for kw in subgr_keywords:
            matched = [c for c in cols if kw in c.lower()]
            if matched:
                default_subgr = matched[0]
                break
            
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
        ui.update_select("sg_treatment", choices=choices_with_labels, selected=default_compare)
        ui.update_select("sg_subgroup", choices=choices_with_labels, selected=default_subgr)
        ui.update_checkbox_group("sg_adjust", choices=choices_with_labels)

    # ==================== 1. CURVES LOGIC (KM/NA) ====================
    @reactive.Effect
    @reactive.event(input.btn_run_curves)
    def _run_curves():
        """Run Kaplan-Meier or Nelson-Aalen curves."""
        data = current_df()
        time_col = input.surv_time()
        event_col = input.surv_event()
        group_col = input.surv_group()
        plot_type = input.plot_type()
        
        if data is None or time_col == "Select..." or event_col == "Select...":
            ui.notification_show("Please select Time and Event variables", type="warning")
            return
            
        if group_col == "None": 
            group_col = None
            
        # ✅ NEW: Parse and validate time points
        time_points: list[float] = []
        raw_tp = input.surv_time_points()
        if raw_tp:
            try:
                # 1. Split and clean input
                parts = [p.strip() for p in raw_tp.split(',') if p.strip()]
                parsed_values = []
                
                # 2. Strict validation loop
                for p in parts:
                    try:
                        val = float(p)
                    except ValueError:
                        # Validation for non-numeric input
                        raise ValueError(f"Non-numeric value detected: '{p}'. Please enter numbers only.") from None
                        
                    if val < 0:
                        # Validation for negative numbers
                        raise ValueError(f"Time points must be non-negative. Found: {val}")
                    
                    parsed_values.append(val)
                
                # 3. Check for duplicates
                unique_points = sorted(set(parsed_values))
                if len(unique_points) < len(parsed_values):
                    # Surface notification for duplication, but proceed with cleanup
                    ui.notification_show("⚠️ Duplicate time points were found and removed.", type="warning")
                
                time_points = unique_points
                
            except ValueError as e:
                # Stop execution if validation fails
                ui.notification_show(f"Input Error: {str(e)}", type="error")
                return 
        
        try:
            ui.notification_show("Generating curves...", duration=None, id="run_curves")
            
            # ✅ NEW: Calculate survival at fixed times if requested
            surv_at_times_df = None
            if time_points:
                surv_at_times_df = survival_lib.calculate_survival_at_times(
                    data, time_col, event_col, group_col, time_points
                )
            
            medians = None # Initialize medians
            
            if plot_type == "km":
                fig, stats = survival_lib.fit_km_logrank(data, time_col, event_col, group_col)
                medians = survival_lib.calculate_median_survival(data, time_col, event_col, group_col)
                
                # ✅ FIXED: Do NOT merge stats (single row test result) and medians (per-group result)
                # Storing them separately prevents NaN and shape misalignment issues
                
            else:
                fig, stats = survival_lib.fit_nelson_aalen(data, time_col, event_col, group_col)
            
            curves_result.set({
                'fig': fig, 
                'stats': stats, 
                'medians': medians, # ✅ Store medians separately in dict
                'surv_at_times': surv_at_times_df,
                'plot_type': plot_type
            })
            ui.notification_remove("run_curves")
            
        except Exception as e:
            ui.notification_remove("run_curves")
            ui.notification_show(f"Error: {e}", type="error")
            logger.exception("Curve error")

    @render.ui
    def out_curves_result():
        """
        Assemble and return the UI card displaying survival-curve outputs and related statistics.
        
        When results are available, the card contains the plot, a summary statistics table, and optionally a median survival table and a survival-at-specific-times table. If no results are present, returns a centered placeholder message informing the user that results will appear here.
        
        Returns:
            ui.Element: A Shiny UI element (card or placeholder div) representing the curves results panel.
        """
        res = curves_result.get()
        if res is None:
            return ui.div(
                ui.markdown("*Results will appear here...*"),
                style=f"color: {COLORS['text_secondary']}; text-align: center; padding: 20px;"
            )
        
        elements = [
            ui.card_header("📈 Plot"),
            ui.output_ui("out_curves_plot"),
            ui.card_header("📄 Log-Rank Test / Summary Statistics"), # Update header title
            ui.output_data_frame("out_curves_table")
        ]
        
        # ✅ NEW: Render Medians Table separately if it exists
        if res.get('medians') is not None:
             elements.append(ui.card_header("⏱️ Median Survival Time"))
             elements.append(ui.output_data_frame("out_medians_table"))
        
        # ✅ NEW: Add table for survival at specific times
        if res.get('surv_at_times') is not None:
            elements.append(ui.card_header("🕰️ Survival Probability at Specific Times"))
            elements.append(ui.output_data_frame("out_surv_times_table"))
            
        return ui.card(*elements)

    @render.ui
    def out_curves_plot():
        """
        Render the survival curves Plotly figure or a waiting placeholder.
        
        If curve results are available in `curves_result`, returns a Shiny UI element containing the Plotly figure HTML; otherwise returns a centered div with a "Waiting for results..." message.
        
        Returns:
            ui element: HTML containing the Plotly figure when results exist, or a div with a waiting message otherwise.
        """
        res = curves_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            res['fig'],
            div_id="plot_curves_km_na",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @render.data_frame
    def out_curves_table():
        """
        Render the curves statistics DataGrid when curve results are available.
        
        Returns:
            DataGrid or None: A DataGrid built from the latest curves result 'stats' entry, or `None` if no results are present.
        """
        res = curves_result.get()
        return render.DataGrid(res['stats']) if res else None

    # ✅ NEW: Renderer for Medians Table
    @render.data_frame
    def out_medians_table():
        res = curves_result.get()
        return render.DataGrid(res['medians']) if res and res.get('medians') is not None else None

    @render.data_frame
    def out_surv_times_table():
        res = curves_result.get()
        return render.DataGrid(res['surv_at_times']) if res and res.get('surv_at_times') is not None else None
    
    @render.download(filename="survival_report.html")
    def btn_dl_curves():
        """Download survival curves report."""
        res = curves_result.get()
        if not res:
            # ✅ FIXED: Changed from b"No results" to "No results" (str) to match success path
            yield "No results"
            return
        
        elements = [
            {'type': 'header', 'data': f"Survival Analysis ({'Kaplan-Meier' if res.get('plot_type', 'km')=='km' else 'Nelson-Aalen'})"},
            {'type': 'plot', 'data': res['fig']},
            {'type': 'header', 'data': 'Statistics'},
            {'type': 'table', 'data': res['stats']}
        ]
        
        # ✅ NEW: Add Medians to download report
        if res.get('medians') is not None:
            elements.append({'type': 'header', 'data': 'Median Survival Time'})
            elements.append({'type': 'table', 'data': res['medians']})
        
        if res.get('surv_at_times') is not None:
            elements.append({'type': 'header', 'data': 'Survival Probability at Fixed Times'})
            elements.append({'type': 'table', 'data': res['surv_at_times']})
            
        yield survival_lib.generate_report_survival("Survival Analysis", elements)

    # ==================== 2. LANDMARK LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_run_landmark)
    def _run_landmark():
        """Run landmark analysis."""
        data = current_df()
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
        Render the landmark analysis result card containing the plot and summary statistics.
        
        Returns:
            ui (shiny UI object) : A card with a header, a summary line showing total N and included N at the landmark time, the landmark plot output, and the landmark statistics table; returns `None` if no landmark results are available.
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
            ui.output_ui("out_landmark_plot"),
            ui.output_data_frame("out_landmark_table")
        )

    @render.ui
    def out_landmark_plot():
        """
        Render the landmark analysis plot or a waiting placeholder if results are not available.
        
        Returns:
            ui_element: A Shiny UI element containing the plot HTML when landmark results exist, or a centered waiting message placeholder otherwise.
        """
        res = landmark_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            res['fig'],
            div_id="plot_landmark_analysis",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @render.data_frame
    def out_landmark_table():
        """
        Render a data grid containing landmark analysis statistics if results are available.
        
        Returns:
            DataGrid or None: A DataGrid built from the stored landmark `stats` when landmark results exist, otherwise `None`.
        """
        res = landmark_result.get()
        return render.DataGrid(res['stats']) if res else None

    @render.download(filename="landmark_report.html")
    def btn_dl_landmark():
        """Download landmark analysis report."""
        res = landmark_result.get()
        if not res:
            # ✅ FIXED: Changed from b"No results" to "No results" (str) to match success path
            yield "No results"
            return
            
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
        data = current_df()
        time_col = input.surv_time()
        event_col = input.surv_event()
        covars = input.cox_covariates()

        if data is None or time_col == "Select..." or event_col == "Select...":
            ui.notification_show("Please select Time and Event variables", type="warning")
            return
        
        if not covars:
            ui.notification_show("Select at least one covariate", type="warning")
            return
            
        try:
            ui.notification_show("Fitting Cox Model...", duration=None, id="run_cox")
            
            # ✅ NEW: Capture model_stats
            cph, res_df, clean_data, err, model_stats = survival_lib.fit_cox_ph(data, time_col, event_col, list(covars))
            
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
                'assumptions_plots': assump_plots,
                'model_stats': model_stats # ✅ NEW
            })
            
            ui.notification_remove("run_cox")
            
        except Exception as e:
            ui.notification_remove("run_cox")
            ui.notification_show(f"Cox error: {e}", type="error")
            logger.exception("Cox Error")

    @render.ui
    def out_cox_result():
        """
        Render the Cox proportional hazards model results card for the UI.
        
        Displays model summary statistics (if available), the results data table, a forest plot, and proportional-hazards (Schoenfeld residuals) outputs. If no Cox results are present, nothing is rendered.
        
        Returns:
            A Shiny UI card containing model stats, the results data frame, the forest plot output placeholder, and the PH-assumption output placeholder, or `None` if no results are available.
        """
        res = cox_result.get()
        if res is None: 
            return None
        
        # ✅ NEW: Format Model Stats using Shiny Tag Helpers instead of raw HTML (Safe interpolation)
        stats_ui = None
        if res.get('model_stats'):
            s = res['model_stats']
            stats_ui = ui.div(
                ui.div(ui.strong("C-index: "), str(s.get('Concordance Index (C-index)', '-'))),
                ui.div(ui.strong("AIC: "), str(s.get('AIC', '-'))),
                ui.div(ui.strong("Events: "), f"{s.get('Number of Events', '-')} / {s.get('Number of Observations', '-')}" ),
                style='display: flex; gap: 20px; padding: 10px; background: #f8f9fa; border-radius: 8px; margin-bottom: 10px;'
            )
        
        return ui.card(
            ui.card_header("📄 Cox Results"),
            stats_ui, # Display stats using safe tag object
            ui.output_data_frame("out_cox_table"),
            
            ui.card_header("🌳 Forest Plot"),
            ui.output_ui("out_cox_forest"),
            
            ui.card_header("🔍 PH Assumption (Schoenfeld Residuals)"),
            ui.output_ui("out_cox_assumptions_ui")
        )

    @render.data_frame
    def out_cox_table():
        """
        Render a data grid showing Cox regression results if available.
        
        Returns:
            DataGrid or None: A DataGrid component containing the Cox results table, or None when no results are present.
        """
        res = cox_result.get()
        return render.DataGrid(res['results_df']) if res else None

    @render.ui
    def out_cox_forest():
        """
        Render the Cox regression forest plot if available; otherwise show a waiting placeholder.
        
        Returns:
            ui_element: A Shiny UI element containing the Plotly-generated forest plot HTML when results exist, or a centered waiting message if results are not yet available.
        """
        res = cox_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            res['forest_fig'],
            div_id="plot_cox_forest",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @render.ui
    def out_cox_assumptions_ui():
        """
        Render the proportional hazards assumption section for Cox regression, including interpretation text and any diagnostic plots.
        
        Returns:
            ui.div or None: A UI container with the interpretation text and zero or more Plotly diagnostic plots when Cox results are available; `None` if no Cox results exist.
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
            for i, fig in enumerate(res['assumptions_plots']):
                include_js = 'cdn' if i == 0 else False
                html_plots += fig.to_html(full_html=False, include_plotlyjs=include_js)
            
            elements.append(ui.HTML(html_plots))
            
        return ui.div(*elements)

    @render.download(filename="cox_report.html")
    def btn_dl_cox():
        """Download Cox regression report."""
        res = cox_result.get()
        if not res:
            # ✅ FIXED: Changed from b"No results" to "No results" (str) to match success path
            yield "No results"
            return

        elements = [
            {'type': 'header', 'data': 'Cox Proportional Hazards Regression'},
        ]
        
        # Add model stats to report
        if res.get('model_stats'):
            s = res['model_stats']
            stats_text = f"C-index: {s.get('Concordance Index (C-index)')}, AIC: {s.get('AIC')}, Events: {s.get('Number of Events')}"
            elements.append({'type': 'text', 'data': stats_text})

        elements.extend([
            {'type': 'table', 'data': res['results_df']},
            {'type': 'plot', 'data': res['forest_fig']},
            {'type': 'header', 'data': 'PH Assumptions'},
            {'type': 'text', 'data': res['assumptions_text']}
        ])
        yield survival_lib.generate_report_survival("Cox Regression", elements)

    # ==================== 4. SUBGROUP LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_run_sg)
    def _run_sg():
        """Run subgroup analysis."""
        if SubgroupAnalysisCox is None:
            ui.notification_show("Subgroup module not found", type="error")
            return
            
        data = current_df()
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
            
            # Ensure SubgroupAnalysisCox is treated as a class we can instantiate
            analyzer = SubgroupAnalysisCox(data) # type: ignore
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
        """
        Builds the UI card displaying subgroup analysis results.
        
        Includes a "Subgroup Forest Plot" section when the result contains a `forest_plot` and an "Interaction Analysis" data table when the result contains an `interaction_table`.
        
        Returns:
            ui.card: Card with available subgroup result sections, or `None` if no results are present.
        """
        res = sg_result.get()
        if res is None: 
            return None
        
        elements = []
        if 'forest_plot' in res:
            elements.append(ui.card_header("🌳 Subgroup Forest Plot"))
            elements.append(ui.output_ui("out_sg_forest"))
            
        if 'interaction_table' in res:
            elements.append(ui.card_header("📄 Interaction Analysis"))
            elements.append(ui.output_data_frame("out_sg_table"))
            
        return ui.card(*elements)

    @render.ui
    def out_sg_forest():
        """
        Render the subgroup analysis forest plot or a placeholder when results are unavailable.
        
        Returns:
            A Shiny UI element containing the subgroup forest plot as HTML when a plot is present; otherwise a centered placeholder message indicating either that results are pending or that no forest plot is available.
        """
        res = sg_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        fig = res.get('forest_plot')
        if fig is None:
            return ui.div(
                ui.markdown("⏳ *No forest plot available...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            fig,
            div_id="plot_subgroup_forest",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)
        
    @render.data_frame
    def out_sg_table():
        """
        Render the subgroup interaction analysis table when subgroup results are available.
        
        Returns:
            DataGrid: A data grid rendering of `interaction_table` from the stored subgroup result, or `None` if no result exists.
        """
        res = sg_result.get()
        return render.DataGrid(res.get('interaction_table')) if res else None

    @render.download(filename="subgroup_report.html")
    def btn_dl_sg():
        """Download subgroup analysis report."""
        res = sg_result.get()
        if not res:
            # ✅ FIXED: Changed from b"No results" to "No results" (str) to match success path
            yield "No results"
            return
            
        elements = [
            {'type': 'header', 'data': 'Cox Subgroup Analysis'},
            {'type': 'plot', 'data': res.get('forest_plot')},
            {'type': 'header', 'data': 'Results'},
            {'type': 'table', 'data': res.get('interaction_table')}
        ]
        elements = [e for e in elements if e.get('data') is not None]
        yield survival_lib.generate_report_survival("Subgroup Analysis", elements)
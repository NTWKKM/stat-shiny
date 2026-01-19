⛳ Survival Analysis Module (Shiny) - MODERN MODULE PATTERN (UPDATED)

Provides UI and server logic for:
- Kaplan-Meier curves with log-rank tests (Enhanced)
- Nelson-Aalen cumulative hazard curves
- Survival Probabilities at specific time points (New)
- Landmark analysis for late endpoints
- Cox proportional hazards regression (with Model Stats)
- Subgroup analysis for treatment heterogeneity
- ⏱️ Time-Varying Covariates Cox (NEW)

Uses Modern Shiny Module Pattern (@module.ui, @module.server decorators)
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from shiny import module, reactive, render, req, ui

from utils import survival_lib
from utils.formatting import create_missing_data_report_html
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.tvc_lib import (
    fit_tvc_cox,
    generate_tvc_report,
    create_tvc_forest_plot,
)
from tabs._tvc_components import (
    tvc_data_format_selector_ui,
    tvc_column_config_ui,
    tvc_risk_interval_picker_ui,
    tvc_data_preview_card_ui,
    tvc_model_config_ui,
    tvc_info_panel_ui,
    detect_tvc_columns,
    detect_static_columns,
    format_interval_preview,
)

try:
    from utils.subgroup_analysis_module import SubgroupAnalysisCox
except ImportError:
    SubgroupAnalysisCox = None  # type: ignore
    
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()

# ============================================================================
# UI Definition - Modern Module Pattern
# ============================================================================
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
                    
                    ui.layout_columns(
                        ui.input_select(
                            "cox_method",
                            "🔧 Fitting Method:",
                            choices={
                                "auto": "Auto (lifelines → Firth fallback)",
                                "lifelines": "Standard (lifelines CoxPHFitter)",
                                "firth": "Firth (for rare events / small samples)"
                            },
                            selected="auto"
                        ),
                        col_widths=[12]
                    ),
                    
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
            
            # TAB 5: Time-Varying Cox (NEW)
            ui.nav_panel(
                "⏱️ Time-Varying Cox",
                ui.card(
                    ui.card_header("Time-Dependent Survival Analysis (Time-Varying Covariates)"),
                    
                    ui.layout_columns(
                        tvc_data_format_selector_ui(),
                        tvc_model_config_ui(),
                        col_widths=[8, 4]
                    ),
                    
                    ui.layout_columns(
                        tvc_column_config_ui(),
                        tvc_info_panel_ui(),
                        col_widths=[8, 4]
                    ),
                    
                    # Risk intervals (for wide format) + data preview
                    ui.layout_columns(
                        tvc_risk_interval_picker_ui(),
                        tvc_data_preview_card_ui(),
                        col_widths=[6, 6]
                    ),
                    
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_tvc",
                            "🚀 Run Time-Varying Cox Model",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_tvc",
                            "📥 Download TVC Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6]
                    ),
                    
                    ui.output_ui("out_tvc_result"),
                    full_screen=True
                )
            ),
            
            # TAB 6: Reference & Interpretation
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
                    | **Time-Varying Cox** | Time-dependent covariates | Dynamic HR, interval-based risk |
                    """)
                )
            )
        )
    )

# ============================================================================
# Server Logic - Modern Module Pattern
# ============================================================================
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
    Initialize server-side logic for the Survival Analysis Shiny module, wiring reactive state, dataset selection, input-choice auto-detection, and handlers for curves, landmark, Cox, subgroup, and time-varying Cox analyses.
    """
    
    # ==================== REACTIVE VALUES ====================
    curves_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    landmark_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    cox_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    sg_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    tvc_result: reactive.Value[Optional[Dict[str, Any]]] = reactive.Value(None)
    tvc_long_data: reactive.Value[Optional[pd.DataFrame]] = reactive.Value(None)
    
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
            
        cols = data.columns.tolist()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # --- AUTO-DETECTION LOGIC ---
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
            
        # Update UI choices with labels
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
        
        # TVC: Column configuration (use same detection logic)
        if len(cols) > 0:
            # ID = first non-numeric column by default
            id_candidates = [c for c in cols if c not in numeric_cols]
            default_id = id_candidates[0] if id_candidates else cols[0]
            
            ui.update_select("tvc_id_col", choices={c: get_label(c) for c in cols}, selected=default_id)
            ui.update_select("tvc_start_col", choices=num_choices_with_labels, selected=default_time)
            ui.update_select("tvc_stop_col", choices=num_choices_with_labels, selected=default_time)
            ui.update_select("tvc_event_col", choices=choices_with_labels, selected=default_event)
            
            tvc_auto = detect_tvc_columns(data)
            static_auto = detect_static_columns(data, exclude_cols=[default_id, default_time, default_event] + tvc_auto)
            
            ui.update_checkbox_group("tvc_tvc_cols", choices={c: get_label(c) for c in tvc_auto}, selected=tvc_auto)
            ui.update_checkbox_group("tvc_static_cols", choices={c: get_label(c) for c in static_auto}, selected=[])

    # ==================== EXISTING LOGIC (CURVES, LANDMARK, COX, SUBGROUP) ====================
    # ... (unchanged code for _run_curves, _run_landmark, _run_cox, _run_sg, and renderers) ...

    # ==================== 5. TIME-VARYING COX LOGIC (NEW) ====================
    @reactive.Effect
    def _update_tvc_manual_interval_visibility():
        """Show/hide manual interval input based on method selection."""
        method = input.tvc_interval_method()
        display_style = "block" if method == "manual" else "none"
        session.send_custom_message("set_element_style", {
            "id": "tvc_manual_interval_div",
            "style": {"display": display_style}
        })

    @reactive.Effect
    def _update_tvc_interval_preview():
        """Update risk interval preview text when settings change."""
        method = input.tvc_interval_method()
        manual_str = input.tvc_manual_intervals()
        data = current_df()
        
        intervals: List[float] = []
        if method == "manual" and manual_str:
            try:
                intervals = sorted(set(float(x.strip()) for x in manual_str.split(",") if x.strip() != ""))
            except ValueError:
                intervals = []
        elif data is not None:
            # Simple auto intervals using quartiles of a numeric time column
            time_col = input.surv_time()
            if time_col in data.columns:
                qs = data[time_col].quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
                intervals = sorted(set(qs))
        
        preview = format_interval_preview(intervals) if intervals else "Intervals will appear here"
        session.send_custom_message("set_inner_text", {
            "id": "tvc_interval_preview",
            "text": preview
        })

    @reactive.Effect
    def _update_tvc_preview_table():
        """Update long-format data preview whenever inputs change significantly."""
        data = current_df()
        if data is None:
            return
        
        id_col = input.tvc_id_col()
        start_col = input.tvc_start_col()
        stop_col = input.tvc_stop_col()
        event_col = input.tvc_event_col()
        tvc_cols = input.tvc_tvc_cols()
        static_cols = input.tvc_static_cols()
        
        # For now, assume long format if user selects start/stop explicitly
        if start_col == "Select..." or stop_col == "Select..." or event_col == "Select...":
            return
        
        try:
            # In long format, we don't transform; just preview the existing data
            long_df = data[[id_col, start_col, stop_col, event_col] + list(tvc_cols) + list(static_cols)].copy()
            long_df = long_df.sort_values([id_col, stop_col])
            tvc_long_data.set(long_df)
        except Exception:
            logger.exception("TVC preview update error")

    @render.ui
    def tvc_preview_summary():
        """Render a summary of the long-format data preview."""
        long_df = tvc_long_data.get()
        if long_df is None:
            return ui.markdown("*No long-format data available yet.*")
        
        id_col = input.tvc_id_col()
        n_patients = long_df[id_col].nunique()
        n_rows = len(long_df)
        n_events = long_df[input.tvc_event_col()].sum()
        
        return ui.markdown(
            f"**Patients:** {n_patients:,} | **Intervals:** {n_rows:,} | **Events:** {int(n_events):,}"
        )

    @render.data_frame
    def tvc_preview_table():
        """Render the first 50 rows of long-format data preview."""
        long_df = tvc_long_data.get()
        if long_df is None:
            return None
        
        preview_df = long_df.head(50)
        return render.DataGrid(preview_df)

    @reactive.Effect
    def _update_tvc_ui_labels():
        """Update input labels based on data format selection."""
        fmt = input.tvc_data_format()
        if fmt == "wide":
            ui.update_select("tvc_stop_col", label="⏱️ Follow-up Time:")
            session.send_custom_message("set_element_style", {"id": "div_tvc_start_col", "style": {"display": "none"}})
        else:
            ui.update_select("tvc_stop_col", label="⏱️ Interval Stop Time:")
            session.send_custom_message("set_element_style", {"id": "div_tvc_start_col", "style": {"display": "block"}})

    @reactive.Effect
    @reactive.event(input.btn_run_tvc)
    def _run_tvc():
        """Run Time-Varying Cox model using long-format data."""
        data = current_df()
        if data is None:
            ui.notification_show("No dataset available", type="warning")
            return
        
        id_col = input.tvc_id_col()
        start_col = input.tvc_start_col()
        stop_col = input.tvc_stop_col() # In Wide format, this acts as 'Time'
        event_col = input.tvc_event_col()
        tvc_cols = list(input.tvc_tvc_cols())
        static_cols = list(input.tvc_static_cols())
        penalizer = float(input.tvc_penalizer())
        
        # Validation
        if any(x == "Select..." for x in [id_col, stop_col, event_col]):
             ui.notification_show("Please configure ID, time, and event columns", type="warning")
             return
             
        if not tvc_cols and not static_cols:
            ui.notification_show("Please select at least one covariate", type="warning")
            return
        
        try:
            ui.notification_show("Fitting Time-Varying Cox Model...", duration=None, id="run_tvc")
            
            # --- 1. Handle Data Format (Long vs Wide) ---
            fmt = input.tvc_data_format()
            
            if fmt == "wide":
                # In Wide format, user selected:
                # - ID: Input ID
                # - Follow-up Time: mapped from tvc_stop_col input (label it clearly in UI if possible, but reusing input is fine)
                # - Event: Input Event
                # - Intervals: From picker
                logger.info("Transforming Wide -> Long for TVC...")
                
                # Get interval method settings
                method = input.tvc_interval_method()
                manual_str = input.tvc_manual_intervals()
                risk_intervals = None
                
                if method == "manual" and manual_str:
                    try:
                        risk_intervals = sorted(set(float(x.strip()) for x in manual_str.split(",") if x.strip() != ""))
                    except ValueError:
                        ui.notification_show("Invalid manual intervals", type="error")
                        ui.notification_remove("run_tvc")
                        return

                from utils.tvc_lib import transform_wide_to_long
                
                long_df, trans_err = transform_wide_to_long(
                    data,
                    id_col=id_col,
                    time_col=stop_col, # Reusing stop_col input as 'Time' for Wide format
                    event_col=event_col,
                    tvc_cols=tvc_cols,
                    static_cols=static_cols,
                    risk_intervals=risk_intervals,
                    interval_method=method
                )
                
                if trans_err:
                    ui.notification_show(trans_err, type="error")
                    logger.error(f"TVC Transformation Error: {trans_err}")
                    ui.notification_remove("run_tvc")
                    return
                    
                fit_data = long_df
                # For long format fitting, start/stop are fixed names from transformation
                fit_start = 'start_time'
                fit_stop = 'stop_time'
                
                # Update cols list for fitting (tvc cols are now single column 'tvc_val'...? 
                # Wait, transform_wide_to_long preserves column names but fills them?
                # ... checking tvc_lib.py -> uses same names for TVC cols in output
                
            else:
                # Long format: Use data directly
                long_df_pre = tvc_long_data.get()
                if long_df_pre is not None:
                    fit_data = long_df_pre
                else:
                    fit_data = data[[id_col, start_col, stop_col, event_col] + tvc_cols + static_cols].copy()
                
                fit_start = start_col
                fit_stop = stop_col
            
            # --- 2. Fit Model ---
            from utils.tvc_lib import check_tvc_assumptions  # Import here to ensure availability
            
            cph, res_df, clean_data, err, stats, missing_info = fit_tvc_cox(
                fit_data,
                start_col=fit_start,
                stop_col=fit_stop,
                event_col=event_col,
                tvc_cols=tvc_cols,
                static_cols=static_cols,
                penalizer=penalizer,
                var_meta=var_meta.get()
            )
            
            if err:
                ui.notification_show(err, type="error")
                ui.notification_remove("run_tvc")
                return
            
            # Forest plot
            forest_fig = create_tvc_forest_plot(res_df)
            
            # --- 3. Diagnostics (Specialized for TVC) ---
            assumption_text, assumption_plots = check_tvc_assumptions(
                cph, 
                clean_data, 
                start_col=fit_start, 
                stop_col=fit_stop, 
                event_col=event_col
            )
            
            tvc_result.set({
                'results_df': res_df,
                'forest_fig': forest_fig,
                'assumptions_text': assumption_text,
                'assumptions_plots': assumption_plots,
                'model_stats': stats,
                'missing_data_info': missing_info
            })
            
            ui.notification_remove("run_tvc")
        except Exception as e:
            ui.notification_remove("run_tvc")
            ui.notification_show(f"TVC model error: {e}", type="error")
            logger.exception("TVC model error")

    @render.ui
    def out_tvc_result():
        """Render the Time-Varying Cox model results card."""
        res = tvc_result.get()
        if res is None:
            return ui.div(
                ui.markdown("*Results will appear here after running the model.*"),
                style=f"color: {COLORS['text_secondary']}; text-align: center; padding: 20px;"
            )
        
        stats_ui = None
        if res.get('model_stats'):
            s = res['model_stats']
            stats_ui = ui.div(
                ui.div(ui.strong("C-index: "), str(s.get('Concordance Index', '-'))),
                ui.div(ui.strong("AIC: "), str(s.get('AIC', '-'))),
                ui.div(ui.strong("Events: "), f"{s.get('N Events', '-')}/{s.get('N Observations', '-')}"),
                style='display: flex; gap: 20px; padding: 10px; background: #f8f9fa; border-radius: 8px; margin-bottom: 10px;'
            )
        
        return ui.card(
            ui.card_header("📄 Time-Varying Cox Results"),
            stats_ui,
            ui.output_data_frame("out_tvc_table"),
            
            ui.card_header("🌳 Forest Plot"),
            ui.output_ui("out_tvc_forest"),
            
            ui.card_header("🔍 Diagnostics"),
            ui.output_ui("out_tvc_assumptions_ui"),
        )

    @render.data_frame
    def out_tvc_table():
        """Render TVC Cox results table."""
        res = tvc_result.get()
        return render.DataGrid(res['results_df']) if res else None

    @render.ui
    def out_tvc_forest():
        """Render TVC Cox forest plot."""
        res = tvc_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for TVC results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            res['forest_fig'],
            div_id="plot_tvc_forest",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @render.ui
    def out_tvc_assumptions_ui():
        """Render diagnostics for TVC model."""
        res = tvc_result.get()
        if not res:
            return None
        
        elements = [
            ui.div(
                ui.markdown(f"**Interpretation:**\n\n{res['assumptions_text']}"),
                style=f"padding: 15px; border-radius: 5px; background-color: {COLORS['primary']}10; border-left: 5px solid {COLORS['primary']};"
            )
        ]
        
        if res['assumptions_plots']:
            html_plots = ""
            for i, fig in enumerate(res['assumptions_plots']):
                include_js = 'cdn' if i == 0 else False
                html_plots += fig.to_html(full_html=False, include_plotlyjs=include_js)
            elements.append(ui.HTML(html_plots))
        
        return ui.div(*elements)

    @render.download(filename="tvc_report.html")
    def btn_dl_tvc():
        """Download TVC Cox analysis report."""
        res = tvc_result.get()
        if not res:
            yield "No results"
            return
        
        elements = [
            {'type': 'header', 'data': 'Time-Varying Cox Regression'},
            {'type': 'table', 'data': res['results_df']},
            {'type': 'plot', 'data': res['forest_fig']},
            {'type': 'header', 'data': 'Diagnostics'},
            {'type': 'text', 'data': res['assumptions_text']}
        ]
        
        html = generate_tvc_report(
            "Time-Varying Cox Regression",
            elements,
            stats=res.get('model_stats', {}),
            missing_data_info=res.get('missing_data_info', {}),
            var_meta=var_meta.get()
        )
        yield html

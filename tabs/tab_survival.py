"""⛳ Survival Analysis Module (Shiny) - MODERN MODULE PATTERN (UPDATED)

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

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from shiny import module, reactive, render, ui

from tabs._tvc_components import (
    detect_static_columns,
    detect_tvc_columns,
    format_interval_preview,
    tvc_column_config_ui,
    tvc_data_format_selector_ui,
    tvc_data_preview_card_ui,
    tvc_info_panel_ui,
    tvc_model_config_ui,
    tvc_risk_interval_picker_ui,
)
from utils import survival_lib
from utils.formatting import create_missing_data_report_html
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.tvc_lib import create_tvc_forest_plot, fit_tvc_cox, generate_tvc_report
from utils.ui_helpers import (
    create_empty_state_ui,
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_results_container,
    create_skeleton_loader_ui,
    create_tooltip_label,
)

try:
    from utils.subgroup_analysis_module import SubgroupAnalysisCox
except ImportError:
    SubgroupAnalysisCox = None  # type: ignore

from logger import get_logger
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)

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
                        create_input_group(
                            "Variable Selection",
                            ui.input_select(
                                "surv_time",
                                create_tooltip_label(
                                    "Time Variable", "Time to event/censoring."
                                ),
                                choices=["Select..."],
                            ),
                            ui.input_select(
                                "surv_event",
                                create_tooltip_label(
                                    "Event Variable (1=Event)",
                                    "Select the binary outcome column. '1' must indicate the event occurred (e.g., Death), '0' indicates censoring.",
                                ),
                                choices=["Select..."],
                            ),
                            ui.input_select(
                                "surv_group",
                                create_tooltip_label(
                                    "Compare Groups",
                                    "Stratify by categorical variable.",
                                ),
                                choices=["None"],
                            ),
                            type="required",
                        ),
                        create_input_group(
                            "Plot Settings",
                            ui.input_radio_buttons(
                                "plot_type",
                                "Select Plot Type:",
                                choices={
                                    "km": "Kaplan-Meier (Survival Function)",
                                    "na": "Nelson-Aalen (Cumulative Hazard)",
                                },
                                selected="km",
                                inline=True,
                            ),
                            ui.input_text(
                                "surv_time_points",
                                create_tooltip_label(
                                    "Survival Probability at (times)",
                                    "Comma-separated time points (e.g., 12, 24).",
                                ),
                                placeholder="e.g. 12, 36, 60 (Non-negative numbers only)",
                            ),
                            type="required",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_curves_validation"),
                    ui.hr(),
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
                        col_widths=[6, 6],
                    ),
                ),
                ui.output_ui(
                    "out_curves_status"
                ),  # Placeholder for loading state if needed
                create_results_container(
                    "Analysis Results", ui.output_ui("out_curves_result")
                ),
            ),
            # TAB 2: Landmark Analysis
            ui.nav_panel(
                "📊 Landmark Analysis",
                ui.card(
                    ui.card_header("Landmark Analysis for Late Endpoints"),
                    ui.div(
                        ui.markdown("""
                            **ℹ️ Principle:** Landmark analysis is useful when the treatment effect is delayed (e.g., immune-oncology) or violates proportional hazards initially.
                            
                            **How it works:**
                            1. Select a "Landmark Time" (t).
                            2. Patients who died/censored *before* t are **excluded**.
                            3. Analysis is performed only on patients who survived to time t, resetting their "start" time to t.
                            """),
                        style=f"padding: 15px; margin-bottom: 20px; background-color: {COLORS['info']}10; border-left: 4px solid {COLORS['info']}; border-radius: 4px;",
                    ),
                    ui.layout_columns(
                        create_input_group(
                            "Settings",
                            create_tooltip_label(
                                "Landmark Time (t)", "Time point for landmark analysis."
                            ),
                            ui.input_slider(
                                "landmark_t",
                                label=None,
                                min=0,
                                max=100,
                                value=10,
                                step=1,
                            ),
                            ui.input_select(
                                "landmark_group",
                                create_tooltip_label(
                                    "Compare Group", "Group variable for comparison."
                                ),
                                choices=["Select..."],
                            ),
                            type="required",
                        ),
                        col_widths=[6],
                    ),
                    ui.output_ui("out_landmark_validation"),
                    ui.hr(),
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
                        col_widths=[6, 6],
                    ),
                ),
                ui.output_ui("out_landmark_status"),
                create_results_container(
                    "Landmark Analysis Results", ui.output_ui("out_landmark_result")
                ),
            ),
            # TAB 3: Cox Regression
            ui.nav_panel(
                "📈 Cox Regression",
                ui.card(
                    ui.card_header("Cox Proportional Hazards Regression"),
                    ui.layout_columns(
                        create_input_group(
                            "Model Configuration",
                            ui.input_select(
                                "cox_method",
                                create_tooltip_label(
                                    "Fitting Method", "Algorithm for model estimation."
                                ),
                                choices={
                                    "auto": "Auto (lifelines → Firth fallback)",
                                    "lifelines": "Standard (lifelines CoxPHFitter)",
                                    "firth": "Firth (for rare events / small samples)",
                                },
                                selected="auto",
                            ),
                            create_tooltip_label(
                                "Select Covariates (Predictors)",
                                "Variables to adjust for in the model.",
                            ),
                            ui.input_selectize(
                                "cox_covariates",
                                label=None,
                                choices=[],
                                selected=[],
                                multiple=True,
                                options={"placeholder": "Select predictors..."},
                            ),
                            type="required",
                        ),
                        col_widths=[12],
                    ),
                    ui.output_ui("out_cox_validation"),
                    ui.hr(),
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
                        col_widths=[6, 6],
                    ),
                ),
                ui.output_ui("out_cox_status"),
                create_results_container(
                    "Cox Model Results", ui.output_ui("out_cox_result")
                ),
            ),
            # TAB 4: Subgroup Analysis
            ui.nav_panel(
                "🔛 Subgroup Analysis",
                ui.card(
                    ui.card_header("Cox Subgroup Analysis - Treatment Heterogeneity"),
                    ui.layout_columns(
                        create_input_group(
                            "Variables",
                            ui.input_select(
                                "sg_time",
                                create_tooltip_label(
                                    "Follow-up Time", "Time to event/censoring."
                                ),
                                choices=["Select..."],
                            ),
                            ui.input_select(
                                "sg_event",
                                create_tooltip_label(
                                    "Event Indicator", "1=Event, 0=Censored."
                                ),
                                choices=["Select..."],
                            ),
                            ui.input_select(
                                "sg_treatment",
                                create_tooltip_label(
                                    "Treatment/Exposure",
                                    "Primary variable of interest.",
                                ),
                                choices=["Select..."],
                            ),
                            type="required",
                        ),
                        create_input_group(
                            "Stratification & Adjustment",
                            ui.input_select(
                                "sg_subgroup",
                                create_tooltip_label(
                                    "Stratify By",
                                    "Categorical variable defining subgroups.",
                                ),
                                choices=["Select..."],
                            ),
                            create_tooltip_label(
                                "Adjustment Variables",
                                "Covariates to adjust for within subgroups.",
                            ),
                            ui.input_checkbox_group(
                                "sg_adjust", label=None, choices=[]
                            ),
                            type="required",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.accordion(
                        ui.accordion_panel(
                            "⚠️ Advanced Settings",
                            create_input_group(
                                "Minimum Counts",
                                ui.input_numeric(
                                    "sg_min_n",
                                    "Min N per subgroup:",
                                    value=5,
                                    min=2,
                                    max=50,
                                ),
                                ui.input_numeric(
                                    "sg_min_events",
                                    "Min events per subgroup:",
                                    value=2,
                                    min=1,
                                    max=50,
                                ),
                                type="advanced",
                            ),
                        ),
                        open=False,
                    ),
                    ui.output_ui("out_sg_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_sg",
                            "🚀 Run Subgroup Analysis",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_sg",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                ui.output_ui("out_sg_status"),
                create_results_container(
                    "Subgroup Analysis Results", ui.output_ui("out_sg_result")
                ),
            ),
            # TAB 5: Time-Varying Cox (NEW)
            ui.nav_panel(
                "⏱️ Time-Varying Cox",
                ui.card(
                    ui.card_header(
                        "Time-Dependent Survival Analysis (Time-Varying Covariates)"
                    ),
                    ui.layout_columns(
                        tvc_data_format_selector_ui(),
                        tvc_model_config_ui(),
                        col_widths=[8, 4],
                    ),
                    ui.layout_columns(
                        tvc_column_config_ui(), tvc_info_panel_ui(), col_widths=[8, 4]
                    ),
                    # Risk intervals (for wide format) + data preview
                    ui.layout_columns(
                        tvc_risk_interval_picker_ui(),
                        tvc_data_preview_card_ui(),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_tvc_validation"),
                    ui.hr(),
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
                        col_widths=[6, 6],
                    ),
                ),
                ui.output_ui("out_tvc_status"),
                create_results_container(
                    "Analysis Results", ui.output_ui("out_tvc_result")
                ),
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
                    """),
                ),
            ),
        ),
    )


# ============================================================================
# Server Logic - Modern Module Pattern
# ============================================================================
@module.server
def survival_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
) -> None:
    """
    Initialize server-side logic for the Survival Analysis Shiny module, wiring reactive state, dataset selection, input-choice auto-detection, and handlers for curves, landmark, Cox, subgroup, and time-varying Cox analyses.
    """

    # ==================== REACTIVE VALUES ====================
    curves_result: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    landmark_result: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    cox_result: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    sg_result: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    tvc_result: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    tvc_long_data: reactive.Value[pd.DataFrame | None] = reactive.Value(None)

    # Running States
    curves_is_running = reactive.Value(False)
    lm_is_running = reactive.Value(False)
    cox_is_running = reactive.Value(False)
    sg_is_running = reactive.Value(False)
    tvc_is_running = reactive.Value(False)

    # ==================== DATASET SELECTION LOGIC ====================
    @reactive.Calc
    def current_df() -> pd.DataFrame | None:
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
                    class_="text-secondary mb-3",
                ),
            )
        return ui.h3("⛳ Survival Analysis")

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
                "radio_survival_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original Data ({original_len:,} rows)",
                    "matched": f"✅ Matched Data ({matched_len:,} rows)",
                },
                selected="matched",
                inline=True,
            )
        return None

    # ==================== LABEL MAPPING LOGIC ====================
    @reactive.Calc
    def label_map() -> dict[str, str]:
        """Build a dictionary mapping raw column names to user-friendly labels from var_meta."""
        meta = var_meta.get()
        if not meta:
            return {}
        try:
            return {
                item.get("name", k): item.get("label", k) for k, item in meta.items()
            }
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
        # 1. Detect Time Variable
        default_time = select_variable_by_keyword(
            numeric_cols,
            ["time_months", "time", "day", "month", "year", "range", "followup", "fu"],
            default_to_first=True,
        )

        # 🟢 Dynamic Slider Max Logic
        max_time_val = 100
        # Get max time from default_time column
        if default_time in data.columns:
            try:
                max_t = data[default_time].max()
                if pd.notna(max_t):
                    max_time_val = int(np.ceil(max_t))
            except Exception:
                pass

        # 2. Detect Event Variable
        default_event = select_variable_by_keyword(
            cols,
            [
                "status_death",
                "status",
                "event",
                "death",
                "cure",
                "survive",
                "died",
                "outcome",
            ],
            default_to_first=True,
        )

        # 3. Detect Group/Treatment
        default_compare = select_variable_by_keyword(
            cols,
            [
                "treatment",
                "group",
                "comorbid",
                "comorb",
                "dz",
                "lab",
                "diag",
                "sex",
                "age",
            ],
            default_to_first=True,
        )

        # 4. Detect Subgroup
        default_subgr = select_variable_by_keyword(
            cols,
            [
                "comorbid",
                "comorb",
                "group",
                "control",
                "contr",
                "ctr",
                "dz",
                "lab",
                "diag",
                "sex",
                "age",
            ],
            default_to_first=True,
        )

        # Update UI choices with labels
        choices_with_labels = {c: get_label(c) for c in cols}
        num_choices_with_labels = {c: get_label(c) for c in numeric_cols}

        # KM Curves
        ui.update_select(
            "surv_time", choices=num_choices_with_labels, selected=default_time
        )
        ui.update_select(
            "surv_event", choices=choices_with_labels, selected=default_event
        )
        ui.update_select(
            "surv_group",
            choices={"None": "None", **choices_with_labels},
            selected=default_compare,
        )

        # Landmark Analysis
        ui.update_select(
            "landmark_group", choices=choices_with_labels, selected=default_compare
        )
        ui.update_slider("landmark_t", max=max_time_val, value=min(10, max_time_val))

        # Cox Regression
        # Auto-select some common covariates if available
        default_cox_covs = []
        possible_covs = [
            "Age_Years",
            "Sex_Male",
            "Treatment_Group",
            "Comorb_",
            "Lab_",
            "BMI",
        ]
        for p in possible_covs:
            for c in cols:
                if p in c and c not in [default_time, default_event, "ID"]:
                    default_cox_covs.append(c)

        # Limit default selection to avoid clutter
        default_cox_covs = default_cox_covs[:5]

        ui.update_selectize(
            "cox_covariates", choices=choices_with_labels, selected=default_cox_covs
        )

        # Subgroup Analysis
        ui.update_select(
            "sg_time", choices=num_choices_with_labels, selected=default_time
        )
        ui.update_select(
            "sg_event", choices=choices_with_labels, selected=default_event
        )
        ui.update_select(
            "sg_treatment", choices=choices_with_labels, selected=default_compare
        )
        ui.update_select(
            "sg_subgroup", choices=choices_with_labels, selected=default_subgr
        )
        ui.update_checkbox_group("sg_adjust", choices=choices_with_labels)

        # TVC: Column configuration (use specialized detection for long-format TVC data)
        if len(cols) > 0:
            # Auto-detect long-format TVC columns by name
            tvc_id_default = "id_tvc" if "id_tvc" in cols else cols[0]
            tvc_start_default = (
                "time_start" if "time_start" in numeric_cols else default_time
            )
            tvc_stop_default = (
                "time_stop" if "time_stop" in numeric_cols else default_time
            )
            tvc_event_default = (
                "status_event" if "status_event" in cols else default_event
            )

            ui.update_select(
                "tvc_id_col",
                choices={c: get_label(c) for c in cols},
                selected=tvc_id_default,
            )
            ui.update_select(
                "tvc_start_col",
                choices=num_choices_with_labels,
                selected=tvc_start_default,
            )
            ui.update_select(
                "tvc_stop_col",
                choices=num_choices_with_labels,
                selected=tvc_stop_default,
            )
            ui.update_select(
                "tvc_event_col", choices=choices_with_labels, selected=tvc_event_default
            )

            # Detect TVC and Static columns
            # Default TVC selection: Try to detect, but allow ALL cols in choices
            tvc_auto = []
            if "TVC_Value" in cols:
                tvc_auto = ["TVC_Value"]
            else:
                tvc_auto = detect_tvc_columns(data)

            # Default Static selection: Prioritize 'Static_Age', 'Static_Sex'
            static_auto = []
            desired_static = ["Static_Age", "Static_Sex"]
            for ds in desired_static:
                if ds in cols:
                    static_auto.append(ds)

            # If no desired static found, fallback to detection
            if not static_auto:
                exclude_for_static = [
                    tvc_id_default,
                    tvc_start_default,
                    tvc_stop_default,
                    tvc_event_default,
                ] + tvc_auto
                static_auto = detect_static_columns(
                    data, exclude_cols=exclude_for_static
                )

            # Update UI: Allow ALL choices, set defaults
            ui.update_selectize(
                "tvc_tvc_cols",
                choices=choices_with_labels,
                selected=tvc_auto,
            )
            ui.update_selectize(
                "tvc_static_cols",
                choices=choices_with_labels,
                selected=static_auto,
            )

    # ==================== 1. CURVES LOGIC (KM / Nelson-Aalen) ====================
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
            ui.notification_show(
                "Please select Time and Event variables", type="warning"
            )
            return

        if group_col == "None":
            group_col = None

        # Parse and validate time points
        time_points: list[float] = []
        raw_tp = input.surv_time_points()
        if raw_tp:
            try:
                parts = [p.strip() for p in raw_tp.split(",") if p.strip()]
                parsed_values = []

                for p in parts:
                    try:
                        val = float(p)
                    except ValueError:
                        raise ValueError(
                            f"Non-numeric value detected: '{p}'. Please enter numbers only."
                        ) from None

                    if val < 0:
                        raise ValueError(
                            f"Time points must be non-negative. Found: {val}"
                        )

                    parsed_values.append(val)

                unique_points = sorted(set(parsed_values))
                if len(unique_points) < len(parsed_values):
                    ui.notification_show(
                        "⚠️ Duplicate time points were found and removed.",
                        type="warning",
                    )

                time_points = unique_points

            except ValueError as e:
                ui.notification_show(f"Input Error: {str(e)}", type="error")
                return

        try:
            curves_is_running.set(True)
            curves_result.set(None)  # Clear previous
            ui.notification_show("Generating curves...", duration=None, id="run_curves")

            surv_at_times_df = None
            if time_points:
                surv_at_times_df = survival_lib.calculate_survival_at_times(
                    data, time_col, event_col, group_col, time_points
                )

            medians = None

            if plot_type == "km":
                fig, stats, missing_info = survival_lib.fit_km_logrank(
                    data, time_col, event_col, group_col, var_meta=var_meta.get()
                )
                medians = survival_lib.calculate_median_survival(
                    data, time_col, event_col, group_col
                )

            else:
                fig, stats, missing_info = survival_lib.fit_nelson_aalen(
                    data, time_col, event_col, group_col, var_meta=var_meta.get()
                )

            curves_result.set(
                {
                    "fig": fig,
                    "stats": stats,
                    "medians": medians,
                    "surv_at_times": surv_at_times_df,
                    "plot_type": plot_type,
                    "missing_data_info": missing_info,
                }
            )
            ui.notification_remove("run_curves")

        except Exception as e:
            ui.notification_remove("run_curves")
            err_msg = f"Curve error: {e!s}"
            curves_result.set({"error": err_msg})
            ui.notification_show("Analysis failed", type="error")
            logger.exception("Curve error")
        finally:
            curves_is_running.set(False)

    @render.ui
    def out_curves_result():
        """Assemble and return the UI card displaying survival-curve outputs and related statistics."""
        if curves_is_running.get():
            return ui.div(
                create_loading_state("Generating survival curves..."),
                create_skeleton_loader_ui(rows=3, show_chart=True),
            )

        res = curves_result.get()
        if res is None:
            return create_empty_state_ui(
                message="No Survival Curves",
                sub_message="Configure settings and click '🚀 Generate Curve' to visualize survival.",
                icon="📈",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        elements = [
            ui.card_header("📈 Plot"),
            ui.output_ui("out_curves_plot"),
            ui.card_header("📄 Log-Rank Test / Summary Statistics"),
            ui.output_data_frame("out_curves_table"),
        ]

        if res.get("medians") is not None:
            elements.append(ui.card_header("⏱️ Median Survival Time"))
            elements.append(ui.output_data_frame("out_medians_table"))

        if res.get("surv_at_times") is not None:
            elements.append(ui.card_header("🕰️ Survival Probability at Specific Times"))
            elements.append(ui.output_data_frame("out_surv_times_table"))

        if "missing_data_info" in res:
            elements.append(ui.card_header("⚠️ Missing Data Report"))
            elements.append(
                ui.HTML(
                    create_missing_data_report_html(
                        res["missing_data_info"], var_meta.get() or {}
                    )
                )
            )

        return ui.div(*elements, class_="fade-in-entry")

    @render.ui
    def out_curves_plot():
        """Render the survival curves Plotly figure or a waiting placeholder."""
        res = curves_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["fig"],
            div_id="plot_curves_km_na",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.data_frame
    def out_curves_table():
        """Render the curves statistics DataGrid when curve results are available."""
        res = curves_result.get()
        return render.DataGrid(res["stats"]) if res else None

    @render.data_frame
    def out_medians_table():
        """Render the median survival table."""
        res = curves_result.get()
        return (
            render.DataGrid(res["medians"])
            if res and res.get("medians") is not None
            else None
        )

    @render.data_frame
    def out_surv_times_table():
        """Render the survival at specific times table."""
        res = curves_result.get()
        return (
            render.DataGrid(res["surv_at_times"])
            if res and res.get("surv_at_times") is not None
            else None
        )

    @render.download(filename="survival_report.html")
    def btn_dl_curves():
        """Download survival curves report."""
        res = curves_result.get()
        if not res:
            yield "No results"
            return

        elements = [
            {
                "type": "header",
                "data": f"Survival Analysis ({'Kaplan-Meier' if res.get('plot_type', 'km') == 'km' else 'Nelson-Aalen'})",
            },
            {"type": "plot", "data": res["fig"]},
            {"type": "header", "data": "Statistics"},
            {"type": "table", "data": res["stats"]},
        ]

        if res.get("medians") is not None:
            elements.append({"type": "header", "data": "Median Survival Time"})
            elements.append({"type": "table", "data": res["medians"]})

        if res.get("surv_at_times") is not None:
            elements.append(
                {"type": "header", "data": "Survival Probability at Fixed Times"}
            )
            elements.append({"type": "table", "data": res["surv_at_times"]})

        yield survival_lib.generate_report_survival(
            "Survival Analysis",
            elements,
            missing_data_info=res.get("missing_data_info"),
            var_meta=var_meta.get(),
        )

    # ==================== 2. LANDMARK ANALYSIS LOGIC ====================
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
            lm_is_running.set(True)
            landmark_result.set(None)
            ui.notification_show(
                "Running Landmark Analysis...", duration=None, id="run_landmark"
            )
            fig, stats, n_pre, n_post, err, missing_info = survival_lib.fit_km_landmark(
                data, time_col, event_col, group_col, t, var_meta=var_meta.get()
            )

            if err:
                ui.notification_show(err, type="error")
            else:
                landmark_result.set(
                    {
                        "fig": fig,
                        "stats": stats,
                        "n_pre": n_pre,
                        "n_post": n_post,
                        "t": t,
                        "missing_data_info": missing_info,
                    }
                )

            ui.notification_remove("run_landmark")
        except Exception as e:
            ui.notification_remove("run_landmark")
            logger.exception("Landmark error")
            err_msg = f"Landmark analysis error: {e!s}"
            landmark_result.set({"error": err_msg})
            ui.notification_show("Analysis failed", type="error")
        finally:
            lm_is_running.set(False)

    @render.ui
    def out_landmark_result():
        """Render the landmark analysis result card containing the plot and summary statistics."""
        if lm_is_running.get():
            return ui.div(
                create_loading_state("Running Landmark Analysis..."),
                create_skeleton_loader_ui(rows=3, show_chart=True),
            )

        res = landmark_result.get()
        if res is None:
            return create_empty_state_ui(
                message="No Landmark Analysis",
                sub_message="Configure parameters and click '🚀 Run Landmark Analysis' to account for immune-delayed effects.",
                icon="📊",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        elements = [
            ui.card_header("📈 Landmark Plot"),
            ui.div(
                ui.markdown(
                    f"**Total N:** {res['n_pre']} | **Included (Survived > {res['t']}):** {res['n_post']}"
                ),
                style=f"padding: 10px; border-radius: 5px; background-color: {COLORS['info']}15; margin-bottom: 15px; border-left: 4px solid {COLORS['info']};",
            ),
            ui.output_ui("out_landmark_plot"),
            ui.output_data_frame("out_landmark_table"),
        ]

        if "missing_data_info" in res:
            elements.append(ui.card_header("⚠️ Missing Data Report"))
            elements.append(
                ui.HTML(
                    create_missing_data_report_html(
                        res["missing_data_info"], var_meta.get() or {}
                    )
                )
            )

        return ui.div(*elements, class_="fade-in-entry")

    @render.ui
    def out_landmark_plot():
        """Render the landmark analysis plot or a waiting placeholder if results are not available."""
        res = landmark_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["fig"],
            div_id="plot_landmark_analysis",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.data_frame
    def out_landmark_table():
        """Render a data grid containing landmark analysis statistics if results are available."""
        res = landmark_result.get()
        return render.DataGrid(res["stats"]) if res else None

    @render.download(filename="landmark_report.html")
    def btn_dl_landmark():
        """Download landmark analysis report."""
        res = landmark_result.get()
        if not res:
            yield "No results"
            return

        elements = [
            {"type": "header", "data": f"Landmark Analysis (t={res['t']})"},
            {"type": "plot", "data": res["fig"]},
            {"type": "header", "data": "Statistics"},
            {"type": "table", "data": res["stats"]},
        ]
        yield survival_lib.generate_report_survival(
            "Landmark Analysis",
            elements,
            missing_data_info=res.get("missing_data_info"),
            var_meta=var_meta.get(),
        )

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
            ui.notification_show(
                "Please select Time and Event variables", type="warning"
            )
            return

        if not covars:
            ui.notification_show("Select at least one covariate", type="warning")
            return

        try:
            cox_is_running.set(True)
            cox_result.set(None)
            ui.notification_show("Fitting Cox Model...", duration=None, id="run_cox")

            cox_method = input.cox_method()

            cph, res_df, clean_data, err, model_stats, missing_info = (
                survival_lib.fit_cox_ph(
                    data,
                    time_col,
                    event_col,
                    list(covars),
                    var_meta=var_meta.get(),
                    method=cox_method,
                )
            )

            if err:
                ui.notification_show(err, type="error")
                ui.notification_remove("run_cox")
                return

            # Forest Plot
            forest_fig = survival_lib.create_forest_plot_cox(res_df)

            # Check Assumptions (Schoenfeld) - only for lifelines CoxPHFitter
            from lifelines import CoxPHFitter

            if isinstance(cph, CoxPHFitter):
                assump_text, assump_plots = survival_lib.check_cph_assumptions(
                    cph, clean_data
                )
            else:
                # Firth Cox PH doesn't support Schoenfeld residuals
                assump_text = "⚠️ Proportional Hazards assumption test is not available for Firth Cox PH models."
                assump_plots = []

            cox_result.set(
                {
                    "results_df": res_df,
                    "forest_fig": forest_fig,
                    "assumptions_text": assump_text,
                    "assumptions_plots": assump_plots,
                    "model_stats": model_stats,
                    "missing_data_info": missing_info,
                }
            )

            ui.notification_remove("run_cox")

        except Exception as e:
            ui.notification_remove("run_cox")
            err_msg = f"Cox error: {e!s}"
            cox_result.set({"error": err_msg})
            ui.notification_show("Analysis failed", type="error")
            logger.exception("Cox Error")
        finally:
            cox_is_running.set(False)

    @render.ui
    def out_cox_result():
        """Render the Cox proportional hazards model results card for the UI."""
        if cox_is_running.get():
            return ui.div(
                create_loading_state("Fitting Cox Proportional Hazards Model..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = cox_result.get()
        if res is None:
            return create_empty_state_ui(
                message="No Cox Model Results",
                sub_message="Select covariates and run the model.",
                icon="📄",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        stats_ui = None
        if res.get("model_stats"):
            s = res["model_stats"]
            stats_ui = ui.div(
                ui.div(
                    ui.strong("C-index: "),
                    str(s.get("Concordance Index (C-index)", "-")),
                ),
                ui.div(ui.strong("AIC: "), str(s.get("AIC", "-"))),
                ui.div(
                    ui.strong("Events: "),
                    f"{s.get('Number of Events', '-')} / {s.get('Number of Observations', '-')}",
                ),
                style="display: flex; gap: 20px; padding: 10px; background: #f8f9fa; border-radius: 8px; margin-bottom: 10px;",
            )

        elements = [
            ui.card_header("📄 Cox Results"),
            stats_ui,
            ui.output_data_frame("out_cox_table"),
            ui.card_header("🌳 Forest Plot"),
            ui.output_ui("out_cox_forest"),
            ui.card_header("🔍 PH Assumption (Schoenfeld Residuals)"),
            ui.output_ui("out_cox_assumptions_ui"),
        ]

        if "missing_data_info" in res:
            elements.append(ui.card_header("⚠️ Missing Data Report"))
            elements.append(
                ui.HTML(
                    create_missing_data_report_html(
                        res["missing_data_info"], var_meta.get() or {}
                    )
                )
            )

        return ui.div(*elements, class_="fade-in-entry")

    @render.data_frame
    def out_cox_table():
        """Render a data grid showing Cox regression results if available."""
        res = cox_result.get()
        return render.DataGrid(res["results_df"]) if res else None

    @render.ui
    def out_cox_forest():
        """Render the Cox regression forest plot if available; otherwise show a waiting placeholder."""
        res = cox_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["forest_fig"],
            div_id="plot_cox_forest",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.ui
    def out_cox_assumptions_ui():
        """Render the proportional hazards assumption section for Cox regression."""
        res = cox_result.get()
        if not res:
            return None

        elements = [
            ui.div(
                ui.markdown(f"**Interpretation:**\n\n{res['assumptions_text']}"),
                style=f"padding: 15px; border-radius: 5px; background-color: {COLORS['primary']}10; border-left: 5px solid {COLORS['primary']};",
            )
        ]

        if res["assumptions_plots"]:
            html_plots = ""
            for i, fig in enumerate(res["assumptions_plots"]):
                include_js = "cdn" if i == 0 else False
                html_plots += fig.to_html(full_html=False, include_plotlyjs=include_js)

            elements.append(ui.HTML(html_plots))

        return ui.div(*elements, class_="fade-in-entry")

    @render.download(filename="cox_report.html")
    def btn_dl_cox():
        """Download Cox regression report."""
        res = cox_result.get()
        if not res:
            yield "No results"
            return

        elements = [
            {"type": "header", "data": "Cox Proportional Hazards Regression"},
        ]

        if res.get("model_stats"):
            s = res["model_stats"]
            stats_text = f"C-index: {s.get('Concordance Index (C-index)')}, AIC: {s.get('AIC')}, Events: {s.get('Number of Events')}"
            elements.append({"type": "text", "data": stats_text})

        elements.extend(
            [
                {"type": "table", "data": res["results_df"]},
                {"type": "plot", "data": res["forest_fig"]},
                {"type": "header", "data": "PH Assumptions"},
                {"type": "text", "data": res["assumptions_text"]},
            ]
        )
        yield survival_lib.generate_report_survival(
            "Cox Regression",
            elements,
            missing_data_info=res.get("missing_data_info"),
            var_meta=var_meta.get(),
        )

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
            sg_is_running.set(True)
            sg_result.set(None)
            ui.notification_show(
                "Running Subgroup Analysis...", duration=None, id="run_sg"
            )

            analyzer = SubgroupAnalysisCox(data)  # type: ignore
            result, _, error = analyzer.analyze(
                duration_col=time,
                event_col=event,
                treatment_col=treat,
                subgroup_col=subgroup,
                adjustment_cols=list(adjust) if adjust else None,
                min_subgroup_n=input.sg_min_n(),
                min_events=input.sg_min_events(),
                var_meta=var_meta.get(),
            )
            if error:
                ui.notification_show(error, type="error")
                ui.notification_remove("run_sg")
                return

            # Generate forest plot
            forest_fig = analyzer.create_forest_plot()
            result["forest_plot"] = forest_fig
            result["interaction_table"] = analyzer.results
            sg_result.set(result)

            ui.notification_remove("run_sg")
        except Exception as e:
            ui.notification_remove("run_sg")
            err_msg = f"Subgroup analysis error: {e!s}"
            sg_result.set({"error": err_msg})
            ui.notification_show("Analysis failed", type="error")
            logger.exception("Subgroup analysis error")
        finally:
            sg_is_running.set(False)

    @render.ui
    def out_sg_result():
        """Builds the UI card displaying subgroup analysis results."""
        if sg_is_running.get():
            return ui.div(
                create_loading_state("Running Subgroup Analysis..."),
                create_skeleton_loader_ui(rows=3, show_chart=True),
            )

        res = sg_result.get()
        if res is None:
            return create_empty_state_ui(
                message="No Subgroup Analysis",
                sub_message="Define subgroups and run analysis to detect treatment heterogeneity.",
                icon="🔛",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        elements = []
        if "forest_plot" in res:
            elements.append(ui.card_header("🌳 Subgroup Forest Plot"))
            elements.append(ui.output_ui("out_sg_forest"))

        if "interaction_table" in res:
            elements.append(ui.card_header("📄 Interaction Analysis"))
            elements.append(ui.output_data_frame("out_sg_table"))

        # Missing Data Report
        if "missing_data_info" in res:
            elements.append(ui.card_header("⚠️ Missing Data Report"))
            elements.append(
                ui.HTML(
                    create_missing_data_report_html(
                        res["missing_data_info"], var_meta.get() or {}
                    )
                )
            )

        return ui.div(*elements, class_="fade-in-entry")

    @render.ui
    def out_sg_forest():
        """Render the subgroup analysis forest plot or a placeholder when results are unavailable."""
        res = sg_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        fig = res.get("forest_plot")
        if fig is None:
            return ui.div(
                ui.markdown("⏳ *No forest plot available...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            fig, div_id="plot_subgroup_forest", include_plotlyjs="cdn", responsive=True
        )
        return ui.HTML(html_str)

    @render.data_frame
    def out_sg_table():
        """Render the subgroup interaction analysis table when subgroup results are available."""
        res = sg_result.get()
        return render.DataGrid(res.get("interaction_table")) if res else None

    @render.download(filename="subgroup_report.html")
    def btn_dl_sg():
        """Download subgroup analysis report."""
        res = sg_result.get()
        if not res:
            yield "No results"
            return

        elements = [
            {"type": "header", "data": "Cox Subgroup Analysis"},
            {"type": "plot", "data": res.get("forest_plot")},
            {"type": "header", "data": "Results"},
            {"type": "table", "data": res.get("interaction_table")},
        ]
        elements = [e for e in elements if e.get("data") is not None]
        yield survival_lib.generate_report_survival("Subgroup Analysis", elements)

    # ==================== 5. TIME-VARYING COX LOGIC (NEW) ====================
    @reactive.Effect
    async def _update_tvc_manual_interval_visibility():
        """Show/hide manual interval input based on method selection."""
        method = input.tvc_interval_method()
        display_style = "block" if method == "manual" else "none"
        await session.send_custom_message(
            "set_element_style",
            {"id": "tvc_manual_interval_div", "style": {"display": display_style}},
        )

    # --- TVC Interval Presets ---
    @reactive.Effect
    @reactive.event(input.btn_tvc_preset_quarterly)
    def _set_quarterly_intervals():
        """Set manual intervals to Quarterly (every 3 months up to 24m)."""
        ui.update_text("tvc_manual_intervals", value="0, 3, 6, 9, 12, 15, 18, 21, 24")

    @reactive.Effect
    @reactive.event(input.btn_tvc_preset_biannual)
    def _set_biannual_intervals():
        """Set manual intervals to Biannual (every 6 months up to 48m)."""
        ui.update_text("tvc_manual_intervals", value="0, 6, 12, 18, 24, 30, 36, 42, 48")

    @reactive.Effect
    @reactive.event(input.btn_tvc_preset_yearly)
    def _set_yearly_intervals():
        """Set manual intervals to Yearly (every 12 months up to 60m)."""
        ui.update_text("tvc_manual_intervals", value="0, 12, 24, 36, 48, 60")

    @reactive.Effect
    async def _update_tvc_interval_preview():
        """Update risk interval preview text when settings change."""
        method = input.tvc_interval_method()
        manual_str = input.tvc_manual_intervals()
        data = current_df()

        intervals: list[float] = []
        if method == "manual" and manual_str:
            try:
                intervals = sorted(
                    set(
                        float(x.strip())
                        for x in manual_str.split(",")
                        if x.strip() != ""
                    )
                )
            except ValueError:
                intervals = []
        elif data is not None:
            # Simple auto intervals using quartiles of a numeric time column
            # In Wide format, tvc_stop_col acts as fallback time col if input.surv_time() is not relevant
            # But let's verify if surv_time() is mapped. Usually user sets tvc_stop_col.
            time_col = input.tvc_stop_col()
            if time_col != "Select..." and time_col in data.columns:
                # Ensure numeric
                if pd.api.types.is_numeric_dtype(data[time_col]):
                    qs = data[time_col].quantile([0, 0.25, 0.5, 0.75, 1.0]).tolist()
                    intervals = sorted(set(qs))

        preview = (
            format_interval_preview(intervals)
            if intervals
            else "Intervals will appear here"
        )
        await session.send_custom_message(
            "set_inner_text", {"id": "tvc_interval_preview", "text": preview}
        )

    @reactive.Effect
    def _update_tvc_preview_table():
        """Update long-format data preview whenever inputs change significantly."""
        data = current_df()
        if data is None:
            return

        fmt = input.tvc_data_format()
        id_col = input.tvc_id_col()
        start_col = input.tvc_start_col()
        stop_col = input.tvc_stop_col()
        event_col = input.tvc_event_col()
        tvc_cols = list(input.tvc_tvc_cols())
        static_cols = list(input.tvc_static_cols())

        # Basic validation
        if any(c == "Select..." for c in [id_col, stop_col, event_col]):
            return

        try:
            long_df = None

            if fmt == "wide":
                # Preview transformation for Wide format
                method = input.tvc_interval_method()
                manual_str = input.tvc_manual_intervals()
                risk_intervals = None

                if method == "manual" and manual_str:
                    try:
                        risk_intervals = sorted(
                            set(
                                float(x.strip())
                                for x in manual_str.split(",")
                                if x.strip() != ""
                            )
                        )
                    except ValueError:
                        pass  # Ignore invalid manual input during preview

                from utils.tvc_lib import transform_wide_to_long

                long_df, _ = transform_wide_to_long(
                    data,
                    id_col=id_col,
                    time_col=stop_col,
                    event_col=event_col,
                    tvc_cols=tvc_cols,
                    static_cols=static_cols,
                    risk_intervals=risk_intervals,
                    interval_method=method,
                )
            else:
                # Long format: Use data directly
                # Ensure start_col is selected
                if start_col == "Select...":
                    return

                # Prevent duplicate columns selection
                req_cols = (
                    [id_col, start_col, stop_col, event_col] + tvc_cols + static_cols
                )
                # Remove duplicates while preserving order
                unique_cols = list(dict.fromkeys(req_cols))

                long_df = data[unique_cols].copy()

            if long_df is not None:
                # Sort if possible (might fail if columns missing or duplicates)
                try:
                    sort_cols = [c for c in [id_col, stop_col] if c in long_df.columns]
                    # Also rename columns to standard start/stop if transformed?
                    # transform_wide_to_long returns 'start_time', 'stop_time'
                    # But input values are retained. If we are in wide mode, id_col is preserved.
                    if fmt == "wide":
                        # Standardize sort column names for transformed data
                        if "start" in long_df.columns:
                            long_df = long_df.sort_values([id_col, "stop"])
                    else:
                        long_df = long_df.sort_values(sort_cols)
                except Exception as e:
                    logger.warning(f"Sort preview failed: {e}")

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
    async def _update_tvc_ui_labels():
        """Update input labels based on data format selection."""
        fmt = input.tvc_data_format()
        if fmt == "wide":
            ui.update_select("tvc_stop_col", label="⏱️ Follow-up Time:")
            await session.send_custom_message(
                "set_element_style",
                {"id": "div_tvc_start_col", "style": {"display": "none"}},
            )
        else:
            ui.update_select("tvc_stop_col", label="⏱️ Interval Stop Time:")
            await session.send_custom_message(
                "set_element_style",
                {"id": "div_tvc_start_col", "style": {"display": "block"}},
            )

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
        stop_col = input.tvc_stop_col()  # In Wide format, this acts as 'Time'
        event_col = input.tvc_event_col()
        tvc_cols = list(input.tvc_tvc_cols())
        static_cols = list(input.tvc_static_cols())
        penalizer = float(input.tvc_penalizer())

        # Validation
        if any(x == "Select..." for x in [id_col, stop_col, event_col]):
            ui.notification_show(
                "Please configure ID, time, and event columns", type="warning"
            )
            return

        if not tvc_cols and not static_cols:
            ui.notification_show("Please select at least one covariate", type="warning")
            return

        try:
            tvc_is_running.set(True)
            tvc_result.set(None)
            ui.notification_show(
                "Fitting Time-Varying Cox Model...", duration=None, id="run_tvc"
            )

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
                        risk_intervals = sorted(
                            set(
                                float(x.strip())
                                for x in manual_str.split(",")
                                if x.strip() != ""
                            )
                        )
                    except ValueError:
                        ui.notification_show("Invalid manual intervals", type="error")
                        ui.notification_remove("run_tvc")
                        return

                from utils.tvc_lib import transform_wide_to_long

                long_df, trans_err = transform_wide_to_long(
                    data,
                    id_col=id_col,
                    time_col=stop_col,  # Reusing stop_col input as 'Time' for Wide format
                    event_col=event_col,
                    tvc_cols=tvc_cols,
                    static_cols=static_cols,
                    risk_intervals=risk_intervals,
                    interval_method=method,
                )

                if trans_err:
                    ui.notification_show(trans_err, type="error")
                    logger.error(f"TVC Transformation Error: {trans_err}")
                    ui.notification_remove("run_tvc")
                    return

                fit_data = long_df
                # For long format fitting, start/stop are fixed names from transformation
                # transform_wide_to_long outputs 'start' and 'stop' columns (see tvc_lib.py lines 284-285)
                fit_start = "start"
                fit_stop = "stop"

                # Update cols list for fitting (tvc cols are now single column 'tvc_val'...?
                # Wait, transform_wide_to_long preserves column names but fills them?
                # ... checking tvc_lib.py -> uses same names for TVC cols in output

            else:
                # Long format: Use data directly
                long_df_pre = tvc_long_data.get()
                if long_df_pre is not None:
                    fit_data = long_df_pre
                else:
                    # Construct from fresh selection, ensuring unique columns
                    req_cols = (
                        [id_col, start_col, stop_col, event_col]
                        + tvc_cols
                        + static_cols
                    )
                    unique_cols = list(dict.fromkeys(req_cols))
                    fit_data = data[unique_cols].copy()

                fit_start = start_col
                fit_stop = stop_col

            # --- 2. Fit Model ---
            from utils.tvc_lib import (  # Import here to ensure availability
                check_tvc_assumptions,
            )

            cph, res_df, clean_data, err, stats, missing_info = fit_tvc_cox(
                fit_data,
                start_col=fit_start,
                stop_col=fit_stop,
                event_col=event_col,
                tvc_cols=tvc_cols,
                static_cols=static_cols,
                penalizer=penalizer,
                var_meta=var_meta.get(),
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
                event_col=event_col,
            )

            tvc_result.set(
                {
                    "results_df": res_df,
                    "forest_fig": forest_fig,
                    "assumptions_text": assumption_text,
                    "assumptions_plots": assumption_plots,
                    "model_stats": stats,
                    "missing_data_info": missing_info,
                }
            )

            ui.notification_remove("run_tvc")
        except Exception as e:
            ui.notification_remove("run_tvc")
            err_msg = f"TVC Error: {e!s}"
            tvc_result.set({"error": err_msg})
            ui.notification_show("Analysis failed", type="error")
            logger.exception("TVC Execution Error")
        finally:
            tvc_is_running.set(False)

    @render.ui
    def out_tvc_result():
        """Render the Time-Varying Cox regression results card."""
        if tvc_is_running.get():
            return ui.div(
                create_loading_state("Fitting Time-Varying Cox Model..."),
                create_skeleton_loader_ui(rows=4, show_chart=True),
            )

        res = tvc_result.get()
        if res is None:
            return create_empty_state_ui(
                message="No Time-Varying Cox Results",
                sub_message="Configure manual intervals or long format data and run model.",
                icon="⏱️",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        stats_ui = None
        if res.get("model_stats"):
            s = res["model_stats"]
            stats_ui = ui.div(
                ui.div(ui.strong("C-index: "), str(s.get("Concordance Index", "-"))),
                ui.div(ui.strong("AIC: "), str(s.get("AIC", "-"))),
                ui.div(
                    ui.strong("Events: "),
                    f"{s.get('N Events', '-')}/{s.get('N Observations', '-')}",
                ),
                style="display: flex; gap: 20px; padding: 10px; background: #f8f9fa; border-radius: 8px; margin-bottom: 10px;",
            )

        elements = [
            ui.card_header("📄 Time-Varying Cox Results"),
            stats_ui,
            ui.output_data_frame("out_tvc_table"),
            ui.card_header("🌳 Forest Plot"),
            ui.output_ui("out_tvc_forest"),
            ui.card_header("🔍 Diagnostics"),
            ui.output_ui("out_tvc_assumptions_ui"),
        ]

        if "missing_data_info" in res:
            elements.append(ui.card_header("⚠️ Missing Data Report"))
            elements.append(
                ui.HTML(
                    create_missing_data_report_html(
                        res["missing_data_info"], var_meta.get() or {}
                    )
                )
            )

        return ui.div(*elements, class_="fade-in-entry")

    @render.data_frame
    def out_tvc_table():
        """Render TVC Cox results table."""
        res = tvc_result.get()
        return render.DataGrid(res["results_df"]) if res else None

    @render.ui
    def out_tvc_forest():
        """Render TVC Cox forest plot."""
        res = tvc_result.get()
        if res is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for TVC results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            res["forest_fig"],
            div_id="plot_tvc_forest",
            include_plotlyjs="cdn",
            responsive=True,
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
                style=f"padding: 15px; border-radius: 5px; background-color: {COLORS['primary']}10; border-left: 5px solid {COLORS['primary']};",
            )
        ]

        if res["assumptions_plots"]:
            html_plots = ""
            for i, fig in enumerate(res["assumptions_plots"]):
                include_js = "cdn" if i == 0 else False
                html_plots += fig.to_html(full_html=False, include_plotlyjs=include_js)
            elements.append(ui.HTML(html_plots))

        return ui.div(*elements, class_="fade-in-entry")

    @render.download(filename="tvc_report.html")
    def btn_dl_tvc():
        """Download TVC Cox analysis report."""
        res = tvc_result.get()
        if not res:
            yield "No results"
            return

        elements = [
            {"type": "header", "data": "Time-Varying Cox Regression"},
            {"type": "table", "data": res["results_df"]},
            {"type": "plot", "data": res["forest_fig"]},
            {"type": "header", "data": "Diagnostics"},
            {"type": "text", "data": res["assumptions_text"]},
        ]

        html = generate_tvc_report(
            "Time-Varying Cox Regression",
            elements,
            stats=res.get("model_stats", {}),
            missing_data_info=res.get("missing_data_info", {}),
            var_meta=var_meta.get(),
        )
        yield html

    # ==================== VALIDATION LOGIC ====================
    @render.ui
    def out_curves_validation():
        d = current_df()
        time_col = input.surv_time()
        event_col = input.surv_event()
        if d is None or d.empty:
            return None
        alerts = []

        if time_col == "Select..." or event_col == "Select...":
            return None  # Wait for selection

        if time_col == event_col:
            alerts.append(
                create_error_alert(
                    "Time and Event variables must be different.",
                    title="Configuration Error",
                )
            )

        if event_col in d.columns and d[event_col].nunique() > 2:
            alerts.append(
                create_error_alert(
                    f"Event variable '{event_col}' should be binary (0/1). It has {d[event_col].nunique()} unique values.",
                    title="Warning",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_landmark_validation():
        d = current_df()
        time_col = input.surv_time()
        lm_t = input.landmark_t()

        if d is None or d.empty or time_col == "Select..." or time_col not in d.columns:
            return None
        alerts = []

        max_time = d[time_col].max()
        if lm_t >= max_time:
            alerts.append(
                create_error_alert(
                    f"Landmark time ({lm_t}) must be less than the maximum follow-up time ({max_time}).",
                    title="Invalid Landmark",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_cox_validation():
        d = current_df()
        covs = input.cox_covariates()
        time_col = input.surv_time()
        event_col = input.surv_event()

        if d is None or d.empty:
            return None
        alerts = []

        if not covs:
            return None  # Silent if empty

        if time_col in covs or event_col in covs:
            alerts.append(
                create_error_alert(
                    "Time or Event variable cannot be used as a covariate.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_sg_validation():
        d = current_df()
        subgroup = input.sg_subgroup()
        treatment = input.sg_treatment()

        if d is None or d.empty:
            return None
        alerts = []

        if subgroup == "Select...":
            return None

        if subgroup == treatment:
            alerts.append(
                create_error_alert(
                    "Subgroup variable cannot be the same as the Treatment variable.",
                    title="Configuration Error",
                )
            )

        if subgroup in d.columns and d[subgroup].nunique() > 10:
            alerts.append(
                create_error_alert(
                    f"Subgroup variable '{subgroup}' has {d[subgroup].nunique()} levels. Recommended < 10.",
                    title="Warning",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_tvc_validation():
        d = current_df()
        fmt = input.tvc_data_format()

        if d is None or d.empty:
            return None
        alerts = []

        if fmt == "wide":
            # Check ID column uniqueness for wide format (should be unique per subject)
            id_col = input.tvc_id_col()
            if id_col and id_col in d.columns:
                if d[id_col].duplicated().any():
                    alerts.append(
                        create_error_alert(
                            f"ID column '{id_col}' contains duplicates. Wide format requires unique IDs.",
                            title="Data Format Error",
                        )
                    )

        if alerts:
            return ui.div(*alerts)
        return None

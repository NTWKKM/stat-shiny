from __future__ import annotations

import html as _html
import re
from typing import Any

import numpy as np
import pandas as pd
from shiny import module, reactive, render, req, ui

from logger import get_logger
from tabs._common import (
    select_variable_by_keyword,
)
from utils import diag_test
from utils.formatting import create_missing_data_report_html
from utils.ui_helpers import (
    create_error_alert,
    create_loading_state,
    create_placeholder_state,
)

logger = get_logger(__name__)


def _safe_filename_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    return s[:80] or "value"


def _auto_detect_icc_vars(cols: list[str]) -> list[str]:
    """Auto-detect ICC/Rater variables based on column name patterns."""
    icc_patterns = ["diagnosis", "icc", "rater", "method", "observer", "judge"]
    detected = []
    for col in cols:
        col_lower = col.lower()
        for pattern in icc_patterns:
            if pattern in col_lower:
                detected.append(col)
                break
    return detected


# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def agreement_ui() -> ui.TagChild:
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
            # TAB 1: Kappa (Categorical Agreement)
            ui.nav_panel(
                "🤝 Cohen's Kappa",
                ui.markdown("##### Cohen's Kappa (Categorical Data Agreement)"),
                ui.row(
                    ui.column(6, ui.output_ui("ui_kappa_v1")),
                    ui.column(6, ui.output_ui("ui_kappa_v2")),
                ),
                ui.output_ui("out_kappa_validation"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_analyze_kappa",
                            "🚀 Calculate Kappa",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.download_button(
                            "btn_dl_kappa_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("out_kappa_results"),
            ),
            # TAB 2: Bland-Altman (Continuous Agreement)
            ui.nav_panel(
                "📉 Bland-Altman",
                ui.markdown("##### Bland-Altman Analysis (Continuous Data Comparison)"),
                ui.row(
                    ui.column(6, ui.output_ui("ui_ba_v1")),
                    ui.column(6, ui.output_ui("ui_ba_v2")),
                ),
                ui.output_ui("out_ba_validation"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_analyze_ba",
                            "🚀 Analyze Agreement",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.download_button(
                            "btn_dl_ba_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("out_ba_results"),
            ),
            # TAB 3: ICC (Reliability)
            ui.nav_panel(
                "🔍 Reliability (ICC)",
                ui.markdown("##### Intraclass Correlation Coefficient (ICC)"),
                ui.row(
                    ui.column(12, ui.output_ui("ui_icc_vars")),
                ),
                ui.output_ui("out_icc_validation"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_run_icc",
                            "🔍 Calculate ICC",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.download_button(
                            "btn_dl_icc_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("out_icc_results"),
            ),
            # TAB 4: Reference & Interpretation
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                    ## 📚 Agreement & Reliability Reference Guide

                    ### 🤝 Cohen's Kappa
                    Used for **categorical (groups)** data to measure agreement between two raters/methods.
                    It accounts for the agreement occurring by chance.
                    - **Kappa > 0.8:** Almost perfect agreement ✅
                    - **Kappa 0.6-0.8:** Substantial agreement
                    - **Kappa 0.4-0.6:** Moderate agreement
                    - **Kappa 0.2-0.4:** Fair agreement ⚠️
                    - **Kappa < 0.2:** Slight/Poor agreement ❌

                    ### 📉 Bland-Altman Plot
                    Used for **continuous** data to compare two measurement methods.
                    - **Bias (Mean Difference):** How much one method differs from the other on average.
                    - **95% Limits of Agreement (LoA):** The range where 95% of differences are expected to lie.
                    - **Proportional Bias:** If the difference increases with the mean (visual check required).

                    ### 🔍 Intraclass Correlation (ICC)
                    Used for **continuous** data to measure reliability among 2 or more raters/measurements.
                    - **ICC(2,1):** Two-way random effects, absolute agreement (Single rater).
                    - **ICC(3,1):** Two-way mixed effects, consistency (Fixed raters).
                    - **Interpretation:**
                        - **> 0.90:** Excellent Reliability 🌟
                        - **0.75 - 0.90:** Good Reliability
                        - **0.50 - 0.75:** Moderate Reliability ⚠️
                        - **< 0.50:** Poor Reliability ❌
                    """),
            ),
        ),
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def agreement_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
) -> None:
    # --- Reactive Results ---
    kappa_html: reactive.Value[str | None] = reactive.Value(None)
    ba_html: reactive.Value[str | None] = reactive.Value(None)
    icc_html: reactive.Value[str | None] = reactive.Value(None)

    kappa_processing: reactive.Value[bool] = reactive.Value(False)
    ba_processing: reactive.Value[bool] = reactive.Value(False)
    icc_processing: reactive.Value[bool] = reactive.Value(False)

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df() -> pd.DataFrame | None:
        if is_matched.get():
            try:
                if input.radio_source() == "matched":
                    return df_matched.get()
            except (AttributeError, KeyError):
                pass
        return df.get()

    @render.ui
    def ui_title_with_summary():
        d = current_df()
        if d is not None:
            return ui.div(
                ui.h3("🤝 Agreement & Reliability"),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3",
                ),
            )
        return ui.h3("🤝 Agreement & Reliability")

    @render.ui
    def ui_matched_info():
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
        if is_matched.get():
            return ui.input_radio_buttons(
                "radio_source",
                "📄 Select Dataset:",
                {
                    "original": "📊 Original Data",
                    "matched": "✅ Matched Data (from PSM)",
                },
                selected="matched",
                inline=True,
            )
        return None

    # Get all columns
    @reactive.Calc
    def all_cols() -> list[str]:
        d = current_df()
        if d is not None:
            return d.columns.tolist()
        return []

    @reactive.Calc
    def num_cols() -> list[str]:
        d = current_df()
        if d is not None:
            return d.select_dtypes(include=[np.number]).columns.tolist()
        return []

    # --- Kappa Inputs ---
    @render.ui
    def ui_kappa_v1():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["diagnosis_dr_a", "rater1", "obs1", "method1"], default_to_first=True
        )
        return ui.input_select(
            "sel_kappa_v1", "Rater/Method 1:", choices=cols, selected=default
        )

    @render.ui
    def ui_kappa_v2():
        cols = all_cols()
        v1 = input.sel_kappa_v1()
        rem_cols = [c for c in cols if c != v1] if v1 else cols
        default = select_variable_by_keyword(
            rem_cols,
            ["diagnosis_dr_b", "rater2", "obs2", "method2"],
            default_to_first=True,
        )
        return ui.input_select(
            "sel_kappa_v2", "Rater/Method 2:", choices=rem_cols, selected=default
        )

    # --- Bland-Altman Inputs ---
    @render.ui
    def ui_ba_v1():
        cols = num_cols()
        default = select_variable_by_keyword(
            cols, ["rater1", "method1", "measure1", "bp_r1"], default_to_first=True
        )
        return ui.input_select(
            "sel_ba_v1", "Method A (Reference):", choices=cols, selected=default
        )

    @render.ui
    def ui_ba_v2():
        cols = num_cols()
        v1 = input.sel_ba_v1()
        rem_cols = [c for c in cols if c != v1] if v1 else cols
        default = select_variable_by_keyword(
            rem_cols, ["rater2", "method2", "measure2", "bp_r2"], default_to_first=True
        )
        return ui.input_select(
            "sel_ba_v2", "Method B (New):", choices=cols, selected=default
        )

    # --- ICC Inputs ---
    @render.ui
    def ui_icc_vars():
        cols = num_cols()
        defaults = _auto_detect_icc_vars(cols)
        return ui.input_selectize(
            "icc_vars",
            "Select Variables (2+ Raters/Methods):",
            choices=cols,
            multiple=True,
            selected=defaults,
        )

    # ==================== KAPPA LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_analyze_kappa)
    def _run_kappa():
        d = current_df()
        v1, v2 = input.sel_kappa_v1(), input.sel_kappa_v2()
        req(d is not None, v1, v2)

        kappa_processing.set(True)

        try:
            res, err, conf, missing_info = diag_test.calculate_kappa(
                d, v1, v2, var_meta=var_meta.get() or {}
            )
            if err:
                kappa_html.set(
                    f"<div class='alert alert-danger'>{_html.escape(str(err))}</div>"
                )
            else:
                rep = [
                    {
                        "type": "text",
                        "data": f"Analysis: Kappa Agreement between {v1} and {v2}",
                    },
                    {"type": "table", "header": "Kappa Statistics", "data": res},
                    {
                        "type": "table",
                        "header": "Confusion Matrix (Crosstab)",
                        "data": conf,
                        "safe_html": True,
                    },
                ]
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )
                kappa_html.set(diag_test.generate_report(f"Kappa: {v1} vs {v2}", rep))
        except Exception as e:
            logger.exception("Kappa failed")
            kappa_html.set(
                f"<div class='alert alert-danger'>Error: {_html.escape(str(e))}</div>"
            )
        finally:
            kappa_processing.set(False)

    @render.ui
    def out_kappa_results():
        if kappa_processing.get():
            return create_loading_state("Calculating Kappa Statistics...")
        if kappa_html.get():
            return ui.HTML(kappa_html.get())
        return create_placeholder_state(
            "Select variables and click 'Calculate Kappa'.", icon="🤝"
        )

    @render.download(
        filename=lambda: f"kappa_{_safe_filename_part(input.sel_kappa_v1())}_{_safe_filename_part(input.sel_kappa_v2())}_report.html"
    )
    def btn_dl_kappa_report():
        req(kappa_html.get())
        yield kappa_html.get()

    # ==================== BLAND-ALTMAN LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_analyze_ba)
    def _run_ba():
        d = current_df()
        v1, v2 = input.sel_ba_v1(), input.sel_ba_v2()
        req(d is not None, v1, v2)

        ba_processing.set(True)

        try:
            result = diag_test.calculate_bland_altman(d, v1, v2)
            if len(result) == 2:
                stats_res, fig = result
                missing_info = {}
            else:
                stats_res, fig, missing_info = result

            if "error" in stats_res:
                ba_html.set(
                    f"<div class='alert alert-danger'>{_html.escape(str(stats_res['error']))}</div>"
                )
            else:
                stats_df = pd.DataFrame(
                    [
                        {
                            "Mean Diff (Bias)": f"{stats_res.get('mean_diff', 0):.4f}",
                            "95% CI Bias": f"{stats_res.get('ci_mean_diff', [0, 0])[0]:.4f} to {stats_res.get('ci_mean_diff', [0, 0])[1]:.4f}",
                            "Upper LoA": f"{stats_res.get('upper_loa', 0):.4f}",
                            "Lower LoA": f"{stats_res.get('lower_loa', 0):.4f}",
                            "N (Valid Pairs)": stats_res.get("n", 0),
                        }
                    ]
                ).T

                rep = [
                    {"type": "text", "data": f"Bland-Altman Comparison: {v1} vs {v2}"},
                    {"type": "plot", "data": fig},
                    {
                        "type": "table",
                        "header": "Agreement Statistics",
                        "data": stats_df,
                    },
                ]
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )
                ba_html.set(
                    diag_test.generate_report(f"Bland-Altman: {v1} vs {v2}", rep)
                )
        except Exception as e:
            logger.exception("Bland-Altman failed")
            ba_html.set(
                f"<div class='alert alert-danger'>Error: {_html.escape(str(e))}</div>"
            )
        finally:
            ba_processing.set(False)

    @render.ui
    def out_ba_results():
        if ba_processing.get():
            return create_loading_state("Generating Bland-Altman Analysis...")
        if ba_html.get():
            return ui.HTML(ba_html.get())
        return create_placeholder_state(
            "Select variables and click 'Analyze Agreement'.", icon="📉"
        )

    @render.download(
        filename=lambda: f"ba_{_safe_filename_part(input.sel_ba_v1())}_{_safe_filename_part(input.sel_ba_v2())}_report.html"
    )
    def btn_dl_ba_report():
        req(ba_html.get())
        yield ba_html.get()

    # ==================== ICC LOGIC ====================
    @reactive.Effect
    @reactive.event(input.btn_run_icc)
    def _run_icc():
        d = current_df()
        cols = input.icc_vars()
        req(d is not None, cols, len(cols) >= 2)

        icc_processing.set(True)

        try:
            res_df, err, anova_df, missing_info = diag_test.calculate_icc(d, list(cols))

            if err:
                icc_html.set(
                    f"<div class='alert alert-danger'>{_html.escape(str(err))}</div>"
                )
            else:
                # Custom Interpretation Block
                interp_parts = []
                for _, row in res_df.iterrows():
                    val = row.get("ICC", 0)
                    if pd.isna(val):
                        continue
                    strength = (
                        "Excellent"
                        if val > 0.9
                        else (
                            "Good"
                            if val > 0.75
                            else "Moderate"
                            if val > 0.5
                            else "Poor"
                        )
                    )
                    icon = "✅" if val > 0.75 else "⚠️" if val > 0.5 else "❌"
                    interp_parts.append(
                        f"<li>{icon} <strong>{_html.escape(str(row.get('Type', '')))}</strong>: {val:.3f} ({strength})</li>"
                    )

                interp_html = f"""
                <div class='alert alert-info mt-3'>
                    <h5>📊 ICC Interpretation</h5>
                    <ul>{"".join(interp_parts)}</ul>
                </div>
                """

                rep = [
                    {
                        "type": "text",
                        "data": f"ICC Reliability Analysis: {', '.join(cols)}",
                    },
                    {"type": "html", "data": interp_html},
                    {"type": "table", "header": "ICC Results", "data": res_df},
                    {"type": "table", "header": "ANOVA Source Table", "data": anova_df},
                ]
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )
                icc_html.set(diag_test.generate_report("ICC Reliability Report", rep))
        except Exception as e:
            logger.exception("ICC failed")
            icc_html.set(
                f"<div class='alert alert-danger'>Error: {_html.escape(str(e))}</div>"
            )
        finally:
            icc_processing.set(False)

    @render.ui
    def out_icc_results():
        if icc_processing.get():
            return create_loading_state("Calculating ICC Reliability...")
        if icc_html.get():
            return ui.HTML(icc_html.get())
        return create_placeholder_state(
            "Select 2+ variables and click 'Calculate ICC'.", icon="🔍"
        )

    @render.download(
        filename=lambda: f"icc_report_{_safe_filename_part('_'.join((input.icc_vars() or [])[:3]) or 'analysis')}.html"
    )
    def btn_dl_icc_report():
        req(icc_html.get())
        yield icc_html.get()

    # --- Validation ---
    @render.ui
    def out_kappa_validation():
        v1, v2 = input.sel_kappa_v1(), input.sel_kappa_v2()
        if v1 and v2 and v1 == v2:
            return create_error_alert(
                "Please select two different variables for comparison."
            )
        return None

    @render.ui
    def out_ba_validation():
        v1, v2 = input.sel_ba_v1(), input.sel_ba_v2()
        if v1 and v2 and v1 == v2:
            return create_error_alert(
                "Please select two different variables for comparison."
            )
        return None

    @render.ui
    def out_icc_validation():
        cols = input.icc_vars()
        if cols and len(cols) < 2:
            return create_error_alert("Please select at least 2 rater/method columns.")
        return None

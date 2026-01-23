"""
🤝 Agreement & Reliability Analysis Module

Consolidating Agreement and Reliability analysis:
- Cohen's Kappa (Inter-rater agreement for categorical data)
- Bland-Altman (Method comparison for continuous data)
- ICC (Intraclass Correlation for continuous reliability)
"""

import re
from typing import Any

import numpy as np
import pandas as pd
from shiny import module, reactive, render, req, ui

from logger import get_logger
from tabs._common import (
    select_variable_by_keyword,
)
from utils import correlation, diag_test
from utils.formatting import create_missing_data_report_html
from utils.ui_helpers import (
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
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
                ui.card(
                    ui.card_header("🤝 Cohen's Kappa (Categorical Data)"),
                    create_input_group(
                        "Rater/Method Selection",
                        ui.row(
                            ui.column(
                                6,
                                ui.output_ui("ui_kappa_v1"),
                            ),
                            ui.column(
                                6,
                                ui.output_ui("ui_kappa_v2"),
                            ),
                        ),
                        type="required",
                    ),
                    ui.output_ui("out_kappa_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_analyze_kappa",
                            "🚀 Calculate Kappa",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_kappa_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container(
                    "Kappa Results",
                    ui.output_ui("out_kappa_results"),
                ),
            ),
            # TAB 2: Bland-Altman (Continuous Agreement)
            ui.nav_panel(
                "📉 Bland-Altman",
                ui.card(
                    ui.card_header("📉 Bland-Altman Analysis (Continuous Data)"),
                    create_input_group(
                        "Variable Selection",
                        ui.row(
                            ui.column(
                                6,
                                ui.output_ui("ui_ba_v1"),
                            ),
                            ui.column(
                                6,
                                ui.output_ui("ui_ba_v2"),
                            ),
                        ),
                        type="required",
                    ),
                    ui.output_ui("out_ba_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_analyze_ba",
                            "🚀 Analyze Agreement",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_ba_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container(
                    "Bland-Altman Results",
                    ui.output_ui("out_ba_results"),
                ),
            ),
            # TAB 3: ICC (Reliability)
            ui.nav_panel(
                "🔍 Reliability (ICC)",
                ui.card(
                    ui.card_header("🔍 Intraclass Correlation Coefficient"),
                    create_input_group(
                        "Variable Selection (2+ Raters/Methods)",
                        ui.output_ui("ui_icc_vars"),
                        type="required",
                    ),
                    ui.output_ui("out_icc_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_icc",
                            "🔍 Calculate ICC",
                            class_="btn-primary",
                            width="100%",
                        ),
                        ui.download_button(
                            "btn_dl_icc",
                            "📥 Download Report",
                            class_="btn-secondary",
                            width="100%",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container(
                    "ICC Results",
                    ui.output_ui("out_icc_result"),
                ),
            ),
            # TAB 4: Reference
            ui.nav_panel(
                "ℹ️ Reference",
                ui.card(
                    ui.card_header("📚 Agreement & Reliability Guide"),
                    ui.markdown("""
                    ### 🤝 Cohen's Kappa
                    Used for **categorical (groups)** data to measure agreement between two raters/methods.
                    - **Kappa > 0.8:** Excellent agreement
                    - **Kappa 0.6-0.8:** Substantial agreement
                    - **Kappa 0.4-0.6:** Moderate agreement
                    - **Kappa < 0.4:** Poor to Fair agreement

                    ### 📉 Bland-Altman Plot
                    Used for **continuous** data to compare two measurement methods.
                    - **Bias (Mean Difference):** How much one method differs from the other on average.
                    - **Limits of Agreement (LoA):** The range (Mean ± 1.96 SD) where 95% of differences lie.
                    - Ideally, 95% of points should be within the LoA lines.

                    ### 🔍 Intraclass Correlation (ICC)
                    Used for **continuous** data to measure reliability among 2 or more raters.
                    - **ICC(2,1) Absolute Agreement:** Use when exact scores must match.
                    - **ICC(3,1) Consistency:** Use when ranking consistency matters.
                    - **Interpretation:**
                        - **> 0.90:** Excellent Reliability ✅
                        - **0.75 - 0.90:** Good Reliability
                        - **0.50 - 0.75:** Moderate Reliability ⚠️
                        - **< 0.50:** Poor Reliability ❌
                    """),
                ),
            ),
        ),
    )


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
    kappa_res: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    ba_res: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    icc_result: reactive.Value[dict[str, Any] | None] = reactive.Value(None)

    kappa_processing: reactive.Value[bool] = reactive.Value(False)
    ba_processing: reactive.Value[bool] = reactive.Value(False)
    icc_processing: reactive.Value[bool] = reactive.Value(False)

    # --- Dataset Logic ---
    @reactive.Calc
    def current_df() -> pd.DataFrame | None:
        source = getattr(input, "radio_source", lambda: None)()
        if is_matched.get() and source == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_title_with_summary():
        d = current_df()
        if d is not None:
            return ui.div(
                ui.h3("🤝 Agreement & Reliability Analysis"),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3",
                ),
            )
        return ui.h3("🤝 Agreement & Reliability Analysis")

    @render.ui
    def ui_matched_info():
        if is_matched.get():
            return ui.div(
                ui.tags.div(
                    "✅ **Matched Dataset Available** - You can select it below",
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
                {"original": "Original Data", "matched": "Matched Data"},
                selected="matched",
                inline=True,
            )
        return None

    # --- Update Dynamic Inputs ---
    @reactive.Effect
    def _update_agreement_inputs():
        """Force update when data changes"""
        _ = current_df()

    @render.ui
    def ui_kappa_v1():
        d = current_df()
        if d is None:
            return ui.input_select(
                "sel_kappa_v1", "Rater/Method 1:", choices=["Select..."]
            )
        cols = d.columns.tolist()
        default_rater1 = select_variable_by_keyword(
            cols,
            ["diagnosis_dr_a", "diagnosis", "dr_a", "rater1", "obs1", "method1"],
            default_to_first=True,
        )
        return ui.input_select(
            "sel_kappa_v1", "Rater/Method 1:", choices=cols, selected=default_rater1
        )

    @render.ui
    def ui_kappa_v2():
        d = current_df()
        if d is None:
            return ui.input_select(
                "sel_kappa_v2", "Rater/Method 2:", choices=["Select..."]
            )
        cols = d.columns.tolist()
        v1 = input.sel_kappa_v1()
        rem_cols = [c for c in cols if c != v1] if v1 else cols
        default_rater2 = select_variable_by_keyword(
            rem_cols,
            ["diagnosis_dr_b", "dr_b", "rater2", "obs2", "method2"],
            default_to_first=True,
        )
        return ui.input_select(
            "sel_kappa_v2", "Rater/Method 2:", choices=cols, selected=default_rater2
        )

    @render.ui
    def ui_ba_v1():
        d = current_df()
        if d is None:
            return ui.input_select(
                "sel_ba_v1", "Variable 1 (Method A):", choices=["Select..."]
            )
        num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        default_ba1 = select_variable_by_keyword(
            num_cols,
            [
                "icc_sysbp_rater1",
                "sysbp_rater1",
                "rater1",
                "method_a",
                "measure_1",
                "test_1",
                "gold",
            ],
            default_to_first=True,
        )
        return ui.input_select(
            "sel_ba_v1",
            "Variable 1 (Method A):",
            choices=num_cols,
            selected=default_ba1,
        )

    @render.ui
    def ui_ba_v2():
        d = current_df()
        if d is None:
            return ui.input_select(
                "sel_ba_v2", "Variable 2 (Method B):", choices=["Select..."]
            )
        num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        v1 = input.sel_ba_v1()
        rem_num = [c for c in num_cols if c != v1] if v1 else num_cols
        default_ba2 = select_variable_by_keyword(
            rem_num,
            [
                "icc_sysbp_rater2",
                "sysbp_rater2",
                "rater2",
                "method_b",
                "measure_2",
                "test_2",
                "new",
            ],
            default_to_first=True,
        )
        if not default_ba2 and num_cols:
            default_ba2 = num_cols[0]
        return ui.input_select(
            "sel_ba_v2",
            "Variable 2 (Method B):",
            choices=num_cols,
            selected=default_ba2,
        )

    @render.ui
    def ui_icc_vars():
        d = current_df()
        if d is None:
            return ui.input_selectize(
                "icc_vars",
                "Select Variables (Raters/Methods):",
                choices=["Select..."],
                multiple=True,
            )
        num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        icc_defaults = _auto_detect_icc_vars(num_cols)
        return ui.input_selectize(
            "icc_vars",
            "Select Variables (Raters/Methods):",
            choices=num_cols,
            multiple=True,
            selected=icc_defaults,
        )

    # ==================== KAPPA ANALYSIS ====================

    # (Kappa Status removed)

    @reactive.Effect
    @reactive.event(input.btn_analyze_kappa)
    def _run_kappa():
        d = current_df()
        v1, v2 = input.sel_kappa_v1(), input.sel_kappa_v2()
        req(d is not None, v1, v2)

        kappa_processing.set(True)
        kappa_res.set(None)

        try:
            res, err, conf, missing_info = diag_test.calculate_kappa(
                d, v1, v2, var_meta=var_meta.get() or {}
            )
            if err:
                kappa_res.set({"error": err})
                ui.notification_show("Analysis failed", type="error")
            else:
                rep = [
                    {"type": "text", "data": f"Agreement: {v1} vs {v2}"},
                    {"type": "table", "header": "Kappa Statistics", "data": res},
                    {
                        "type": "contingency_table",
                        "header": "Confusion Matrix",
                        "data": conf,
                    },
                ]
                if missing_info:
                    rep.append(
                        {
                            "type": "raw_html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )
                kappa_res.set(
                    {"html": diag_test.generate_report(f"Kappa: {v1} vs {v2}", rep)}
                )
        except Exception as e:
            logger.exception("Kappa failed")
            kappa_res.set({"error": f"Error: {str(e)}"})
            ui.notification_show("Analysis failed", type="error")
        finally:
            kappa_processing.set(False)

    @render.ui
    def out_kappa_results():
        if kappa_processing.get():
            return create_loading_state("Calculating Kappa Statistics...")

        res = kappa_res.get()
        if res is None:
            return create_placeholder_state(
                "Select two variables and click 'Calculate Kappa' to see results.",
                icon="🤝",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        if "html" in res:
            return ui.HTML(res["html"])
        return None

    @render.download(filename="kappa_report.html")
    def btn_dl_kappa_report():
        res = kappa_res.get()
        if res and "html" in res:
            yield res["html"]
        else:
            yield "No report available."

    # ==================== BLAND-ALTMAN ANALYSIS ====================
    # (Bland-Altman Status removed)

    @reactive.Effect
    @reactive.event(input.btn_analyze_ba)
    def _run_ba():
        d = current_df()
        v1, v2 = input.sel_ba_v1(), input.sel_ba_v2()
        req(d is not None, v1, v2)

        ba_processing.set(True)
        ba_res.set(None)

        try:
            stats_res, fig, missing_info = diag_test.calculate_bland_altman(d, v1, v2)

            if "error" in stats_res:
                ba_res.set({"error": stats_res["error"]})
                ui.notification_show("Analysis failed", type="error")
            else:
                stats_df = pd.DataFrame(
                    [
                        {
                            "Mean Diff (Bias)": f"{stats_res['mean_diff']:.4f}",
                            "95% CI Bias": f"{stats_res['ci_mean_diff'][0]:.4f} - {stats_res['ci_mean_diff'][1]:.4f}",
                            "Upper LoA": f"{stats_res['upper_loa']:.4f}",
                            "Lower LoA": f"{stats_res['lower_loa']:.4f}",
                            "N": stats_res["n"],
                        }
                    ]
                ).T
                rep = [
                    {
                        "type": "text",
                        "data": f"Bland-Altman: {v1} vs {v2}",
                    },
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
                            "type": "raw_html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )

                ba_res.set(
                    {
                        "html": diag_test.generate_report(
                            f"Bland-Altman: {v1} vs {v2}", rep
                        ),
                        "missing_data_info": missing_info,
                    }
                )
        except Exception as e:
            logger.exception("Bland-Altman failed")
            ba_res.set({"error": f"Error: {str(e)}"})
            ui.notification_show("Analysis failed", type="error")
        finally:
            ba_processing.set(False)

    @render.ui
    def out_ba_results():
        if ba_processing.get():
            return create_loading_state("Generating Bland-Altman analysis...")

        res = ba_res.get()
        if res is None:
            return create_placeholder_state(
                "Select two variables and click 'Analyze Agreement' to see results.",
                icon="📉",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        if "html" in res:
            return ui.div(
                ui.HTML(res["html"]),
                ui.hr(),
                ui.HTML(
                    create_missing_data_report_html(
                        res.get("missing_data_info", {}), var_meta.get() or {}
                    )
                ),
            )

        return None

    @render.download(filename="bland_altman_report.html")
    def btn_dl_ba_report():
        res = ba_res.get()
        if res and "html" in res:
            yield res["html"]
        else:
            yield "No report available."

    # ==================== ICC ANALYSIS ====================
    @reactive.Effect
    @reactive.event(input.btn_run_icc)
    def _run_icc():
        d = current_df()
        cols = input.icc_vars()
        if d is None or not cols or len(cols) < 2:
            ui.notification_show("Select at least 2 variables", type="warning")
            return

        icc_processing.set(True)
        icc_result.set(None)

        try:
            # Show notification as fallback/supplement
            ui.notification_show("Calculating ICC...", duration=None, id="run_icc")

            res_df, err, anova_df, missing_info = diag_test.calculate_icc(d, list(cols))

            if err:
                ui.notification_show("Analysis failed", type="error")
                ui.notification_remove("run_icc")
                icc_result.set({"error": err})
            else:
                icc_result.set(
                    {
                        "results_df": res_df,
                        "anova_df": anova_df,
                        "variables": list(cols),
                        "data_label": (
                            "Matched Data"
                            if is_matched.get()
                            and getattr(input, "radio_source", lambda: None)()
                            == "matched"
                            else "Original Data"
                        ),
                        "missing_data_info": missing_info,
                    }
                )
                ui.notification_remove("run_icc")
        except Exception as e:
            ui.notification_remove("run_icc")
            ui.notification_show("Analysis failed", type="error")
            icc_result.set({"error": f"Error: {str(e)}"})
        finally:
            icc_processing.set(False)

    @render.ui
    def out_icc_result():
        if icc_processing.get():
            return create_loading_state(
                "Calculating Intraclass Correlation Coefficient..."
            )

        result = icc_result.get()
        if result is None:
            return create_placeholder_state(
                "Select raters/methods and click 'Calculate ICC' to see results.",
                icon="🔍",
            )

        if "error" in result:
            return create_error_alert(result["error"])

        res_df = result["results_df"]
        interp_parts = []
        for _, row in res_df.iterrows():
            val = row.get("ICC", 0)
            if pd.isna(val):
                continue
            strength = (
                "Excellent"
                if val > 0.9
                else "Good"
                if val > 0.75
                else "Moderate"
                if val > 0.5
                else "Poor"
            )
            icon = "✅" if val > 0.75 else "⚠️" if val > 0.5 else "❌"
            interp_parts.append(
                f"{icon} <strong>{row.get('Type')}</strong> = {val:.3f}: {strength}"
            )

        interp_html = f"""
        <div class="alert alert-light border-start border-4 border-primary">
            <h5>📊 Interpretation</h5>
            {"<br>".join(interp_parts)}
        </div>
        """

        return ui.div(
            ui.h5("ICC Results"),
            ui.markdown(f"**Data:** {result['data_label']}"),
            ui.HTML(interp_html),
            ui.h5("ICC Table"),
            ui.output_data_frame("icc_results_table"),
            ui.h5("ANOVA Table"),
            ui.output_data_frame("icc_anova_table"),
            ui.hr(),
            ui.HTML(
                create_missing_data_report_html(
                    result.get("missing_data_info", {}), var_meta.get() or {}
                )
            ),
        )

    @render.data_frame
    def icc_results_table():
        res = icc_result.get()
        if res and "results_df" in res:
            return render.DataGrid(res["results_df"], width="100%")
        return None

    @render.data_frame
    def icc_anova_table():
        res = icc_result.get()
        if res and "anova_df" in res:
            return render.DataGrid(res["anova_df"], width="100%")
        return None

    @render.download(filename="icc_report.html")
    def btn_dl_icc():
        result = icc_result.get()
        if not result or "error" in result:
            yield b"No Data"
            return
        elements = [
            {"type": "text", "data": f"Data: {result['data_label']}"},
            {"type": "table", "header": "ICC Results", "data": result["results_df"]},
            {"type": "table", "header": "ANOVA Table", "data": result["anova_df"]},
        ]
        if result.get("missing_data_info"):
            elements.append(
                {
                    "type": "raw_html",
                    "data": create_missing_data_report_html(
                        result["missing_data_info"], var_meta.get() or {}
                    ),
                }
            )
        yield correlation.generate_report("ICC Report", elements).encode("utf-8")

    # ==================== VALIDATION LOGIC ====================
    @render.ui
    def out_kappa_validation():
        d = current_df()
        v1 = input.sel_kappa_v1()
        v2 = input.sel_kappa_v2()

        if d is None or d.empty:
            return None
        alerts = []

        if not v1 or not v2:
            return None

        if v1 == v2:
            alerts.append(
                create_error_alert(
                    "Rater 1 and Rater 2 must be different variables.",
                    title="Configuration Error",
                )
            )

        # Check cardinality - Kappa is for categorical
        for v in [v1, v2]:
            if v in d.columns:
                if d[v].nunique() > 20:
                    alerts.append(
                        create_error_alert(
                            f"Variable '{v}' has {d[v].nunique()} unique values. Kappa is intended for categorical/nominal data with few categories.",
                            title="Warning",
                        )
                    )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_ba_validation():
        d = current_df()
        v1 = input.sel_ba_v1()
        v2 = input.sel_ba_v2()

        if d is None or d.empty:
            return None
        alerts = []

        if not v1 or not v2:
            return None

        if v1 == v2:
            alerts.append(
                create_error_alert(
                    "Method A and Method B must be different variables.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_icc_validation():
        d = current_df()
        cols = input.icc_vars()

        if d is None or d.empty:
            return None
        alerts = []

        if not cols:
            return None

        if len(cols) < 2:
            alerts.append(
                create_error_alert(
                    "Please select at least 2 raters/methods for ICC.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

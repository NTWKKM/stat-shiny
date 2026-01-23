from __future__ import annotations

from typing import Any

import pandas as pd
from shiny import module, reactive, render, req, ui

from logger import get_logger
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from utils import decision_curve_lib, diag_test
from utils.formatting import create_missing_data_report_html
from utils.ui_helpers import (
    create_empty_state_ui,
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_results_container,
    create_skeleton_loader_ui,
)

logger = get_logger(__name__)

COLORS = get_color_palette()


# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def diag_ui() -> ui.TagChild:
    return ui.div(
        # Title + Data Summary inline
        ui.output_ui("ui_title_with_summary"),
        # Dataset Info Box
        ui.output_ui("ui_matched_info"),
        ui.br(),
        # Dataset Selector
        ui.output_ui("ui_dataset_selector"),
        ui.br(),
        # Tabs for different analyses
        ui.navset_tab(
            # TAB 1: ROC Curve & AUC
            ui.nav_panel(
                "📈 ROC Curve & AUC",
                ui.card(
                    ui.card_header("Receiver Operating Characteristic (ROC)"),
                    ui.layout_columns(
                        create_input_group(
                            "Variables",
                            ui.output_ui("ui_roc_truth"),
                            ui.output_ui("ui_roc_score"),
                            type="required",
                        ),
                        create_input_group(
                            "Settings",
                            ui.output_ui("ui_roc_method"),
                            ui.output_ui("ui_roc_pos_label"),
                            type="optional",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_roc_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_analyze_roc",
                            "🚀 Analyze ROC",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_roc_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container(
                    "ROC Results", ui.output_ui("out_roc_results")
                ),
                ui.div(
                    ui.h5("💡 Interpretation Guide"),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.strong("Sensitivity (True Positive Rate)"),
                            ": How well the test detects the disease/outcome.",
                        ),
                        ui.tags.li(
                            ui.strong("Specificity (True Negative Rate)"),
                            ": How well the test correctly excludes healthy people.",
                        ),
                        ui.tags.li(
                            ui.strong("AUC (Area Under Curve)"),
                            ": 0.5 = Random Guessing, 1.0 = Perfect. >0.8 is generally considered good.",
                        ),
                    ),
                    class_="alert alert-light",
                    style="margin-top: 20px; border: 1px solid #eee;",
                ),
            ),
            # TAB 2: Chi-Square & Risk Analysis
            ui.nav_panel(
                "🎲 Chi-Square & Risk (2x2)",
                ui.card(
                    ui.card_header("Chi-Square & Risk Analysis (2x2)"),
                    ui.layout_columns(
                        create_input_group(
                            "Variables",
                            ui.output_ui("ui_chi_v1"),
                            ui.output_ui("ui_chi_v1_pos"),
                            ui.output_ui("ui_chi_v2"),
                            ui.output_ui("ui_chi_v2_pos"),
                            type="required",
                        ),
                        create_input_group(
                            "Configuration",
                            ui.output_ui("ui_chi_method"),
                            ui.output_ui("ui_chi_caption"),
                            ui.output_ui("ui_chi_note"),
                            type="optional",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_chi_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_analyze_chi",
                            "🚀 Analyze Chi-Square",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_chi_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container("Results", ui.output_ui("out_chi_results")),
                ui.div(
                    ui.h5("💡 Interpretation Guide (2x2)"),
                    ui.tags.ul(
                        ui.tags.li(
                            ui.strong("P-Value < 0.05"),
                            ": Statistically significant association in the 2x2 table.",
                        ),
                        ui.tags.li(
                            ui.strong("Odds Ratio (OR)"),
                            ": Likelihood of outcome in exposed vs unexposed group.",
                        ),
                        ui.tags.li(
                            ui.strong("Risk Ratio (RR) / Relative Risk"),
                            ": Probability of outcome in exposed vs unexposed group.",
                        ),
                    ),
                    class_="alert alert-light",
                    style="margin-top: 20px; border: 1px solid #eee;",
                ),
            ),
            # TAB 4: Descriptive
            ui.nav_panel(
                "📊 Descriptive",
                ui.card(
                    ui.card_header("Descriptive Statistics"),
                    create_input_group(
                        "Variables", ui.output_ui("ui_desc_var"), type="required"
                    ),
                    ui.output_ui("out_desc_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_desc",
                            "Show Stats",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_desc_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container("Results", ui.output_ui("out_desc_results")),
            ),
            # TAB 5: Decision Curve Analysis (DCA)
            ui.nav_panel(
                "📉 Decision Curve (DCA)",
                ui.card(
                    ui.card_header("Decision Curve Analysis"),
                    create_input_group(
                        "Variables",
                        ui.output_ui("ui_dca_truth"),
                        ui.output_ui("ui_dca_prob"),
                        type="required",
                    ),
                    ui.output_ui("out_dca_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_dca",
                            "🚀 Run DCA",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_dca_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container("Results", ui.output_ui("out_dca_results")),
            ),
            # TAB 6: Reference & Interpretation
            ui.nav_panel(
                "ℹ️ Reference & Interpretation",
                ui.card(
                    ui.card_header("Reference & Interpretation"),
                    ui.markdown("""
                        ## 📚 Reference & Interpretation Guide
                        ... (คงเดิม) ...
                    """),
                ),
            ),
        ),
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def diag_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
) -> None:
    # --- Reactive Results Storage ---
    roc_res: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    chi_res: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    # --- Descriptive Reactives ---
    desc_res: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    desc_processing: reactive.Value[bool] = reactive.Value(False)

    # --- DCA Reactives ---
    dca_res: reactive.Value[dict[str, Any] | None] = reactive.Value(None)
    dca_processing: reactive.Value[bool] = reactive.Value(False)

    roc_processing: reactive.Value[bool] = reactive.Value(False)
    chi_processing: reactive.Value[bool] = reactive.Value(False)

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df() -> pd.DataFrame | None:
        # ✅ FIX: ใช้ radio_diag_source ให้ตรงกับ UI
        source = getattr(input, "radio_diag_source", lambda: None)()
        if is_matched.get() and source == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_title_with_summary():
        d = current_df()
        if d is not None:
            return ui.div(
                ui.h3("🧪 Diagnostic Tests (ROC)"),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3",
                ),
            )
        return ui.h3("🧪 Diagnostic Tests (ROC)")

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
                "radio_diag_source",  # ✅ ID นี้ต้องตรงกับที่ current_df เรียก
                "📄 Select Dataset:",
                {
                    "original": "📊 Original Data",
                    "matched": "✅ Matched Data (from PSM)",
                },
                selected="matched",
                inline=True,
            )
        return None

    # --- Update Dynamic Inputs ---
    @reactive.Effect
    def _update_diag_inputs():
        """Force update when data changes"""
        _ = current_df()

    # Get all columns
    @reactive.Calc
    def all_cols() -> list[str]:
        d = current_df()
        if d is not None:
            return d.columns.tolist()
        return []

    # --- ROC Inputs UI ---
    @render.ui
    def ui_roc_truth():
        cols = all_cols()
        default_v = select_variable_by_keyword(
            cols,
            ["gold_standard_disease", "gold", "standard", "outcome", "truth"],
            default_to_first=True,
        )
        # ✅ FIX: ใส่ session.ns เสมอ สำหรับ Dynamic UI
        return ui.input_select(
            session.ns("sel_roc_truth"),
            "Gold Standard (Binary):",
            choices=cols,
            selected=default_v,
        )

    @render.ui
    def ui_roc_score():
        cols = all_cols()
        default_v = select_variable_by_keyword(
            cols,
            ["test_score_rapid", "score", "prob", "predict"],
            default_to_first=True,
        )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_roc_score"),
            "Test Score (Continuous):",
            choices=cols,
            selected=default_v,
        )

    @render.ui
    def ui_roc_method():
        # ✅ FIX: ใส่ session.ns
        return ui.input_radio_buttons(
            session.ns("radio_roc_method"),
            "CI Method:",
            {"delong": "DeLong et al.", "hanley": "Binomial (Hanley)"},
            inline=True,
        )

    @render.ui
    def ui_roc_pos_label():
        truth_col = input.sel_roc_truth()
        d = current_df()
        if d is not None and truth_col and truth_col in d.columns:
            unique_vals = sorted([str(x) for x in d[truth_col].dropna().unique()])
            default_pos_idx = 0
            if "1" in unique_vals:
                default_pos_idx = unique_vals.index("1")
            elif "1.0" in unique_vals:
                default_pos_idx = unique_vals.index("1.0")

            # ✅ FIX: ใส่ session.ns
            return ui.input_select(
                session.ns("sel_roc_pos_label"),
                "Positive Label (1):",
                choices=unique_vals,
                selected=unique_vals[default_pos_idx] if unique_vals else None,
            )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_roc_pos_label"),
            "Positive Label (1):",
            choices=[],
        )

    # --- Chi-Square Inputs UI ---
    @render.ui
    def ui_chi_v1():
        cols = all_cols()
        default_v = select_variable_by_keyword(
            cols,
            ["treatment_group", "treatment", "group", "exposure"],
            default_to_first=True,
        )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_chi_v1"),
            "Variable 1 (Exposure/Row):",
            choices=cols,
            selected=default_v,
        )

    @render.ui
    def ui_chi_v2():
        cols = all_cols()
        default_v = select_variable_by_keyword(
            cols,
            ["outcome_cured", "outcome", "cured", "disease", "status"],
            default_to_first=True,
        )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_chi_v2"),
            "Variable 2 (Outcome/Col):",
            choices=cols,
            selected=default_v,
        )

    @render.ui
    def ui_chi_method():
        # ✅ FIX: ใส่ session.ns
        return ui.input_radio_buttons(
            session.ns("radio_chi_method"),
            "Test Method:",
            {
                "Pearson (Standard)": "Pearson",
                "Yates' correction": "Yates",
                "Fisher's Exact Test": "Fisher",
            },
            inline=True,
        )

    @render.ui
    def ui_chi_caption():
        return ui.div(
            ui.markdown("*(Choose method in Tab 5 for guidance)*"),
            class_="text-secondary text-sm",
        )

    def get_pos_label_settings(
        df_input: pd.DataFrame, col_name: str
    ) -> tuple[list[str], int]:
        if df_input is None or col_name not in df_input.columns:
            return [], 0
        unique_vals = [str(x) for x in df_input[col_name].dropna().unique()]
        unique_vals.sort()
        default_idx = 0

        if "1" in unique_vals:
            default_idx = unique_vals.index("1")
        elif "1.0" in unique_vals:
            default_idx = unique_vals.index("1.0")
        elif "0" in unique_vals:
            default_idx = unique_vals.index("0")

        return unique_vals, default_idx

    @render.ui
    def ui_chi_v1_pos():
        v1_col = input.sel_chi_v1()
        d = current_df()
        if d is None:
            return None
        v1_uv, v1_idx = get_pos_label_settings(d, v1_col)
        if not v1_uv:
            return ui.div(
                ui.markdown(f"⚠️ No values in {v1_col}"),
                class_="text-danger text-sm",
            )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_chi_v1_pos"),
            f"Positive Label (Row: {v1_col}):",
            choices=v1_uv,
            selected=v1_uv[v1_idx] if v1_uv else None,
        )

    @render.ui
    def ui_chi_v2_pos():
        v2_col = input.sel_chi_v2()
        d = current_df()
        if d is None:
            return None
        v2_uv, v2_idx = get_pos_label_settings(d, v2_col)
        if not v2_uv:
            return ui.div(
                ui.markdown(f"⚠️ No values in {v2_col}"),
                class_="text-danger text-sm",
            )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_chi_v2_pos"),
            f"Positive Label (Col: {v2_col}):",
            choices=v2_uv,
            selected=v2_uv[v2_idx] if v2_uv else None,
        )

    @render.ui
    def ui_chi_note():
        return ui.div(
            ui.markdown("*Select for Risk/OR calculation*"),
            class_="text-secondary text-sm",
        )

    @render.ui
    def ui_chi_empty():
        return ui.div()

    # --- Descriptive Inputs UI ---
    @render.ui
    def ui_desc_var():
        cols = all_cols()
        default_v = select_variable_by_keyword(
            cols, ["age_years", "age", "years"], default_to_first=True
        )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_desc_var"),
            "Select Variable:",
            choices=cols,
            selected=default_v,
        )

    # --- ROC Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_roc)
    def _run_roc():
        d = current_df()
        req(d is not None, input.sel_roc_truth(), input.sel_roc_score())

        roc_processing.set(True)
        roc_res.set(None)

        try:
            truth_col = input.sel_roc_truth()
            score_col = input.sel_roc_score()
            method = input.radio_roc_method()
            pos_label = input.sel_roc_pos_label()

            res, err, fig, coords = diag_test.analyze_roc(
                d,
                truth_col,
                score_col,
                method=method,
                pos_label_user=pos_label,
                var_meta=var_meta.get() or {},
            )

            if err:
                roc_res.set({"error": err})
            else:
                rep = [
                    {
                        "type": "text",
                        "data": f"📊 Analysis: {score_col} vs {truth_col}",
                    },
                ]

                if fig is not None:
                    rep.append({"type": "plot", "data": fig})
                else:
                    rep.append(
                        {
                            "type": "text",
                            "data": "⚠️ Warning: ROC plot could not be generated",
                        }
                    )

                if res is not None:
                    res_display = {
                        k: v for k, v in res.items() if k != "missing_data_info"
                    }
                    rep.append(
                        {
                            "type": "table",
                            "header": "ROC Statistics",
                            "data": pd.DataFrame([res_display]).T,
                        }
                    )

                if coords is not None:
                    rep.append(
                        {
                            "type": "table",
                            "header": "Performance at Different Thresholds",
                            "data": coords,
                        }
                    )

                if res and "missing_data_info" in res:
                    rep.append(
                        {
                            "type": "raw_html",
                            "data": create_missing_data_report_html(
                                res["missing_data_info"], var_meta.get() or {}
                            ),
                        }
                    )

                html_report = diag_test.generate_report(
                    f"ROC Analysis Report ({method})", rep
                )
                roc_res.set({"html": html_report})

        except Exception as e:
            logger.exception("ROC analysis failed")
            roc_res.set({"error": f"Error: {str(e)}"})
        finally:
            roc_processing.set(False)

    @render.ui
    def out_roc_results():
        if roc_processing.get():
            return ui.div(
                create_loading_state("Generating ROC curve and statistics..."),
                create_skeleton_loader_ui(rows=3, show_chart=True),
            )

        res = roc_res.get()
        if res is None:
            return create_empty_state_ui(
                message="No ROC Analysis Results",
                sub_message="Select variables and click '🚀 Analyze ROC' to view the curve and metrics.",
                icon="📈",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        if "html" in res:
            return ui.HTML(res["html"])
        return None

    @render.download(filename="roc_report.html")
    def btn_dl_roc_report():
        res = roc_res.get()
        if res and "html" in res:
            yield res["html"]
        else:
            yield "No report available."

    # --- Chi-Square Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_chi)
    def _run_chi():
        d = current_df()
        req(d is not None, input.sel_chi_v1(), input.sel_chi_v2())

        chi_processing.set(True)
        chi_res.set(None)

        try:
            tab, stats, msg, risk, missing_info = diag_test.calculate_chi2(
                d,
                input.sel_chi_v1(),
                input.sel_chi_v2(),
                method=input.radio_chi_method(),
                v1_pos=input.sel_chi_v1_pos(),
                v2_pos=input.sel_chi_v2_pos(),
                var_meta=var_meta.get() or {},
            )

            if tab is not None:
                status_text = (
                    f"Note: {msg.strip()}"
                    if msg.strip()
                    else "Analysis Status: Completed successfully."
                )
                rep = [
                    {
                        "type": "text",
                        "data": "Analysis: Diagnostic Test / Chi-Square",
                    },
                    {
                        "type": "text",
                        "data": f"Variables: {input.sel_chi_v1()} vs {input.sel_chi_v2()}",
                    },
                    {"type": "text", "data": status_text},
                    {
                        "type": "contingency_table",
                        "header": "Contingency Table",
                        "data": tab,
                    },
                ]
                if stats is not None:
                    rep.append(
                        {
                            "type": "table",
                            "header": "Statistics",
                            "data": stats,
                        }
                    )
                if risk is not None:
                    rep.append(
                        {
                            "type": "table",
                            "header": "Risk & Effect Measures",
                            "data": risk,
                        }
                    )

                if missing_info:
                    rep.append(
                        {
                            "type": "raw_html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )

                chi_res.set(
                    {
                        "html": diag_test.generate_report(
                            f"Chi2: {input.sel_chi_v1()} vs {input.sel_chi_v2()}",
                            rep,
                        )
                    }
                )
            else:
                chi_res.set({"error": f"Analysis failed: {msg}"})

        except Exception as e:
            logger.exception("Chi-Square analysis failed")
            chi_res.set({"error": f"Error: {str(e)}"})
        finally:
            chi_processing.set(False)

    @render.ui
    def out_chi_results():
        if chi_processing.get():
            return ui.div(
                create_loading_state("Calculating Chi-Square statistics..."),
                create_skeleton_loader_ui(rows=4, show_chart=False),
            )

        res = chi_res.get()
        if res is None:
            return create_empty_state_ui(
                message="No Chi-Square Results",
                sub_message="Select variables and click '🚀 Analyze Chi-Square' to view association statistics.",
                icon="🎲",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        if "html" in res:
            return ui.HTML(res["html"])
        return None

    @render.download(filename="chi2_report.html")
    def btn_dl_chi_report():
        res = chi_res.get()
        if res and "html" in res:
            yield res["html"]
        else:
            yield "No report available."

    # --- Descriptive Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_run_desc)
    def _run_desc():
        d = current_df()
        req(d is not None, input.sel_desc_var())

        desc_processing.set(True)
        desc_res.set(None)

        try:
            stats_df, missing_info = diag_test.calculate_descriptive(
                d, input.sel_desc_var(), var_meta=var_meta.get() or {}
            )
            if stats_df is not None:
                rep = [{"type": "table", "data": stats_df}]

                if missing_info:
                    rep.append(
                        {
                            "type": "raw_html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )

                desc_res.set(
                    {
                        "html": diag_test.generate_report(
                            f"Descriptive: {input.sel_desc_var()}",
                            rep,
                        )
                    }
                )
            else:
                desc_res.set(
                    {"error": "No data available for {}".format(input.sel_desc_var())}
                )

        except Exception as e:
            logger.exception("Descriptive analysis failed")
            desc_res.set({"error": f"Error: {str(e)}"})
        finally:
            desc_processing.set(False)

    @render.ui
    def out_desc_results():
        if desc_processing.get():
            return ui.div(
                create_loading_state("Calculating descriptive statistics..."),
                create_skeleton_loader_ui(rows=5, show_chart=False),
            )

        res = desc_res.get()
        if res is None:
            return create_empty_state_ui(
                message="No Descriptive Statistics",
                sub_message="Select a variable and click 'Show Stats' to view summary.",
                icon="📊",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        if "html" in res:
            return ui.HTML(res["html"])
        return None

    @render.download(filename="descriptive_report.html")
    def btn_dl_desc_report():
        res = desc_res.get()
        if res and "html" in res:
            yield res["html"]
        else:
            yield "No report available."

    # --- DCA Logic ---
    @render.ui
    def ui_dca_truth():
        cols = all_cols()
        default_v = select_variable_by_keyword(
            cols,
            ["outcome_cured", "outcome", "truth", "gold"],
            default_to_first=True,
        )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_dca_truth"),
            "Outcome (Truth):",
            choices=cols,
            selected=default_v,
        )

    @render.ui
    def ui_dca_prob():
        cols = all_cols()
        default_v = select_variable_by_keyword(
            cols,
            [
                "gold_standard_disease",
                "test_score_rapid",
                "score",
                "prob",
                "pred",
                "gold",
            ],
            default_to_first=True,
        )
        # ✅ FIX: ใส่ session.ns
        return ui.input_select(
            session.ns("sel_dca_prob"),
            "Prediction/Score (Probability):",
            choices=cols,
            selected=default_v,
        )

    @reactive.Effect
    @reactive.event(input.btn_run_dca)
    def _run_dca():
        d = current_df()
        req(d is not None, input.sel_dca_truth(), input.sel_dca_prob())

        dca_processing.set(True)
        dca_res.set(None)

        try:
            truth = input.sel_dca_truth()
            prob = input.sel_dca_prob()

            nb_model, missing_info = decision_curve_lib.calculate_net_benefit(
                d,
                truth,
                prob,
                model_name="Current Model",
                var_meta=var_meta.get() or {},
            )

            if "error" in missing_info:
                raise ValueError(missing_info["error"])

            d_clean = d.loc[missing_info.get("analyzed_indices", d.index)]
            nb_all = decision_curve_lib.calculate_net_benefit_all(d_clean, truth)
            nb_none = decision_curve_lib.calculate_net_benefit_none()

            df_dca = pd.concat([nb_model, nb_all, nb_none])
            fig = decision_curve_lib.create_dca_plot(df_dca)

            df_disp = nb_model[
                ["threshold", "net_benefit", "tp_rate", "fp_rate"]
            ].copy()
            df_disp = df_disp[
                df_disp["threshold"].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            ]

            missing_report_html = create_missing_data_report_html(
                missing_info, var_meta.get() or {}
            )

            rep = [
                {
                    "type": "text",
                    "data": f"Decision Curve Analysis: {prob} (Model) vs {truth} (Outcome)",
                },
                {"type": "plot", "data": fig},
                {
                    "type": "table",
                    "header": "Net Benefit (Model) at selected thresholds",
                    "data": df_disp,
                },
                {
                    "type": "raw_html",
                    "data": f"<hr>{missing_report_html}",
                },
            ]

            html_content = diag_test.generate_report(f"DCA: {prob} vs {truth}", rep)
            dca_res.set({"html": html_content})

        except Exception as e:
            logger.exception("DCA analysis failed")
            dca_res.set({"error": f"Error: {str(e)}"})
        finally:
            dca_processing.set(False)

    @render.ui
    def out_dca_results():
        if dca_processing.get():
            return ui.div(
                create_loading_state("Calculating Decision Curve..."),
                create_skeleton_loader_ui(rows=3, show_chart=True),
            )

        res = dca_res.get()
        if res is None:
            return create_empty_state_ui(
                message="No Decision Curve Analysis",
                sub_message="Select variables and click '🚀 Run DCA' to view net benefit.",
                icon="📉",
            )

        if "error" in res:
            return create_error_alert(res["error"])

        if "html" in res:
            return ui.HTML(res["html"])
        return None

    @render.download(filename="dca_report.html")
    def btn_dl_dca_report():
        res = dca_res.get()
        if res and "html" in res:
            yield res["html"]
        else:
            yield "No report available."

    # ==================== VALIDATION LOGIC ====================
    @render.ui
    def out_roc_validation():
        d = current_df()
        truth = input.sel_roc_truth()
        score = input.sel_roc_score()

        if d is None or d.empty:
            return None
        alerts = []

        if not truth or not score:
            return None

        if truth == score:
            alerts.append(
                create_error_alert(
                    "Gold Standard and Test Score must be different variables.",
                    title="Configuration Error",
                )
            )

        if truth in d.columns and d[truth].nunique() > 2:
            alerts.append(
                create_error_alert(
                    f"Gold Standard '{truth}' should be binary (2 unique values). It has {d[truth].nunique()}.",
                    title="Warning",
                )
            )

        if score in d.columns and not pd.api.types.is_numeric_dtype(d[score]):
            alerts.append(
                create_error_alert(
                    f"Test Score '{score}' should be numeric/continuous.",
                    title="Warning",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_chi_validation():
        d = current_df()
        v1 = input.sel_chi_v1()
        v2 = input.sel_chi_v2()

        if d is None or d.empty:
            return None
        alerts = []

        if not v1 or not v2:
            return None

        if v1 == v2:
            alerts.append(
                create_error_alert(
                    "Variables must be different.", title="Configuration Error"
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_desc_validation():
        return None

    @render.ui
    def out_dca_validation():
        d = current_df()
        truth = input.sel_dca_truth()
        prob = input.sel_dca_prob()

        if d is None or d.empty:
            return None
        alerts = []

        if not truth or not prob:
            return None

        if truth == prob:
            alerts.append(
                create_error_alert(
                    "Outcome and Probability variables must be different.",
                    title="Configuration Error",
                )
            )

        if prob in d.columns:
            if not pd.api.types.is_numeric_dtype(d[prob]):
                alerts.append(
                    create_error_alert(
                        f"Probability '{prob}' must be numeric.",
                        title="Invalid Variable",
                    )
                )
            elif (d[prob] < 0).any() or (d[prob] > 1).any():
                alerts.append(
                    create_error_alert(
                        f"Probability '{prob}' values should be between 0 and 1.",
                        title="Warning",
                    )
                )

        if alerts:
            return ui.div(*alerts)
        return None

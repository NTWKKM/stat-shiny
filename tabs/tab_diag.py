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
                ui.markdown("##### ROC Curve Analysis"),
                ui.row(
                    ui.column(3, ui.output_ui("ui_roc_truth")),
                    ui.column(3, ui.output_ui("ui_roc_score")),
                    ui.column(3, ui.output_ui("ui_roc_method")),
                    ui.column(3, ui.output_ui("ui_roc_pos_label")),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_analyze_roc",
                            "🚀 Analyze ROC",
                            class_="btn-primary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.download_button(
                            "btn_dl_roc_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                            width="100%",
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_roc_status"),  # Status message area
                ui.br(),
                ui.output_ui("out_roc_results"),
            ),
            # TAB 2: Chi-Square & Risk Analysis
            ui.nav_panel(
                "🎲 Chi-Square & Risk (2x2)",
                ui.markdown("##### Chi-Square & Risk Analysis (2x2 Contingency Table)"),
                ui.row(
                    ui.column(3, ui.output_ui("ui_chi_v1")),
                    ui.column(3, ui.output_ui("ui_chi_v2")),
                    ui.column(3, ui.output_ui("ui_chi_method")),
                    ui.column(3, ui.output_ui("ui_chi_caption")),
                ),
                ui.row(
                    ui.column(3, ui.output_ui("ui_chi_v1_pos")),
                    ui.column(3, ui.output_ui("ui_chi_v2_pos")),
                    ui.column(3, ui.output_ui("ui_chi_note")),
                    ui.column(3, ui.output_ui("ui_chi_empty")),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_analyze_chi",
                            "🚀 Analyze Chi-Square",
                            class_="btn-primary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.download_button(
                            "btn_dl_chi_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                            width="100%",
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_chi_status"),  # Status message area
                ui.br(),
                ui.output_ui("out_chi_results"),
            ),
            # TAB 3: Descriptive
            ui.nav_panel(
                "📊 Descriptive",
                ui.markdown("##### Descriptive Statistics"),
                ui.row(ui.column(12, ui.output_ui("ui_desc_var"))),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_run_desc",
                            "Show Stats",
                            class_="btn-primary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.download_button(
                            "btn_dl_desc_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                            width="100%",
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_desc_status"),  # Status message area
                ui.br(),
                ui.output_ui("out_desc_results"),
            ),
            # TAB 4: Decision Curve Analysis (DCA)
            ui.nav_panel(
                "📉 Decision Curve (DCA)",
                ui.markdown("##### Decision Curve Analysis"),
                ui.row(
                    ui.column(6, ui.output_ui("ui_dca_truth")),
                    ui.column(6, ui.output_ui("ui_dca_prob")),
                ),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_run_dca",
                            "🚀 Run DCA",
                            class_="btn-primary w-100",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.download_button(
                            "btn_dl_dca_report",
                            "📥 Download Report",
                            class_="btn-outline-primary w-100",
                            width="100%",
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_dca_status"),
                ui.br(),
                ui.output_ui("out_dca_results"),
            ),
            # TAB 5: Reference & Interpretation
            ui.nav_panel(
                "ℹ️ Reference & Interpretation",
                ui.markdown("""
                    ## 📚 Reference & Interpretation Guide

                    💡 **Tip:** This section provides detailed explanations and interpretation rules for all the diagnostic tests.

                    ### 🚦 Quick Decision Guide

                    | **Question** | **Recommended Test** | **Example** |
                    | :--- | :--- | :--- |
                    | My test is a **score** (e.g., 0-100) and I want to see how well it predicts a **disease** (Yes/No)? | **ROC Curve & AUC** | Risk Score vs Diabetes |
                    | I want to find the **best cut-off** value for my test score? | **ROC Curve (Youden Index)** | Finding optimal BP for Hypertension |
                    | Are these two **groups** (e.g., Treatment vs Control) different in outcome (Cured vs Not Cured)? | **Chi-Square** | Drug A vs Placebo on Recovery |
                    | Do two doctors **agree** on the same diagnosis? | **Cohen's Kappa** | Radiologist A vs Radiologist B |
                    | I just want to summarize **one variable** (Mean, Count)? | **Descriptive** | Age distribution |

                    ### ⚖️ Interpretation Guidelines

                    #### ROC Curve & AUC
                    - **AUC > 0.9:** Excellent discrimination
                    - **AUC 0.8-0.9:** Good discrimination
                    - **AUC 0.7-0.8:** Fair discrimination
                    - **AUC 0.5-0.7:** Poor discrimination
                    - **AUC = 0.5:** No discrimination (random chance)
                    - **Youden J Index:** Sensitivity + Specificity - 1 (higher is better, max = 1)

                    #### Chi-Square Test
                    - **P < 0.05:** Statistically significant association
                    - **Odds Ratio (OR):** If 95% CI doesn't include 1.0, it's significant
                    - **Risk Ratio (RR):** Similar interpretation as OR
                    - Use **Fisher's Exact Test** when expected counts < 5

                    #### Cohen's Kappa
                    - **Kappa > 0.8:** Almost perfect/excellent agreement
                    - **Kappa 0.6-0.8:** Substantial agreement
                    - **Kappa 0.4-0.6:** Moderate agreement
                    - **Kappa 0.2-0.4:** Fair agreement
                    - **Kappa 0.0-0.2:** Slight agreement
                    - **Kappa < 0:** Poor agreement (worse than chance)

                    ### 📊 Descriptive Statistics
                    - **Mean:** Average value (affected by outliers)
                    - **Median:** Middle value (robust to outliers)
                    - **SD (Standard Deviation):** Spread of data around mean
                    - **Q1/Q3:** 25th and 75th percentiles
                    """),
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
    roc_html: reactive.Value[str | None] = reactive.Value(None)
    chi_html: reactive.Value[str | None] = reactive.Value(None)
    desc_html: reactive.Value[str | None] = reactive.Value(None)

    # --- DCA Reactives ---
    dca_html: reactive.Value[str | None] = reactive.Value(None)
    dca_processing: reactive.Value[bool] = reactive.Value(False)

    roc_processing: reactive.Value[bool] = reactive.Value(False)
    chi_processing: reactive.Value[bool] = reactive.Value(False)
    desc_processing: reactive.Value[bool] = reactive.Value(False)

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df() -> pd.DataFrame | None:
        if is_matched.get() and input.radio_diag_source() == "matched":
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
                "radio_diag_source",
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

    # --- ROC Inputs UI ---
    @render.ui
    def ui_roc_truth():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["gold", "standard", "truth"], default_to_first=True
        )
        return ui.input_select(
            "sel_roc_truth",
            "Gold Standard (Binary):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_roc_score():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["score", "rapid", "test"], default_to_first=True
        )
        return ui.input_select(
            "sel_roc_score",
            "Test Score (Continuous):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_roc_method():
        return ui.input_radio_buttons(
            "radio_roc_method",
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
            # Force default to "1" or "1.0" if available
            if "1" in unique_vals:
                default_pos_idx = unique_vals.index("1")
            elif "1.0" in unique_vals:
                default_pos_idx = unique_vals.index("1.0")

            return ui.input_select(
                "sel_roc_pos_label",
                "Positive Label (1):",
                choices=unique_vals,
                selected=unique_vals[default_pos_idx] if unique_vals else None,
            )
        return ui.input_select(
            "sel_roc_pos_label",
            "Positive Label (1):",
            choices=[],
        )

    # --- Chi-Square Inputs UI ---
    @render.ui
    def ui_chi_v1():
        cols = all_cols()
        v1_idx = next((i for i, c in enumerate(cols) if c == "Treatment_Group"), 0)
        return ui.input_select(
            "sel_chi_v1",
            "Variable 1 (Exposure/Row):",
            choices=cols,
            selected=cols[v1_idx] if cols else None,
        )

    @render.ui
    def ui_chi_v2():
        cols = all_cols()
        v2_idx = next(
            (i for i, c in enumerate(cols) if c == "Outcome_Cured"),
            min(1, len(cols) - 1),
        )
        return ui.input_select(
            "sel_chi_v2",
            "Variable 2 (Outcome/Col):",
            choices=cols,
            selected=cols[v2_idx] if cols else None,
        )

    @render.ui
    def ui_chi_method():
        return ui.input_radio_buttons(
            "radio_chi_method",
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

        # Prioritize "1" then "1.0", then "0"
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
        return ui.input_select(
            "sel_chi_v1_pos",
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
        return ui.input_select(
            "sel_chi_v2_pos",
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
        return ui.input_select("sel_desc_var", "Select Variable:", choices=cols)

    # --- ROC Status & Results ---
    @render.ui
    def ui_roc_status():
        if roc_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "📄 Generating ROC curve and statistics... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_chi_status():
        if chi_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "📄 Calculating Chi-Square statistics... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_desc_status():
        if desc_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "📄 Calculating descriptive statistics... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    # --- ROC Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_roc)
    def _run_roc():
        d = current_df()
        req(d is not None, input.sel_roc_truth(), input.sel_roc_score())

        # Set processing flag
        roc_processing.set(True)

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
                roc_html.set(f"<div class='alert alert-danger'>📄 Error: {err}</div>")
            else:
                rep = [
                    {
                        "type": "text",
                        "data": f"📊 Analysis: {score_col} vs {truth_col}",
                    },
                ]

                # Add plot if available
                if fig is not None:
                    rep.append({"type": "plot", "data": fig})
                else:
                    rep.append(
                        {
                            "type": "text",
                            "data": "⚠️ Warning: ROC plot could not be generated",
                        }
                    )

                # Add statistics table
                if res is not None:
                    # Filter out missing_data_info/metadata from the main stats table
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

                # Add performance coordinates table
                if coords is not None:
                    rep.append(
                        {
                            "type": "table",
                            "header": "Performance at Different Thresholds",
                            "data": coords,
                        }
                    )

                # Missing Data Report
                if res and "missing_data_info" in res:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                res["missing_data_info"], var_meta.get() or {}
                            ),
                        }
                    )

                html_report = diag_test.generate_report(
                    f"ROC Analysis Report ({method})", rep
                )
                roc_html.set(html_report)

        except Exception as e:
            logger.exception("ROC analysis failed")
            roc_html.set(f"<div class='alert alert-danger'>📄 Error: {str(e)}</div>")
        finally:
            # Clear processing flag
            roc_processing.set(False)

    @render.ui
    def out_roc_results():
        if roc_html.get():
            return ui.div(ui.HTML(roc_html.get()), class_="fade-in-entry")
        return ui.div(
            "Click 'Analyze ROC' to view results.",
            class_="text-secondary p-3",
        )

    @render.download(filename="roc_report.html")
    def btn_dl_roc_report():
        yield roc_html.get()

    # --- Chi-Square Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_chi)
    def _run_chi():
        d = current_df()
        req(d is not None, input.sel_chi_v1(), input.sel_chi_v2())

        # Set processing flag
        chi_processing.set(True)

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

                # Missing Data Report
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )

                chi_html.set(
                    diag_test.generate_report(
                        f"Chi2: {input.sel_chi_v1()} vs {input.sel_chi_v2()}",
                        rep,
                    )
                )
            else:
                chi_html.set(
                    f"<div class='alert alert-danger'>Analysis failed: {msg}</div>"
                )

        except Exception as e:
            logger.exception("Chi-Square analysis failed")
            chi_html.set(f"<div class='alert alert-danger'>📄 Error: {str(e)}</div>")
        finally:
            # Clear processing flag
            chi_processing.set(False)

    @render.ui
    def out_chi_results():
        if chi_html.get():
            return ui.div(ui.HTML(chi_html.get()), class_="fade-in-entry")
        return ui.div("Results will appear here.", class_="text-secondary p-3")

    @render.download(filename="chi2_report.html")
    def btn_dl_chi_report():
        yield chi_html.get()

    # --- Descriptive Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_run_desc)
    def _run_desc():
        d = current_df()
        req(d is not None, input.sel_desc_var())

        # Set processing flag
        desc_processing.set(True)

        try:
            res, missing_info = diag_test.calculate_descriptive(
                d,
                input.sel_desc_var(),
                var_meta=var_meta.get() or {},
            )

            if res is not None:
                rep = [{"type": "table", "data": res}]
                if missing_info:
                    rep.append(
                        {
                            "type": "html",
                            "data": create_missing_data_report_html(
                                missing_info, var_meta.get() or {}
                            ),
                        }
                    )
                desc_html.set(
                    diag_test.generate_report(
                        f"Descriptive: {input.sel_desc_var()}",
                        rep,
                    )
                )
            else:
                desc_html.set(
                    "<div class='alert alert-danger'>No data available for {}</div>".format(
                        input.sel_desc_var()
                    )
                )

        except Exception as e:
            logger.exception("Descriptive analysis failed")
            desc_html.set(f"<div class='alert alert-danger'>📄 Error: {str(e)}</div>")
        finally:
            # Clear processing flag
            desc_processing.set(False)

    @render.ui
    def out_desc_results():
        if desc_html.get():
            return ui.div(ui.HTML(desc_html.get()), class_="fade-in-entry")
        return ui.div("Results will appear here.", class_="text-secondary p-3")

    @render.download(filename="descriptive_report.html")
    def btn_dl_desc_report():
        content = desc_html.get()
        if content:
            yield content
        else:
            yield "<html><body><p>No results available. Please run the analysis first.</p></body></html>"

    # --- DCA Logic ---
    @render.ui
    def ui_dca_truth():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["gold", "standard", "outcome"], default_to_first=True
        )
        return ui.input_select(
            "sel_dca_truth",
            "Outcome (Truth):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_dca_prob():
        cols = all_cols()
        default = select_variable_by_keyword(
            cols, ["score", "prob", "pred"], default_to_first=True
        )
        return ui.input_select(
            "sel_dca_prob",
            "Prediction/Score (Probability):",
            choices=cols,
            selected=default,
        )

    @render.ui
    def ui_dca_status():
        if dca_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2"),
                    "📄 Calculating Decision Curve... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    @reactive.Effect
    @reactive.event(input.btn_run_dca)
    def _run_dca():
        d = current_df()
        req(d is not None, input.sel_dca_truth(), input.sel_dca_prob())

        dca_processing.set(True)

        try:
            truth = input.sel_dca_truth()
            prob = input.sel_dca_prob()

            # Calculate Net Benefits
            nb_model, missing_info = decision_curve_lib.calculate_net_benefit(
                d,
                truth,
                prob,
                model_name="Current Model",
                var_meta=var_meta.get() or {},
            )
            if missing_info and "error" in missing_info:
                raise ValueError(missing_info["error"])
            nb_all = decision_curve_lib.calculate_net_benefit_all(d, truth)
            nb_none = decision_curve_lib.calculate_net_benefit_none()

            # Combine
            df_dca = pd.concat([nb_model, nb_all, nb_none])

            # Plot
            fig = decision_curve_lib.create_dca_plot(df_dca)

            # Report Generation
            df_disp = nb_model[
                ["threshold", "net_benefit", "tp_rate", "fp_rate"]
            ].copy()
            df_disp = df_disp[
                df_disp["threshold"].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            ]

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
            ]

            # Missing Data Report
            if missing_info:
                rep.append(
                    {
                        "type": "html",
                        "data": create_missing_data_report_html(
                            missing_info, var_meta.get() or {}
                        ),
                    }
                )

            html_content = diag_test.generate_report(f"DCA: {prob} vs {truth}", rep)
            dca_html.set(html_content)

        except Exception as e:
            logger.exception("DCA analysis failed")
            dca_html.set(f"<div class='alert alert-danger'>Error: {str(e)}</div>")
        finally:
            dca_processing.set(False)

    @render.ui
    def out_dca_results():
        if dca_html.get():
            return ui.div(ui.HTML(dca_html.get()), class_="fade-in-entry")
        return ui.div("Click 'Run DCA' to view results.", class_="text-secondary p-3")

    @render.download(filename="dca_report.html")
    def btn_dl_dca_report():
        content = dca_html.get()
        if content:
            yield content
        else:
            yield "<html><body><p>No results available. Please run the analysis first.</p></body></html>"

from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import diag_test

from tabs._common import get_color_palette

COLORS = get_color_palette()

# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def diag_ui():
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
        ui.navset_card_tab(
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
                ui.markdown(
                    "##### Chi-Square & Risk Analysis (2x2 Contingency Table)"
                ),
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
            # TAB 3: Agreement (Kappa)
            ui.nav_panel(
                "🤝 Agreement (Kappa)",
                ui.markdown("##### Agreement Analysis (Cohen's Kappa)"),
                ui.row(
                    ui.column(6, ui.output_ui("ui_kappa_v1")),
                    ui.column(6, ui.output_ui("ui_kappa_v2")),
                ),
                ui.output_ui("ui_kappa_warning"),
                ui.row(
                    ui.column(
                        6,
                        ui.input_action_button(
                            "btn_analyze_kappa",
                            "🚀 Calculate Kappa",
                            class_="btn-primary w-100",
                            width="100%",
                        ),
                    ),
                    ui.column(
                        6,
                        ui.download_button(
                            "btn_dl_kappa_report",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                            width="100%",
                        ),
                    ),
                ),
                ui.br(),
                ui.output_ui("ui_kappa_status"),  # Status message area
                ui.br(),
                ui.output_ui("out_kappa_results"),
            ),
            # TAB 4: Descriptive
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
            # TAB 5: Reference & Interpretation
            ui.nav_panel(
                "ℹ️ Reference & Interpretation",
                ui.markdown(
                    """
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
                """
                ),
            ),
        ),
    )


# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def diag_server(input, output, session, df, var_meta, df_matched, is_matched):

    # --- Reactive Results Storage ---
    roc_html = reactive.Value(None)
    chi_html = reactive.Value(None)
    kappa_html = reactive.Value(None)
    desc_html = reactive.Value(None)

    # --- Processing Status Indicators ---
    roc_processing = reactive.Value(False)
    chi_processing = reactive.Value(False)
    kappa_processing = reactive.Value(False)
    desc_processing = reactive.Value(False)

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df():
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
    def all_cols():
        d = current_df()
        if d is not None:
            return d.columns.tolist()
        return []

    # --- ROC Inputs UI ---
    @render.ui
    def ui_roc_truth():
        cols = all_cols()
        def_idx = 0
        for i, c in enumerate(cols):
            if "gold" in c.lower() or "standard" in c.lower():
                def_idx = i
                break
        return ui.input_select(
            "sel_roc_truth",
            "Gold Standard (Binary):",
            choices=cols,
            selected=cols[def_idx] if cols else None,
        )

    @render.ui
    def ui_roc_score():
        cols = all_cols()
        score_idx = 0
        for i, c in enumerate(cols):
            if "score" in c.lower():
                score_idx = i
                break
        return ui.input_select(
            "sel_roc_score",
            "Test Score (Continuous):",
            choices=cols,
            selected=cols[score_idx] if cols else None,
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
            unique_vals = sorted(
                [str(x) for x in d[truth_col].dropna().unique()]
            )
            default_pos_idx = 0
            if "1" in unique_vals:
                default_pos_idx = unique_vals.index("1")
            return ui.input_select(
                "sel_roc_pos_label",
                "Positive Label (1):",
                choices=unique_vals,
                selected=unique_vals[default_pos_idx]
                if unique_vals
                else None,
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
        v1_idx = next(
            (i for i, c in enumerate(cols) if c == "Treatment_Group"), 0
        )
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

    def get_pos_label_settings(df_input, col_name):
        if df_input is None or col_name not in df_input.columns:
            return [], 0
        unique_vals = [str(x) for x in df_input[col_name].dropna().unique()]
        unique_vals.sort()
        default_idx = 0
        if "1" in unique_vals:
            default_idx = unique_vals.index("1")
        elif "0" in unique_vals:
            default_idx = unique_vals.index("0")
        return unique_vals, default_idx

    @render.ui
    def ui_chi_v1_pos():
        v1_col = input.sel_chi_v1()
        d = current_df()
        v1_uv, v1_idx = get_pos_label_settings(d, v1_col)
        if not v1_uv:
            return ui.div(
                ui.markdown(f"⚠️ No values in {v1_col}"),
                class_="text-error text-sm",
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
        v2_uv, v2_idx = get_pos_label_settings(d, v2_col)
        if not v2_uv:
            return ui.div(
                ui.markdown(f"⚠️ No values in {v2_col}"),
                class_="text-error text-sm",
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

    # --- Kappa Inputs UI ---
    @render.ui
    def ui_kappa_v1():
        cols = all_cols()
        kv1_idx = 0
        for i, col in enumerate(cols):
            if (
                "dr_a" in col.lower()
                or "rater_1" in col.lower()
                or "diagnosis_a" in col.lower()
            ):
                kv1_idx = i
                break
        return ui.input_select(
            "sel_kappa_v1",
            "Rater/Method 1:",
            choices=cols,
            selected=cols[kv1_idx] if cols else None,
        )

    @render.ui
    def ui_kappa_v2():
        cols = all_cols()
        kv2_idx = min(1, len(cols) - 1)
        for i, col in enumerate(cols):
            if (
                "dr_b" in col.lower()
                or "rater_2" in col.lower()
                or "diagnosis_b" in col.lower()
            ):
                kv2_idx = i
                break
        kv1_idx = next(
            (i for i, c in enumerate(cols) if c == input.sel_kappa_v1()), 0
        )
        if kv2_idx == kv1_idx and len(cols) > 1:
            kv2_idx = min(kv1_idx + 1, len(cols) - 1)
        return ui.input_select(
            "sel_kappa_v2",
            "Rater/Method 2:",
            choices=cols,
            selected=cols[kv2_idx] if cols else None,
        )

    @render.ui
    def ui_kappa_warning():
        if input.sel_kappa_v1() == input.sel_kappa_v2():
            return ui.div(
                ui.markdown("⚠️ Please select two different columns for Kappa."),
                class_="alert alert-warning",
            )
        return None

    # --- Descriptive Inputs UI ---
    @render.ui
    def ui_desc_var():
        cols = all_cols()
        return ui.input_select(
            "sel_desc_var", "Select Variable:", choices=cols
        )

    # --- ROC Status & Results ---
    @render.ui
    def ui_roc_status():
        if roc_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(
                        class_="spinner-border spinner-border-sm me-2"
                    ),
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
                    ui.tags.span(
                        class_="spinner-border spinner-border-sm me-2"
                    ),
                    "📄 Calculating Chi-Square statistics... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_kappa_status():
        if kappa_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(
                        class_="spinner-border spinner-border-sm me-2"
                    ),
                    "📄 Calculating Kappa statistics... Please wait",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_desc_status():
        if desc_processing.get():
            return ui.div(
                ui.tags.div(
                    ui.tags.span(
                        class_="spinner-border spinner-border-sm me-2"
                    ),
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
            )

            if err:
               roc_html.set(
                    f"<div class='alert alert-danger'>📄 Error: {err}</div>"


                )
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
                    rep.append(
                        {
                            "type": "table",
                            "header": "ROC Statistics",
                            "data": pd.DataFrame([res]).T,
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

                html_report = diag_test.generate_report(
                    f"ROC Analysis Report ({method})", rep
                )
                roc_html.set(html_report)

        finally:
            # Clear processing flag
            roc_processing.set(False)

    @render.ui
    def out_roc_results():
        if roc_html.get():
            return ui.HTML(roc_html.get())
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
            tab, stats, msg, risk = diag_test.calculate_chi2(
                d,
                input.sel_chi_v1(),
                input.sel_chi_v2(),
                method=input.radio_chi_method(),
                v1_pos=input.sel_chi_v1_pos(),
                v2_pos=input.sel_chi_v2_pos(),
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



        finally:
            # Clear processing flag
            chi_processing.set(False)

    @render.ui
    def out_chi_results():
        if chi_html.get():
            return ui.HTML(chi_html.get())
        return ui.div(
            "Results will appear here.", class_="text-secondary p-3"
        )

    @render.download(filename="chi2_report.html")
    def btn_dl_chi_report():
        yield chi_html.get()

    # --- Kappa Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_kappa)
    def _run_kappa():
        d = current_df()
        req(d is not None, input.sel_kappa_v1(), input.sel_kappa_v2())

        # Set processing flag
        kappa_processing.set(True)

        try:
            res, err, conf = diag_test.calculate_kappa(
                d, input.sel_kappa_v1(), input.sel_kappa_v2()
            )
            if err:
                kappa_html.set(
                    "<div class='alert alert-danger'>{}</div>".format(err)
                )
            else:
                rep = [
                    {
                        "type": "text",
                        "data": f"Agreement Analysis: {input.sel_kappa_v1()} vs {input.sel_kappa_v2()}",
                    },
                    {
                        "type": "table",
                        "header": "Kappa Statistics",
                        "data": res,
                    },
                    {
                        "type": "contingency_table",
                        "header": "Confusion Matrix (Crosstab)",
                        "data": conf,
                    },
                ]
                kappa_html.set(
                    diag_test.generate_report(
                        f"Kappa: {input.sel_kappa_v1()} vs {input.sel_kappa_v2()}",
                        rep,
                    )
                )

        finally:
            # Clear processing flag
            kappa_processing.set(False)

    @render.ui
    def out_kappa_results():
        if kappa_html.get():
            return ui.HTML(kappa_html.get())
        return ui.div(
            "Results will appear here.", class_="text-secondary p-3"
        )

    @render.download(filename="kappa_report.html")
    def btn_dl_kappa_report():
        yield kappa_html.get()

    # --- Descriptive Analysis Logic ---
    @reactive.Effect
    @reactive.event(input.btn_run_desc)
    def _run_desc():
        d = current_df()
        req(d is not None, input.sel_desc_var())

        # Set processing flag
        desc_processing.set(True)

        try:
            res = diag_test.calculate_descriptive(
                d, input.sel_desc_var()
            )
            if res is not None:
                desc_html.set(
                    diag_test.generate_report(
                        f"Descriptive: {input.sel_desc_var()}",
                        [{"type": "table", "data": res}],
                    )
                )
            else:
                desc_html.set(
                    "<div class='alert alert-danger'>No data available for {}</div>".format(
                        input.sel_desc_var()
                    )
                )

        finally:
            # Clear processing flag
            desc_processing.set(False)

    @render.ui
    def out_desc_results():
        if desc_html.get():
            return ui.HTML(desc_html.get())
        return ui.div(
            "Results will appear here.", class_="text-secondary p-3"
        )

    @render.download(filename="descriptive_report.html")
    def btn_dl_desc_report():
        yield desc_html.get()

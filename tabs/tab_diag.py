from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import diag_test  # ดึง Logic จากไฟล์ diag_test.py ที่เราปรับปรุงแล้ว
from typing import List, Tuple

# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def diag_ui():
    return ui.navset_card_tab(
        # ---------------------------------------------------------------------
        # TAB 1: ROC Curve
        # ---------------------------------------------------------------------
        ui.nav_panel("📈 ROC Curve & AUC",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("ROC Analysis Options"),
                    ui.output_ui("ui_roc_dataset_source"),
                    ui.hr(),
                    ui.input_select("sel_roc_truth", "Gold Standard (Binary):", choices=[]),
                    ui.input_select("sel_roc_score", "Test Score (Continuous):", choices=[]),
                    ui.input_radio_buttons("radio_roc_method", "CI Method:", 
                                         {"delong": "DeLong et al.", "hanley": "Binomial (Hanley)"}),
                    ui.input_select("sel_roc_pos_label", "Positive Label (1):", choices=[]),
                    ui.hr(),
                    ui.input_action_button("btn_analyze_roc", "🚀 Analyze ROC", class_="btn-primary"),
                    ui.br(),
                    ui.download_button("btn_dl_roc_report", "📥 Download Report", class_="btn-secondary"),
                    width=350
                ),
                ui.output_ui("out_roc_results")
            )
        ),

        # ---------------------------------------------------------------------
        # TAB 2: Chi-Square
        # ---------------------------------------------------------------------
        ui.nav_panel("🎲 Chi-Square & Risk (2x2)",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("2x2 Analysis Options"),
                    ui.input_select("sel_chi_v1", "Variable 1 (Row/Exposure):", choices=[]),
                    ui.input_select("sel_chi_v2", "Variable 2 (Col/Outcome):", choices=[]),
                    ui.input_radio_buttons("radio_chi_method", "Test Method:", 
                        {"Pearson (Standard)": "Pearson", "Yates' correction": "Yates", "Fisher's Exact Test": "Fisher"}),
                    ui.input_select("sel_chi_v1_pos", "Pos Label (Row):", choices=[]),
                    ui.input_select("sel_chi_v2_pos", "Pos Label (Col):", choices=[]),
                    ui.hr(),
                    ui.input_action_button("btn_analyze_chi", "🚀 Run Analysis", class_="btn-primary"),
                    ui.br(),
                    ui.download_button("btn_dl_chi_report", "📥 Download Report"),
                    width=350
                ),
                ui.output_ui("out_chi_results")
            )
        ),

        # ---------------------------------------------------------------------
        # TAB 3: Agreement
        # ---------------------------------------------------------------------
        ui.nav_panel("🤝 Agreement (Kappa)",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Kappa Options"),
                    ui.input_select("sel_kappa_v1", "Rater 1:", choices=[]),
                    ui.input_select("sel_kappa_v2", "Rater 2:", choices=[]),
                    ui.hr(),
                    ui.input_action_button("btn_analyze_kappa", "🚀 Calculate Kappa", class_="btn-primary"),
                    ui.br(),
                    ui.download_button("btn_dl_kappa_report", "📥 Download Report"),
                    width=350
                ),
                ui.output_ui("out_kappa_results")
            )
        ),

        # ---------------------------------------------------------------------
        # TAB 4: Descriptive
        # ---------------------------------------------------------------------
        ui.nav_panel("📊 Descriptive",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("sel_desc_var", "Select Variable:", choices=[]),
                    ui.input_action_button("btn_run_desc", "Show Stats", class_="btn-primary"),
                    ui.br(),
                    ui.download_button("btn_dl_desc_report", "📥 Download Report"),
                    width=350
                ),
                ui.output_ui("out_desc_results")
            )
        ),

        # ---------------------------------------------------------------------
        # TAB 5: Reference
        # ---------------------------------------------------------------------
        ui.nav_panel("ℹ️ Reference",
            ui.markdown("""
            ## 📚 Reference & Interpretation Guide
            
            ### 🚦 Quick Decision Guide
            | Question | Recommended Test | Goal |
            | :--- | :--- | :--- |
            | How well does a score predict disease? | **ROC Curve** | Find AUC and Best Cut-off |
            | Are two groups different in outcome? | **Chi-Square** | Calculate OR, RR, p-value |
            | Do two doctors agree? | **Kappa** | Measure Inter-rater reliability |
            
            ### ⚖️ Interpretation
            * **AUC > 0.8:** Good discrimination.
            * **Kappa > 0.6:** Substantial agreement.
            * **P < 0.05:** Statistically significant.
            """)
        )
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

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df():
        # ถ้ามี matched data และเลือกใช้ matched
        if is_matched.get() and input.radio_diag_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_roc_dataset_source():
        if is_matched.get():
            return ui.input_radio_buttons("radio_diag_source", "📄 Select Dataset:",
                                        {"original": "📊 Original", "matched": "✅ Matched"},
                                        selected="matched", inline=True)
        return None

    # --- Update Dropdowns ---
    @reactive.Effect
    def _update_dropdowns():
        d = current_df()
        if d is None: return
        cols = d.columns.tolist()
        ui.update_select("sel_roc_truth", choices=cols)
        ui.update_select("sel_roc_score", choices=cols)
        ui.update_select("sel_chi_v1", choices=cols)
        ui.update_select("sel_chi_v2", choices=cols)
        ui.update_select("sel_kappa_v1", choices=cols)
        ui.update_select("sel_kappa_v2", choices=cols)
        ui.update_select("sel_desc_var", choices=cols)

    # --- Dynamic Label Updates ---
    @reactive.Effect
    def _update_pos_labels():
        d = current_df()
        if d is None: return
        
        # ROC Pos Label
        truth_col = input.sel_roc_truth()
        if truth_col in d.columns:
            u_vals = [str(x) for x in d[truth_col].dropna().unique()]
            ui.update_select("sel_roc_pos_label", choices=u_vals)
            
        # Chi Pos Labels
        v1, v2 = input.sel_chi_v1(), input.sel_chi_v2()
        if v1 in d.columns:
            ui.update_select("sel_chi_v1_pos", choices=[str(x) for x in d[v1].dropna().unique()])
        if v2 in d.columns:
            ui.update_select("sel_chi_v2_pos", choices=[str(x) for x in d[v2].dropna().unique()])

    # --- 📈 ROC Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_roc)
    def _run_roc():
        d = current_df()
        # FIX: Check 'd is not None' explicitly to avoid ValueError with DataFrame
        req(d is not None, input.sel_roc_truth(), input.sel_roc_score())
        
        res, err, fig, coords = diag_test.analyze_roc(
            d, input.sel_roc_truth(), input.sel_roc_score(),
            method=input.radio_roc_method(),
            pos_label_user=input.sel_roc_pos_label()
        )
        
        if err:
            roc_html.set(f"<div class='alert alert-danger'>{err}</div>")
        else:
            rep = [
                {'type':'text', 'data': f"Analysis: {input.sel_roc_score()} vs {input.sel_roc_truth()}"},
                {'type':'plot', 'data': fig},
                {'type':'table', 'header':'Key Statistics', 'data': pd.DataFrame([res]).T},
                {'type':'table', 'header':'Performance Table', 'data': coords.head(20)}
            ]
            roc_html.set(diag_test.generate_report("ROC Analysis Report", rep))

    @render.ui
    def out_roc_results():
        if roc_html.get():
            return ui.HTML(roc_html.get())
        return ui.div("Click 'Analyze ROC' to view results.", class_="text-muted p-3")

    @render.download(filename="roc_report.html")
    def btn_dl_roc_report():
        yield roc_html.get()

    # --- 🎲 Chi-Square Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_chi)
    def _run_chi():
        d = current_df()
        # FIX: Check 'd is not None' explicitly
        req(d is not None, input.sel_chi_v1(), input.sel_chi_v2())
        
        tab, stats, msg, risk = diag_test.calculate_chi2(
            d, input.sel_chi_v1(), input.sel_chi_v2(),
            method=input.radio_chi_method(),
            v1_pos=input.sel_chi_v1_pos(),
            v2_pos=input.sel_chi_v2_pos()
        )
        
        rep = [
            {'type': 'contingency_table', 'header': 'Contingency Table', 'data': tab},
            {'type': 'table', 'header': 'Test Statistics', 'data': stats}
        ]
        if risk is not None:
            rep.append({'type': 'table', 'header': 'Risk & Diagnostic Metrics', 'data': risk})
        
        chi_html.set(diag_test.generate_report("Chi-Square Report", rep))

    @render.ui
    def out_chi_results():
        if chi_html.get(): return ui.HTML(chi_html.get())
        return ui.p("Results will appear here.", class_="p-3")

    # --- 🤝 Kappa Logic ---
    @reactive.Effect
    @reactive.event(input.btn_analyze_kappa)
    def _run_kappa():
        d = current_df()
        # FIX: Check 'd is not None' explicitly
        req(d is not None)
        res, err, conf = diag_test.calculate_kappa(d, input.sel_kappa_v1(), input.sel_kappa_v2())
        if err: kappa_html.set(err)
        else:
            rep = [{'type':'table', 'header':'Agreement', 'data':res}, {'type':'table', 'header':'Matrix', 'data':conf}]
            kappa_html.set(diag_test.generate_report("Kappa Report", rep))

    @render.ui
    def out_kappa_results():
        if kappa_html.get(): return ui.HTML(kappa_html.get())
        return ui.p("Kappa results...", class_="p-3")

    # --- 📊 Descriptive Logic ---
    @reactive.Effect
    @reactive.event(input.btn_run_desc)
    def _run_desc():
        d = current_df()
        # FIX: Check 'd is not None' explicitly
        req(d is not None, input.sel_desc_var())
        res = diag_test.calculate_descriptive(d, input.sel_desc_var())
        desc_html.set(diag_test.generate_report(f"Stats: {input.sel_desc_var()}", [{'type':'table', 'data':res}]))

    @render.ui
    def out_desc_results():
        if desc_html.get(): return ui.HTML(desc_html.get())
        return ui.p("Descriptive stats...", class_="p-3")

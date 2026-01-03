from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from htmltools import HTML, div
import gc

# Import internal modules
# แก้ไขการเรียกใช้ให้สัมพันธ์กับพารามิเตอร์ใหม่ใน logic.py
from logic import process_data_and_generate_html, HAS_FIRTH
from forest_plot_lib import create_forest_plot
from subgroup_analysis_module import SubgroupAnalysisLogit
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()

# ==============================================================================
# Helper Functions
# ==============================================================================
def check_perfect_separation(df, target_col):
    """Identify columns causing perfect separation."""
    risky_vars = []
    try:
        y = pd.to_numeric(df[target_col], errors='coerce').dropna()
        if y.nunique() < 2: 
            return []
    except (KeyError, TypeError, ValueError): 
        return []

    for col in df.columns:
        if col == target_col: continue
        # ตรวจสอบเฉพาะตัวแปรที่มีกลุ่มน้อยๆ
        if df[col].nunique() < 10: 
            try:
                tab = pd.crosstab(df[col], y)
                if (tab == 0).any().any():
                    risky_vars.append(col)
            except (ValueError, TypeError): 
                pass
    return risky_vars

# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def logit_ui():
    return ui.navset_card_tab(
        # =====================================================================
        # TAB 1: Regression Analysis (Binary/Poisson)
        # =====================================================================
        ui.nav_panel(
            "📈 Regression Analysis",
            
            ui.card(
                ui.card_header("📈 Analysis Options"),
                
                ui.output_ui("ui_dataset_selector"),
                ui.hr(),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Variable Selection:"),
                        ui.input_select("sel_outcome", "Select Outcome (Y):", choices=[]),
                        ui.output_ui("ui_separation_warning"),
                        ui.output_ui("ui_firth_status"), # แสดงสถานะ Firth
                    ),
                    
                    ui.card(
                        ui.card_header("Model & Method:"),
                        ui.input_radio_buttons(
                            "radio_model_type",
                            "Model Type:",
                            {
                                "logistic": "Binary Logistic (0/1)",
                                "poisson": "Poisson (Counts/Rates)"
                            }
                        ),
                        ui.input_radio_buttons(
                            "radio_method",
                            "Fitting Method:",
                            {
                                "auto": "Auto (Recommended)",
                                "bfgs": "Standard (MLE)",
                                "firth": "Firth's (Penalized)"
                            }
                        ),
                    ),
                    col_widths=[6, 6]
                ),

                ui.layout_columns(
                    ui.div(
                        ui.h6("Exclude Variables (Optional):"),
                        ui.input_selectize("sel_exclude", label=None, choices=[], multiple=True),
                    ),
                    ui.div(
                        ui.h6("Interaction Terms (Optional):"),
                        ui.input_selectize(
                            "sel_interactions", 
                            label=None, 
                            choices=[], 
                            multiple=True,
                            options={"placeholder": "Select pairs e.g. age:sex"}
                        ),
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.hr(),
                
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run_logit",
                        "🚀 Run Regression",
                        class_="btn-primary btn-sm w-100"
                    ),
                    ui.download_button(
                        "btn_dl_report",
                        "📥 Download Report",
                        class_="btn-secondary btn-sm w-100"
                    ),
                    col_widths=[6, 6]
                ),
            ),
            
            ui.output_ui("out_logit_status"),
            ui.navset_card_underline(
                ui.nav_panel(
                    "🌳 Forest Plots",
                    ui.output_ui("ui_forest_tabs")
                ),
                ui.nav_panel(
                    "📋 Detailed Report",
                    ui.output_ui("out_html_report")
                )
            )
        ),

        # =====================================================================
        # TAB 2: Subgroup Analysis (คงเดิม)
        # =====================================================================
        ui.nav_panel(
            "🗣️ Subgroup Analysis",
            ui.card(
                ui.card_header("🗣️ Subgroup Settings"),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Core Variables:"),
                        ui.input_select("sg_outcome", "Outcome (Binary):", choices=[]),
                        ui.input_select("sg_treatment", "Treatment/Exposure:", choices=[]),
                        ui.input_select("sg_subgroup", "Stratify By:", choices=[]),
                    ),
                    ui.card(
                        ui.card_header("Adjustment & Advanced:"),
                        ui.input_selectize("sg_adjust", "Adjustment Covariates:", choices=[], multiple=True),
                        ui.input_numeric("sg_min_n", "Min N per subgroup:", value=5, min=2),
                    ),
                    col_widths=[6, 6]
                ),
                ui.accordion(
                    ui.accordion_panel(
                        "✏️ Custom Settings",
                        ui.input_text("sg_title", "Custom Title:", placeholder="Subgroup Analysis..."),
                    ),
                    open=False
                ),
                ui.hr(),
                ui.input_action_button(
                    "btn_run_subgroup",
                    "🚀 Run Subgroup Analysis",
                    class_="btn-primary btn-sm w-100"
                ),
            ),
            ui.output_ui("out_subgroup_status"),
            ui.navset_card_underline(
                ui.nav_panel(
                    "🌳 Forest Plot",
                    output_widget("out_sg_forest_plot"),
                    ui.hr(),
                    ui.input_text("txt_edit_forest_title", "Edit Plot Title:", placeholder="Enter new title..."),
                    ui.input_action_button("btn_update_plot_title", "Update Title", class_="btn-sm"),
                ),
                ui.nav_panel(
                    "📂 Summary & Interpretation",
                    ui.layout_columns(
                        ui.value_box("Overall OR", ui.output_text("val_overall_or")),
                        ui.value_box("Overall P-value", ui.output_text("val_overall_p")),
                        ui.value_box("Interaction P-value", ui.output_text("val_interaction_p")),
                        col_widths=[4, 4, 4]
                    ),
                    ui.hr(),
                    ui.output_ui("out_interpretation_box"),
                    ui.h5("Detailed Results"),
                    ui.output_data_frame("out_sg_table")
                ),
                ui.nav_panel(
                    "💾 Exports",
                    ui.h5("Download Results"),
                    ui.layout_columns(
                        ui.download_button("dl_sg_html", "💿 HTML Plot", class_="btn-sm w-100"),
                        ui.download_button("dl_sg_csv", "📋 CSV Results", class_="btn-sm w-100"),
                        ui.download_button("dl_sg_json", "📝 JSON Data", class_="btn-sm w-100"),
                        col_widths=[4, 4, 4]
                    )
                )
            )
        ),

        # =====================================================================
        # TAB 3: Reference
        # =====================================================================
        ui.nav_panel(
            "ℹ️ Reference",
            ui.markdown("""
## 📚 Regression Reference

### Model Types:
* **Binary Logistic**: For 0/1 outcomes. Reports **Odds Ratios (OR)**.
* **Poisson**: For count data or rates. Reports **Incidence Rate Ratios (IRR)**.

### Interpretation:
* **Value > 1**: Positive association (Higher risk/rate) 🔴
* **Value < 1**: Negative association (Protective) 🟢
* **P-value < 0.05**: Statistically significant.
* **VIF > 10**: High multicollinearity (Variables might be redundant).

### Interaction Terms:
* Testing if the effect of one variable depends on another (e.g., `age:sex`).
            """)
        )
    )

# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def logit_server(input, output, session, df, var_meta, df_matched, is_matched):
    
    logit_res = reactive.Value(None)
    subgroup_res = reactive.Value(None)
    subgroup_analyzer = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.btn_run_logit, input.btn_run_subgroup)
    def _cleanup_after_analysis():
        gc.collect()

    @reactive.Calc
    def current_df():
        if is_matched.get() and input.radio_dataset_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_dataset_selector():
        d_orig = df.get()
        if is_matched.get():
            d_match = df_matched.get()
            return ui.input_radio_buttons(
                "radio_dataset_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original ({len(d_orig) if d_orig is not None else 0})",
                    "matched": f"✅ Matched ({len(d_match) if d_match is not None else 0})"
                },
                selected="matched",
                inline=True
            )
        return ui.p(f"📊 Using Original Data ({len(d_orig) if d_orig is not None else 0} rows)", class_="text-muted")

    @reactive.Effect
    def _update_inputs():
        d = current_df()
        if d is None or d.empty: return
        
        cols = d.columns.tolist()
        binary_cols = [c for c in cols if d[c].nunique() == 2]
        sg_cols = [c for c in cols if 2 <= d[c].nunique() <= 10]
        
        # Create interaction pairs list (e.g., var1:var2)
        interaction_choices = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                interaction_choices.append(f"{cols[i]}:{cols[j]}")

        ui.update_select("sel_outcome", choices=cols) # Poisson supports non-binary
        ui.update_selectize("sel_exclude", choices=cols)
        ui.update_selectize("sel_interactions", choices=interaction_choices)
        
        ui.update_select("sg_outcome", choices=binary_cols)
        ui.update_select("sg_treatment", choices=cols)
        ui.update_select("sg_subgroup", choices=sg_cols)
        ui.update_selectize("sg_adjust", choices=cols)

    @render.ui
    def ui_firth_status():
        if not HAS_FIRTH:
            return ui.div("⚠️ Firthlogist not installed (Firth method disabled)", class_="text-danger", style="font-size: 0.8em;")
        return None

    @render.ui
    def ui_separation_warning():
        d = current_df()
        target = input.sel_outcome()
        if d is None or d.empty or not target: return None
        if input.radio_model_type() != "logistic": return None
        
        risky = check_perfect_separation(d, target)
        if risky:
            return ui.div(
                ui.h6("⚠️ Perfect Separation Risk", class_="text-warning"),
                ui.p(f"Variables: {', '.join(risky)}", style="font-size: 0.8em;")
            )
        return None

    # ==========================================================================
    # LOGIC: Regression Execution
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_logit)
    def _run_logit():
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude()
        method = input.radio_method()
        model_type = input.radio_model_type()
        interactions = list(input.sel_interactions())
        
        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        
        # เตรียมข้อมูลตัดตัวแปรที่ Exclude ออก
        final_df = d.drop(columns=exclude, errors='ignore')
        
        with ui.Progress(min=0, max=1) as p:
            p.set(message=f"Running {model_type.capitalize()} Regression...", detail="Computing results & VIF...")
            
            try:
                # ส่งพารามิเตอร์ใหม่ไปยัง logic.py
                html_rep, or_res, aor_res = process_data_and_generate_html(
                    final_df, 
                    target, 
                    var_meta=var_meta.get(), 
                    method=method,
                    model_type=model_type,
                    interaction_terms=interactions
                )
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Regression error")
                return
            
            # Label สำหรับ Forest Plot (OR หรือ IRR)
            eff_label = "IRR" if model_type == "poisson" else "OR"
            
            fig_adj = None
            fig_crude = None
            
            if aor_res:
                df_adj = pd.DataFrame([{'variable': k, **v} for k, v in aor_res.items()])
                if not df_adj.empty:
                    fig_adj = create_forest_plot(
                        df_adj, 'aor', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                        title=f"<b>Multivariable: Adjusted {eff_label}</b>", x_label=f"Adjusted {eff_label}"
                    )
            
            if or_res:
                df_crude = pd.DataFrame([{'variable': k, **v} for k, v in or_res.items()])
                if not df_crude.empty:
                    fig_crude = create_forest_plot(
                        df_crude, 'or', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                        title=f"<b>Univariable: Crude {eff_label}</b>", x_label=f"Crude {eff_label}"
                    )

            logit_res.set({
                "html": html_rep,
                "fig_adj": fig_adj,
                "fig_crude": fig_crude
            })
            
            ui.notification_show("✅ Analysis Complete!", type="message")

    # --- Renderers ---
    @render.ui
    def out_logit_status():
        if logit_res.get():
            return ui.div(ui.h5("✅ Regression Complete"), class_="alert alert-success")
        return None

    @render.ui
    def out_html_report():
        res = logit_res.get()
        if res:
            return ui.card(ui.HTML(res['html']))
        return ui.p("Run analysis to see detailed report.", class_="text-muted text-center p-4")

    @render.ui
    def ui_forest_tabs():
        res = logit_res.get()
        if not res: return ui.p("Run analysis to see forest plots.", class_="text-muted text-center p-4")
        
        tabs = []
        if res['fig_crude']:
            tabs.append(ui.nav_panel("Crude Effect", output_widget("out_forest_crude")))
        if res['fig_adj']:
            tabs.append(ui.nav_panel("Adjusted Effect", output_widget("out_forest_adj")))
            
        return ui.navset_card_tab(*tabs) if tabs else ui.p("No forest plots available.")

    @render_widget
    def out_forest_adj():
        res = logit_res.get()
        return res['fig_adj'] if res else None

    @render_widget
    def out_forest_crude():
        res = logit_res.get()
        return res['fig_crude'] if res else None

    @render.download(filename="regression_report.html")
    def btn_dl_report():
        res = logit_res.get()
        if res: yield res['html']

    # ==========================================================================
    # Subgroup Analysis Logic (คงเดิม)
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_subgroup)
    def _run_subgroup():
        d = current_df()
        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        
        analyzer = SubgroupAnalysisLogit(d)
        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Subgroup Analysis...")
            try:
                results = analyzer.analyze(
                    outcome_col=input.sg_outcome(),
                    treatment_col=input.sg_treatment(),
                    subgroup_col=input.sg_subgroup(),
                    adjustment_cols=list(input.sg_adjust()),
                    min_subgroup_n=input.sg_min_n()
                )
                subgroup_res.set(results)
                subgroup_analyzer.set(analyzer)
                ui.notification_show("✅ Subgroup Analysis Complete!", type="message")
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")

    @render_widget
    def out_sg_forest_plot():
        analyzer = subgroup_analyzer.get()
        if analyzer:
            title = input.txt_edit_forest_title() or input.sg_title() or None
            return analyzer.create_forest_plot(title=title)
        return None

    @render.text
    def val_overall_or():
        res = subgroup_res.get()
        if res: return f"{res.get('overall', {}).get('or', 0):.3f}"
        return "-"

    @render.text
    def val_overall_p():
        res = subgroup_res.get()
        if res: return f"{res['overall']['p_value']:.4f}"
        return "-"
    
    @render.text
    def val_interaction_p():
        res = subgroup_res.get()
        if res: return f"{res['interaction']['p_value']:.4f}"
        return "-"

    @render.ui
    def out_interpretation_box():
        res = subgroup_res.get()
        analyzer = subgroup_analyzer.get()
        if res and analyzer:
            interp = analyzer.get_interpretation()
            color = "alert-warning" if res['interaction']['significant'] else "alert-success"
            return ui.div(f"{interp}", class_=f"alert {color}")
        return None

    @render.data_frame
    def out_sg_table():
        res = subgroup_res.get()
        if res:
            df_res = res['results_df'].copy()
            cols = ['group', 'n', 'events', 'or', 'ci_low', 'ci_high', 'p_value']
            available_cols = [c for c in cols if c in df_res.columns]
            return render.DataGrid(df_res[available_cols].round(4))
        return None

    @render.download(filename=lambda: f"subgroup_plot_{input.sg_subgroup()}.html")
    def dl_sg_html():
        analyzer = subgroup_analyzer.get()
        if analyzer and analyzer.figure:
            yield analyzer.figure.to_html(include_plotlyjs='cdn')

    @render.download(filename=lambda: f"subgroup_res_{input.sg_subgroup()}.csv")
    def dl_sg_csv():
        res = subgroup_res.get()
        if res: yield res['results_df'].to_csv(index=False)

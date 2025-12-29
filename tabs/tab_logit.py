from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from htmltools import HTML, div

# Import internal modules
from logic import process_data_and_generate_html
from forest_plot_lib import create_forest_plot
from subgroup_analysis_module import SubgroupAnalysisLogit
from logger import get_logger

logger = get_logger(__name__)

# ==============================================================================
# Helper Functions (Pure Logic)
# ==============================================================================
def check_perfect_separation(df, target_col):
    """Identify columns causing perfect separation."""
    risky_vars = []
    try:
        y = pd.to_numeric(df[target_col], errors='coerce').dropna()
        if y.nunique() < 2: return []
    except: return []

    for col in df.columns:
        if col == target_col: continue
        if df[col].nunique() < 10: 
            try:
                tab = pd.crosstab(df[col], y)
                if (tab == 0).any().any():
                    risky_vars.append(col)
            except: pass
    return risky_vars

# ==============================================================================
# UI Definition
# ==============================================================================
@module.ui
def logit_ui():
    return ui.navset_card_tab(
        # ---------------------------------------------------------------------
        # TAB 1: Binary Logistic Regression
        # ---------------------------------------------------------------------
        ui.nav_panel("📈 Binary Logistic Regression",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Analysis Options"),
                    
                    # Dataset Source Selection (Dynamic)
                    ui.output_ui("ui_dataset_selector"),
                    ui.hr(),

                    # Variable Selection
                    ui.input_select("sel_outcome", "Select Outcome (Y):", choices=[]),
                    ui.output_ui("ui_separation_warning"), # Warning box
                    ui.input_selectize("sel_exclude", "Exclude Variables:", choices=[], multiple=True),
                    
                    # Method Selection
                    ui.input_radio_buttons("radio_method", "Regression Method:", 
                        {"auto": "Auto (Recommended)", "bfgs": "Standard (MLE)", "firth": "Firth's (Penalized)"}),
                    
                    ui.hr(),
                    ui.input_action_button("btn_run_logit", "🚀 Run Regression", class_="btn-primary"),
                    ui.br(), ui.br(),
                    ui.download_button("btn_dl_report", "📥 Download Report", class_="btn-secondary"),
                    width=350
                ),
                
                # Main Content Area
                ui.output_ui("out_logit_status"),
                ui.navset_card_underline(
                    ui.nav_panel("🌳 Forest Plots",
                        ui.output_ui("ui_forest_tabs") # Dynamic tabs for Crude/Adjusted
                    ),
                    ui.nav_panel("📋 Detailed Report",
                        ui.output_ui("out_html_report")
                    )
                )
            )
        ),

        # ---------------------------------------------------------------------
        # TAB 2: Subgroup Analysis
        # ---------------------------------------------------------------------
        ui.nav_panel("🗒️ Subgroup Analysis",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Subgroup Settings"),
                    ui.input_select("sg_outcome", "Outcome (Binary):", choices=[]),
                    ui.input_select("sg_treatment", "Treatment/Exposure:", choices=[]),
                    ui.input_select("sg_subgroup", "Stratify By:", choices=[]),
                    ui.input_selectize("sg_adjust", "Adjustment Covariates:", choices=[], multiple=True),
                    
                    ui.accordion(
                        ui.accordion_panel("⚙️ Advanced",
                            ui.input_numeric("sg_min_n", "Min N per subgroup:", value=5, min=2),
                            ui.input_text("sg_title", "Custom Title:", placeholder="Subgroup Analysis...")
                        ),
                        open=False
                    ),
                    
                    ui.hr(),
                    ui.input_action_button("btn_run_subgroup", "🚀 Run Subgroup", class_="btn-primary"),
                    width=350
                ),
                
                # Subgroup Results Area
                ui.output_ui("out_subgroup_status"),
                ui.navset_card_underline(
                    ui.nav_panel("🌳 Forest Plot",
                        output_widget("out_sg_forest_plot"),
                        ui.input_text("txt_edit_forest_title", "Edit Plot Title:", placeholder="Enter new title..."),
                        ui.input_action_button("btn_update_plot_title", "Update Title", class_="btn-sm")
                    ),
                    ui.nav_panel("📊 Summary & Interpretation",
                        ui.layout_columns(
                            ui.value_box("Overall OR", ui.output_text("val_overall_or")),
                            ui.value_box("Overall P-value", ui.output_text("val_overall_p")),
                            ui.value_box("Interaction P-value", ui.output_text("val_interaction_p"))
                        ),
                        ui.hr(),
                        ui.output_ui("out_interpretation_box"),
                        ui.h5("Detailed Results"),
                        ui.output_data_frame("out_sg_table")
                    ),
                    ui.nav_panel("💾 Exports",
                        ui.h5("Download Results"),
                        ui.layout_columns(
                            ui.download_button("dl_sg_html", "💿 HTML Plot"),
                            ui.download_button("dl_sg_csv", "📋 CSV Results"),
                            ui.download_button("dl_sg_json", "📝 JSON Data")
                        )
                    )
                )
            )
        ),

        # ---------------------------------------------------------------------
        # TAB 3: Reference
        # ---------------------------------------------------------------------
        ui.nav_panel("ℹ️ Reference",
            ui.markdown("""
            ### 📚 Logistic Regression Reference
            
            **When to Use:**
            * Predicting binary outcomes (Disease/No Disease)
            * Understanding risk/protective factors (Odds Ratios)
            
            **Interpretation:**
            * **OR > 1**: Risk Factor (Increased odds) 🔴
            * **OR < 1**: Protective Factor (Decreased odds) 🟢
            * **OR = 1**: No Effect
            * **CI crosses 1**: Not statistically significant
            
            **Perfect Separation:**
            * Occurs when a predictor perfectly predicts the outcome (e.g., all smokers died).
            * **Solution:** Use **Auto** or **Firth's** method, or exclude the variable.
            
            **Subgroup Analysis:**
            * Tests if treatment effect varies by group (Interaction).
            * **P-interaction < 0.05**: Significant heterogeneity (Report subgroups separately).
            """)
        )
    )

# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def logit_server(input, output, session, df, var_meta, df_matched, is_matched):
    
    # --- State Management ---
    logit_res = reactive.Value(None)     # Store main logit results
    subgroup_res = reactive.Value(None)  # Store subgroup results
    subgroup_analyzer = reactive.Value(None) # Store analyzer instance
    
    # --- 1. Dataset Selection Logic ---
    @reactive.Calc
    def current_df():
        if is_matched.get() and input.radio_dataset_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_dataset_selector():
        if is_matched.get():
            return ui.input_radio_buttons("radio_dataset_source", "Select Dataset:",
                                        {"original": f"📊 Original ({len(df.get())})", 
                                         "matched": f"✅ Matched ({len(df_matched.get())})"},
                                        selected="matched")
        return ui.p(f"📊 Using Original Data ({len(df.get())} rows)", class_="text-muted")

    # --- 2. Dynamic Input Updates ---
    @reactive.Effect
    def _update_inputs():
        d = current_df()
        if d is None: return
        
        cols = d.columns.tolist()
        
        # Identify binary columns for outcomes
        binary_cols = [c for c in cols if d[c].nunique() == 2]
        
        # Identify potential subgroups (2-10 levels)
        sg_cols = [c for c in cols if 2 <= d[c].nunique() <= 10]
        
        # Update Tab 1 Inputs
        ui.update_select("sel_outcome", choices=binary_cols)
        ui.update_selectize("sel_exclude", choices=cols)
        
        # Update Tab 2 Inputs
        ui.update_select("sg_outcome", choices=binary_cols)
        ui.update_select("sg_treatment", choices=cols)
        ui.update_select("sg_subgroup", choices=sg_cols)
        ui.update_selectize("sg_adjust", choices=cols)

    # --- 3. Separation Warning Logic ---
    @render.ui
    def ui_separation_warning():
        d = current_df()
        target = input.sel_outcome()
        if d is None or not target: return None
        
        risky = check_perfect_separation(d, target)
        if risky:
            return ui.div(
                ui.h6("⚠️ Perfect Separation Risk", class_="text-warning"),
                ui.p(f"Variables: {', '.join(risky)}"),
                ui.p("Recommendation: Use 'Auto' method or exclude variables.", style="font-size: 0.8em;")
            )
        return None

    # ==========================================================================
    # LOGIC: Main Logistic Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_logit)
    def _run_logit():
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude()
        method = input.radio_method()
        
        req(d, target)
        
        # Prepare data
        final_df = d.drop(columns=exclude, errors='ignore')
        
        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Logistic Regression...", detail="Calculating...")
            
            # Run Logic from logic.py
            html_rep, or_res, aor_res = process_data_and_generate_html(
                final_df, target, var_meta=var_meta, method=method
            )
            
            # Generate Forest Plots using library
            fig_adj = None
            fig_crude = None
            
            if aor_res:
                df_adj = pd.DataFrame([{'variable': k, **v} for k, v in aor_res.items()])
                if not df_adj.empty:
                    fig_adj = create_forest_plot(
                        df_adj, 'aor', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                        title="<b>Multivariable: Adjusted OR</b>", x_label="Adjusted OR"
                    )
            
            if or_res:
                df_crude = pd.DataFrame([{'variable': k, **v} for k, v in or_res.items()])
                if not df_crude.empty:
                    fig_crude = create_forest_plot(
                        df_crude, 'or', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                        title="<b>Univariable: Crude OR</b>", x_label="Crude OR"
                    )

            # Store Results
            logit_res.set({
                "html": html_rep,
                "fig_adj": fig_adj,
                "fig_crude": fig_crude
            })
            
            ui.notification_show("Analysis Complete!", type="message")

    # --- Render Main Results ---
    @render.ui
    def out_html_report():
        res = logit_res.get()
        if res: return ui.HTML(res['html'])
        return ui.div("Run analysis to see detailed report.", class_="text-muted p-3")

    @render.ui
    def ui_forest_tabs():
        res = logit_res.get()
        if not res: return None
        
        tabs = []
        if res['fig_adj']:
            tabs.append(ui.nav_panel("Adjusted OR", output_widget("out_forest_adj")))
        if res['fig_crude']:
            tabs.append(ui.nav_panel("Crude OR", output_widget("out_forest_crude")))
            
        if not tabs: return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render_widget
    def out_forest_adj():
        res = logit_res.get()
        if res and res['fig_adj']: return res['fig_adj']
        return None

    @render_widget
    def out_forest_crude():
        res = logit_res.get()
        if res and res['fig_crude']: return res['fig_crude']
        return None

    @render.download(filename="logit_report.html")
    def btn_dl_report():
        res = logit_res.get()
        if res: yield res['html']

    # ==========================================================================
    # LOGIC: Subgroup Analysis
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_subgroup)
    def _run_subgroup():
        d = current_df()
        req(d, input.sg_outcome(), input.sg_treatment(), input.sg_subgroup())
        
        analyzer = SubgroupAnalysisLogit(d)
        
        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Subgroup Analysis...", detail="Testing interactions...")
            
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
                ui.notification_show("Subgroup Analysis Complete!", type="message")
                
            except Exception as e:
                ui.notification_show(f"Error: {str(e)}", type="error")
                logger.error(f"Subgroup error: {e}")

    # --- Render Subgroup Results ---
    @render_widget
    def out_sg_forest_plot():
        analyzer = subgroup_analyzer.get()
        if analyzer:
            # Check if user updated title
            title = input.sg_title() if input.sg_title() else None
            return analyzer.create_forest_plot(title=title)
        return None

    @reactive.Effect
    @reactive.event(input.btn_update_plot_title)
    def _update_sg_title():
        # This just triggers re-render of widget via reactive dependency on input.txt_edit_forest_title implies
        # But we need to update the analyzer's stored figure or call create again.
        # The widget function above calls create_forest_plot dynamically, so just invalidating is enough.
        pass

    @render.text
    def val_overall_or():
        res = subgroup_res.get()
        if res: return f"{res['overall']['or']:.3f}"
        return "-"

    @render.text
    def val_overall_p():
        res = subgroup_res.get()
        if res: return f"{res['overall']['p_value']:.4f}"
        return "-"
    
    @render.text
    def val_interaction_p():
        res = subgroup_res.get()
        if res:
             p_int = res['interaction']['p_value']
             return f"{p_int:.4f}" if p_int is not None else "N/A"
        return "-"

    @render.ui
    def out_interpretation_box():
        res = subgroup_res.get()
        analyzer = subgroup_analyzer.get()
        if res and analyzer:
            interp = analyzer.get_interpretation()
            is_het = res['interaction']['significant']
            color = "alert-warning" if is_het else "alert-success"
            icon = "⚠️" if is_het else "✅"
            return ui.div(f"{icon} {interp}", class_=f"alert {color}")
        return None

    @render.data_frame
    def out_sg_table():
        res = subgroup_res.get()
        if res:
            df_res = res['results_df'].copy()
            # Simple formatting for display
            cols = ['group', 'n', 'events', 'or', 'ci_low', 'ci_high', 'p_value']
            return render.DataGrid(df_res[cols].round(4))
        return None

    # --- Subgroup Downloads ---
    @render.download(filename=lambda: f"subgroup_plot_{input.sg_subgroup()}.html")
    def dl_sg_html():
        analyzer = subgroup_analyzer.get()
        if analyzer and analyzer.figure:
            yield analyzer.figure.to_html(include_plotlyjs='cdn')

    @render.download(filename=lambda: f"subgroup_res_{input.sg_subgroup()}.csv")
    def dl_sg_csv():
        res = subgroup_res.get()
        if res:
            yield res['results_df'].to_csv(index=False)

    @render.download(filename=lambda: f"subgroup_data_{input.sg_subgroup()}.json")
    def dl_sg_json():
        res = subgroup_res.get()
        if res:
            # Need to handle numpy types for JSON serialization
            yield json.dumps(res, indent=2, default=str)

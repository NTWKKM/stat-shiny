from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from htmltools import HTML, div
import gc

# Import internal modules
from logic import process_data_and_generate_html
from forest_plot_lib import create_forest_plot
from subgroup_analysis_module import SubgroupAnalysisLogit
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()

# ==============================================================================
# Helper Functions (Pure Logic) - OPTIMIZED
# ==============================================================================
def check_perfect_separation(df, target_col):
    """
    Identify columns causing perfect separation.
    Optimized to match logic.py's robust handling.
    """
    risky_vars = []
    
    # Check Target
    if target_col not in df.columns:
        return []
        
    y = pd.to_numeric(df[target_col], errors='coerce').dropna()
    if y.nunique() < 2: 
        return []

    # Check Predictors
    for col in df.columns:
        if col == target_col: continue
        
        # Optimize: Vectorized numeric conversion like in logic.py
        X_num = pd.to_numeric(df[col], errors='coerce')
        
        # Check separation only if variable has variance
        if X_num.nunique() > 1: 
            try:
                # Align X and y indices
                valid_idx = X_num.index.intersection(y.index)
                if len(valid_idx) == 0: continue
                
                tab = pd.crosstab(X_num.loc[valid_idx], y.loc[valid_idx])
                
                # If any cell is 0, it suggests potential separation
                if (tab == 0).any().any():
                    risky_vars.append(col)
            except Exception as e:
                logger.debug(f"Could not check separation for {col}: {e}")
                
    return risky_vars

# ==============================================================================
# UI Definition - Stacked Layout (Controls Top + Content Bottom)
# ==============================================================================
@module.ui
def logit_ui():
    return ui.navset_card_tab(
        # =====================================================================
        # TAB 1: Binary Logistic Regression
        # =====================================================================
        ui.nav_panel(
            "📈 Binary Logistic Regression",
            
            # Control section (top)
            ui.card(
                ui.card_header("📈 Analysis Options"),
                
                ui.output_ui("ui_dataset_selector"),
                ui.hr(),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Variable Selection:"),
                        ui.input_select("sel_outcome", "Select Outcome (Y):", choices=[]),
                        ui.output_ui("ui_separation_warning"),
                    ),
                    
                    ui.card(
                        ui.card_header("Method & Settings:"),
                        ui.input_radio_buttons(
                            "radio_method",
                            "Regression Method:",
                            {
                                "auto": "Auto (Recommended)",
                                "bfgs": "Standard (MLE)",
                                "firth": "Firth's (Penalized)"
                            }
                        ),
                    ),
                    
                    col_widths=[6, 6]
                ),
                
                ui.h6("Exclude Variables (Optional):"),
                ui.input_selectize("sel_exclude", label=None, choices=[], multiple=True),
                
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
            
            # Content section (bottom)
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
        # TAB 2: Subgroup Analysis
        # =====================================================================
        ui.nav_panel(
            "🗣️ Subgroup Analysis",
            
            # Control section (top)
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
            
            # Content section (bottom)
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
## 📚 Logistic Regression Reference

### When to Use:
* Predicting binary outcomes (Disease/No Disease)
* Understanding risk/protective factors (Odds Ratios)
* Adjustment for confounders in observational studies

### Interpretation:

**Odds Ratios (OR):**
* **OR > 1**: Risk Factor (Increased odds) 🔴
* **OR < 1**: Protective Factor (Decreased odds) 🟢
* **OR = 1**: No Effect
* **CI crosses 1**: Not statistically significant

**Example:**
* OR = 2.5 (CI 1.2-5.0): Exposure increases odds of outcome by 2.5× (Range: 1.2× to 5×)

### Regression Methods:

**Standard (MLE)** - Most common
* Uses Maximum Likelihood Estimation
* Fast and reliable for most datasets
* Issues: Perfect separation causes failure

**Firth's (Penalized)** - For separation issues
* Reduces bias using penalized likelihood
* Better for rare outcomes or small samples
* Handles perfect separation well

**Auto** - Recommended
* Automatically detects separation
* Uses Firth if needed, Standard otherwise

### VIF (Multicollinearity):
The report now automatically checks for Variance Inflation Factor.
* **VIF > 10**: Indicates high multicollinearity (variables are too similar).
* **Impact**: Coefficients become unstable and standard errors inflate.
* **Solution**: Remove one of the correlated variables.

### Subgroup Analysis:
* Tests if treatment effect varies by group (Interaction test)
* **P-interaction < 0.05**: Significant heterogeneity → Report subgroups separately
* **P-interaction ≥ 0.05**: Homogeneous effect → Report overall effect
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
            original = df.get()
            matched = df_matched.get()
            original_len = len(original) if original is not None else 0
            matched_len = len(matched) if matched is not None else 0
            return ui.input_radio_buttons(
                "radio_dataset_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original ({original_len})",
                    "matched": f"✅ Matched ({matched_len})"
                },
                selected="matched",
                inline=True
            )
        # Fallback for non-matched data
        d = df.get()
        row_count = len(d) if d is not None else 0
        return ui.p(f"📊 Using Original Data ({row_count} rows)", class_="text-muted")

    # --- 2. Dynamic Input Updates ---
    @reactive.Effect
    def _update_inputs():
        d = current_df()
        if d is None or d.empty: return
        
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
        if d is None or d.empty or not target: return None
        
        risky = check_perfect_separation(d, target)
        if risky:
            return ui.div(
                ui.h6("⚠️ Perfect Separation Risk", class_="text-warning"),
                ui.p(f"Variables: {', '.join(risky)}"),
                ui.p("Recommendation: Use 'Auto' method or exclude variables.", style="font-size: 0.8em;")
            )
        return None

    # ==========================================================================
    # OPTIMIZED: Cache Logistic Regression Results & Forest Plots
    # ==========================================================================
    @reactive.Calc
    def logit_cached_res():
        """
        OPTIMIZATION: Cache the logistic regression results.
        This prevents multiple calls to logit_res.get() from triggering re-renders.
        Only triggers when button is clicked.
        """
        return logit_res.get()

    @reactive.Effect
    @reactive.event(input.btn_run_logit)
    def _run_logit():
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude()
        method = input.radio_method()
        
        # FIX: Check if DataFrame is None or empty using proper methods
        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select an outcome variable", type="error")
            return
        
        # Prepare data
        final_df = d.drop(columns=exclude, errors='ignore')
        
        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Logistic Regression...", detail="Calculating...")
            
            try:
                # Run Logic from logic.py
                html_rep, or_res, aor_res = process_data_and_generate_html(
                    final_df, target, var_meta=var_meta.get(), method=method
                )
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Logistic regression error")
                return
            
            # Generate Forest Plots using library
            fig_adj = None
            fig_crude = None
            
            try:
                if aor_res:
                    df_adj = pd.DataFrame([{'variable': k, **v} for k, v in aor_res.items()])
                    if not df_adj.empty:
                        fig_adj = create_forest_plot(
                            df_adj, 'aor', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                            title="<b>Multivariable: Adjusted OR</b>", x_label="Adjusted OR"
                        )
                        # Immediate cleanup after creating figure
                        del df_adj
                        gc.collect()
                
                if or_res:
                    df_crude = pd.DataFrame([{'variable': k, **v} for k, v in or_res.items()])
                    if not df_crude.empty:
                        fig_crude = create_forest_plot(
                            df_crude, 'or', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                            title="<b>Univariable: Crude OR</b>", x_label="Crude OR"
                        )
                        # Immediate cleanup after creating figure
                        del df_crude
            
            except Exception as e:
                logger.exception("Forest plot generation error")
                ui.notification_show(f"Forest plot error: {e!s}", type="warning")

            # Store Results ONCE
            logit_res.set({
                "html": html_rep,
                "fig_adj": fig_adj,
                "fig_crude": fig_crude
            })
            
            # Cleanup after successful analysis
            del html_rep, or_res, aor_res, fig_adj, fig_crude
            gc.collect()
            
            ui.notification_show("✅ Analysis Complete!", type="message")

    # --- Render Main Results (Use Cached Results) ---
    @render.ui
    def out_logit_status():
        res = logit_cached_res()
        if res:
            return ui.div(
                ui.h5("✅ Regression Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;"
            )
        return None

    @render.ui
    def out_html_report():
        res = logit_cached_res()
        if res:
            return ui.card(
                ui.card_header("📋 Detailed Report"),
                ui.HTML(res['html'])
            )
        return ui.card(
            ui.card_header("📋 Detailed Report"),
            ui.div(
                "Run analysis to see detailed report.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;"
            )
        )

    @render.ui
    def ui_forest_tabs():
        res = logit_cached_res()
        if not res: 
            return ui.div(
                "Run analysis to see forest plots.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;"
            )
        
        tabs = []
        if res['fig_crude']:
            tabs.append(ui.nav_panel("Crude OR", output_widget("out_forest_crude")))
        if res['fig_adj']:
            tabs.append(ui.nav_panel("Adjusted OR", output_widget("out_forest_adj")))
            
        if not tabs: 
            return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render_widget
    def out_forest_adj():
        res = logit_cached_res()
        if res and res['fig_adj']: return res['fig_adj']
        return None

    @render_widget
    def out_forest_crude():
        res = logit_cached_res()
        if res and res['fig_crude']: return res['fig_crude']
        return None

    @render.download(filename="logit_report.html")
    def btn_dl_report():
        res = logit_cached_res()
        if res: yield res['html']

    # ==========================================================================
    # OPTIMIZED: Cache Subgroup Analysis Results
    # ==========================================================================
    @reactive.Calc
    def subgroup_cached_res():
        """
        OPTIMIZATION: Cache the subgroup analysis results.
        Prevents redundant rendering calls.
        """
        return subgroup_res.get()

    @reactive.Calc
    def subgroup_cached_analyzer():
        """
        OPTIMIZATION: Cache the analyzer instance.
        """
        return subgroup_analyzer.get()

    @reactive.Effect
    @reactive.event(input.btn_run_subgroup)
    def _run_subgroup():
        d = current_df()
        
        # FIX: Check if DataFrame is None or empty using proper methods
        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not input.sg_outcome() or not input.sg_treatment() or not input.sg_subgroup():
            ui.notification_show("Please fill all required fields", type="error")
            return
        
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
                ui.notification_show("✅ Subgroup Analysis Complete!", type="message")
                
                # Cleanup after successful analysis
                gc.collect()
                
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Subgroup analysis error")

    # --- Render Subgroup Results (Use Cached Results) ---
    @render.ui
    def out_subgroup_status():
        res = subgroup_cached_res()
        if res:
            return ui.div(
                ui.h5("✅ Subgroup Analysis Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;"
            )
        return None

    @render_widget
    def out_sg_forest_plot():
        analyzer = subgroup_cached_analyzer()
        if analyzer:
            # Use txt_edit_forest_title if provided, fallback to sg_title
            title = input.txt_edit_forest_title() or input.sg_title() or None
            return analyzer.create_forest_plot(title=title)
        return None

    @reactive.Effect
    @reactive.event(input.btn_update_plot_title)
    def _update_sg_title():
        # Invalidate to trigger re-render of the forest plot widget
        subgroup_analyzer.set(subgroup_analyzer.get())

    @render.text
    def val_overall_or():
        res = subgroup_cached_res()
        if res:
            overall = res.get('overall', {})
            or_val = overall.get('or')
            return f"{or_val:.3f}" if or_val is not None else "N/A"
        return "-"

    @render.text
    def val_overall_p():
        res = subgroup_cached_res()
        if res: return f"{res['overall']['p_value']:.4f}"
        return "-"
    
    @render.text
    def val_interaction_p():
        res = subgroup_cached_res()
        if res:
             p_int = res['interaction']['p_value']
             return f"{p_int:.4f}" if p_int is not None else "N/A"
        return "-"

    @render.ui
    def out_interpretation_box():
        res = subgroup_cached_res()
        analyzer = subgroup_cached_analyzer()
        if res and analyzer:
            interp = analyzer.get_interpretation()
            is_het = res['interaction']['significant']
            color = "alert-warning" if is_het else "alert-success"
            icon = "⚠️" if is_het else "✅"
            return ui.div(f"{icon} {interp}", class_=f"alert {color}")
        return None

    @render.data_frame
    def out_sg_table():
        res = subgroup_cached_res()
        if res:
            df_res = res['results_df'].copy()
            # Simple formatting for display
            cols = ['group', 'n', 'events', 'or', 'ci_low', 'ci_high', 'p_value']
            available_cols = [c for c in cols if c in df_res.columns]
            return render.DataGrid(df_res[available_cols].round(4))
        return None

    # --- Subgroup Downloads ---
    @render.download(filename=lambda: f"subgroup_plot_{input.sg_subgroup()}.html")
    def dl_sg_html():
        analyzer = subgroup_cached_analyzer()
        if analyzer and analyzer.figure:
            yield analyzer.figure.to_html(include_plotlyjs='cdn')

    @render.download(filename=lambda: f"subgroup_res_{input.sg_subgroup()}.csv")
    def dl_sg_csv():
        res = subgroup_cached_res()
        if res:
            yield res['results_df'].to_csv(index=False)

    @render.download(filename=lambda: f"subgroup_data_{input.sg_subgroup()}.json")
    def dl_sg_json():
        res = subgroup_cached_res()
        if res:
            # Need to handle numpy types for JSON serialization
            yield json.dumps(res, indent=2, default=str)

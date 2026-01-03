from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from htmltools import HTML, div
import gc

# Import internal modules
from logic import process_data_and_generate_html, analyze_outcome
from poisson_lib import analyze_poisson_outcome, check_count_outcome
from interaction_lib import test_interaction_significance, interpret_interaction
from forest_plot_lib import create_forest_plot
from subgroup_analysis_module import SubgroupAnalysisLogit
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()

# ==============================================================================
# Helper Functions (Pure Logic)
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

        if df[col].nunique() < 10: 
            try:
                tab = pd.crosstab(df[col], y)
                if (tab == 0).any().any():
                    risky_vars.append(col)
            except (ValueError, TypeError): 
                pass
    return risky_vars

# ==============================================================================
# UI Definition - Enhanced with Poisson & Interactions
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
                
                # ✅ Interaction Terms Selection
                ui.accordion(
                    ui.accordion_panel(
                        "🔗 Interaction Terms (Advanced)",
                        ui.p("Test if the effect of one variable depends on another:", 
                             style="font-size: 0.9em; color: #666;"),
                        ui.input_selectize(
                            "sel_interactions",
                            "Select Variable Pairs for Interactions:",
                            choices=[],
                            multiple=True
                        ),
                        ui.p("Format: var1 × var2", style="font-size: 0.8em; font-style: italic; color: #999;")
                    ),
                    open=False
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
        # TAB 2: Poisson Regression
        # =====================================================================
        ui.nav_panel(
            "📊 Poisson Regression",
            
            ui.card(
                ui.card_header("📊 Count Data Analysis"),
                
                ui.output_ui("ui_dataset_selector_poisson"),
                ui.hr(),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Variable Selection:"),
                        ui.input_select("poisson_outcome", "Select Count Outcome:", choices=[]),
                        ui.input_select("poisson_offset", "Exposure Offset (Optional):", choices=["None"]),
                        ui.output_ui("ui_count_validation"),
                    ),
                    
                    ui.card(
                        ui.card_header("Settings:"),
                        ui.p("Poisson regression for count/rate data", 
                             style="font-size: 0.9em; color: #666; margin-top: 10px;"),
                        ui.p("✓ Hospital visits<br>✓ Event counts<br>✓ Disease incidence", 
                             style="font-size: 0.85em; color: #999;")
                    ),
                    
                    col_widths=[6, 6]
                ),
                
                ui.h6("Exclude Variables (Optional):"),
                ui.input_selectize("poisson_exclude", label=None, choices=[], multiple=True),
                
                # ✅ Interaction Terms for Poisson
                ui.accordion(
                    ui.accordion_panel(
                        "🔗 Interaction Terms (Advanced)",
                        ui.p("Test if the effect of one variable depends on another:", 
                             style="font-size: 0.9em; color: #666;"),
                        ui.input_selectize(
                            "poisson_interactions",
                            "Select Variable Pairs for Interactions:",
                            choices=[],
                            multiple=True
                        ),
                        ui.p("Format: var1 × var2", style="font-size: 0.8em; font-style: italic; color: #999;")
                    ),
                    open=False
                ),
                
                ui.hr(),
                
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run_poisson",
                        "🚀 Run Poisson Regression",
                        class_="btn-primary btn-sm w-100"
                    ),
                    ui.download_button(
                        "btn_dl_poisson_report",
                        "📥 Download Report",
                        class_="btn-secondary btn-sm w-100"
                    ),
                    col_widths=[6, 6]
                ),
            ),
            
            # Results
            ui.output_ui("out_poisson_status"),
            ui.navset_card_underline(
                ui.nav_panel(
                    "🌳 Forest Plots",
                    ui.output_ui("ui_poisson_forest_tabs")
                ),
                ui.nav_panel(
                    "📋 Detailed Report",
                    ui.output_ui("out_poisson_html_report")
                )
            )
        ),

        # =====================================================================
        # TAB 3: Subgroup Analysis
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
                        ui.download_button("dl_sg_json", "📁 JSON Data", class_="btn-sm w-100"),
                        col_widths=[4, 4, 4]
                    )
                )
            )
        ),

        # =====================================================================
        # TAB 4: Reference (Updated with Poisson & Interaction info)
        # =====================================================================
        ui.nav_panel(
            "ℹ️ Reference",
            ui.markdown("""
## 📚 Regression Models Reference

### 📈 Binary Logistic Regression
**When to Use:**
* Binary outcomes (Disease/No Disease, Success/Failure)
* Returns **Odds Ratios (OR)**

**Interpretation:**
* **OR > 1**: Risk Factor (Increased odds) 🔴
* **OR < 1**: Protective Factor (Decreased odds) 🟢
* **OR = 1**: No Effect
* **CI crosses 1**: Not statistically significant

**Methods:**
* **Auto** - Recommended (detects separation automatically)
* **Standard (MLE)** - Fast, reliable for most datasets
* **Firth's** - For rare outcomes or perfect separation

---

### 📊 Poisson Regression
**When to Use:**
* Count outcomes (# of events, hospital visits)
* Rate data (events per person-year)
* Returns **Incidence Rate Ratios (IRR)**

**Interpretation:**
* **IRR > 1**: Increased rate of events 🔴
* **IRR < 1**: Decreased rate of events 🟢
* **IRR = 1**: No effect on rate

**Example:**
* IRR = 2.5 → Exposed group has 2.5× the rate of events

**Note:** If variance >> mean (overdispersion), consider Negative Binomial model.

---

### 🔗 Interaction Terms
**Purpose:** Test if the effect of Variable A depends on Variable B

**Interpretation:**
* **P-interaction < 0.05**: Effect differs across groups → Report separately
* **P-interaction ≥ 0.05**: Consistent effect → Report overall

**Example:**
* Drug × Age interaction: Drug effectiveness may vary by age group

**Caution:** Interactions complicate interpretation—use only when justified.

**Supported in:**
* ✅ Binary Logistic Regression (OR)
* ✅ Poisson Regression (IRR)

---

### Perfect Separation (Logistic Only)
Occurs when a predictor perfectly predicts outcome.

**Solution:** Use **Auto** or **Firth's** method, or exclude the variable.
            """)
        )
    )

# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def logit_server(input, output, session, df, var_meta, df_matched, is_matched):

    # --- State Management ---
    logit_res = reactive.Value(None)
    poisson_res = reactive.Value(None)
    subgroup_res = reactive.Value(None)
    subgroup_analyzer = reactive.Value(None)

    # --- Cache Clearing ---
    @reactive.Effect
    @reactive.event(input.btn_run_logit, input.btn_run_poisson, input.btn_run_subgroup)
    def _cleanup_after_analysis():
        try:
            gc.collect()
            logger.debug("Post-analysis cache cleared")
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
    
    # --- Dataset Selection ---
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
        d = df.get()
        row_count = len(d) if d is not None else 0
        return ui.p(f"📊 Using Original Data ({row_count} rows)", class_="text-muted")

    @render.ui
    def ui_dataset_selector_poisson():
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
        d = df.get()
        row_count = len(d) if d is not None else 0
        return ui.p(f"📊 Using Original Data ({row_count} rows)", class_="text-muted")

    # --- Dynamic Input Updates ---
    @reactive.Effect
    def _update_inputs():
        d = current_df()
        if d is None or d.empty: return

        cols = d.columns.tolist()
        binary_cols = [c for c in cols if d[c].nunique() == 2]
        sg_cols = [c for c in cols if 2 <= d[c].nunique() <= 10]
        
        # Count columns for Poisson
        count_cols = []
        for c in cols:
            try:
                numeric_vals = pd.to_numeric(d[c], errors='coerce').dropna()
                if len(numeric_vals) > 0 and (numeric_vals >= 0).all():
                    count_cols.append(c)
            except Exception:
                pass

        # Update Logistic Inputs
        ui.update_select("sel_outcome", choices=binary_cols)
        ui.update_selectize("sel_exclude", choices=cols)
        
        # Interaction pairs for logistic
        interaction_choices = {f"{v1} × {v2}": f"{v1} × {v2}" 
                              for i, v1 in enumerate(cols) 
                              for v2 in cols[i+1:] if v1 != v2}
        ui.update_selectize("sel_interactions", choices=interaction_choices)

        # Update Poisson Inputs
        ui.update_select("poisson_outcome", choices=count_cols)
        ui.update_select("poisson_offset", choices=["None"] + cols)
        ui.update_selectize("poisson_exclude", choices=cols)
        ui.update_selectize("poisson_interactions", choices=interaction_choices)

        # Update Subgroup Inputs
        ui.update_select("sg_outcome", choices=binary_cols)
        ui.update_select("sg_treatment", choices=cols)
        ui.update_select("sg_subgroup", choices=sg_cols)
        ui.update_selectize("sg_adjust", choices=cols)

    # --- Separation Warning ---
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

    # Count validation for Poisson
    @render.ui
    def ui_count_validation():
        d = current_df()
        outcome = input.poisson_outcome()
        if d is None or d.empty or not outcome: return None
        
        is_valid, msg = check_count_outcome(d[outcome])
        if not is_valid:
            return ui.div(
                ui.h6("⚠️ Invalid Count Data", class_="text-danger"),
                ui.p(msg, style="font-size: 0.9em;")
            )
        elif "overdispersion" in msg.lower():
            return ui.div(
                ui.h6("⚠️ Warning", class_="text-warning"),
                ui.p(msg, style="font-size: 0.9em;")
            )
        return None

    # ==========================================================================
    # ✅ LOGIC: Binary Logistic Regression (WITH INTERACTIONS - BACKWARD COMPATIBLE)
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_logit)
    def _run_logit():
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude()
        method = input.radio_method()
        
        # ✅ Parse interaction pairs
        interaction_pairs = []
        if input.sel_interactions():
            for pair_str in input.sel_interactions():
                parts = pair_str.split(' × ')
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"✅ Logistic: {len(interaction_pairs)} interaction pairs selected")

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select an outcome variable", type="error")
            return

        final_df = d.drop(columns=exclude, errors='ignore')

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Logistic Regression...", detail="Calculating...")

            try:
                # ✅ Try to get result (could be 3 or 4 values)
                result = process_data_and_generate_html(
                    final_df, target, var_meta=var_meta.get(), 
                    method=method, interaction_pairs=interaction_pairs or None
                )
                
                # ✅ Handle both 3-value and 4-value returns (backward compatibility)
                if len(result) == 4:
                    html_rep, or_res, aor_res, int_res = result
                    if interaction_pairs:
                        logger.info(f"✅ Logistic regression completed with {len(int_res)} interactions")
                elif len(result) == 3:
                    html_rep, or_res, aor_res = result
                    int_res = {}
                    logger.info("⚠️ Using legacy 3-value return (no interaction support)")
                else:
                    raise ValueError(f"Unexpected return value count: {len(result)}")
                
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Logistic regression error")
                return

            fig_adj = None
            fig_crude = None

            if aor_res:
                df_adj = pd.DataFrame([{'variable': k, **v} for k, v in aor_res.items()])
                if not df_adj.empty:
                    fig_adj = create_forest_plot(
                        df_adj, 'aor', 'ci_low', 'ci_high', 'variable', 
                        pval_col='p_value',
                        title="Multivariable: Adjusted OR",  # ✅ FIX: Remove HTML tags
                        x_label="Adjusted OR"
                    )

            if or_res:
                df_crude = pd.DataFrame([{'variable': k, **v} for k, v in or_res.items()])
                if not df_crude.empty:
                    fig_crude = create_forest_plot(
                        df_crude, 'or', 'ci_low', 'ci_high', 'variable', 
                        pval_col='p_value',
                        title="Univariable: Crude OR",  # ✅ FIX: Remove HTML tags
                        x_label="Crude OR"
                    )

            logit_res.set({
                "html": html_rep,
                "fig_adj": fig_adj,
                "fig_crude": fig_crude,
                "interaction_results": int_res  # ✅ Store interaction results
            })

            ui.notification_show("✅ Analysis Complete!", type="message")

    # --- Render Logistic Results ---
    @render.ui
    def out_logit_status():
        res = logit_res.get()
        if res:
            int_count = len(res.get('interaction_results', {}))
            int_text = f" ({int_count} interactions)" if int_count > 0 else ""
            return ui.div(
                ui.h5(f"✅ Regression Complete{int_text}"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;"
            )
        return None

    @render.ui
    def out_html_report():
        res = logit_res.get()
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
        res = logit_res.get()
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
    # ✅ LOGIC: Poisson Regression (WITH INTERACTIONS - BACKWARD COMPATIBLE)
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_poisson)
    def _run_poisson():
        d = current_df()
        target = input.poisson_outcome()
        exclude = input.poisson_exclude()
        offset_col = input.poisson_offset() if input.poisson_offset() != "None" else None
        
        # ✅ Parse interaction pairs
        interaction_pairs = []
        if input.poisson_interactions():
            for pair_str in input.poisson_interactions():
                parts = pair_str.split(' × ')
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"✅ Poisson: {len(interaction_pairs)} interaction pairs selected")

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select an outcome variable", type="error")
            return

        final_df = d.drop(columns=exclude, errors='ignore')

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Poisson Regression...", detail="Calculating...")

            try:
                # ✅ Try to get result (could be 3 or 4 values)
                result = analyze_poisson_outcome(
                    target, final_df, var_meta=var_meta.get(), 
                    offset_col=offset_col, interaction_pairs=interaction_pairs or None
                )
                
                # ✅ Handle both 3-value and 4-value returns (backward compatibility)
                if len(result) == 4:
                    html_rep, irr_res, airr_res, int_res = result
                    if interaction_pairs:
                        logger.info(f"✅ Poisson regression completed with {len(int_res)} interactions")
                elif len(result) == 3:
                    html_rep, irr_res, airr_res = result
                    int_res = {}
                    logger.info("⚠️ Using legacy 3-value return (no interaction support)")
                else:
                    raise ValueError(f"Unexpected return value count: {len(result)}")
                
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Poisson regression error")
                return

            fig_adj = None
            fig_crude = None

            if airr_res:
                df_adj = pd.DataFrame([{'variable': k, **v} for k, v in airr_res.items()])
                if not df_adj.empty:
                    fig_adj = create_forest_plot(
                        df_adj, 'airr', 'ci_low', 'ci_high', 'variable', 
                        pval_col='p_value',
                        title="Multivariable: Adjusted IRR",  # ✅ FIX: Remove HTML tags
                        x_label="Adjusted IRR"
                    )

            if irr_res:
                df_crude = pd.DataFrame([{'variable': k, **v} for k, v in irr_res.items()])
                if not df_crude.empty:
                    fig_crude = create_forest_plot(
                        df_crude, 'irr', 'ci_low', 'ci_high', 'variable', 
                        pval_col='p_value',
                        title="Univariable: Crude IRR",  # ✅ FIX: Remove HTML tags
                        x_label="Crude IRR"
                    )

            poisson_res.set({
                "html": html_rep,
                "fig_adj": fig_adj,
                "fig_crude": fig_crude,
                "interaction_results": int_res  # ✅ Store interaction results
            })

            ui.notification_show("✅ Poisson Analysis Complete!", type="message")

    # --- Render Poisson Results ---
    @render.ui
    def out_poisson_status():
        res = poisson_res.get()
        if res:
            int_count = len(res.get('interaction_results', {}))
            int_text = f" ({int_count} interactions)" if int_count > 0 else ""
            return ui.div(
                ui.h5(f"✅ Poisson Regression Complete{int_text}"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;"
            )
        return None

    @render.ui
    def out_poisson_html_report():
        res = poisson_res.get()
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
    def ui_poisson_forest_tabs():
        res = poisson_res.get()
        if not res: 
            return ui.div(
                "Run analysis to see forest plots.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;"
            )

        tabs = []
        if res['fig_crude']:
            tabs.append(ui.nav_panel("Crude IRR", output_widget("out_poisson_forest_crude")))
        if res['fig_adj']:
            tabs.append(ui.nav_panel("Adjusted IRR", output_widget("out_poisson_forest_adj")))

        if not tabs: 
            return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render_widget
    def out_poisson_forest_adj():
        res = poisson_res.get()
        if res and res['fig_adj']: return res['fig_adj']
        return None

    @render_widget
    def out_poisson_forest_crude():
        res = poisson_res.get()
        if res and res['fig_crude']: return res['fig_crude']
        return None

    @render.download(filename="poisson_report.html")
    def btn_dl_poisson_report():
        res = poisson_res.get()
        if res: yield res['html']

    # ==========================================================================
    # LOGIC: Subgroup Analysis (Unchanged)
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_subgroup)
    def _run_subgroup():
        d = current_df()
        
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
                
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Subgroup analysis error")

    @render.ui
    def out_subgroup_status():
        res = subgroup_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Subgroup Analysis Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;"
            )
        return None

    @render_widget
    def out_sg_forest_plot():
        analyzer = subgroup_analyzer.get()
        if analyzer:
            title = input.txt_edit_forest_title() or input.sg_title() or None
            return analyzer.create_forest_plot(title=title)
        return None

    @reactive.Effect
    @reactive.event(input.btn_update_plot_title)
    def _update_sg_title():
        subgroup_analyzer.set(subgroup_analyzer.get())

    @render.text
    def val_overall_or():
        res = subgroup_res.get()
        if res:
            overall = res.get('overall', {})
            or_val = overall.get('or')
            return f"{or_val:.3f}" if or_val is not None else "N/A"
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
        if res:
            yield res['results_df'].to_csv(index=False)

    @render.download(filename=lambda: f"subgroup_data_{input.sg_subgroup()}.json")
    def dl_sg_json():
        res = subgroup_res.get()
        if res:
            yield json.dumps(res, indent=2, default=str)

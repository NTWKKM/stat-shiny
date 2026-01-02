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

def _find_smart_outcome(outcome_cols):
    """Find best outcome column using priorities: Example names > Keywords > First available."""
    if not outcome_cols:
        return None
        
    # 1. Example Data specific names
    example_targets = ["Outcome_Cured", "Status_Death", "Target", "Class"]
    for target in example_targets:
        if target in outcome_cols:
            return target
            
    # 2. Fuzzy scan for common keywords
    outcome_keywords = ['outcome', 'status', 'event', 'target', 'result', 'died', 'survived', 'death']
    for col in outcome_cols:
        if any(k in col.lower() for k in outcome_keywords):
            return col
            
    # 3. Default
    return outcome_cols[0]

def _find_smart_treatment(binary_cols):
    """Find best treatment column using priorities: Example names > Keywords."""
    if not binary_cols:
        return None
        
    treat_keywords = ['treat', 'group', 'drug', 'arm', 'interv', 'exposure']
    
    # 1. Example Data Priority
    if "Treatment_Group" in binary_cols:
        return "Treatment_Group"
    
    # 2. Keyword scan
    for col in binary_cols:
        if any(k in col.lower() for k in treat_keywords):
            return col
            
    return None

def _find_smart_subgroup(sg_cols, selected_outcome, selected_treatment):
    """Find best subgroup column, avoiding outcome/treatment vars."""
    if not sg_cols:
        return None
        
    # 1. Filter candidates
    candidates = [c for c in sg_cols if c != selected_outcome and c != selected_treatment]
    if not candidates:
        return None
        
    # 2. Prefer common subgroups
    sg_keywords = ['sex', 'gender', 'age_group', 'bmi_group', 'stage', 'grade']
    for col in candidates:
        if any(k in col.lower() for k in sg_keywords):
            return col
            
    # 3. Default
    return candidates[0]

# ==============================================================================
# UI Definition - Stacked Layout (Controls Top + Content Bottom)
# ==============================================================================
@module.ui
def logit_ui():
    return ui.navset_card_tab(
        # =====================================================================
        # TAB 1: Regression Analysis (Renamed from Binary Logit)
        # =====================================================================
        ui.nav_panel(
            "📈 Regression Analysis",
            
            # Control section (top)
            ui.card(
                ui.card_header("📈 Analysis Options"),
                
                ui.output_ui("ui_dataset_selector"),
                ui.hr(),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Variable Selection:"),
                        ui.input_select("sel_model_type", "Model Type:", 
                                        {"logistic": "Logistic (Binary Outcome)", "poisson": "Poisson (Count/Rate)"}),
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
                                "firth": "Firth's (Penalized)" # Only valid for Logit
                            }
                        ),
                    ),
                    
                    col_widths=[6, 6]
                ),
                
                ui.accordion(
                    ui.accordion_panel(
                        "🔗 Interaction Terms (Advanced)",
                        ui.layout_columns(
                            ui.input_select("int_var1", "Variable 1:", choices=[], selected=None),
                            ui.input_select("int_var2", "Variable 2:", choices=[], selected=None),
                            col_widths=[6, 6]
                        ),
                        ui.input_action_button("btn_add_interaction", "➕ Add Interaction Term", class_="btn-sm"),
                        ui.p("Active Interactions:", class_="mt-2 mb-1 fw-bold"),
                        ui.output_ui("ui_interaction_list"),
                        open=False
                    )
                ),
                
                ui.h6("Exclude Variables (Optional):", class_="mt-3"),
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
        # TAB 2: Subgroup Analysis (Kept as is)
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
## 📚 Regression Analysis Reference

### Model Types:

**1. Logistic Regression (Binary)**
* **Target:** Yes/No, Dead/Alive (0/1)
* **Output:** Odds Ratios (OR)
* **Interpretation:** OR > 1 means increased odds of outcome.

**2. Poisson Regression (Count)**
* **Target:** Counts (e.g., Days in hospital, Number of events)
* **Output:** Incidence Rate Ratios (IRR)
* **Interpretation:** IRR > 1 means increased rate of events.

### Interaction Terms:
* Used to test if the effect of one variable depends on another (e.g., Does treatment work differently in older people?).
* **Caution:** Interactions can make models harder to interpret.

### Regression Methods:

**Standard (MLE)**
* Standard Maximum Likelihood. Works for both Logistic and Poisson.

**Firth's (Penalized)**
* Only for **Logistic**. Reduces bias in small samples or separation.

**Auto**
* Automatically selects Firth if separation is detected (Logistic only), otherwise uses Standard.
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
    interaction_list = reactive.Value([]) # Store active interaction terms
    
    # Track state for dynamic removal buttons
    last_remove_clicks = reactive.Value({})

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

    # --- 2. Dynamic Input Updates (Optimized with Smart Selection) ---
    
    # Notify user when model type changes
    @reactive.Effect
    @reactive.event(input.sel_model_type, ignore_init=True)
    def _notify_on_model_type_change():
        m_type = input.sel_model_type()
        ui.notification_show(
            f"Model type changed to {m_type.capitalize()}. Please verify outcome selection.",
            type="warning",
            duration=5
        )

    @reactive.Effect
    def _update_inputs():
        d = current_df()
        if d is None or d.empty: return
        
        cols = d.columns.tolist()
        
        # Identify columns based on Model Type
        model_type = input.sel_model_type()
        
        outcome_cols = []
        if model_type == 'logistic':
             # Binary for Logistic
             outcome_cols = [c for c in cols if d[c].nunique() == 2]
        else:
             # Numeric/Integer for Poisson
             outcome_cols = [c for c in cols if pd.api.types.is_numeric_dtype(d[c])]

        # Identify potential subgroups (2-10 levels)
        sg_cols = [c for c in cols if 2 <= d[c].nunique() <= 10]
        
        # --- SMART SELECTION LOGIC ---
        
        # 1. Smart Outcome Selection
        selected_outcome = _find_smart_outcome(outcome_cols)

        # 2. Smart Treatment Selection (For Subgroup)
        binary_cols = [c for c in cols if d[c].nunique() == 2]
        selected_treatment = _find_smart_treatment(binary_cols)
        
        # 3. Smart Subgroup Selection (For Subgroup)
        selected_sg = _find_smart_subgroup(sg_cols, selected_outcome, selected_treatment)


        # --- UPDATE UI ---
        
        # Update Tab 1 Inputs
        ui.update_select("sel_outcome", choices=outcome_cols, selected=selected_outcome)
        ui.update_selectize("sel_exclude", choices=cols)
        
        # Update Interaction Inputs
        ui.update_select("int_var1", choices=[""] + cols)
        ui.update_select("int_var2", choices=[""] + cols)
        
        # Update Tab 2 Inputs (Subgroup)
        ui.update_select("sg_outcome", choices=binary_cols, selected=selected_outcome if selected_outcome in binary_cols else None)
        ui.update_select("sg_treatment", choices=cols, selected=selected_treatment)
        ui.update_select("sg_subgroup", choices=sg_cols, selected=selected_sg)
        ui.update_selectize("sg_adjust", choices=cols)

    # --- Interaction Logic ---
    @reactive.Effect
    @reactive.event(input.btn_add_interaction)
    def _add_interaction():
        v1 = input.int_var1()
        v2 = input.int_var2()
        if v1 and v2 and v1 != v2:
            current = interaction_list.get()
            term = f"{v1}:{v2}"
            if term not in current:
                updated = current + [term]
                interaction_list.set(updated)
                ui.notification_show(f"Added interaction: {term}", type="message")
            else:
                ui.notification_show("Interaction already exists", type="warning")

    @render.ui
    def ui_interaction_list():
        terms = interaction_list.get()
        if not terms:
            return ui.span("No interactions added.", style="color: gray; font-size: 0.9em;")
        
        items = []
        for i, term in enumerate(terms):
            # Create a removal button for each term
            btn_id = f"btn_remove_int_{i}"
            items.append(
                ui.div(
                    ui.span(term, class_="badge bg-info text-dark me-2"),
                    ui.input_action_button(
                        btn_id, "✕", 
                        class_="btn-link text-danger p-0", 
                        style="text-decoration: none; font-weight: bold; font-size: 1.1em; line-height: 1;"
                    ),
                    class_="d-inline-flex align-items-center me-2 mb-2 px-2 py-1 border rounded bg-light"
                )
            )
        
        return ui.div(
            ui.div(*items, class_="d-flex flex-wrap"),
            ui.div(
                ui.input_action_button("btn_clear_interactions", "Clear All", class_="btn-xs btn-outline-danger mt-2"),
            )
        )
        
    @reactive.Effect
    def _handle_interaction_removal():
        """
        Dynamically handle removal of interaction terms.
        This polls the buttons associated with current interactions.
        """
        terms = interaction_list.get()
        # Create dependency on potential buttons
        current_clicks = {}
        triggered_idx = None
        
        # Scan inputs for all potential buttons in the current list
        for i in range(len(terms)):
            btn_id = f"btn_remove_int_{i}"
            try:
                # Check if input exists and get its value
                if hasattr(input, btn_id):
                    val = getattr(input, btn_id)()
                    current_clicks[btn_id] = val
            except Exception as e:
                # Log failure but continue loop so one bad button doesn't break everything
                logger.debug(f"Failed to read interaction removal button {btn_id}: {e}")
        
        # Compare with previous state to detect clicks
        prev_clicks = last_remove_clicks.get()
        for btn_id, val in current_clicks.items():
            if val > prev_clicks.get(btn_id, 0):
                try:
                    # Extract index from button ID
                    idx = int(btn_id.split('_')[-1])
                    triggered_idx = idx
                except ValueError:
                    pass
                break
        
        # Update state
        last_remove_clicks.set(current_clicks)
        
        # Apply removal if triggered
        if triggered_idx is not None and 0 <= triggered_idx < len(terms):
            removed_term = terms[triggered_idx]
            new_terms = list(terms)
            new_terms.pop(triggered_idx)
            interaction_list.set(new_terms)
            ui.notification_show(f"Removed interaction: {removed_term}", type="message")

    @reactive.Effect
    @reactive.event(input.btn_clear_interactions)
    def _clear_interactions():
        interaction_list.set([])


    # --- 3. Separation Warning Logic ---
    @render.ui
    def ui_separation_warning():
        # Only relevant for Logistic
        if input.sel_model_type() != 'logistic':
            return None
            
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
    # OPTIMIZED: Cache Regression Results & Forest Plots
    # ==========================================================================
    @reactive.Calc
    def logit_cached_res():
        return logit_res.get()

    @reactive.Effect
    @reactive.event(input.btn_run_logit)
    def _run_logit():
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude()
        method = input.radio_method()
        model_type = input.sel_model_type()
        interactions = interaction_list.get()
        
        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select an outcome variable", type="error")
            return
        
        # Prepare data
        # Note: We do NOT drop interaction vars from exclude here, logic handles exclusion
        final_df = d.drop(columns=exclude, errors='ignore')
        
        msg = "Running Logistic Regression..." if model_type == 'logistic' else "Running Poisson Regression..."
        
        with ui.Progress(min=0, max=1) as p:
            p.set(message=msg, detail="Calculating...")
            
            try:
                # Run Logic
                html_rep, or_res, aor_res = process_data_and_generate_html(
                    final_df, target, var_meta=var_meta.get(), method=method, 
                    model_type=model_type, interaction_terms=interactions
                )
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Regression error")
                return
            
            # Generate Forest Plots using library
            fig_adj = None
            fig_crude = None
            
            x_label = "Adjusted IRR" if model_type == 'poisson' else "Adjusted OR"
            x_label_crude = "Crude IRR" if model_type == 'poisson' else "Crude OR"
            col_key = 'aor' # The key in dict is still 'aor'/'or' from logic for consistency
            
            try:
                if aor_res:
                    df_adj = pd.DataFrame([{'variable': k, **v} for k, v in aor_res.items()])
                    if not df_adj.empty:
                        fig_adj = create_forest_plot(
                            df_adj, 'aor', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                            title=f"<b>Multivariable: {x_label}</b>", x_label=x_label
                        )
                        del df_adj
                        # Optimized: Removed intermediate gc.collect()
                
                if or_res:
                    df_crude = pd.DataFrame([{'variable': k, **v} for k, v in or_res.items()])
                    if not df_crude.empty:
                        fig_crude = create_forest_plot(
                            df_crude, 'or', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                            title=f"<b>Univariable: {x_label_crude}</b>", x_label=x_label_crude
                        )
                        del df_crude
            
            except Exception as e:
                logger.exception("Forest plot generation error")
                ui.notification_show(f"Forest plot error: {e!s}", type="warning")

            # Store Results ONCE
            logit_res.set({
                "html": html_rep,
                "fig_adj": fig_adj,
                "fig_crude": fig_crude,
                "model_type": model_type
            })
            
            del html_rep, or_res, aor_res, fig_adj, fig_crude
            gc.collect() # Only run GC once at the end
            
            ui.notification_show("✅ Analysis Complete!", type="message")

    # --- Render Main Results (Use Cached Results) ---
    @render.ui
    def out_logit_status():
        res = logit_cached_res()
        if res:
            m_type = res.get('model_type', 'Regression').capitalize()
            return ui.div(
                ui.h5(f"✅ {m_type} Analysis Complete"),
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
        
        m_type = res.get('model_type', 'logistic')
        label = "IRR" if m_type == 'poisson' else "OR"
        
        tabs = []
        if res['fig_crude']:
            tabs.append(ui.nav_panel(f"Crude {label}", output_widget("out_forest_crude")))
        if res['fig_adj']:
            tabs.append(ui.nav_panel(f"Adjusted {label}", output_widget("out_forest_adj")))
            
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

    @render.download(filename="regression_report.html")
    def btn_dl_report():
        res = logit_cached_res()
        if res: yield res['html']

    # ==========================================================================
    # OPTIMIZED: Cache Subgroup Analysis Results (Kept Original Logic)
    # ==========================================================================
    @reactive.Calc
    def subgroup_cached_res():
        return subgroup_res.get()

    @reactive.Calc
    def subgroup_cached_analyzer():
        return subgroup_analyzer.get()

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
                
                gc.collect()
                
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Subgroup analysis error")

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
            title = input.txt_edit_forest_title() or input.sg_title() or None
            return analyzer.create_forest_plot(title=title)
        return None

    @reactive.Effect
    @reactive.event(input.btn_update_plot_title)
    def _update_sg_title():
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
        if res: 
            overall = res.get('overall', {})
            p = overall.get('p_value')
            return f"{p:.4f}" if isinstance(p, (int, float)) else "N/A"
        return "-"
    
    @render.text
    def val_interaction_p():
        res = subgroup_cached_res()
        if res:
             # Defensively access nested keys
             interaction = res.get('interaction', {})
             p_int = interaction.get('p_value')
             return f"{p_int:.4f}" if isinstance(p_int, (int, float)) else "N/A"
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
            yield json.dumps(res, indent=2, default=str)

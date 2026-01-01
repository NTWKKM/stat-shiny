from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import table_one  # Import from root
import psm_lib  # Import from root
from logger import get_logger
import io

from tabs._common import get_color_palette

logger = get_logger(__name__)
COLORS = get_color_palette()

# ==============================================================================
# Helper Function (Pure Python)
# ==============================================================================
def _calculate_categorical_smd(df: pd.DataFrame, treatment_col: str, cat_cols: list) -> pd.DataFrame:
    """
    Compute standardized mean differences (SMD) for categorical covariates between treated and control groups.
    """
    if not cat_cols:
        return pd.DataFrame(columns=['Variable', 'SMD'])
    
    smd_data = []
    # Ensure columns exist before filtering
    if treatment_col not in df.columns:
        return pd.DataFrame(columns=['Variable', 'SMD'])

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    n_treated = len(treated)
    n_control = len(control)
    
    if n_treated == 0 or n_control == 0:
        return pd.DataFrame(columns=['Variable', 'SMD'])
    
    for col in cat_cols:
        if col not in df.columns: continue
        try:
            categories = df[col].dropna().unique()
            smd_squared_sum = 0
            
            for cat in categories:
                p_treated = (treated[col] == cat).sum() / n_treated
                p_control = (control[col] == cat).sum() / n_control
                
                p_pooled = (n_treated * p_treated + n_control * p_control) / (n_treated + n_control)
                variance = p_pooled * (1 - p_pooled) + 1e-8
                
                smd_level = (p_treated - p_control) / np.sqrt(variance)
                smd_squared_sum += smd_level ** 2
            
            smd = np.sqrt(smd_squared_sum)
            smd_data.append({'Variable': col, 'SMD': smd})
            
        except Exception as e:
            logger.warning("Error calculating categorical SMD for %s: %s", col, e)
            continue
    
    return pd.DataFrame(smd_data)

# ==============================================================================
# UI Definition - Stacked Layout (Controls Top + Content Bottom)
# ==============================================================================
@module.ui
def baseline_matching_ui():
    return ui.navset_card_tab(
        
        # ===== SUBTAB 1: BASELINE CHARACTERISTICS (TABLE 1) =====
        ui.nav_panel(
            "📊 Baseline Characteristics (Table 1)",
            
            # Control section (top)
            ui.card(
                ui.card_header("📊 Table 1 Options"),
                
                ui.output_ui("ui_matched_status_banner_t1"),
                ui.output_ui("ui_dataset_selector_t1"),
                ui.output_ui("ui_data_info_t1"),
                
                ui.hr(),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Configuration"),
                        ui.h6("Group By (Column):"),
                        ui.input_select("sel_group_col", label=None, choices=[]),
                        
                        ui.h6("Choose OR Style:"),
                        ui.input_radio_buttons(
                            "radio_or_style",
                            label=None,
                            choices={
                                "all_levels": "All Levels (Every Level vs Ref)",
                                "simple": "Simple (Single Line/Risk vs Ref)"
                            }
                        ),
                    ),
                    
                    ui.card(
                        ui.card_header("Variables"),
                        ui.h6("Include Variables:"),
                        ui.input_selectize("sel_t1_vars", label=None, choices=[], multiple=True),
                    ),
                    
                    col_widths=[6, 6]
                ),
                
                ui.hr(),
                
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_gen_table1",
                        "📊 Generate Table 1",
                        class_="btn-primary btn-sm w-100",
                    ),
                    
                    ui.download_button(
                        "btn_dl_table1",
                        "📥 Download HTML",
                        class_="btn-success btn-sm w-100"
                    ),
                    
                    col_widths=[6, 6]
                ),
            ),
            
            # Content section (bottom)
            ui.output_ui("out_table1_html"),
        ),
        
        # ===== SUBTAB 2: PROPENSITY SCORE MATCHING =====
        ui.nav_panel(
            "⚖️ Propensity Score Matching",
            
            # Control section (top)
            ui.card(
                ui.card_header("⚖️ PSM Configuration"),
                
                ui.h5("Step 1️⃣: Configure Variables"),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Quick Presets:"),
                        ui.input_radio_buttons(
                            "radio_preset",
                            label=None,
                            choices={
                                "custom": "🔧 Custom (Manual)",
                                "demographics": "👥 Demographics",
                                "full_medical": "🏥 Full Medical"
                            },
                            selected="custom"
                        ),
                        
                        ui.p(
                            ui.strong("Presets include:"),
                            ui.br(),
                            "👥 Demographics: Age, Sex, BMI",
                            ui.br(),
                            "🏥 Full Medical: Age, Sex, BMI, Comorbidities, Lab values",
                            ui.br(),
                            "🔧 Custom: You choose all variables",
                            style=f"font-size: 0.85em; color: {COLORS['text_secondary']};"
                        ),
                    ),
                    
                    ui.card(
                        ui.card_header("Manual Selection:"),
                        ui.input_select("sel_treat_col", "💊 Treatment Variable (Binary):", choices=[]),
                        ui.input_select("sel_outcome_col", "🎯 Outcome Variable (Optional):", choices=[]),
                        ui.input_selectize("sel_covariates", "📊 Confounding Variables:", choices=[], multiple=True),
                    ),
                    
                    col_widths=[6, 6]
                ),
                
                ui.hr(),
                
                ui.output_ui("ui_psm_config_summary"),
                
                ui.hr(),
                
                # Advanced Settings
                ui.accordion(
                    ui.accordion_panel(
                        "⚙️ Advanced Settings",
                        ui.p(ui.strong("Caliper Width (Matching Tolerance)")),
                        ui.input_select(
                            "sel_caliper_preset",
                            label=None,
                            choices={
                                "1.0": "🔓 Very Loose (1.0×SD) - Most matches, weaker balance",
                                "0.5": "📊 Loose (0.5×SD) - Balanced approach",
                                "0.25": "⚖️ Standard (0.25×SD) - RECOMMENDED ← START HERE",
                                "0.1": "🔒 Strict (0.1×SD) - Fewer matches, excellent balance"
                            },
                            selected="0.25"
                        ),
                        ui.p(
                            "📌 Caliper = max distance to match treated with control. Wider = more matches, less balance.",
                            style=f"font-size: 0.8em; color: {COLORS['text_secondary']};"
                        ),
                    ),
                    open=False
                ),
                
                ui.hr(),
                
                ui.h5("Step 2️⃣: Run Matching"),
                
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run_psm",
                        "🚀 Run Propensity Score Matching",
                        class_="btn-danger btn-sm w-100"
                    ),
                    ui.output_ui("ui_psm_run_status"),
                    
                    col_widths=[9, 3]
                ),
            ),
            
            # Content section (bottom) - with nested tabs for results
            ui.output_ui("ui_psm_main_content")
        ),
        
        # ===== SUBTAB 3: MATCHED DATA VIEW =====
        ui.nav_panel(
            "✅ Matched Data View",
            
            # Control section (top)
            ui.card(
                ui.card_header("✅ Matched Data Actions"),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Export Options:"),
                        ui.download_button(
                            "btn_dl_matched_csv_view",
                            "📥 CSV Format",
                            class_="w-100 btn-sm"
                        ),
                        ui.br(),
                        ui.download_button(
                            "btn_dl_matched_xlsx_view",
                            "📥 Excel Format",
                            class_="w-100 btn-sm"
                        ),
                    ),
                    
                    ui.card(
                        ui.card_header("Filter & Display:"),
                        ui.input_slider(
                            "slider_matched_rows",
                            "Rows to display:",
                            min=1,
                            max=100,
                            value=50,
                            step=10
                        ),
                    ),
                    
                    ui.card(
                        ui.card_header("Compare Variable:"),
                        ui.input_select("sel_stat_var_tab3", label=None, choices=[]),
                    ),
                    
                    ui.card(
                        ui.card_header("Reset:"),
                        ui.input_action_button(
                            "btn_clear_matched_tab3",
                            "🔄 Clear Matched Data",
                            class_="btn-warning btn-sm w-100"
                        ),
                    ),
                    
                    col_widths=[3, 3, 3, 3]
                ),
            ),
            
            # Content section (bottom)
            ui.output_ui("ui_matched_status_tab3"),
            
            ui.card(
                ui.card_header("📊 Summary Statistics"),
                ui.output_ui("ui_matched_summary_stats"),
            ),
            
            ui.card(
                ui.card_header("🔍 Data Preview"),
                ui.output_data_frame("out_matched_df_preview")
            ),
            
            ui.card(
                ui.card_header("📈 Statistics by Group"),
                ui.navset_card_underline(
                    ui.nav_panel(
                        "📊 Descriptive Stats",
                        ui.output_data_frame("out_matched_stats")
                    ),
                    ui.nav_panel(
                        "📉 Visualization",
                        output_widget("out_matched_boxplot")
                    ),
                )
            )
        ),
        
        # ===== SUBTAB 4: REFERENCE & INTERPRETATION =====
        ui.nav_panel(
            "ℹ️ Reference & Interpretation",
            
            ui.markdown("""
## 📚 Reference & Interpretation Guide

💡 **Tip:** This section provides detailed explanations and interpretation rules for Table 1 and Propensity Score Matching.

### 🚦 Quick Decision Guide

| **Question** | **Recommended Action** | **Goal** |
| :--- | :--- | :--- |
| Do my groups differ at baseline? | **Generate Table 1** (Subtab 1) | Check for significant p-values (< 0.05). |
| My groups are imbalanced. Can I fix? | **Run PSM** (Subtab 2) | Create a "synthetic" RCT where groups are balanced. |
| Did the matching work? | **Check SMD** (Subtab 2 - Results) | Look for **SMD < 0.1** in the Love Plot. |
| What do I do with matched data? | **Export / Use Matched Data** | Go to **Subtab 3** to export, or select "✅ Matched Data" in other analysis tabs. |

---
            """),
            
            ui.layout_columns(
                ui.card(
                    ui.card_header("📊 Baseline Characteristics (Table 1)"),
                    ui.markdown("""
**Concept:** A standard table in medical research that compares the demographic and clinical characteristics of two or more groups (e.g., Treatment vs Placebo).

**Interpretation:**

* **P-value:** Tests if there is a statistically significant difference between groups.
* **p < 0.05:** Significant difference (Imbalance) ⚠️. This suggests confounding may be present.
* **p ≥ 0.05:** No significant difference (Balanced) ✅.

**Reporting Standards:**

* **Numeric Data (Normal):** Report **Mean ± SD**. (e.g., Age: 45.2 ± 10.1)
* **Numeric Data (Skewed):** Report **Median (IQR)**. (e.g., LOS: 5 (3-10))
* **Categorical Data:** Report **Count (%)**. (e.g., Male: 50 (45%))
                    """)
                ),
                
                ui.card(
                    ui.card_header("⚖️ Propensity Score Matching (PSM)"),
                    ui.markdown("""
**Concept:** A statistical technique used in observational studies to reduce selection bias. It pairs patients in the treated group with patients in the control group who have similar "propensity scores" (probability of receiving treatment).

**Key Metric: Standardized Mean Difference (SMD):**

* The gold standard for checking balance after matching.
* **SMD < 0.1:** Excellent Balance ✅ (Groups are comparable).
* **SMD 0.1 - 0.2:** Acceptable.
* **SMD > 0.2:** Imbalanced ❌.

**Caliper (Tolerance):**

* Determines how "close" a match must be.
* **Stricter (0.1×SD):** Better balance, but you might lose more patients (fewer matches).
* **Looser (0.5×SD):** More matches, but balance might be worse.
                    """)
                ),
                col_widths=[6, 6]
            ),
            
            ui.hr(),
            
            ui.markdown("""
### 📝 Common Workflow

1. **Check Original Data:** Run Table 1 on the "Original Data". Note any variables with p < 0.05.
2. **Match:** Go to Subtab 2, select Treatment, Outcome, and **all confounding variables** (especially those with p < 0.05).
3. **Verify:** After matching, check the **Love Plot**. Ensure all dots (Matched) are within the < 0.1 zone.
4. **Re-check Table 1:** Go back to Subtab 1, switch the dataset selector to **"✅ Matched Data"**, and generate Table 1 again. P-values should now be non-significant (or SMDs low).
            """)
        ),
        
        id="baseline_matching_tabs"
    )

# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def baseline_matching_server(input, output, session, df, var_meta, df_matched, is_matched, matched_treatment_col, matched_covariates):
    
    # -------------------------------------------------------------------------
    # SHARED REACTIVE VALUES
    # -------------------------------------------------------------------------
    psm_results = reactive.Value(None)
    html_content = reactive.Value(None)
    
    # -------------------------------------------------------------------------
    # HELPER: Get Current Data for Table 1
    # -------------------------------------------------------------------------
    @reactive.Calc
    def current_t1_data():
        if is_matched.get() and input.radio_dataset_source() == "matched" and df_matched.get() is not None:
            return df_matched.get(), "✅ Matched Data"
        return df.get(), "📊 Original Data"

    # -------------------------------------------------------------------------
    # UI UPDATERS (Dropdowns, Selectors)
    # -------------------------------------------------------------------------
    @reactive.Effect
    def _update_common_dropdowns():
        d = df.get()
        if d is None: return
        cols = d.columns.tolist()
        
        # Table 1
        ui.update_select("sel_group_col", choices=["None"] + cols)
        ui.update_selectize("sel_t1_vars", choices=cols, selected=cols)
        
        # PSM
        ui.update_select("sel_treat_col", choices=cols)
        ui.update_select("sel_outcome_col", choices=["⊘ None / Skip", *cols], selected="⊘ None / Skip")
        
        # FIX: Update covariates dropdown with all columns
        ui.update_selectize("sel_covariates", choices=cols, selected=[])
        
        # Matched View
        numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        ui.update_select("sel_stat_var_tab3", choices=numeric_cols)

    # =========================================================================
    # TAB 1: TABLE 1 LOGIC
    # =========================================================================
    
    @render.ui
    def ui_dataset_selector_t1():
        if is_matched.get():
            return ui.input_radio_buttons(
                "radio_dataset_source",
                "📄 Select Dataset:",
                choices={
                    "original": "📊 Original Data",
                    "matched": "✅ Matched Data (from PSM)"
                },
                selected="original",
                inline=True
            )
        return None

    @render.ui
    def ui_data_info_t1():
        data, label = current_t1_data()
        if data is None:
            return None
        return ui.p(
            f"**Using:** {label}",
            ui.br(),
            f"**Rows:** {len(data)} | **Columns:** {len(data.columns)}",
            style=f"font-size: 0.9em; color: {COLORS['text_secondary']};"
        )

    @render.ui
    def ui_matched_status_banner_t1():
        if is_matched.get():
            return ui.div(
                ui.p(
                    ui.strong("✅ Matched Dataset Available"),
                    " - You can select it above for analysis",
                    style=f"color: {COLORS['success']}; margin-bottom: 5px;"
                ),
                style=(
                    "padding: 10px; border-radius: 6px; margin-bottom: 15px; "
                    "background-color: rgba(34,167,101,0.08); "
                    f"border: 1px solid {COLORS['success']};"
                )
            )
        return None

    @reactive.Effect
    @reactive.event(input.btn_gen_table1)
    def _generate_table1():
        data, label = current_t1_data()
        if data is None: return
        
        group_col = input.sel_group_col()
        if group_col == "None": group_col = None
        
        selected_vars = input.sel_t1_vars()
        if not selected_vars:
            ui.notification_show("Please select at least one variable", type="warning")
            return

        ui.notification_show("Generating Table 1...", duration=None, id="gen_t1_notif")
        try:
            html = table_one.generate_table(
                data,
                selected_vars,
                group_col,
                var_meta.get(),
                or_style=input.radio_or_style()
            )
            html_content.set(html)
            ui.notification_remove("gen_t1_notif")
        except Exception as e:
            ui.notification_remove("gen_t1_notif")
            ui.notification_show(f"Error: {e}", type="error")
            logger.exception("Table 1 Generation Error")

    @render.ui
    def out_table1_html():
        if html_content.get():
            return ui.card(
                ui.card_header("📊 Table 1 Results"),
                ui.HTML(html_content.get()),
            )
        return ui.card(
            ui.card_header("📊 Table 1 Results"),
            ui.div(
                "Click '📊 Generate Table 1' to view results.",
                style="color: gray; font-style: italic; padding: 20px; text-align: center;"
            )
        )

    @render.download(filename="table1.html")
    def btn_dl_table1():
        if html_content.get():
            yield html_content.get()

    # =========================================================================
    # TAB 2: PSM LOGIC
    # =========================================================================
    
    @reactive.Effect
    def _apply_psm_presets():
        d = df.get()
        if d is None: return
        
        preset = input.radio_preset()
        treat = input.sel_treat_col()
        outcome = input.sel_outcome_col()
        
        # Build list of excluded columns
        excluded = []
        if treat:
            excluded.append(treat)
        if outcome and outcome != "⊘ None / Skip":
            excluded.append(outcome)
        
        # Get candidate columns (exclude treatment, outcome, and ID-like columns)
        candidates = [c for c in d.columns if c not in excluded and c.lower() not in ['id', 'index']]
        selected = []
        
        if preset == "demographics":
            # Match columns containing age, sex, bmi
            selected = [c for c in candidates if any(
                x in c.lower() for x in ['age', 'sex', 'male', 'bmi']
            )]
        elif preset == "full_medical":
            # Match columns for demographics + comorbidities + lab values
            selected = [c for c in candidates if any(
                x in c.lower() for x in ['age', 'sex', 'male', 'bmi', 'comorb', 'hyper', 'diab', 'lab', 'glucose', 'hba1c']
            )]
        
        # Only update if preset is not custom
        if preset != "custom":
            ui.update_selectize("sel_covariates", selected=selected)
            logger.info(f"Preset '{preset}' applied: selected {len(selected)} variables")

    @render.ui
    def ui_psm_config_summary():
        covs = input.sel_covariates() or []
        treat = input.sel_treat_col()
        outcome = input.sel_outcome_col()
        
        config_valid = len(covs) > 0
        
        summary_items = [
            f"💊 **Treatment:** `{treat if treat else '(not selected)'}`",
            f"🎯 **Outcome:** `{outcome if outcome != '⊘ None / Skip' else 'Skip'}`",
            f"📊 **Confounders:** {len(covs)} selected"
        ]
        
        if not config_valid:
            summary_items.append("❌ **Error:** Please select at least one covariate")
        
        summary_text = "**✅ Configuration Summary:**\n" + "\n".join(summary_items)
        
        return ui.info_message(summary_text, class_="bg-primary-light")

    @render.ui
    def ui_psm_run_status():
        covs = input.sel_covariates() or []
        if not covs:
            return ui.span(
                "⚠️ Select covariates",
                class_="text-danger fw-bold"
            )
        return ui.span(
            "✅ Ready to run",
            class_="text-success fw-bold"
        )

    @reactive.Effect
    @reactive.event(input.btn_run_psm)
    def _run_psm():
        d = df.get()
        treat_col = input.sel_treat_col()
        cov_cols = list(input.sel_covariates() or [])
        caliper = float(input.sel_caliper_preset())
        
        if d is None or not treat_col or not cov_cols:
            ui.notification_show("Please configure all required fields", type="warning")
            return

        ui.notification_show("Running Propensity Score Matching...", duration=None, id="psm_running")
        
        try:
            df_analysis = d.copy()
            
            # Pre-processing
            unique_treat = df_analysis[treat_col].dropna().unique()
            if len(unique_treat) != 2:
                raise ValueError(f"Treatment variable must have exactly 2 values. Found {len(unique_treat)}.")
            
            # Encode if categorical
            final_treat_col = treat_col
            if not pd.api.types.is_numeric_dtype(df_analysis[treat_col]):
                minor_val = df_analysis[treat_col].value_counts().idxmin()
                final_treat_col = f"{treat_col}_encoded"
                df_analysis[final_treat_col] = np.where(df_analysis[treat_col] == minor_val, 1, 0)
            
            # Handle categorical covariates
            cat_covs = [c for c in cov_cols if not pd.api.types.is_numeric_dtype(df_analysis[c])]
            if cat_covs:
                df_analysis = pd.get_dummies(df_analysis, columns=cat_covs, drop_first=True)
                new_cols = [c for c in df_analysis.columns if c not in d.columns and c != final_treat_col]
                final_cov_cols = [c for c in cov_cols if c not in cat_covs] + new_cols
            else:
                final_cov_cols = cov_cols

            # --- 🟢 FIXED: Updated to match new psm_lib API ---
            # Calculation
            # 1. Calculate Propensity Score (Returns DF, not Tuple)
            df_ps = psm_lib.calculate_propensity_score(df_analysis, final_treat_col, final_cov_cols)
            
            # 2. Perform Matching (Signature: df, treatment, caliper)
            df_m = psm_lib.perform_matching(df_ps, final_treat_col, caliper=caliper)
            
            msg = "Matching successful" # Default success message
            
            if df_m is None or df_m.empty:
                raise ValueError("No matches found within the specified caliper.")

            # SMD
            smd_pre = psm_lib.calculate_smd(df_ps, final_treat_col, final_cov_cols)
            smd_post = psm_lib.calculate_smd(df_m, final_treat_col, final_cov_cols)
            
            # Cat SMD
            if cat_covs:
                smd_pre_cat = _calculate_categorical_smd(df_ps, final_treat_col, cat_covs)
                smd_post_cat = _calculate_categorical_smd(df_m, final_treat_col, cat_covs)
                smd_pre = pd.concat([smd_pre, smd_pre_cat], ignore_index=True)
                smd_post = pd.concat([smd_post, smd_post_cat], ignore_index=True)

            # Save results
            results = {
                "df_matched": df_m,
                "smd_pre": smd_pre,
                "smd_post": smd_post,
                "final_treat_col": final_treat_col,
                "msg": msg,
                "df_ps_len": len(df_ps),
                "df_matched_len": len(df_m),
                "treat_pre_sum": df_ps[final_treat_col].sum(),
                "treat_post_sum": df_m[final_treat_col].sum()
            }
            psm_results.set(results)
            
            # Update Global State
            df_matched.set(df_m)
            is_matched.set(True)
            matched_treatment_col.set(final_treat_col)
            matched_covariates.set(cov_cols)
            
            ui.notification_remove("psm_running")
            ui.notification_show("✅ Matching Successful!", type="message")
            logger.info(f"💾 Matched data stored. Rows: {len(df_m)}")

        except Exception as e:
            ui.notification_remove("psm_running")
            ui.notification_show(f"❌ Matching Failed: {e}", type="error")
            logger.error(f"PSM Error: {e}")

    # --- PSM Main Content Output ---
    
    @render.ui
    def ui_psm_main_content():
        res = psm_results.get()
        
        if res is None:
            return ui.card(
                ui.card_header("📊 Results"),
                ui.p("Click '🚀 Run Propensity Score Matching' to view results.", style="color: gray; font-style: italic; padding: 20px; text-align: center;")
            )
        
        # Display results with nested tabs
        return ui.navset_card_underline(
            # Tab 1: Match Quality
            ui.nav_panel(
                "📊 Match Quality",
                
                ui.h5("Step 3️⃣: Match Quality Summary"),
                ui.layout_columns(
                    ui.value_box("Pairs Matched", ui.output_ui("val_pairs"), theme="bg-teal"),
                    ui.value_box("Sample Retained", ui.output_ui("val_retained"), theme="bg-teal"),
                    ui.value_box("Good Balance", ui.output_ui("val_balance"), theme="bg-green"),
                    ui.value_box("SMD Improvement", ui.output_ui("val_smd_imp"), theme="bg-green"),
                    col_widths=[3, 3, 3, 3]
                ),
                
                ui.output_ui("ui_balance_alert"),
                
                ui.hr(),
                
                ui.h5("Step 4️⃣: Balance Assessment"),
                ui.navset_card_underline(
                    ui.nav_panel(
                        "📉 Love Plot",
                        output_widget("out_love_plot"),
                        ui.p("Green (diamond) = matched, Red (circle) = unmatched. Target: All on left (SMD < 0.1)", style=f"font-size: 0.85em; color: {COLORS['text_secondary']}; margin-top: 10px;")
                    ),
                    ui.nav_panel(
                        "📋 SMD Table",
                        ui.output_data_frame("out_smd_table"),
                        ui.p("✅ Good balance: SMD < 0.1 after matching", style=f"font-size: 0.85em; color: {COLORS['text_secondary']}; margin-top: 10px;")
                    ),
                    ui.nav_panel(
                        "📊 Group Comparison",
                        ui.output_data_frame("out_group_comparison_table")
                    ),
                ),
            ),
            
            # Tab 2: Export
            ui.nav_panel(
                "📥 Export & Next Steps",
                
                ui.h5("Step 5️⃣: Export & Next Steps"),
                ui.layout_columns(
                    ui.download_button(
                        "btn_dl_psm_csv",
                        "📥 Download CSV",
                        class_="w-100 btn-sm"
                    ),
                    ui.download_button(
                        "btn_dl_psm_report",
                        "📥 Report HTML",
                        class_="w-100 btn-sm"
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.p(
                    "✅ Full matched data available in **Subtab 3 (Matched Data View)**",
                    style="background-color: #f0fdf4; padding: 10px; border-radius: 5px; border: 1px solid #bbf7d0; margin-top: 10px;"
                ),
            ),
            
            id="psm_results_tabs"
        )

    # --- PSM Output Components ---
    
    @render.ui
    def val_pairs():
        res = psm_results.get()
        if not res: return "-"
        return f"{res['treat_post_sum']:.0f}"

    @render.ui
    def val_retained():
        res = psm_results.get()
        if not res: return "-"
        pct = (res['df_matched_len'] / res['df_ps_len'] * 100)
        return f"{pct:.1f}%"

    @render.ui
    def val_balance():
        res = psm_results.get()
        if not res: return "-"
        good = (res['smd_post']['SMD'] < 0.1).sum()
        total = len(res['smd_post'])
        return f"{good}/{total}"

    @render.ui
    def val_smd_imp():
        res = psm_results.get()
        if not res: return "-"
        merged = res['smd_pre'].merge(res['smd_post'], on='Variable', suffixes=('_pre', '_post'))
        avg_pre = merged['SMD_pre'].mean()
        avg_post = merged['SMD_post'].mean()
        imp = ((avg_pre - avg_post)/avg_pre * 100) if avg_pre > 0 else 0
        return f"{imp:.1f}%"

    @render.ui
    def ui_balance_alert():
        res = psm_results.get()
        if not res:
            return None
        
        good = (res['smd_post']['SMD'] < 0.1).sum()
        total = len(res['smd_post'])
        
        if good == total:
            return ui.div(
                ui.strong("✅ Excellent balance achieved!"),
                " All variables have SMD < 0.1",
                style=(
                    "padding: 10px; border-radius: 6px; "
                    "background-color: rgba(34,167,101,0.08); "
                    f"border: 1px solid {COLORS['success']}; "
                    f"color: {COLORS['success']};"
                )
            )
        else:
            bad_count = total - good
            return ui.div(
                ui.strong("⚠️ Imbalance remains"),
                f" on {bad_count} variable(s). Try increasing caliper width or checking for outliers.",
                style=(
                    "padding: 10px; border-radius: 6px; "
                    "background-color: rgba(255,185,0,0.08); "
                    f"border: 1px solid {COLORS['warning']}; "
                    "color: #000;"
                )
            )

    @render_widget
    def out_love_plot():
        res = psm_results.get()
        if not res: return None
        return psm_lib.plot_love_plot(res['smd_pre'], res['smd_post'])

    @render.data_frame
    def out_smd_table():
        res = psm_results.get()
        if not res: return None
        merged = res['smd_pre'].merge(res['smd_post'], on='Variable', suffixes=('_before', '_after'))
        merged['Improvement %'] = ((merged['SMD_before'] - merged['SMD_after']) / merged['SMD_before'].replace(0, np.nan) * 100).round(1).fillna(0)
        return render.DataGrid(merged.style.format({'SMD_before': '{:.4f}', 'SMD_after': '{:.4f}', 'Improvement %': '{:.1f}%'}))

    @render.data_frame
    def out_group_comparison_table():
        res = psm_results.get()
        if not res: return None
        
        treat_col = res['final_treat_col']
        comp_data = pd.DataFrame({
            'Stage': ['Before', 'After'],
            'Treated (1)': [
                (res['treat_pre_sum'] if 'treat_pre_sum' in res else '-'),
                res['treat_post_sum']
            ],
            'Control (0)': [
                (res['df_ps_len'] - res['treat_pre_sum'] if 'treat_pre_sum' in res else '-'),
                (res['df_matched_len'] - res['treat_post_sum'])
            ]
        })
        return render.DataGrid(comp_data)

    @render.download(filename="matched_data.csv")
    def btn_dl_psm_csv():
        res = psm_results.get()
        if res:
            yield res['df_matched'].to_csv(index=False)
              
    @render.download(filename="psm_report.html")
    def btn_dl_psm_report():
        res = psm_results.get()
        if res:
            fig = psm_lib.plot_love_plot(res['smd_pre'], res['smd_post'])
            merged = res['smd_pre'].merge(res['smd_post'], on='Variable', suffixes=('_before', '_after'))
            elements = [
                {'type': 'text', 'data': f"PSM Report"},
                {'type': 'table', 'data': merged},
                {'type': 'plot', 'data': fig}
            ]
            html = psm_lib.generate_psm_report("Propensity Score Matching Report", elements)
            yield html

    # =========================================================================
    # TAB 3: MATCHED DATA VIEW
    # =========================================================================
    
    @render.ui
    def ui_matched_status_tab3():
        if df_matched.get() is not None:
            df_m = df_matched.get()
            treat_col = matched_treatment_col.get()
            return ui.div(
                ui.h5(
                    ui.span("✅ Matched Dataset Ready", style=f"color: {COLORS['success']};"),
                    style="margin-bottom: 10px;"
                ),
                ui.p(
                    f"• Total rows: **{len(df_m):,}**",
                    ui.br(),
                    f"• Treatment variable: **{treat_col}**",
                    style="font-size: 0.95em;"
                ),
                style=(f"background-color: rgba(34,167,101,0.08); padding: 15px; border-radius: 5px; "
                       f"border: 1px solid {COLORS['success']}; margin-bottom: 20px;")
            )
        else:
            return ui.info_message(
                "ℹ️ **No matched data available yet.**\n\n"
                "1. Go to **Subtab 2 (Propensity Score Matching)**\n\n"
                "2. Configure variables and run PSM matching\n\n"
                "3. Return here to view and export matched data"
            )

    @render.ui
    def ui_matched_summary_stats():
        if df_matched.get() is None:
            return None
        
        df_m = df_matched.get()
        treat_col = matched_treatment_col.get()
        
        # Show group sizes
        if treat_col and treat_col in df_m.columns:
            grp_counts = df_m[treat_col].value_counts().sort_index()
            return ui.p(
                ui.strong(f"Group Sizes ({treat_col}):"),
                ui.br(),
                ", ".join([f"{idx}: {count}" for idx, count in grp_counts.items()]),
                style=f"font-size: 0.9em; color: {COLORS['text_secondary']};"
            )
        return None

    @render.data_frame
    def out_matched_df_preview():
        if df_matched.get() is not None:
            n_rows = input.slider_matched_rows() or 50
            return render.DataGrid(df_matched.get().head(n_rows), filters=True)
        return None

    @render.data_frame
    def out_matched_stats():
        d = df_matched.get()
        var = input.sel_stat_var_tab3()
        treat = matched_treatment_col.get()
        
        if d is not None and var and treat and var in d.columns and treat in d.columns:
            return render.DataGrid(d.groupby(treat)[var].describe().reset_index())
        return None

    @render_widget
    def out_matched_boxplot():
        d = df_matched.get()
        var = input.sel_stat_var_tab3()
        treat = matched_treatment_col.get()
        
        if d is not None and var and treat:
            return px.box(d, x=treat, y=var, title=f"{var} by {treat}")
        return None

    @reactive.Effect
    @reactive.event(input.btn_clear_matched_tab3)
    def _clear_matched():
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])
        psm_results.set(None)
        html_content.set(None)
        ui.notification_show("Matched data cleared", type="warning")
        logger.info("🔄 Matched data cleared")

    # Exports for Tab 3
    @render.download(filename="matched_data.csv")
    def btn_dl_matched_csv_view():
        if df_matched.get() is not None:
            yield df_matched.get().to_csv(index=False)
              
    @render.download(filename="matched_data.xlsx")
    def btn_dl_matched_xlsx_view():
        if df_matched.get() is not None:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_matched.get().to_excel(writer, index=False)
            yield buffer.getvalue()

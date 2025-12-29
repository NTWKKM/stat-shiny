from shiny import ui, module, reactive, render, req
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import table_one  # Import from root
import psm_lib  # Import from root
from logger import get_logger
import io

logger = get_logger(__name__)

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
# UI Definition
# ==============================================================================
@module.ui
def baseline_matching_ui():
    return ui.navset_card_tab(
        # ---------------------------------------------------------------------
        # TAB 1: Baseline Characteristics (Table 1)
        # ---------------------------------------------------------------------
        ui.nav_panel("📊 Baseline Characteristics (Table 1)",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Table 1 Options"),
                    ui.output_ui("ui_dataset_selector"),
                    ui.hr(),
                    ui.input_select("sel_group_col", "Group By (Column):", choices=[]),
                    ui.input_radio_buttons("radio_or_style", "Choose OR Style:",
                                         choices={"all_levels": "All Levels (Every Level vs Ref)",
                                                  "simple": "Simple (Single Line/Risk vs Ref)"}),
                    ui.input_selectize("sel_t1_vars", "Include Variables:", choices=[], multiple=True),
                    ui.hr(),
                    ui.input_action_button("btn_gen_table1", "📊 Generate Table 1", class_="btn-primary"),
                    ui.br(),
                    ui.download_button("btn_dl_table1", "📥 Download HTML", class_="btn-secondary"),
                    width=350
                ),
                ui.output_ui("out_table1_html"),
                ui.output_ui("ui_matched_status_banner")
            )
        ),

        # ---------------------------------------------------------------------
        # TAB 2: Propensity Score Matching
        # ---------------------------------------------------------------------
        ui.nav_panel("⚖️ Propensity Score Matching",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Step 1️⃣: Configure"),
                    ui.input_radio_buttons("radio_preset", "Start with template:",
                                         choices=["🔧 Custom", "👥 Demographics", "🏥 Full Medical"]),
                    ui.input_select("sel_treat_col", "💊 Treatment (Binary):", choices=[]),
                    ui.input_select("sel_outcome_col", "🎯 Outcome (Optional):", choices=[]),
                    ui.input_selectize("sel_covariates", "📊 Confounders:", choices=[], multiple=True),
                    
                    ui.accordion(
                        ui.accordion_panel("⚙️ Advanced Settings",
                            ui.input_select("sel_caliper_preset", "Matching Strictness (Caliper):",
                                          choices={
                                              "1.0": "🔓 Very Loose (1.0×SD)",
                                              "0.5": "📊 Loose (0.5×SD)",
                                              "0.25": "⚖️ Standard (0.25×SD)",
                                              "0.1": "🔒 Strict (0.1×SD)"
                                          }, selected="0.25"),
                             ui.p("Caliper = max distance to match. Wider = more matches, less balance.", class_="text-muted", style="font-size: 0.8em;")
                        ),
                        open=False
                    ),
                    ui.hr(),
                    ui.h5("Step 2️⃣: Run"),
                    ui.input_action_button("btn_run_psm", "🚀 Run Matching", class_="btn-danger"),
                    ui.output_text("out_config_status"),
                    width=350
                ),
                
                # Main Result Area
                ui.navset_card_underline(
                    ui.nav_panel("📊 Match Quality",
                        ui.layout_columns(
                             ui.value_box("Pairs Matched", ui.output_ui("val_pairs"), theme="primary"),
                             ui.value_box("Sample Retained", ui.output_ui("val_retained"), theme="primary"),
                             ui.value_box("Good Balance", ui.output_ui("val_balance"), theme="teal"),
                             ui.value_box("SMD Improvement", ui.output_ui("val_smd_imp"), theme="teal"),
                             col_widths=[3, 3, 3, 3]
                        ),
                        ui.output_ui("ui_psm_error"),
                        ui.hr(),
                        ui.h5("Balance Assessment"),
                        ui.layout_columns(
                            ui.card(output_widget("out_love_plot")),
                            ui.card(ui.output_data_frame("out_smd_table")),
                            col_widths=[6, 6]
                        ),
                        ui.h5("Export"),
                        ui.layout_columns(
                             ui.download_button("btn_dl_psm_csv", "📥 Download Matched CSV"),
                             ui.download_button("btn_dl_psm_report", "📥 Download Report HTML"),
                        )
                    ),
                )
            )
        ),

        # ---------------------------------------------------------------------
        # TAB 3: Matched Data View
        # ---------------------------------------------------------------------
        ui.nav_panel("✅ Matched Data View",
             ui.layout_sidebar(
                ui.sidebar(
                    ui.h5("Actions"),
                    ui.input_action_button("btn_clear_matched_tab3", "🔄 Clear Matched Data", class_="btn-warning"),
                    ui.hr(),
                    ui.h6("Export"),
                    ui.download_button("btn_dl_matched_csv_view", "📥 CSV Format"),
                    ui.download_button("btn_dl_matched_xlsx_view", "📥 Excel Format"),
                ),
                ui.card(
                    ui.card_header("Matched Data Preview"),
                    ui.output_data_frame("out_matched_df_preview")
                ),
                ui.card(
                    ui.card_header("Statistics by Group"),
                    ui.layout_columns(
                        ui.input_select("sel_stat_var_tab3", "Compare Variable:", choices=[]),
                        col_widths=[4]
                    ),
                    ui.navset_card_underline(
                         ui.nav_panel("📊 Descriptive Stats", ui.output_data_frame("out_matched_stats")),
                         ui.nav_panel("📉 Visualization", output_widget("out_matched_boxplot"))
                    )
                )
             )
        ),

        # ---------------------------------------------------------------------
        # TAB 4: Reference
        # ---------------------------------------------------------------------
        ui.nav_panel("ℹ️ Reference",
            ui.markdown("""
            ## 📚 Reference & Interpretation Guide
            
            ### 🚦 Quick Decision Guide
            | Question | Recommended Action | Goal |
            | :--- | :--- | :--- |
            | Do my groups differ at baseline? | **Generate Table 1** | Check for p < 0.05. |
            | My groups are imbalanced. Can I fix this? | **Run PSM** | Create a balanced "synthetic" RCT. |
            | Did the matching work? | **Check SMD** | Look for **SMD < 0.1** in Love Plot. |
            
            ### ⚖️ Propensity Score Matching (PSM)
            * **SMD < 0.1:** Excellent Balance ✅
            * **SMD 0.1 - 0.2:** Acceptable
            * **SMD > 0.2:** Imbalanced ❌
            """)
        )
    )

# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def baseline_matching_server(input, output, session, df, var_meta, df_matched, is_matched, matched_treatment_col, matched_covariates):
    
    # -------------------------------------------------------------------------
    # SHARED REACTIVE VALUES
    # -------------------------------------------------------------------------
    # Store PSM results locally to persist between tab switches
    psm_results = reactive.Value(None) 
    
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
        ui.update_select("sel_outcome_col", choices=["⊘ None / Skip"] + cols)
        
        # Matched View
        numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
        ui.update_select("sel_stat_var_tab3", choices=numeric_cols)

    @render.ui
    def ui_dataset_selector():
        if is_matched.get():
            return ui.input_radio_buttons("radio_dataset_source", "📄 Select Dataset:",
                                        choices={"original": "📊 Original Data", 
                                                 "matched": "✅ Matched Data"},
                                        selected="original")
        return None

    @render.ui
    def ui_matched_status_banner():
        if is_matched.get():
             return ui.div(
                 ui.h5("✅ Matched Dataset Available", style="color: green; margin-bottom: 0px;"),
                 ui.p("You can select it above for analysis.", style="font-size: 0.9em;"),
                 style="background-color: #f0fdf4; padding: 10px; border-radius: 5px; margin-top: 10px; border: 1px solid #bbf7d0;"
             )
        return None

    # -------------------------------------------------------------------------
    # TAB 1 LOGIC: GENERATE TABLE 1
    # -------------------------------------------------------------------------
    html_content = reactive.Value(None)

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
            # Generate HTML using table_one library
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
            return ui.HTML(html_content.get())
        return ui.div("Click 'Generate Table 1' to view results.", style="color: gray; font-style: italic; padding: 20px;")

    @render.download(filename="table1.html")
    def btn_dl_table1():
        if html_content.get():
            yield html_content.get()

    # -------------------------------------------------------------------------
    # TAB 2 LOGIC: PSM CONFIG & RUN
    # -------------------------------------------------------------------------
    
    # Auto-select covariates based on presets
    @reactive.Effect
    def _apply_psm_presets():
        d = df.get()
        if d is None: return
        
        preset = input.radio_preset()
        treat = input.sel_treat_col()
        outcome = input.sel_outcome_col()
        
        excluded = [treat]
        if outcome != "⊘ None / Skip": excluded.append(outcome)
        
        candidates = [c for c in d.columns if c not in excluded]
        selected = []
        
        if preset == "👥 Demographics":
            selected = [c for c in candidates if any(x in c.lower() for x in ['age', 'sex', 'bmi'])]
        elif preset == "🏥 Full Medical":
            selected = [c for c in candidates if any(x in c.lower() for x in ['age', 'sex', 'bmi', 'comorb', 'hyper', 'diab', 'lab'])]
        
        if preset != "🔧 Custom":
             ui.update_selectize("sel_covariates", selected=selected)

    @render.text
    def out_config_status():
        covs = input.sel_covariates()
        if not covs:
            return "⚠️ Please select covariates"
        return f"✅ Ready to match with {len(covs)} confounders"

    # Run PSM
    @reactive.Effect
    @reactive.event(input.btn_run_psm)
    def _run_psm():
        d = df.get()
        treat_col = input.sel_treat_col()
        cov_cols = list(input.sel_covariates())
        caliper = float(input.sel_caliper_preset())
        
        if not d is not None or not treat_col or not cov_cols:
            return

        ui.notification_show("Running Propensity Score Matching...", duration=None, id="psm_running")
        
        try:
            df_analysis = d.copy()
            
            # --- Pre-processing Logic (Simplified from original) ---
            unique_treat = df_analysis[treat_col].dropna().unique()
            if len(unique_treat) != 2:
                raise ValueError(f"Treatment variable must have exactly 2 values. Found {len(unique_treat)}.")
            
            # Encode if categorical
            final_treat_col = treat_col
            # (Assuming numeric 0/1 for simplicity in Shiny adaptation, but keeping safe logic)
            if not pd.api.types.is_numeric_dtype(df_analysis[treat_col]):
                minor_val = df_analysis[treat_col].value_counts().idxmin()
                final_treat_col = f"{treat_col}_encoded"
                df_analysis[final_treat_col] = np.where(df_analysis[treat_col] == minor_val, 1, 0)
            
            # Handle categorical covariates (One-Hot Encoding)
            cat_covs = [c for c in cov_cols if not pd.api.types.is_numeric_dtype(df_analysis[c])]
            if cat_covs:
                df_analysis = pd.get_dummies(df_analysis, columns=cat_covs, drop_first=True)
                new_cols = [c for c in df_analysis.columns if c not in d.columns and c != final_treat_col]
                final_cov_cols = [c for c in cov_cols if c not in cat_covs] + new_cols
            else:
                final_cov_cols = cov_cols

            # --- Calculation ---
            df_ps, _ = psm_lib.calculate_ps(df_analysis, final_treat_col, final_cov_cols)
            df_m, msg = psm_lib.perform_matching(df_ps, final_treat_col, 'ps_logit', caliper)
            
            if df_m is None:
                raise ValueError(msg)

            # SMD Calc
            smd_pre = psm_lib.calculate_smd(df_ps, final_treat_col, final_cov_cols)
            smd_post = psm_lib.calculate_smd(df_m, final_treat_col, final_cov_cols)
            
            # Cat SMD
            if cat_covs:
                smd_pre_cat = _calculate_categorical_smd(df_ps, final_treat_col, cat_covs)
                smd_post_cat = _calculate_categorical_smd(df_m, final_treat_col, cat_covs)
                smd_pre = pd.concat([smd_pre, smd_pre_cat], ignore_index=True)
                smd_post = pd.concat([smd_post, smd_post_cat], ignore_index=True)

            # Save results to local reactive
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
            ui.notification_show("Matching Successful!", type="message")

        except Exception as e:
            ui.notification_remove("psm_running")
            ui.notification_show(f"Matching Failed: {e}", type="error")
            logger.error(f"PSM Error: {e}")

    # --- PSM Outputs ---
    
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
        # Match variables for comparison
        merged = res['smd_pre'].merge(res['smd_post'], on='Variable', suffixes=('_pre', '_post'))
        avg_pre = merged['SMD_pre'].mean()
        avg_post = merged['SMD_post'].mean()
        imp = ((avg_pre - avg_post)/avg_pre * 100) if avg_pre > 0 else 0
        return f"{imp:.1f}%"

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
        merged['Improvement %'] = ((merged['SMD_before'] - merged['SMD_after']) / merged['SMD_before'] * 100).round(1)
        return render.DataGrid(merged)

    @render.download(filename="matched_data.csv")
    def btn_dl_psm_csv():
        res = psm_results.get()
        if res:
             yield res['df_matched'].to_csv(index=False)
             
    @render.download(filename="psm_report.html")
    def btn_dl_psm_report():
        res = psm_results.get()
        if res:
            # Generate report HTML
            fig = psm_lib.plot_love_plot(res['smd_pre'], res['smd_post'])
            merged = res['smd_pre'].merge(res['smd_post'], on='Variable', suffixes=('_before', '_after'))
            elements = [
                {'type': 'text', 'data': f"PSM Report"},
                {'type': 'table', 'data': merged},
                {'type': 'plot', 'data': fig}
            ]
            html = psm_lib.generate_psm_report("Propensity Score Matching Report", elements)
            yield html

    # -------------------------------------------------------------------------
    # TAB 3 LOGIC: MATCHED VIEW
    # -------------------------------------------------------------------------
    
    @render.data_frame
    def out_matched_df_preview():
        if df_matched.get() is not None:
            return render.DataGrid(df_matched.get().head(100), filters=True)
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
        psm_results.set(None) # Clear local results too
        ui.notification_show("Matched data cleared", type="warning")

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

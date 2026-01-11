from shiny import ui, module, reactive, render, req
from utils.plotly_html_renderer import plotly_figure_to_html
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import html
from htmltools import HTML, div
import gc
# Use built-in list/dict/tuple for Python 3.9+ and typing for complex types
from typing import Optional, Any, Union, cast

# Import internal modules
from logic import analyze_outcome
from poisson_lib import analyze_poisson_outcome
from forest_plot_lib import create_forest_plot
from subgroup_analysis_module import SubgroupAnalysisLogit, SubgroupResult
from logger import get_logger
from tabs._common import get_color_palette
from config import CONFIG

logger = get_logger(__name__)
COLORS = get_color_palette()

# ==============================================================================
# Helper Functions (Pure Logic)
# ==============================================================================
def check_perfect_separation(df: pd.DataFrame, target_col: str) -> list[str]:
    """Identify columns causing perfect separation."""
    risky_vars: list[str] = []
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
                # Use crosstab to find cells with 0 count (perfect separation indicator)
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
def logit_ui() -> ui.TagChild:
    return ui.div(
        # Title + Data Summary inline
        ui.output_ui("ui_title_with_summary"),
        
        # Dataset Info Box
        ui.output_ui("ui_matched_info"),
        ui.br(),
        
        # Dataset Selector
        ui.output_ui("ui_dataset_selector"),
        ui.br(),
        
        # Main Analysis Tabs
        ui.navset_tab(
            # =====================================================================
            # TAB 1: Binary Logistic Regression
            # =====================================================================
            ui.nav_panel(
                "📈 Binary Logistic Regression",

                # Control section (top)
                ui.card(
                    ui.card_header("📈 Analysis Options"),

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
                    
                    # Interaction Pairs selector
                    ui.h6("🔗 Interaction Pairs (Optional):"),
                    ui.input_selectize(
                        "sel_interactions", 
                        label=None, 
                        choices=[], 
                        multiple=True,
                        options={"placeholder": "Select variable pairs to test interactions..."}
                    ),
                    ui.p(
                        "💡 Select pairs of variables to test for interaction effects (e.g., 'age × sex')",
                        style="font-size: 0.8em; color: #666; margin-top: 4px;"
                    ),

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
                ui.navset_tab(
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

                # Control section (top)
                ui.card(
                    ui.card_header("📊 Poisson Analysis Options"),

                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Variable Selection:"),
                            ui.input_select("poisson_outcome", "Select Count Outcome (Y):", choices=[]),
                            ui.input_select("poisson_offset", "Offset Column (Optional):", choices=["None"]),
                            ui.p(
                                "💡 Offset: Use for rate calculations (e.g., person-years, population)",
                                style="font-size: 0.8em; color: #666; margin-top: 4px;"
                            ),
                        ),

                        ui.card(
                            ui.card_header("Advanced Settings:"),
                            ui.h6("Exclude Variables (Optional):"),
                            ui.input_selectize("poisson_exclude", label=None, choices=[], multiple=True),
                        ),
                        
                        col_widths=[6, 6]
                    ),
                    
                    # Interaction Pairs selector
                    ui.h6("🔗 Interaction Pairs (Optional):"),
                    ui.input_selectize(
                        "poisson_interactions", 
                        label=None, 
                        choices=[], 
                        multiple=True,
                        options={"placeholder": "Select variable pairs to test interactions..."}
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

                # Content section (bottom)
                ui.output_ui("out_poisson_status"),
                ui.navset_tab(
                    ui.nav_panel(
                        "🌳 Forest Plots",
                        ui.output_ui("ui_poisson_forest_tabs")
                    ),
                    ui.nav_panel(
                        "📋 Detailed Report",
                        ui.output_ui("out_poisson_html_report")
                    ),
                    ui.nav_panel(
                        "📚 Reference",
                        ui.markdown("""
                        ### Poisson Regression Reference
                        
                        **When to Use:**
                        * Count outcomes (e.g., number of events, visits, infections)
                        * Rate data with exposure offset (e.g., events per person-year)
                        
                        **Interpretation:**
                        * **IRR > 1**: Higher incidence rate (Risk factor) 🔴
                        * **IRR < 1**: Lower incidence rate (Protective) 🟢
                        * **IRR = 1**: No effect on rate
                        
                        **Overdispersion:**
                        If variance >> mean, consider Negative Binomial regression.
                        """)
                    )
                )
            ),

            # =====================================================================
            # TAB 3: Subgroup Analysis
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
                ui.navset_tab(
                    ui.nav_panel(
                        "🌳 Forest Plot",
                        ui.output_ui("out_sg_forest_plot"),
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
            # TAB 4: Reference
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
                
                ### Perfect Separation:
                Occurs when a predictor perfectly predicts the outcome (e.g., all smokers died).
                * **Solution:** Use **Auto** or **Firth's** method, or exclude the variable.
                
                ### Subgroup Analysis:
                * Tests if treatment effect varies by group (Interaction test)
                * **P-interaction < 0.05**: Significant heterogeneity → Report subgroups separately
                * **P-interaction ≥ 0.05**: Homogeneous effect → Report overall effect
                """)
            )
        )
    )

# ==============================================================================
# Server Logic
# ==============================================================================
@module.server
def logit_server(
    input: Any, 
    output: Any, 
    session: Any, 
    df: reactive.Value[Optional[pd.DataFrame]], 
    var_meta: reactive.Value[dict[str, Any]], 
    df_matched: reactive.Value[Optional[pd.DataFrame]], 
    is_matched: reactive.Value[bool]
) -> None:

    # --- State Management ---
    # Store main logit results: {'html': str, 'fig_adj': FigureWidget, 'fig_crude': FigureWidget}
    logit_res = reactive.Value(None)     
    # Store Poisson results: {'html': str, 'fig_adj': FigureWidget, 'fig_crude': FigureWidget}
    poisson_res = reactive.Value(None)   
    # Store subgroup results: SubgroupResult
    subgroup_res: reactive.Value[Optional[SubgroupResult]] = reactive.Value(None)
    # Store analyzer instance: SubgroupAnalysisLogit
    subgroup_analyzer: reactive.Value[Optional[SubgroupAnalysisLogit]] = reactive.Value(None)

    # --- Cache Clearing on Tab Change ---
    @reactive.Effect
    @reactive.event(input.btn_run_logit, input.btn_run_poisson, input.btn_run_subgroup)
    def _cleanup_after_analysis():
        """
        OPTIMIZATION: Clear cache after completing analysis.
        This prevents memory buildup from heavy computations.
        """
        try:
            gc.collect()  # Force garbage collection
            logger.debug("Post-analysis cache cleared")
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")
    
    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df() -> Optional[pd.DataFrame]:
        if is_matched.get() and input.radio_logit_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_title_with_summary():
        """Display title with dataset summary."""
        d = current_df()
        if d is not None:
            return ui.div(
                ui.h3("📈 Logistic Regression"),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3"
                )
            )
        return ui.h3("📈 Logistic Regression")

    @render.ui
    def ui_matched_info():
        """Display matched dataset availability info."""
        if is_matched.get():
            return ui.div(
                ui.tags.div(
                    "✅ **Matched Dataset Available** - You can select it below for analysis",
                    class_="alert alert-info"
                )
            )
        return None

    @render.ui
    def ui_dataset_selector():
        """Render dataset selector radio buttons."""
        if is_matched.get():
            original = df.get()
            matched = df_matched.get()
            original_len = len(original) if original is not None else 0
            matched_len = len(matched) if matched is not None else 0
            return ui.input_radio_buttons(
                "radio_logit_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original ({original_len:,} rows)",
                    "matched": f"✅ Matched ({matched_len:,} rows)"
                },
                selected="matched",
                inline=True
            )
        return None

    # --- Dynamic Input Updates ---
    @reactive.Effect
    def _update_inputs():
        d = current_df()
        if d is None or d.empty: return

        cols = d.columns.tolist()
        
        # Identify binary columns for outcomes
        binary_cols = [c for c in cols if d[c].nunique() == 2]
        
        # Identify potential subgroups (2-10 levels)
        sg_cols = [c for c in cols if 2 <= d[c].nunique() <= 10]

        # Update Tab 1 (Binary Logit) Inputs
        ui.update_select("sel_outcome", choices=binary_cols)
        ui.update_selectize("sel_exclude", choices=cols)
        
        # Generate interaction pair choices for Logit
        from itertools import islice, combinations
        interaction_choices = list(islice((f"{a} × {b}" for a, b in combinations(cols, 2)), 50))
        ui.update_selectize("sel_interactions", choices=interaction_choices)
        
        # Update Tab 2 (Poisson) Inputs
        # Identify count columns (non-negative integers)
        count_cols = [
            c for c in cols 
            if pd.api.types.is_numeric_dtype(d[c]) and (d[c].dropna() >= 0).all() and (d[c].dropna() % 1 == 0).all()
        ]
        ui.update_select("poisson_outcome", choices=count_cols if count_cols else cols)
        ui.update_select("poisson_offset", choices=["None"] + cols)
        ui.update_selectize("poisson_exclude", choices=cols)
        ui.update_selectize("poisson_interactions", choices=interaction_choices[:50])

        # Update Tab 3 (Subgroup) Inputs
        ui.update_select("sg_outcome", choices=binary_cols)
        ui.update_select("sg_treatment", choices=cols)
        ui.update_select("sg_subgroup", choices=sg_cols)
        ui.update_selectize("sg_adjust", choices=cols)

    # --- Separation Warning Logic ---
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
    # LOGIC: Main Logistic Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_logit)
    def _run_logit():
        d = current_df()
        target = input.sel_outcome()
        exclude = input.sel_exclude()
        method = input.radio_method()
        interactions_raw = input.sel_interactions()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select an outcome variable", type="error")
            return

        # Prepare data
        final_df = d.drop(columns=exclude, errors='ignore')
        
        # Parse interaction pairs from "var1 × var2" format
        interaction_pairs: Optional[list[tuple[str, str]]] = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"Logit: Using {len(interaction_pairs)} interaction pairs")

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Logistic Regression...", detail="Calculating...")

            try:
                # Run Logic from logic.py
                # Note: updated logic.py returns 4 values and html_rep typically includes the plot + css
                html_rep, or_res, aor_res, interaction_res = analyze_outcome(
                    target, final_df, var_meta=var_meta.get(), method=method,
                    interaction_pairs=interaction_pairs,
                    adv_stats=CONFIG
                )
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Logistic regression error")
                return

            # Generate Forest Plots using library (for interactive widgets)
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

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Table + Plots) ---
            # 1. Create Fragment for UI (Table + Plots)
            logit_fragment_html = html_rep
            
            # Append Adjusted Plot if available
            if fig_adj:
                plot_html = plotly_figure_to_html(fig_adj, include_plotlyjs='cdn')
                logit_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Adjusted Forest Plot</h3>{plot_html}</div>"
            
            # Append Crude Plot if available
            if fig_crude:
                plot_html = plotly_figure_to_html(fig_crude, include_plotlyjs='cdn')
                logit_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Crude Forest Plot</h3>{plot_html}</div>"

            # 2. Create Full HTML for Download (Wrapped)
            full_logit_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Logistic Regression Report: {html.escape(target)}</title>
            </head>
            <body>
                <div class="report-container">
                    {logit_fragment_html}
                </div>
            </body>
            </html>
            """

            # Store Results
            logit_res.set({
                "html_fragment": logit_fragment_html, # For UI
                "html_full": full_logit_html,         # For Download
                "fig_adj": fig_adj,
                "fig_crude": fig_crude
            })

            ui.notification_show("✅ Analysis Complete!", type="message")

    # --- Render Main Results ---
    @render.ui
    def out_logit_status():
        res = logit_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Regression Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;"
            )
        return None

    @render.ui
    def out_html_report():
        res = logit_res.get()
        if res:
            return ui.card(
                ui.card_header("📋 Detailed Report"),
                ui.HTML(res['html_fragment'])
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
            tabs.append(ui.nav_panel("Crude OR", ui.output_ui("out_forest_crude")))
        if res['fig_adj']:
            tabs.append(ui.nav_panel("Adjusted OR", ui.output_ui("out_forest_adj")))

        if not tabs: 
            return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render.ui
    def out_forest_adj():
        res = logit_res.get()
        if res is None or not res.get('fig_adj'):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            res['fig_adj'],
            div_id="plot_logit_forest_adj",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @render.ui
    def out_forest_crude():
        res = logit_res.get()
        if res is None or not res.get('fig_crude'):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            res['fig_crude'],
            div_id="plot_logit_forest_crude",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @render.download(filename="logit_report.html")
    def btn_dl_report():
        res = logit_res.get()
        if res: yield res['html_full']

    # ==========================================================================
    # LOGIC: Poisson Regression
    # ==========================================================================
    @reactive.Effect
    @reactive.event(input.btn_run_poisson)
    def _run_poisson():
        d = current_df()
        target = input.poisson_outcome()
        exclude = input.poisson_exclude()
        offset_col = input.poisson_offset()
        interactions_raw = input.poisson_interactions()

        if d is None or d.empty:
            ui.notification_show("Please load data first", type="error")
            return
        if not target:
            ui.notification_show("Please select a count outcome variable", type="error")
            return

        # Prepare data
        final_df = d.drop(columns=exclude, errors='ignore')
        offset = offset_col if offset_col != "None" else None
        
        # Parse interaction pairs
        interaction_pairs: Optional[list[tuple[str, str]]] = None
        if interactions_raw:
            interaction_pairs = []
            for pair_str in interactions_raw:
                parts = pair_str.split(" × ")
                if len(parts) == 2:
                    interaction_pairs.append((parts[0].strip(), parts[1].strip()))
            logger.info(f"Poisson: Using {len(interaction_pairs)} interaction pairs")

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Running Poisson Regression...", detail="Calculating...")

            try:
                # Run Poisson Logic
                # Expecting 4 values from the updated poisson_lib.py
                html_rep, irr_res, airr_res, interaction_res = analyze_poisson_outcome(
                    target, final_df, var_meta=var_meta.get(),
                    offset_col=offset, interaction_pairs=interaction_pairs
                )
            except Exception as e:
                ui.notification_show(f"Error: {e!s}", type="error")
                logger.exception("Poisson regression error")
                return

            # Generate Forest Plots for IRR
            fig_adj = None
            fig_crude = None

            if airr_res:
                df_adj = pd.DataFrame([{'variable': k, **v} for k, v in airr_res.items()])
                if not df_adj.empty:
                    fig_adj = create_forest_plot(
                        df_adj, 'airr', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                        title="<b>Multivariable: Adjusted IRR</b>", x_label="Adjusted IRR"
                    )

            if irr_res:
                df_crude = pd.DataFrame([{'variable': k, **v} for k, v in irr_res.items()])
                if not df_crude.empty:
                    fig_crude = create_forest_plot(
                        df_crude, 'irr', 'ci_low', 'ci_high', 'variable', pval_col='p_value',
                        title="<b>Univariable: Crude IRR</b>", x_label="Crude IRR"
                    )

            # --- MANUALLY CONSTRUCT COMPLETE REPORT (Combined Table + Plot) ---
            # Unlike logic.py, poisson_lib might return just the table HTML.
            # We inject CSS and append the Forest Plot HTML here to match the requested format.
            
            # Keep a fragment for in-app rendering
            poisson_fragment_html = html_rep
            
            # Append Adjusted Plot if available, else Crude
            plot_html = ""
            if fig_adj:
                plot_html = fig_adj.to_html(full_html=False, include_plotlyjs='cdn')
                poisson_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Adjusted Forest Plot</h3>{plot_html}</div>"
            elif fig_crude:
                plot_html = fig_crude.to_html(full_html=False, include_plotlyjs='cdn')
                poisson_fragment_html += f"<div class='forest-plot-section' style='margin-top: 30px; padding: 10px; border-top: 2px solid #eee;'><h3>🌲 Crude Forest Plot</h3>{plot_html}</div>"

            # Wrap in standard HTML structure for standalone download correctness
            wrapped_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Poisson Regression Report: {html.escape(target)}</title>
            </head>
            <body>
                <div class="report-container">
                    {poisson_fragment_html}
                </div>
            </body>
            </html>
            """
            full_poisson_html = wrapped_html

            # Store Results
            poisson_res.set({
                "html_fragment": poisson_fragment_html,  # For UI rendering
                "html_full": full_poisson_html,          # For downloads
                "fig_adj": fig_adj,
                "fig_crude": fig_crude
            })

            ui.notification_show("✅ Poisson Analysis Complete!", type="message")

    # --- Render Poisson Results ---
    @render.ui
    def out_poisson_status():
        res = poisson_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Poisson Regression Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;"
            )
        return None

    @render.ui
    def out_poisson_html_report():
        res = poisson_res.get()
        if res:
            return ui.card(
                ui.card_header("📋 Poisson Regression Report"),
                ui.HTML(res['html_fragment'])
            )
        return ui.card(
            ui.card_header("📋 Poisson Regression Report"),
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
            tabs.append(ui.nav_panel("Crude IRR", ui.output_ui("out_poisson_forest_crude")))
        if res['fig_adj']:
            tabs.append(ui.nav_panel("Adjusted IRR", ui.output_ui("out_poisson_forest_adj")))

        if not tabs:
            return ui.div("No forest plots available.", class_="text-muted")
        return ui.navset_card_tab(*tabs)

    @render.ui
    def out_poisson_forest_adj():
        res = poisson_res.get()
        if res is None or not res.get('fig_adj'):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            res['fig_adj'],
            div_id="plot_poisson_forest_adj",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @render.ui
    def out_poisson_forest_crude():
        res = poisson_res.get()
        if res is None or not res.get('fig_crude'):
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            res['fig_crude'],
            div_id="plot_poisson_forest_crude",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @render.download(filename="poisson_report.html")
    def btn_dl_poisson_report():
        res = poisson_res.get()
        if res: 
            yield res['html_full']

    # ==========================================================================
    # LOGIC: Subgroup Analysis
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

    # --- Render Subgroup Results ---
    @render.ui
    def out_subgroup_status():
        res = subgroup_res.get()
        if res:
            return ui.div(
                ui.h5("✅ Subgroup Analysis Complete"),
                style=f"background-color: {COLORS['primary_light']}; padding: 15px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px;"
            )
        return None

    @render.ui
    def out_sg_forest_plot():
        analyzer = subgroup_analyzer.get()
        if analyzer is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        # Use txt_edit_forest_title if provided, fallback to sg_title
        title = input.txt_edit_forest_title() or input.sg_title() or None
        fig = analyzer.create_forest_plot(title=title)
        if fig is None:
            return ui.div(
                ui.markdown("⏳ *No forest plot available...*"),
                style="color: #999; text-align: center; padding: 20px;"
            )
        html_str = plotly_figure_to_html(
            fig,
            div_id="plot_logit_subgroup",
            include_plotlyjs='cdn',
            responsive=True
        )
        return ui.HTML(html_str)

    @reactive.Effect
    @reactive.event(input.btn_update_plot_title)
    def _update_sg_title():
        # Invalidate to trigger re-render of the forest plot widget
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
            # Simple formatting for display
            cols = ['group', 'n', 'events', 'or', 'ci_low', 'ci_high', 'p_value']
            available_cols = [c for c in cols if c in df_res.columns]
            return render.DataGrid(df_res[available_cols].round(4))
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

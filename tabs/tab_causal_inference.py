import numpy as np
import pandas as pd
import plotly.express as px
from shiny import module, reactive, render, ui

from tabs._common import (
    get_color_palette,
)
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.psm_lib import calculate_ipw, calculate_ps, check_balance
from utils.sensitivity_lib import calculate_e_value
from utils.stratified_lib import breslow_day, mantel_haenszel

COLORS = get_color_palette()


@module.ui
def causal_inference_ui():
    """UI for Causal Inference."""
    return ui.div(
        ui.h3("🎯 Causal Inference"),
        # Dataset info and selector
        ui.output_ui("ui_matched_info_ci"),
        ui.output_ui("ui_dataset_selector_ci"),
        ui.br(),
        ui.navset_tab(
            ui.nav_panel(
                "⚖️ PSM & IPW",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_select(
                            "psm_treatment", "Treatment Variable (Binary)", choices=[]
                        ),
                        ui.input_select("psm_outcome", "Outcome Variable", choices=[]),
                        ui.input_selectize(
                            "psm_covariates",
                            "Covariates for Matching",
                            choices=[],
                            multiple=True,
                        ),
                        ui.input_action_button(
                            "btn_run_psm", "Run Analysis", class_="btn-primary"
                        ),
                    ),
                    ui.card(
                        ui.div(
                            "💡 ",
                            ui.strong("Need matched dataset?"),
                            " Use ",
                            ui.strong("Table 1 & Matching → PSM"),
                            " to create balanced paired data for further analysis.",
                            style="padding: 8px 12px; margin-bottom: 12px; background-color: rgba(23, 162, 184, 0.1); border-left: 3px solid #17a2b8; border-radius: 4px; font-size: 0.85em;",
                        ),
                        ui.h4("Inverse Probability Weighting (IPW) Results"),
                        ui.output_ui("out_ipw_results"),
                        ui.h5("Standardized Mean Differences (Balance Check)"),
                        ui.output_data_frame("out_balance_table"),
                    ),
                ),
            ),
            ui.nav_panel(
                "📊 Stratified Analysis",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_select(
                            "strat_treatment", "Treatment/Exposure (Binary)", choices=[]
                        ),
                        ui.input_select(
                            "strat_outcome", "Outcome (Binary)", choices=[]
                        ),
                        ui.input_select(
                            "strat_stratum", "Stratification Variable", choices=[]
                        ),
                        ui.input_action_button(
                            "btn_run_strat",
                            "Run Stratified Analysis",
                            class_="btn-primary",
                        ),
                    ),
                    ui.card(
                        ui.h4("Mantel-Haenszel Odds Ratio"),
                        ui.output_ui("out_mh_results"),
                        ui.h4("Test for Homogeneity (Breslow-Day / Interaction)"),
                        ui.output_ui("out_homogeneity_results"),
                        ui.h5("Stratum-Specific Estimates"),
                        ui.output_data_frame("out_stratum_table"),
                    ),
                ),
            ),
            ui.nav_panel(
                "🔍 Sensitivity Analysis",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_numeric(
                            "sens_est", "Observed Estimate (RR or OR)", 2.0, step=0.1
                        ),
                        ui.input_numeric("sens_lower", "Lower CI Limit", 1.5, step=0.1),
                        ui.input_numeric("sens_upper", "Upper CI Limit", 3.0, step=0.1),
                        ui.input_action_button(
                            "btn_calc_evalue", "Calculate E-Value", class_="btn-primary"
                        ),
                    ),
                    ui.card(
                        ui.h4("E-Value for Unmeasured Confounding"),
                        ui.output_ui("out_evalue_results"),
                        ui.markdown("""
                        **Interpretation:** The E-value is the minimum strength of association that an unmeasured confounder would need to have with both the treatment and the outcome to fully explain away a specific treatment-outcome association, conditional on the measured covariates.
                        """),
                    ),
                ),
            ),
            ui.nav_panel(
                "⚖️ Balance Diagnostics",
                ui.card(
                    ui.card_header("Love Plot (Covariate Balance)"),
                    ui.output_ui("plot_love"),
                ),
            ),
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                    ### Causal Inference Methods
                    
                    #### 1. Propensity Score Methods (IPW)
                    *   **Goal**: Estimate the Average Treatment Effect (ATE) by adjusting for confounding.
                    *   **Mechanism**: Observations are weighted by the inverse of their probability of receiving the received treatment.
                    *   **Diagnostics**: Check Standardized Mean Differences (SMD). Ideally, all SMD < 0.1 after weighting.
                    
                    #### 2. Stratified Analysis (Mantel-Haenszel)
                    *   **Goal**: adjust for confounding by a categorical variable (stratum).
                    *   **Mantel-Haenszel OR**: A weighted average of stratum-specific odds ratios.
                    *   **Homogeneity Test**: Checks if the effect of treatment is consistent across all strata.
                    
                    #### 3. Sensitivity Analysis (E-Value)
                    *   **Goal**: Assess robustness to unmeasured confounding.
                    *   **E-Value**: The strength of association an unmeasured confounder must have to explain away the observed effect. Larger E-values imply more robust results.
                 """),
            ),
        ),
    )


@module.server
def causal_inference_server(
    input, output, session, df, var_meta, df_matched, is_matched
):
    """Server logic for Causal Inference."""

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df():
        if is_matched.get() and input.radio_ci_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_matched_info_ci():
        """Display matched dataset availability info."""
        if is_matched.get():
            return ui.div(
                ui.tags.div(
                    "✅ **Matched Dataset Available** - You can select it below for analysis",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_dataset_selector_ci():
        """Render dataset selector radio buttons."""
        if is_matched.get():
            original = df.get()
            matched = df_matched.get()
            original_len = len(original) if original is not None else 0
            matched_len = len(matched) if matched is not None else 0
            return ui.input_radio_buttons(
                "radio_ci_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original ({original_len:,} rows)",
                    "matched": f"✅ Matched ({matched_len:,} rows)",
                },
                selected="original",
                inline=True,
            )
        return None

    # --- Update Choices ---
    @reactive.Effect
    def update_inputs():
        d = current_df()
        if d is None:
            return
        cols = list(d.columns)
        num_cols = list(d.select_dtypes(include=np.number).columns)
        cat_cols = list(d.select_dtypes(include=["object", "category"]).columns)

        # Heuristic for binary cols (0/1 or similar small unique count)
        binary_cols = [c for c in cols if d[c].nunique() == 2]

        ui.update_select("psm_treatment", choices=binary_cols)
        ui.update_select("psm_outcome", choices=num_cols + binary_cols)
        ui.update_selectize("psm_covariates", choices=num_cols)  # Simplified for now

        ui.update_select("strat_treatment", choices=binary_cols)
        ui.update_select("strat_outcome", choices=binary_cols)
        ui.update_select("strat_stratum", choices=cat_cols + binary_cols)

    # --- PSM & IPW Logic ---
    psm_res = reactive.Value(None)
    balance_res = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_run_psm)
    def run_psm_analysis():
        d = current_df()
        if d is None:
            return

        try:
            treatment = input.psm_treatment()
            outcome = input.psm_outcome()
            covs = list(input.psm_covariates())

            if not treatment or not outcome or not covs:
                ui.notification_show("Please select all PSM variables.", type="warning")
                return

            # Calulate PS
            d_clean = d.dropna(subset=[treatment, outcome] + covs).copy()
            # Ensure treatment is 0/1
            d_clean[treatment] = d_clean[treatment].astype(int)

            ps = calculate_ps(d_clean, treatment, covs)
            d_clean["ps"] = ps

            # IPW
            ipw_res = calculate_ipw(d_clean, treatment, outcome, "ps")
            psm_res.set(ipw_res)

            # Balance
            # Calculate weights for balance check
            T = d_clean[treatment]
            ps_vals = d_clean["ps"]
            weights = np.where(T == 1, 1 / ps_vals, 1 / (1 - ps_vals))

            bal = check_balance(
                d_clean,
                treatment,
                covs,
                weights=pd.Series(weights, index=d_clean.index),
            )
            balance_res.set(bal)

            ui.notification_show("PSM/IPW Analysis Complete", type="message")

        except Exception as e:
            ui.notification_show(f"Analysis Error: {str(e)}", type="error")

    @render.ui
    def out_ipw_results():
        res = psm_res.get()
        if res is None:
            return None
        if "error" in res:
            return ui.div(f"Error: {res['error']}", style="color: red;")

        return ui.div(
            ui.p(f"ATE Estimate: {res['ATE']:.4f}"),
            ui.p(f"95% CI: [{res['CI_Lower']:.4f}, {res['CI_Upper']:.4f}]"),
            ui.p(f"P-value: {res['p_value']:.4f}"),
        )

    @render.data_frame
    def out_balance_table():
        return balance_res.get()

    @render.ui
    def plot_love():
        bal = balance_res.get()
        if bal is None:
            return ui.div("Run PSM analysis to see balance diagnostics.")

        fig = px.scatter(
            bal,
            x="SMD",
            y="Covariate",
            title="Standardized Mean Differences (Weighted)",
            range_x=[0, max(0.5, bal["SMD"].max() * 1.1)],
        )
        fig.add_vline(
            x=0.1, line_dash="dash", line_color="red", annotation_text="Threshold 0.1"
        )
        return ui.HTML(plotly_figure_to_html(fig))

    # --- Stratified Logic ---
    strat_res = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_run_strat)
    def run_stratified():
        d = current_df()
        if d is None:
            return
        try:
            mh = mantel_haenszel(
                d, input.strat_outcome(), input.strat_treatment(), input.strat_stratum()
            )
            bd = breslow_day(
                d, input.strat_outcome(), input.strat_treatment(), input.strat_stratum()
            )

            strat_res.set({"mh": mh, "bd": bd})
        except Exception as e:
            ui.notification_show(f"Stratified Error: {str(e)}", type="error")

    @render.ui
    def out_mh_results():
        res = strat_res.get()
        if not res:
            return None
        mh = res["mh"]
        if "error" in mh:
            return ui.div(f"Error: {mh['error']}")

        return ui.div(ui.h5(f"Mantel-Haenszel OR: {mh['MH_OR']:.4f}"))

    @render.ui
    def out_homogeneity_results():
        res = strat_res.get()
        if not res:
            return None
        bd = res["bd"]
        if "error" in bd:
            return ui.div(f"Error: {bd['error']}")

        return ui.div(
            ui.p(f"Test: {bd.get('test')}"),
            ui.p(f"P-value: {bd.get('p_value'):.4f}"),
            ui.p(f"Conclusion: {bd.get('conclusion')}"),
        )

    @render.data_frame
    def out_stratum_table():
        res = strat_res.get()
        if not res or "error" in res["mh"]:
            return None
        return res["mh"]["Strata_Results"]

    # --- Sensitivity Logic ---
    eval_res = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_calc_evalue)
    def run_evalue():
        try:
            res = calculate_e_value(
                input.sens_est(), input.sens_lower(), input.sens_upper()
            )
            eval_res.set(res)
        except Exception as e:
            ui.notification_show(str(e), type="error")

    @render.ui
    def out_evalue_results():
        res = eval_res.get()
        if not res:
            return None
        return ui.div(
            ui.h5(f"E-Value for Estimate: {res['e_value_estimate']}"),
            ui.p(f"E-Value for CI Limit: {res['e_value_ci_limit']}"),
        )

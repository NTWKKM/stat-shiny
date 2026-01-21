import numpy as np
import pandas as pd
import plotly.express as px
from shiny import module, reactive, render, ui

from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.psm_lib import calculate_ipw, calculate_ps, check_balance
from utils.sensitivity_lib import calculate_e_value
from utils.stratified_lib import breslow_day, mantel_haenszel
from utils.ui_helpers import (
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
)

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
                        create_input_group(
                            "Model Specification",
                            ui.input_select(
                                "psm_treatment",
                                "Treatment Variable (Binary)",
                                choices=[],
                            ),
                            ui.input_select(
                                "psm_outcome", "Outcome Variable", choices=[]
                            ),
                            ui.input_selectize(
                                "psm_covariates",
                                "Covariates for Matching",
                                choices=[],
                                multiple=True,
                            ),
                            type="required",
                        ),
                        ui.output_ui("out_psm_validation"),
                        ui.br(),
                        ui.input_action_button(
                            "btn_run_psm", "Run Analysis", class_="btn-primary w-100"
                        ),
                    ),
                    ui.div(
                        ui.div(
                            "💡 ",
                            ui.strong("Need matched dataset?"),
                            " Use ",
                            ui.strong("Clinical → Table 1 & Matching"),
                            " to create balanced paired data first.",
                            style="padding: 10px; margin-bottom: 20px; background-color: var(--bs-info-bg-subtle, #e0faff); border-left: 4px solid var(--bs-info); border-radius: 4px;",
                        ),
                        create_results_container(
                            "IPW & Balance Results", ui.output_ui("out_psm_container")
                        ),
                    ),
                ),
            ),
            ui.nav_panel(
                "📊 Stratified Analysis",
                ui.layout_sidebar(
                    ui.sidebar(
                        create_input_group(
                            "Variables",
                            ui.input_select(
                                "strat_treatment",
                                "Treatment/Exposure (Binary)",
                                choices=[],
                            ),
                            ui.input_select(
                                "strat_outcome", "Outcome (Binary)", choices=[]
                            ),
                            ui.input_select(
                                "strat_stratum", "Stratification Variable", choices=[]
                            ),
                            type="required",
                        ),
                        ui.output_ui("out_strat_validation"),
                        ui.br(),
                        ui.input_action_button(
                            "btn_run_strat",
                            "Run Stratified Analysis",
                            class_="btn-primary w-100",
                        ),
                    ),
                    create_results_container(
                        "Stratified Results", ui.output_ui("out_strat_container")
                    ),
                ),
            ),
            ui.nav_panel(
                "🔍 Sensitivity Analysis",
                ui.layout_sidebar(
                    ui.sidebar(
                        create_input_group(
                            "Parameters",
                            ui.input_numeric(
                                "sens_est",
                                "Observed Estimate (RR or OR)",
                                2.0,
                                step=0.1,
                            ),
                            ui.input_numeric(
                                "sens_lower", "Lower CI Limit", 1.5, step=0.1
                            ),
                            ui.input_numeric(
                                "sens_upper", "Upper CI Limit", 3.0, step=0.1
                            ),
                            type="required",
                        ),
                        ui.output_ui("out_sens_validation"),
                        ui.br(),
                        ui.input_action_button(
                            "btn_calc_evalue",
                            "Calculate E-Value",
                            class_="btn-primary w-100",
                        ),
                    ),
                    create_results_container(
                        "E-Value Results", ui.output_ui("out_eval_container")
                    ),
                ),
            ),
            ui.nav_panel(
                "⚖️ Balance Diagnostics",
                create_results_container(
                    "Love Plot (Covariate Balance)",
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

        # Defaults for PSM
        def_psm_treat = select_variable_by_keyword(
            binary_cols, ["treatment", "group", "exposure"], default_to_first=True
        )
        def_psm_out = select_variable_by_keyword(
            cols, ["outcome", "cured", "death", "event"], default_to_first=True
        )

        ui.update_select("psm_treatment", choices=binary_cols, selected=def_psm_treat)
        ui.update_select(
            "psm_outcome", choices=num_cols + binary_cols, selected=def_psm_out
        )
        ui.update_selectize("psm_covariates", choices=num_cols)

        # Defaults for Stratified
        def_strat_treat = select_variable_by_keyword(
            binary_cols, ["treatment", "group", "exposure"], default_to_first=True
        )
        # Try to find different binary col for outcome
        rem_binary = [c for c in binary_cols if c != def_strat_treat]
        def_strat_out = select_variable_by_keyword(
            rem_binary, ["outcome", "cured", "event", "response"], default_to_first=True
        )
        # Find categorical stratum
        def_strat_stratum = select_variable_by_keyword(
            cat_cols,
            ["strata", "category", "stage", "gender", "sex"],
            default_to_first=True,
        )

        ui.update_select(
            "strat_treatment", choices=binary_cols, selected=def_strat_treat
        )
        ui.update_select("strat_outcome", choices=binary_cols, selected=def_strat_out)
        ui.update_select(
            "strat_stratum", choices=cat_cols + binary_cols, selected=def_strat_stratum
        )

    # --- PSM & IPW Logic ---
    psm_res = reactive.Value(None)
    balance_res = reactive.Value(None)

    # Running States
    psm_is_running = reactive.Value(False)
    strat_is_running = reactive.Value(False)
    eval_is_running = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.btn_run_psm)
    def run_psm_analysis():
        d = current_df()
        if d is None:
            return

        try:
            psm_is_running.set(True)
            psm_res.set(None)
            ui.notification_show("Running PSM/IPW...", duration=None, id="run_psm")

            treatment = input.psm_treatment()
            outcome = input.psm_outcome()
            covs = list(input.psm_covariates())

            if not treatment or not outcome or not covs:
                ui.notification_show("Please select all PSM variables.", type="warning")
                ui.notification_remove("run_psm")
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

            ui.notification_remove("run_psm")
            ui.notification_show("PSM/IPW Analysis Complete", type="message")

        except Exception as e:
            ui.notification_remove("run_psm")
            ui.notification_show("Analysis failed", type="error")
            psm_res.set({"error": f"Analysis Error: {str(e)}"})
        finally:
            psm_is_running.set(False)

    @render.ui
    def out_psm_container():
        if psm_is_running.get():
            return create_loading_state("Running PSM & IPW Analysis...")

        res = psm_res.get()
        if res is None:
            return create_placeholder_state(
                "Select variables and run analysis.", icon="⚖️"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        ipw_div = ui.div(
            ui.p(f"ATE Estimate: {res['ATE']:.4f}"),
            ui.p(f"95% CI: [{res['CI_Lower']:.4f}, {res['CI_Upper']:.4f}]"),
            ui.p(f"P-value: {res['p_value']:.4f}"),
        )

        return ui.div(
            ui.h5("Inverse Probability Weighting (IPW)"),
            ipw_div,
            ui.hr(),
            ui.h5("Standardized Mean Differences (Balance Check)"),
            ui.output_data_frame("out_balance_table"),
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
            strat_is_running.set(True)
            strat_res.set(None)
            ui.notification_show(
                "Running Stratified Analysis...", duration=None, id="run_strat"
            )

            mh = mantel_haenszel(
                d, input.strat_outcome(), input.strat_treatment(), input.strat_stratum()
            )
            bd = breslow_day(
                d, input.strat_outcome(), input.strat_treatment(), input.strat_stratum()
            )

            strat_res.set({"mh": mh, "bd": bd})
            ui.notification_remove("run_strat")
        except Exception as e:
            ui.notification_remove("run_strat")
            ui.notification_show("Analysis failed", type="error")
            strat_res.set({"error": f"Stratified Error: {str(e)}"})
        finally:
            strat_is_running.set(False)

    @render.ui
    def out_strat_container():
        if strat_is_running.get():
            return create_loading_state("Running Stratified Analysis...")

        res = strat_res.get()
        if not res:
            return create_placeholder_state(
                "Select variables and run stratified analysis.", icon="📊"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        # Prepare helper text
        mh = res["mh"]
        bd = res["bd"]

        mh_ui = (
            create_error_alert(mh["error"])
            if "error" in mh
            else ui.div(ui.h5(f"Mantel-Haenszel OR: {mh['MH_OR']:.4f}"))
        )

        bd_ui = None
        if "error" in bd:
            bd_ui = create_error_alert(bd["error"])
        else:
            bd_ui = ui.div(
                ui.p(f"Test: {bd.get('test')}"),
                ui.p(f"P-value: {bd.get('p_value'):.4f}"),
                ui.p(f"Conclusion: {bd.get('conclusion')}"),
            )

        return ui.div(
            ui.h5("Mantel-Haenszel Odds Ratio"),
            mh_ui,
            ui.br(),
            ui.h5("Test for Homogeneity (Breslow-Day / Interaction)"),
            bd_ui,
            ui.br(),
            ui.h5("Stratum-Specific Estimates"),
            ui.output_data_frame("out_stratum_table"),
        )

    @render.data_frame
    def out_stratum_table():
        res = strat_res.get()
        if not res or "error" in res or "error" in res["mh"]:
            return None
        return res["mh"]["Strata_Results"]

    # --- Sensitivity Logic ---
    eval_res = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_calc_evalue)
    def run_evalue():
        try:
            eval_is_running.set(True)
            eval_res.set(None)

            res = calculate_e_value(
                input.sens_est(), input.sens_lower(), input.sens_upper()
            )
            eval_res.set(res)
        except Exception as e:
            ui.notification_show("Analysis failed", type="error")
            eval_res.set({"error": f"Error: {str(e)}"})
        finally:
            eval_is_running.set(False)

    @render.ui
    def out_eval_container():
        if eval_is_running.get():
            return create_loading_state("Calculating E-Value...")

        res = eval_res.get()
        if not res:
            return create_placeholder_state(
                "Enter parameters and calculate E-Value.", icon="🔍"
            )

        if "error" in res:
            return create_error_alert(res["error"])

        return ui.div(
            ui.h5("E-Value for Unmeasured Confounding"),
            ui.div(
                ui.h5(f"E-Value for Estimate: {res['e_value_estimate']}"),
                ui.p(f"E-Value for CI Limit: {res['e_value_ci_limit']}"),
            ),
            ui.br(),
            ui.markdown("""
            **Interpretation:** The E-value is the minimum strength of association that an unmeasured confounder would need to have with both the treatment and the outcome to fully explain away a specific treatment-outcome association, conditional on the measured covariates.
            """),
        )

    # ==================== VALIDATION LOGIC ====================
    @render.ui
    def out_psm_validation():
        outcome = input.psm_outcome()
        treatment = input.psm_treatment()
        covs = input.psm_covariates()

        alerts = []
        if not outcome or not treatment:
            return None

        if outcome == treatment:
            alerts.append(
                create_error_alert(
                    "Outcome and Treatment variables must be different.",
                    title="Configuration Error",
                )
            )

        if covs and (outcome in covs or treatment in covs):
            alerts.append(
                create_error_alert(
                    "Covariates cannot include Outcome or Treatment variables.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_strat_validation():
        outcome = input.strat_outcome()
        treatment = input.strat_treatment()
        stratum = input.strat_stratum()

        alerts = []
        if not outcome or not treatment or not stratum:
            return None

        if len({outcome, treatment, stratum}) < 3:
            alerts.append(
                create_error_alert(
                    "Outcome, Treatment, and Stratum variables must be different.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_sens_validation():
        est = input.sens_est()
        lower = input.sens_lower()
        upper = input.sens_upper()

        alerts = []
        if est < 0 or lower < 0 or upper < 0:
            alerts.append(
                create_error_alert(
                    "Values should generally be non-negative (ratios).", title="Warning"
                )
            )

        if not (lower <= est <= upper) and not (upper <= est <= lower):
            alerts.append(
                create_error_alert(
                    "Estimate usually falls between Lower and Upper CI limits.",
                    title="Check Values",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

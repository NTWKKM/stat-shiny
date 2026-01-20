import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from shiny import module, reactive, render, ui

from tabs._common import (
    get_color_palette,
)
from utils.collinearity_lib import calculate_vif
from utils.heterogeneity_lib import calculate_heterogeneity
from utils.mediation_lib import analyze_mediation
from utils.model_diagnostics_lib import (
    calculate_cooks_distance,
    get_diagnostic_plot_data,
    run_heteroscedasticity_test,
    run_reset_test,
)
from utils.plotly_html_renderer import plotly_figure_to_html

COLORS = get_color_palette()


@module.ui
def advanced_inference_ui():
    """UI for Advanced Inference."""
    return ui.div(
        ui.h3("🔍 Advanced Inference"),
        # Dataset info and selector
        ui.output_ui("ui_matched_info_ai"),
        ui.output_ui("ui_dataset_selector_ai"),
        ui.br(),
        ui.navset_tab(
            ui.nav_panel(
                "🎯 Mediation Analysis",
                ui.card(
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Variables"),
                            ui.input_select("med_outcome", "Outcome (Y):", choices=[]),
                            ui.input_select(
                                "med_treatment", "Treatment (X):", choices=[]
                            ),
                            ui.input_select(
                                "med_mediator", "Mediator (M):", choices=[]
                            ),
                            ui.input_selectize(
                                "med_confounders",
                                "Confounders (C):",
                                choices=[],
                                multiple=True,
                            ),
                        ),
                        ui.card(
                            ui.card_header("Action"),
                            ui.input_action_button(
                                "btn_run_mediation",
                                "🚀 Run Mediation",
                                class_="btn-primary w-100",
                            ),
                        ),
                    ),
                    ui.output_ui("out_mediation_status"),
                ),
                ui.output_data_frame("tbl_mediation"),
            ),
            ui.nav_panel(
                "🔬 Collinearity",
                ui.card(
                    ui.input_selectize(
                        "coll_vars", "Predictors:", choices=[], multiple=True
                    ),
                    ui.input_action_button(
                        "btn_run_collinearity", "Analyze", class_="btn-primary"
                    ),
                ),
                ui.output_data_frame("tbl_vif"),
                ui.output_ui("plot_corr_heatmap"),
            ),
            ui.nav_panel(
                "📊 Model Diagnostics",
                ui.card(
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Input Data"),
                            ui.input_select("diag_outcome", "Outcome (Y):", choices=[]),
                            ui.input_select(
                                "diag_predictor", "Main Predictor (X):", choices=[]
                            ),
                            ui.input_selectize(
                                "diag_covariates",
                                "Covariates:",
                                choices=[],
                                multiple=True,
                            ),
                        ),
                        ui.card(
                            ui.card_header("Action"),
                            ui.input_action_button(
                                "btn_run_diag",
                                "🚀 Run Diagnostics",
                                class_="btn-primary w-100",
                            ),
                        ),
                    ),
                    ui.output_ui("out_diag_status"),
                ),
                ui.navset_tab(
                    ui.nav_panel("📋 Tests", ui.output_data_frame("tbl_diagnostics")),
                    ui.nav_panel(
                        "📉 Plots",
                        ui.layout_columns(
                            ui.output_plot("plot_residuals"),
                            ui.output_plot("plot_qq"),
                        ),
                    ),
                    ui.nav_panel(
                        "⚠️ Influence",
                        ui.output_data_frame("tbl_cooks"),
                        ui.output_text_verbatim("txt_cooks_summary"),
                    ),
                ),
            ),
            ui.nav_panel(
                "🏥 Heterogeneity Testing",
                ui.card(
                    ui.card_header("Meta-Analysis Data Entry"),
                    ui.p(
                        "Enter comma-separated values for effect sizes and variances."
                    ),
                    ui.input_text("het_effects", "Effect Sizes (e.g., 0.5, 0.4, 0.6):"),
                    ui.input_text(
                        "het_variances", "Variances (e.g., 0.01, 0.02, 0.015):"
                    ),
                    ui.input_action_button(
                        "btn_run_het",
                        "🚀 Calculate Heterogeneity",
                        class_="btn-primary",
                    ),
                ),
                ui.output_data_frame("tbl_heterogeneity"),
            ),
            ui.nav_panel(
                "ℹ️ Reference",
                ui.markdown("""
                    ### Advanced Inference Reference
                    
                    **Mediation Analysis:**
                    * **ACME**: Average Causal Mediation Effect (Indirect Effect)
                    * **ADE**: Average Direct Effect
                    * **Total Effect**: ACME + ADE
                    
                    **Collinearity:**
                    * **VIF > 5-10**: High multicollinearity
                    * **Tolerance < 0.1**: High multicollinearity
                 """),
            ),
        ),
    )


@module.server
def advanced_inference_server(
    input, output, session, df, var_meta, df_matched, is_matched
):
    """Server logic for Advanced Inference."""

    mediation_results = reactive.Value(None)
    vif_results = reactive.Value(None)

    # --- Dataset Selection Logic ---
    @reactive.Calc
    def current_df():
        if is_matched.get() and input.radio_ai_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_matched_info_ai():
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
    def ui_dataset_selector_ai():
        """Render dataset selector radio buttons."""
        if is_matched.get():
            original = df.get()
            matched = df_matched.get()
            original_len = len(original) if original is not None else 0
            matched_len = len(matched) if matched is not None else 0
            return ui.input_radio_buttons(
                "radio_ai_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original ({original_len:,} rows)",
                    "matched": f"✅ Matched ({matched_len:,} rows)",
                },
                selected="original",
                inline=True,
            )
        return None

    @reactive.Effect
    def _update_inputs():
        d = current_df()
        if d is not None:
            cols = d.columns.tolist()
            ui.update_select("med_outcome", choices=cols)
            ui.update_select("med_treatment", choices=cols)
            ui.update_select("med_mediator", choices=cols)
            ui.update_selectize("med_confounders", choices=cols)
            ui.update_selectize("coll_vars", choices=cols)
            ui.update_select("diag_outcome", choices=cols)
            ui.update_select("diag_predictor", choices=cols)
            ui.update_selectize("diag_covariates", choices=cols)

    @reactive.Effect
    @reactive.event(input.btn_run_mediation)
    def _run_mediation():
        if current_df() is None:
            ui.notification_show("Load data first", type="error")
            return

        try:
            results = analyze_mediation(
                data=current_df(),
                outcome=input.med_outcome(),
                treatment=input.med_treatment(),
                mediator=input.med_mediator(),
                confounders=list(input.med_confounders())
                if input.med_confounders()
                else None,
            )
            mediation_results.set(results)
            ui.notification_show("Mediation Analysis Complete", type="message")
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

    @render.ui
    def out_mediation_status():
        if mediation_results.get():
            return ui.div("✅ Analysis Complete", class_="alert alert-success mt-2")
        return None

    @render.data_frame
    def tbl_mediation():
        res = mediation_results.get()
        if res:
            return pd.DataFrame(
                {
                    "Effect": [
                        "Total Effect",
                        "Direct Effect (ADE)",
                        "Indirect Effect (ACME)",
                        "Proportion Mediated",
                    ],
                    "Estimate": [
                        f"{res['total_effect']:.4f}",
                        f"{res['direct_effect']:.4f}",
                        f"{res['indirect_effect']:.4f}",
                        f"{res['proportion_mediated']:.4%}",
                    ],
                }
            )
        return None

    @reactive.Effect
    @reactive.event(input.btn_run_collinearity)
    def _run_collinearity():
        if current_df() is None:
            ui.notification_show("Load data first", type="error")
            return

        predictors = list(input.coll_vars())
        if not predictors:
            ui.notification_show("Select predictors first", type="error")
            return

        try:
            res = calculate_vif(current_df(), predictors)
            vif_results.set(res)
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

    @render.data_frame
    def tbl_vif():
        return vif_results.get()

    @render.ui
    def plot_corr_heatmap():
        """Render correlation heatmap for selected predictors using Plotly."""
        if current_df() is None:
            return None
        predictors = list(input.coll_vars())
        if not predictors or len(predictors) < 2:
            return None

        # Calculate correlation matrix
        d = current_df()[predictors].select_dtypes(include=[np.number]).dropna()
        if d.empty:
            return None

        corr = d.corr()

        # Create interactive heatmap with Plotly
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(title="Correlation Heatmap")

        return plotly_figure_to_html(fig)

    # --- Model Diagnostics Logic ---
    diag_results = reactive.Value(None)
    diag_plot_data = reactive.Value(None)
    cooks_results = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_run_diag)
    def _run_diagnostics():
        if current_df() is None:
            ui.notification_show("Load data first", type="error")
            return

        try:
            d = current_df().dropna()
            y_col = input.diag_outcome()
            x_col = input.diag_predictor()
            covars = list(input.diag_covariates()) if input.diag_covariates() else []

            # Fit OLS for diagnostics
            X = d[[x_col] + covars]
            X = sm.add_constant(X)
            Y = d[y_col]

            # Ensure numeric
            X = X.select_dtypes(include=[np.number])
            Y = pd.to_numeric(Y, errors="coerce")

            model = sm.OLS(Y, X).fit()

            # Run tests
            res_reset = run_reset_test(model)
            res_bp = run_heteroscedasticity_test(model)
            res_cooks = calculate_cooks_distance(model)
            res_plots = get_diagnostic_plot_data(model)

            diag_results.set([res_reset, res_bp])
            diag_plot_data.set(res_plots)
            cooks_results.set(res_cooks)

            ui.notification_show("Diagnostics Complete", type="message")

        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

    @render.data_frame
    def tbl_diagnostics():
        res = diag_results.get()
        if res:
            return pd.DataFrame(res)
        return None

    @render.plot
    def plot_residuals():
        data = diag_plot_data.get()
        if not data:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=data["fitted_values"], y=data["residuals"], ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
        return fig

    @render.plot
    def plot_qq():
        data = diag_plot_data.get()
        if not data:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        # Use standardized residuals for Q-Q plot
        sm.qqplot(np.array(data["std_residuals"]), line="45", fit=True, ax=ax)
        ax.set_title("Q-Q Plot")
        return fig

    @render.data_frame
    def tbl_cooks():
        res = cooks_results.get()
        if not res or not res["influential_points"]:
            return None

        # Show top influential points
        d = current_df().iloc[res["influential_points"]]
        return d.head(10)  # Show top 10

    @render.text
    def txt_cooks_summary():
        res = cooks_results.get()
        if not res:
            return ""
        if res["n_influential"] == 0:
            return "No influential points detected (Cook's D < 4/n)"
        return f"Detected {res['n_influential']} influential points (Cook's D > {res['threshold']:.4f})"

    # --- Heterogeneity Logic ---
    het_results = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.btn_run_het)
    def _run_heterogeneity():
        try:
            eff_str = input.het_effects()
            var_str = input.het_variances()

            if not eff_str or not var_str:
                ui.notification_show("Enter effects and variances", type="error")
                return

            effects = [float(x.strip()) for x in eff_str.split(",") if x.strip()]
            variances = [float(x.strip()) for x in var_str.split(",") if x.strip()]

            if len(effects) != len(variances):
                ui.notification_show("Mismatched lengths", type="error")
                return

            res = calculate_heterogeneity(effects, variances)
            het_results.set(res)

        except ValueError:
            ui.notification_show("Invalid numeric input", type="error")
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

    @render.data_frame
    def tbl_heterogeneity():
        res = het_results.get()
        if res:
            # Format single row DF
            return pd.DataFrame([res])
        return None

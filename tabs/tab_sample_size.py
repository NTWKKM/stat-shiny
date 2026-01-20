from __future__ import annotations

from typing import Any

from shiny import module, reactive, render, ui
from utils import sample_size_lib

@module.ui
def sample_size_ui() -> ui.TagChild:
    return ui.div(
        ui.h4("🔢 Sample Size & Power Calculator"),
        ui.br(),
        ui.navset_card_pill(
            # --- 1. MEANS (T-Test) ---
            ui.nav_panel(
                "📊 Means (T-test)",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Input Parameters"),
                        ui.input_numeric("ss_mean1", "Mean Group 1:", value=0),
                        ui.input_numeric("ss_mean2", "Mean Group 2:", value=5),
                        ui.input_numeric("ss_sd1", "SD Group 1:", value=10),
                        ui.input_numeric("ss_sd2", "SD Group 2:", value=10),
                        ui.input_slider("ss_means_power", "Power (1-β):", min=0.5, max=0.99, value=0.8, step=0.01),
                        ui.input_slider("ss_means_alpha", "Alpha (Sig. Level):", min=0.001, max=0.2, value=0.05, step=0.001),
                        ui.input_numeric("ss_means_ratio", "Ratio (N2/N1):", value=1, min=0.1, step=0.1),
                        ui.input_action_button("btn_calc_means", "🚀 Calculate N", class_="btn-primary w-100"),
                    ),
                    ui.card(
                        ui.card_header("Results"),
                        ui.output_ui("out_means_result")
                    )
                )
            ),
            # --- 2. PROPORTIONS (Chi-Sq) ---
            ui.nav_panel(
                "🎲 Proportions",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Input Parameters"),
                        ui.input_numeric("ss_p1", "Proportion 1 (0-1):", value=0.1, min=0, max=1, step=0.01),
                        ui.input_numeric("ss_p2", "Proportion 2 (0-1):", value=0.2, min=0, max=1, step=0.01),
                        ui.input_slider("ss_props_power", "Power (1-β):", min=0.5, max=0.99, value=0.8, step=0.01),
                        ui.input_slider("ss_props_alpha", "Alpha (Sig. Level):", min=0.001, max=0.2, value=0.05, step=0.001),
                        ui.input_numeric("ss_props_ratio", "Ratio (N2/N1):", value=1, min=0.1, step=0.1),
                        ui.input_action_button("btn_calc_props", "🚀 Calculate N", class_="btn-primary w-100"),
                    ),
                    ui.card(
                        ui.card_header("Results"),
                        ui.output_ui("out_props_result")
                    )
                )
            ),
            # --- 3. SURVIVAL (Log-Rank) ---
            ui.nav_panel(
                "⏳ Survival (Log-Rank)",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Input Parameters"),
                        ui.input_radio_buttons("ss_surv_mode", "Input Mode:", {"hr": "Hazard Ratio", "median": "Median Survival Time"}),
                        ui.panel_conditional(
                            "input.ss_surv_mode == 'hr'",
                            ui.input_numeric("ss_surv_hr", "Hazard Ratio (HR):", value=2.0, min=0.001)
                        ),
                        ui.panel_conditional(
                            "input.ss_surv_mode == 'median'",
                            ui.input_numeric("ss_surv_m1", "Median Survival 1:", value=12),
                            ui.input_numeric("ss_surv_m2", "Median Survival 2:", value=24)
                        ),
                        ui.input_slider("ss_surv_power", "Power (1-β):", min=0.5, max=0.99, value=0.8, step=0.01),
                        ui.input_slider("ss_surv_alpha", "Alpha (Sig. Level):", min=0.001, max=0.2, value=0.05, step=0.001),
                        ui.input_numeric("ss_surv_ratio", "Ratio (N2/N1):", value=1, min=0.1, step=0.1),
                        ui.input_action_button("btn_calc_surv", "🚀 Calculate Events", class_="btn-primary w-100"),
                    ),
                    ui.card(
                        ui.card_header("Results"),
                        ui.output_ui("out_surv_result"),
                        ui.markdown("*Note: This calculates required number of EVENTS, not total subjects.*")
                    )
                )
            ),
             # --- 4. CORRELATION ---
            ui.nav_panel(
                "📈 Correlation",
                 ui.layout_columns(
                    ui.card(
                        ui.card_header("Input Parameters"),
                        ui.input_numeric("ss_corr_r", "Expected Correlation (r):", value=0.3, min=-1, max=1, step=0.05),
                        ui.input_slider("ss_corr_power", "Power (1-β):", min=0.5, max=0.99, value=0.8, step=0.01),
                        ui.input_slider("ss_corr_alpha", "Alpha (Sig. Level):", min=0.001, max=0.2, value=0.05, step=0.001),
                        ui.input_action_button("btn_calc_corr", "🚀 Calculate N", class_="btn-primary w-100"),
                    ),
                    ui.card(
                        ui.card_header("Results"),
                        ui.output_ui("out_corr_result")
                    )
                 )
            )
        )
    )

@module.server
def sample_size_server(input: Any, output: Any, session: Any) -> None:
    
    # --- RESULT HOLDERS ---
    res_means = reactive.Value(None)
    res_props = reactive.Value(None)
    res_surv = reactive.Value(None)
    res_corr = reactive.Value(None)

    # --- CALCULATORS ---
    
    @reactive.Effect
    @reactive.event(input.btn_calc_means)
    def _calc_means():
        try:
            res = sample_size_lib.calculate_sample_size_means(
                power=input.ss_means_power(),
                ratio=input.ss_means_ratio(),
                mean1=input.ss_mean1(),
                mean2=input.ss_mean2(),
                sd1=input.ss_sd1(),
                sd2=input.ss_sd2(),
                alpha=input.ss_means_alpha()
            )
            res_means.set(res)
        except Exception as e:
            res_means.set(f"Error: {e}")

    @reactive.Effect
    @reactive.event(input.btn_calc_props)
    def _calc_props():
        try:
            res = sample_size_lib.calculate_sample_size_proportions(
                power=input.ss_props_power(),
                ratio=input.ss_props_ratio(),
                p1=input.ss_p1(),
                p2=input.ss_p2(),
                alpha=input.ss_props_alpha()
            )
            res_props.set(res)
        except Exception as e:
            res_props.set(f"Error: {e}")

    @reactive.Effect
    @reactive.event(input.btn_calc_surv)
    def _calc_surv():
        try:
            h0 = input.ss_surv_hr() if input.ss_surv_mode() == 'hr' else input.ss_surv_m1()
            h1 = 0 if input.ss_surv_mode() == 'hr' else input.ss_surv_m2()
            
            res = sample_size_lib.calculate_sample_size_survival(
                power=input.ss_surv_power(),
                ratio=input.ss_surv_ratio(),
                h0=h0,
                h1=h1,
                alpha=input.ss_surv_alpha(),
                mode=input.ss_surv_mode()
            )
            res_surv.set(res)
        except Exception as e:
            res_surv.set(f"Error: {e}")

    @reactive.Effect
    @reactive.event(input.btn_calc_corr)
    def _calc_corr():
        try:
            res = sample_size_lib.calculate_sample_size_correlation(
                power=input.ss_corr_power(),
                r=input.ss_corr_r(),
                alpha=input.ss_corr_alpha()
            )
            res_corr.set(res)
        except Exception as e:
            res_corr.set(f"Error: {e}")

    # --- RENDERERS ---

    def _render_n_result(res):
        if res is None:
            return ui.p("Enter parameters and click Calculate.", style="color: grey;")
        if isinstance(res, str): # Error
            return ui.div(res, class_="text-danger")
        
        return ui.div(
            ui.h2(f"Total N = {int(res['total'])}", class_="text-primary text-center"),
            ui.hr(),
            ui.p(f"Group 1 (n1): {int(res['n1'])}"),
            ui.p(f"Group 2 (n2): {int(res['n2'])}"),
            style="background: #f8f9fa; padding: 20px; border-radius: 10px;"
        )

    @render.ui
    def out_means_result():
        return _render_n_result(res_means.get())

    @render.ui
    def out_props_result():
        return _render_n_result(res_props.get())
        
    @render.ui
    def out_surv_result():
        res = res_surv.get()
        if res is None:
            return ui.p("Enter parameters and click Calculate.", style="color: grey;")
        if isinstance(res, str):
            return ui.div(res, class_="text-danger")
            
        return ui.div(
            ui.h2(f"Total Events = {int(res['total_events'])}", class_="text-primary text-center"),
            ui.p(f"Hazard Ratio detected: {res['hr']:.2f}", class_="text-center text-muted"),
            style="background: #f8f9fa; padding: 20px; border-radius: 10px;"
        )

    @render.ui
    def out_corr_result():
        res = res_corr.get()
        if res is None:
            return ui.p("Enter parameters and click Calculate.", style="color: grey;")
        if isinstance(res, str):
            return ui.div(res, class_="text-danger")
            
        return ui.div(
            ui.h2(f"Total N = {int(res)}", class_="text-primary text-center"),
            style="background: #f8f9fa; padding: 20px; border-radius: 10px;"
        )

from shiny import ui, reactive, render, module
from config import CONFIG
from logger import get_logger
from tabs._common import get_color_palette
from typing import Any

logger = get_logger(__name__)
COLORS = get_color_palette()

@module.ui
def advanced_stats_ui() -> ui.TagChild:
    """
    UI for Advanced Statistics Module.
    """
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Statistical Corrections"),
            
            # Section 1: MCC
            ui.h6("🔹 Multiple Comparison Correction"),
            ui.input_switch("mcc_enable", "Enable MCC", value=CONFIG.get('stats.mcc_enable', False)),
            ui.panel_conditional(
                "input.mcc_enable",
                ui.input_radio_buttons(
                    "mcc_method", "Method",
                    choices={
                        "bonferroni": "Bonferroni",
                        "holm": "Holm",
                        "fdr_bh": "Benjamini-Hochberg (FDR)",
                        "sidak": "Sidak"
                    },
                    selected=CONFIG.get('stats.mcc_method', 'fdr_bh')
                ),
                ui.input_select(
                    "mcc_alpha", "Significance Level (Alpha)",
                    choices=["0.01", "0.05", "0.10"],
                    selected=str(CONFIG.get('stats.mcc_alpha', '0.05'))
                )
            ),
            
            ui.br(),

            # Section 2: VIF
            ui.h6("🔹 Collinearity (VIF)"),
            ui.input_switch("vif_enable", "Enable VIF Check", value=CONFIG.get('stats.vif_enable', False)),
            ui.panel_conditional(
                "input.vif_enable",
                ui.input_slider(
                    "vif_threshold", "VIF Threshold",
                    min=2, max=20, value=CONFIG.get('stats.vif_threshold', 10), step=1
                )
            ),

            ui.br(),

            # Section 3: CI Method
            ui.h6("🔹 Confidence Intervals"),
            ui.input_radio_buttons(
                "ci_method", "Method",
                choices={
                    "wald": "Wald",
                    "profile": "Profile Likelihood",
                    "exact": "Exact (where applicable)"
                },
                selected=CONFIG.get('stats.ci_method', 'wald')
            ),
            
            ui.br(),
            
            ui.input_action_button("btn_save_stats", "💾 Save Stats Settings", class_="btn-primary", width="100%"),
            
            width=300,
            bg=COLORS['smoke_white']
        ),
        
        ui.card(
            ui.card_header("📋 Analysis Log & Guide"),
            ui.layout_columns(
                ui.div(
                    ui.h6("Recent Settings Used"),
                    ui.output_text("txt_stats_summary"),
                    class_="p-3 border rounded bg-light"
                ),
                ui.div(
                    ui.markdown("""
                    ### Guide
                    - **MCC**: Adjusts P-values to control family-wise error rate or FDR.
                        - *Bonferroni*: Conservative.
                        - *FDR (BH)*: Good balance for discovery.
                    - **VIF**: Detects multicollinearity.
                        - *VIF > 10*: High collinearity (consider removing variable).
                    """),
                    class_="p-3"
                ),
                col_widths=(6, 6)
            ),
            full_screen=True
        )
    )

@module.server
def advanced_stats_server(input, output, session, config: Any):
    """
    Server logic for Advanced Statistics Module.
    """
    
    @render.text
    def txt_stats_summary():
        mcc = "ON" if input.mcc_enable() else "OFF"
        vif = "ON" if input.vif_enable() else "OFF"
        return f"""
        MCC Status: {mcc}
        Method: {input.mcc_method()} (Alpha: {input.mcc_alpha()})
        VIF Check: {vif} (Threshold: {input.vif_threshold()})
        CI Method: {input.ci_method()}
        """

    @reactive.Effect
    @reactive.event(input.btn_save_stats)
    def _save_stats_settings():
        try:
            config.update('stats.mcc_enable', input.mcc_enable())
            config.update('stats.mcc_method', input.mcc_method())
            config.update('stats.mcc_alpha', float(input.mcc_alpha()))
            config.update('stats.vif_enable', input.vif_enable())
            config.update('stats.vif_threshold', int(input.vif_threshold()))
            config.update('stats.ci_method', input.ci_method())
            
            logger.info("✅ Advanced stats settings saved")
            ui.notification_show("✅ Advanced stats settings saved", type="message")
        except Exception as e:
            logger.exception("Error saving stats settings")
            ui.notification_show(f"❌ Error: {e}", type="error")

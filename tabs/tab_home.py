from shiny import module, ui


@module.ui
def home_ui():
    return ui.div(
        # Features Grid
        ui.div(
            ui.div(
                # Card: Data
                _feature_card(
                    "📁 Data Management",
                    "Import datasets (CSV, Excel), view summaries, and cleaning.",
                    "1. Data",
                    tab_value="data",
                ),
                # Card: Table 1
                _feature_card(
                    "📋 Table 1 & Matching",
                    "Generate baseline tables and perform propensity score matching.",
                    "2. Table 1",
                    tab_value="bm",
                ),
                # Card: General Stats
                _feature_card(
                    "📊 General Statistics",
                    "Diagnostic tests, Correlation, Agreement (Kappa, Bland-Altman).",
                    "3. General Stats",
                    tab_value="Diagnostic Tests",
                ),
                # Card: Modeling
                _feature_card(
                    "🔬 Advanced Modeling",
                    "Regression (Linear, Logistic, Firth), Survival, Advanced Inference.",
                    "4. Modeling",
                    tab_value="Regression Analysis",
                ),
                # Card: Clinical
                _feature_card(
                    "🏥 Clinical Tools",
                    "Sample Size Calculator, Causal Inference methods.",
                    "5. Clinical",
                    tab_value="Sample Size Calculator",
                ),
                # Card: Settings
                _feature_card(
                    "⚙️ Settings",
                    "Configure application preferences and defaults.",
                    "6. Settings",
                    tab_value="settings",
                ),
                style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px;",
            ),
            class_="app-container",  # Re-use container padding if needed, or just div
            style="padding: 24px 0;",
        ),
    )


def _feature_card(title, description, subtitle, tab_value=None):
    onclick_js = ""
    if tab_value:
        onclick_js = f"var el = document.querySelector('.navbar-nav .nav-link[data-value=\\'{tab_value}\\']'); if (el) el.click();"

    return ui.div(
        ui.div(
            subtitle,
            style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-bottom: 8px; font-weight: 600;",
        ),
        ui.h4(title, style="margin-top: 0; margin-bottom: 12px; font-size: 18px;"),
        ui.p(description, style="color: #555; font-size: 14px; margin-bottom: 0;"),
        class_="card-body feature-card",
        style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; height: 100%; transition: transform 0.2s, box-shadow 0.2s; cursor: pointer;",
        onclick=onclick_js,
    )


@module.server
def home_server(input, output, session):
    pass

from shiny import module, ui

from tabs._styling import get_color_code


@module.ui
def home_ui():
    primary_color = get_color_code("primary")
    primary_dark = get_color_code("primary_dark")

    return ui.div(
        # Hero Section
        ui.div(
            ui.h1("Medical Stat Tool", style="color: white; margin-bottom: 16px;"),
            ui.p(
                "A comprehensive platform for clinical data analysis, visualization, and advanced statistical modeling.",
                class_="lead",
                style="color: rgba(255,255,255,0.9); font-size: 18px; margin-bottom: 32px; max-width: 800px; margin-left: auto; margin-right: auto;",
            ),
            # Note: Navigation links in Shiny without Router/JS is tricky, these serve as visual anchors or we can use custom JS to switch tabs
            # For now, just styling.
            class_="hero-section",
            style=f"text-align: center; padding: 60px 24px; background: linear-gradient(135deg, {primary_color} 0%, {primary_dark} 100%); border-radius: 16px; margin-bottom: 40px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);",
        ),
        # Features Grid
        ui.div(
            ui.h3(
                "Available Modules",
                class_="results-title",
                style="text-align: center; margin-bottom: 32px;",
            ),
            ui.div(
                # Card: Data
                _feature_card(
                    "📁 Data Management",
                    "Import datasets (CSV, Excel), view summaries, and cleaning.",
                    "1. Data",
                ),
                # Card: Table 1
                _feature_card(
                    "📋 Table 1 & Matching",
                    "Generate baseline tables and perform propensity score matching.",
                    "2. Table 1",
                ),
                # Card: General Stats
                _feature_card(
                    "📊 General Statistics",
                    "Diagnostic tests, Correlation, Agreement (Kappa, Bland-Altman).",
                    "3. General Stats",
                ),
                # Card: Modeling
                _feature_card(
                    "🔬 Advanced Modeling",
                    "Regression (Linear, Logistic, Firth), Survival, Advanced Inference.",
                    "4. Modeling",
                ),
                # Card: Clinical
                _feature_card(
                    "🏥 Clinical Tools",
                    "Sample Size Calculator, Causal Inference methods.",
                    "5. Clinical",
                ),
                # Card: Settings
                _feature_card(
                    "⚙️ Settings",
                    "Configure application preferences and defaults.",
                    "6. Settings",
                ),
                style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px;",
            ),
            class_="app-container",  # Re-use container padding if needed, or just div
        ),
        # Workflow Guide
        ui.div(
            ui.h3(
                "Recommended Workflow",
                class_="results-title",
                style="text-align: center; margin-top: 48px; margin-bottom: 32px;",
            ),
            ui.div(
                _step_card(
                    1, "Import Data", "Upload your dataset in Data Management tab."
                ),
                ui.div(
                    "→",
                    style="font-size: 24px; color: #ccc; align-self: center; display: none; @media(min-width: 768px){display: block;}",
                ),
                _step_card(
                    2, "Explore & Clean", "Check missing values and variable types."
                ),
                ui.div(
                    "→",
                    style="font-size: 24px; color: #ccc; align-self: center; display: none; @media(min-width: 768px){display: block;}",
                ),
                _step_card(
                    3, "Generate Table 1", "Create baseline characteristics table."
                ),
                ui.div(
                    "→",
                    style="font-size: 24px; color: #ccc; align-self: center; display: none; @media(min-width: 768px){display: block;}",
                ),
                _step_card(4, "Analyze", "Run regression, survival, or other models."),
                style="display: flex; flex-direction: column; gap: 20px; @media(min-width: 768px){flex-direction: row; justify-content: center;}",
            ),
            style="margin-bottom: 48px;",
        ),
    )


def _feature_card(title, description, subtitle):
    return ui.div(
        ui.div(
            subtitle,
            style="font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-bottom: 8px; font-weight: 600;",
        ),
        ui.h4(title, style="margin-top: 0; margin-bottom: 12px; font-size: 18px;"),
        ui.p(description, style="color: #555; font-size: 14px; margin-bottom: 0;"),
        class_="card-body",
        style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; height: 100%; transition: transform 0.2s, box-shadow 0.2s; cursor: pointer;",
        # Hover effect handling via CSS usually, inline style has limits
    )


def _step_card(number, title, text):
    return ui.div(
        ui.div(
            str(number),
            style="width: 32px; height: 32px; background: #1E3A5F; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-bottom: 12px;",
        ),
        ui.strong(title, style="display: block; margin-bottom: 4px; color: #1E3A5F;"),
        ui.span(text, style="font-size: 13px; color: #666;"),
        style="background: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb; flex: 1; min-width: 200px;",
    )


@module.server
def home_server(input, output, session):
    pass

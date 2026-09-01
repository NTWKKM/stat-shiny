from __future__ import annotations

from shiny import module, ui

from tabs._common import get_color_palette


@module.ui
def home_ui():
    colors = get_color_palette()

    return ui.div(
        ui.div(
            # --- Hero Header Section ---
            ui.div(
                ui.div(
                    ui.span(
                        "🏥 Offline-First Clinical Analytics",
                        class_="badge",
                        style=f"background-color: {colors['primary_light']}; color: {colors['primary']}; border: 1px solid {colors['border']}; padding: 6px 14px; font-size: 13px; font-weight: 600; border-radius: 20px; margin-bottom: 16px; display: inline-block;",
                    ),
                    ui.h1(
                        "Medical Statistical Platform",
                        style=f"color: {colors['primary']}; font-weight: 700; font-size: 28px; margin-bottom: 10px; letter-spacing: -0.02em;",
                    ),
                    ui.p(
                        "Comprehensive biostatistical toolkit for clinical research, observational trials, diagnostic test evaluation, and multivariable modeling.",
                        class_="text-muted",
                        style="max-width: 780px; font-size: 15px; line-height: 1.6; margin-bottom: 24px;",
                    ),
                ),
                class_="home-hero-container mb-4",
            ),
            # --- Features Grid ---
            ui.div(
                ui.h4(
                    "Statistical Analysis Modules",
                    style=f"color: {colors['primary']}; font-weight: 600; font-size: 18px; margin-bottom: 16px; letter-spacing: -0.01em;",
                ),
                ui.div(
                    # Card 1: Data
                    _feature_card(
                        "📁 Data Management",
                        "Import datasets (CSV, Excel), automated type casting, and missing data imputation.",
                        "1. Data Pipeline",
                        tab_value="data",
                    ),
                    # Card 2: Table 1
                    _feature_card(
                        "📋 Table 1 & Matching",
                        "Generate baseline patient tables (SMD, non-parametric) and propensity score matching.",
                        "2. Baseline & Matching",
                        tab_value="bm",
                    ),
                    # Card 3: General Stats
                    _feature_card(
                        "📊 General Statistics",
                        "Diagnostic test accuracy, ROC/AUC curves, correlation matrices, and agreement tests.",
                        "3. General Analytics",
                        tab_value="diagnostic",
                    ),
                    # Card 4: Modeling
                    _feature_card(
                        "🔬 Advanced Modeling",
                        "Multivariable regression (Linear, Logistic, Firth), time-to-event survival, and meta-analysis.",
                        "4. Multivariable Models",
                        tab_value="regression",
                    ),
                    # Card 5: Clinical
                    _feature_card(
                        "🏥 Clinical Tools",
                        "Clinical trial sample size & power estimation, and causal inference methods.",
                        "5. Study Design",
                        tab_value="sample_size",
                    ),
                    # Card 6: Settings
                    _feature_card(
                        "⚙️ Settings",
                        "Configure precision settings, p-value display thresholds, and export formats.",
                        "6. System Config",
                        tab_value="settings",
                    ),
                    class_="home-grid",
                ),
            ),
            class_="app-container",
        ),
    )


def _feature_card(
    title: str, description: str, subtitle: str, tab_value: str | None = None
) -> ui.Tag:
    onclick_js = ""
    if tab_value:
        onclick_js = f"var el = document.querySelector('.navbar-nav [data-value=\\'{tab_value}\\']'); if (el) el.click();"

    return ui.div(
        ui.div(
            subtitle,
            class_="text-muted-sm",
            style="margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 11px;",
        ),
        ui.h4(
            title,
            style="margin-top: 0; margin-bottom: 10px; font-size: 16px; font-weight: 600;",
        ),
        ui.p(
            description,
            class_="text-muted-sm",
            style="font-size: 13px; line-height: 1.5; margin-bottom: 0;",
        ),
        class_="feature-card",
        onclick=onclick_js,
    )


@module.server
def home_server(input, output, session):
    pass

from __future__ import annotations

from shiny import module, ui

from tabs._common import get_color_palette


@module.ui
def home_ui():
    colors = get_color_palette()

    return ui.layout_sidebar(
        # =========================================================================
        # 🤖 LEFT SIDEBAR: AI AGENT SKILLS PROMOTION
        # =========================================================================
        ui.sidebar(
            ui.div(
                ui.div(
                    ui.span(
                        "✨ OPEN SOURCE",
                        class_="badge",
                        style="background-color: #0F172A; color: #FFFFFF; font-size: 10px; font-weight: 700; border-radius: 12px; padding: 3px 8px; margin-bottom: 10px; display: inline-block; letter-spacing: 0.05em;",
                    ),
                    ui.h4(
                        "🤖 Shinystat AI Skills",
                        style=f"color: {colors['primary']}; font-weight: 700; font-size: 16px; margin-bottom: 8px; line-height: 1.3;",
                    ),
                    ui.p(
                        "Autonomous biostatistical intelligence synthesized directly from this platform for modern AI coding agents.",
                        class_="text-muted-sm",
                        style="font-size: 12px; line-height: 1.5; margin-bottom: 14px;",
                    ),
                    # Quick install command box
                    ui.div(
                        ui.div(
                            "⚡ Install with one command:",
                            style="font-size: 11px; font-weight: 600; color: #475569; margin-bottom: 6px;",
                        ),
                        ui.div(
                            ui.code(
                                "npx skills add https://github.com/ntwkkm/shinystat-skills",
                                style="font-size: 11px; background: #0F172A; color: #38BDF8; padding: 8px 10px; border-radius: 6px; display: block; word-break: break-all; font-family: monospace; margin-bottom: 8px; border: 1px solid #1E293B;",
                            ),
                            ui.tags.button(
                                "📋 Copy Command",
                                class_="btn btn-sm btn-outline-primary w-100 mb-2",
                                style="font-size: 11px; padding: 5px 8px; font-weight: 600;",
                                onclick="navigator.clipboard.writeText('npx skills add https://github.com/ntwkkm/shinystat-skills'); this.innerText='✅ Copied!'; var b=this; setTimeout(function(){b.innerText='📋 Copy Command';}, 2000);",
                            ),
                        ),
                    ),
                    ui.hr(style="margin: 14px 0; border-color: #E2E8F0;"),
                    # Agents supported
                    ui.div(
                        ui.div(
                            "Compatible Agents:",
                            style="font-size: 11px; font-weight: 600; color: #475569; margin-bottom: 6px;",
                        ),
                        ui.div(
                            ui.span(
                                "Antigravity",
                                class_="badge me-1 mb-1",
                                style="background: #F1F5F9; color: #0F172A; border: 1px solid #CBD5E1; font-size: 10.5px;",
                            ),
                            ui.span(
                                "Claude Code",
                                class_="badge me-1 mb-1",
                                style="background: #F1F5F9; color: #0F172A; border: 1px solid #CBD5E1; font-size: 10.5px;",
                            ),
                            ui.span(
                                "Cursor",
                                class_="badge me-1 mb-1",
                                style="background: #F1F5F9; color: #0F172A; border: 1px solid #CBD5E1; font-size: 10.5px;",
                            ),
                            ui.span(
                                "Cline / Windsurf",
                                class_="badge me-1 mb-1",
                                style="background: #F1F5F9; color: #0F172A; border: 1px solid #CBD5E1; font-size: 10.5px;",
                            ),
                            ui.span(
                                "Claude.ai Web",
                                class_="badge me-1 mb-1",
                                style="background: #F1F5F9; color: #0F172A; border: 1px solid #CBD5E1; font-size: 10.5px;",
                            ),
                            style="margin-bottom: 12px;",
                        ),
                    ),
                    # Features checklist
                    ui.div(
                        ui.div(
                            "✓ Autonomous model & EPV selection",
                            style="font-size: 11.5px; color: #334155; margin-bottom: 4px;",
                        ),
                        ui.div(
                            "✓ Firth logistic & profile CIs",
                            style="font-size: 11.5px; color: #334155; margin-bottom: 4px;",
                        ),
                        ui.div(
                            "✓ Restricted cubic splines (RCS)",
                            style="font-size: 11.5px; color: #334155; margin-bottom: 4px;",
                        ),
                        ui.div(
                            "✓ 300 DPI publication plots",
                            style="font-size: 11.5px; color: #334155; margin-bottom: 4px;",
                        ),
                        ui.div(
                            "✓ NEJM/JAMA 3-rule tables",
                            style="font-size: 11.5px; color: #334155; margin-bottom: 4px;",
                        ),
                        ui.div(
                            "✓ PSM matching & MICE imputation",
                            style="font-size: 11.5px; color: #334155; margin-bottom: 14px;",
                        ),
                    ),
                    # GitHub button
                    ui.tags.a(
                        "⭐ View on GitHub →",
                        href="https://github.com/ntwkkm/shinystat-skills",
                        target="_blank",
                        rel="noopener noreferrer",
                        class_="btn btn-sm btn-dark w-100",
                        style="background: #0F172A; color: #FFFFFF; font-size: 11.5px; padding: 6px 10px; font-weight: 600; text-decoration: none; border-radius: 6px; display: block; text-align: center;",
                    ),
                    class_="p-2",
                ),
            ),
            width=290,
            bg=colors["smoke_white"],
            title="AI Agent Skills",
        ),
        # =========================================================================
        # 🏠 MAIN CONTENT
        # =========================================================================
        ui.div(
            # --- 1. Hero Header Section ---
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
            # --- 2. Featured AI Agent Skills Showcase Section ---
            ui.div(
                ui.div(
                    ui.div(
                        ui.span(
                            "🤖 AI-NATIVE BIOSTATISTICS",
                            class_="badge",
                            style="background-color: #0F172A; color: #FFFFFF; font-size: 11px; font-weight: 700; border-radius: 12px; padding: 4px 10px; margin-bottom: 10px; display: inline-block; letter-spacing: 0.05em;",
                        ),
                        ui.tags.a(
                            "⭐ GitHub: ntwkkm/shinystat-skills",
                            href="https://github.com/ntwkkm/shinystat-skills",
                            target="_blank",
                            rel="noopener noreferrer",
                            class_="badge ms-2",
                            style="background-color: #F8FAFC; color: #0F172A; border: 1px solid #CBD5E1; font-size: 11px; font-weight: 600; border-radius: 12px; padding: 4px 10px; display: inline-block; text-decoration: none;",
                        ),
                    ),
                    ui.h3(
                        "Empower Your AI Agents with Shinystat Skills",
                        style=f"color: {colors['primary']}; font-weight: 700; font-size: 22px; margin-bottom: 8px; letter-spacing: -0.01em;",
                    ),
                    ui.p(
                        "The biostatistical decision heuristics, algorithms, and publication renderers from this webapp have been synthesized into open-source agent skills. Bring autonomous biostatistical authority directly into your AI pair programmer with one command:",
                        class_="text-muted",
                        style="font-size: 14px; line-height: 1.6; margin-bottom: 16px; max-width: 880px;",
                    ),
                    # Terminal install bar
                    ui.div(
                        ui.span(
                            "$ npx skills add https://github.com/ntwkkm/shinystat-skills",
                            class_="skill-terminal-code",
                        ),
                        ui.tags.button(
                            "📋 Copy Command",
                            class_="btn btn-sm btn-light",
                            style="font-size: 12px; padding: 4px 10px; font-weight: 600; border-radius: 4px; white-space: nowrap;",
                            onclick="navigator.clipboard.writeText('npx skills add https://github.com/ntwkkm/shinystat-skills'); this.innerText='✅ Copied!'; var b=this; setTimeout(function(){b.innerText='📋 Copy Command';}, 2000);",
                        ),
                        class_="skill-terminal-box",
                    ),
                    # Two Specialized Skills Grid
                    ui.div(
                        # Skill 1: shinystat (Local / CLI)
                        ui.div(
                            ui.div(
                                ui.div(
                                    ui.span(
                                        "⚡ LOCAL & CLI",
                                        class_="badge",
                                        style="background: #0F172A; color: #FFFFFF; font-size: 10px; font-weight: 700; border-radius: 4px; padding: 2px 6px; margin-bottom: 6px; display: inline-block;",
                                    ),
                                    ui.h4(
                                        "shinystat",
                                        style="font-size: 17px; font-weight: 700; margin-bottom: 6px; color: #0F172A;",
                                    ),
                                    ui.p(
                                        "For Antigravity, Claude Code, Cursor, Cline, Codex",
                                        style="font-size: 12px; color: #64748B; margin-bottom: 12px; font-weight: 500;",
                                    ),
                                ),
                                ui.tags.ul(
                                    ui.tags.li(
                                        "Bundled complete medstat CLI calculation engine"
                                    ),
                                    ui.tags.li(
                                        "Autonomous Firth penalized logistic regression with profile likelihood CIs"
                                    ),
                                    ui.tags.li(
                                        "Restricted cubic splines (RCS) & non-linear curves"
                                    ),
                                    ui.tags.li(
                                        "Austin 2009 SMD & Propensity Score Matching (PSM)"
                                    ),
                                    ui.tags.li(
                                        "Austin & Steyerberg 2019 ICI & Vickers DCA Net Benefit"
                                    ),
                                    ui.tags.li(
                                        "300 DPI publication plots & interactive Plotly dual-view HTML reports"
                                    ),
                                    style="font-size: 12.5px; color: #334155; line-height: 1.6; padding-left: 18px; margin-bottom: 14px;",
                                ),
                                ui.div(
                                    ui.code(
                                        "npx skills add https://github.com/ntwkkm/shinystat-skills --skill shinystat",
                                        style="font-size: 10.5px; background: #F1F5F9; color: #0F172A; padding: 6px 8px; border-radius: 4px; display: block; word-break: break-all; margin-bottom: 8px; border: 1px solid #E2E8F0; font-family: monospace;",
                                    ),
                                    ui.tags.button(
                                        "📋 Copy Install Command",
                                        class_="btn btn-sm btn-outline-primary w-100",
                                        style="font-size: 11px; padding: 4px 8px; font-weight: 600;",
                                        onclick="navigator.clipboard.writeText('npx skills add https://github.com/ntwkkm/shinystat-skills --skill shinystat'); this.innerText='✅ Copied!'; var b=this; setTimeout(function(){b.innerText='📋 Copy Install Command';}, 2000);",
                                    ),
                                ),
                                class_="skill-card-sub",
                            ),
                            class_="col-12 col-md-6 mb-3",
                        ),
                        # Skill 2: shinystat-cloud (Cloud / Sandbox)
                        ui.div(
                            ui.div(
                                ui.div(
                                    ui.span(
                                        "☁️ CLOUD SANDBOX",
                                        class_="badge",
                                        style="background: #475569; color: #FFFFFF; font-size: 10px; font-weight: 700; border-radius: 4px; padding: 2px 6px; margin-bottom: 6px; display: inline-block;",
                                    ),
                                    ui.h4(
                                        "shinystat-cloud",
                                        style="font-size: 17px; font-weight: 700; margin-bottom: 6px; color: #0F172A;",
                                    ),
                                    ui.p(
                                        "For Claude.ai Web, Web Chats, Jupyter, Google Colab",
                                        style="font-size: 12px; color: #64748B; margin-bottom: 12px; font-weight: 500;",
                                    ),
                                ),
                                ui.tags.ul(
                                    ui.tags.li(
                                        "Zero-install, standalone pure-Python recipes"
                                    ),
                                    ui.tags.li(
                                        "Runs in Claude Web Analysis Tool without custom dependencies"
                                    ),
                                    ui.tags.li(
                                        "Standard scientific stack: pandas, scipy, statsmodels, lifelines"
                                    ),
                                    ui.tags.li(
                                        "Automated EPV estimation & sparse-data handling"
                                    ),
                                    ui.tags.li(
                                        "Little's MCAR missing data test & MICE Rubin's rules pooling"
                                    ),
                                    ui.tags.li(
                                        "NEJM/JAMA 3-rule tables & multi-format export (HTML, Word, Markdown)"
                                    ),
                                    style="font-size: 12.5px; color: #334155; line-height: 1.6; padding-left: 18px; margin-bottom: 14px;",
                                ),
                                ui.div(
                                    ui.code(
                                        "npx skills add https://github.com/ntwkkm/shinystat-skills --skill shinystat-cloud",
                                        style="font-size: 10.5px; background: #F1F5F9; color: #0F172A; padding: 6px 8px; border-radius: 4px; display: block; word-break: break-all; margin-bottom: 8px; border: 1px solid #E2E8F0; font-family: monospace;",
                                    ),
                                    ui.tags.button(
                                        "📋 Copy Install Command",
                                        class_="btn btn-sm btn-outline-secondary w-100",
                                        style="font-size: 11px; padding: 4px 8px; font-weight: 600;",
                                        onclick="navigator.clipboard.writeText('npx skills add https://github.com/ntwkkm/shinystat-skills --skill shinystat-cloud'); this.innerText='✅ Copied!'; var b=this; setTimeout(function(){b.innerText='📋 Copy Install Command';}, 2000);",
                                    ),
                                ),
                                class_="skill-card-sub",
                            ),
                            class_="col-12 col-md-6 mb-3",
                        ),
                        class_="row",
                    ),
                    # Collapsible Quickstart Guide
                    ui.tags.details(
                        ui.tags.summary(
                            "📖 How to Use in Antigravity, Claude Code, Cursor, and Claude Web",
                            style="font-weight: 600; font-size: 13px; color: #0F172A; cursor: pointer; padding: 10px 0;",
                        ),
                        ui.div(
                            ui.div(
                                ui.h6(
                                    "1. Antigravity & Claude Code (Recommended)",
                                    style="font-weight: 700; font-size: 13px; margin-top: 10px; margin-bottom: 4px;",
                                ),
                                ui.p(
                                    "Open your project terminal and run `npx skills add https://github.com/ntwkkm/shinystat-skills`. Antigravity and Claude Code auto-detect and register the skills into `.agents/skills/`. Prompt your agent naturally: 'Analyze my clinical trial cohort, compute Table 1 with SMDs, and fit a multivariable Cox model with Schoenfeld assumption tests.'",
                                    style="font-size: 12.5px; color: #475569; line-height: 1.5; margin-bottom: 10px;",
                                ),
                                ui.h6(
                                    "2. Cursor & Cline",
                                    style="font-weight: 700; font-size: 13px; margin-bottom: 4px;",
                                ),
                                ui.p(
                                    "Run `npx skills add https://github.com/ntwkkm/shinystat-skills` or copy the `shinystat` directory into `.cursor/skills/` or `.agents/skills/`. Cursor will reference the decision heuristics and recipes automatically during analysis.",
                                    style="font-size: 12.5px; color: #475569; line-height: 1.5; margin-bottom: 10px;",
                                ),
                                ui.h6(
                                    "3. Claude.ai Web Analysis Tool",
                                    style="font-weight: 700; font-size: 13px; margin-bottom: 4px;",
                                ),
                                ui.p(
                                    "For Claude Web without a local terminal, use `shinystat-cloud`. Copy the system prompt into Claude Project Knowledge or Custom Instructions to grant Claude Web the complete biostatistical decision authority.",
                                    style="font-size: 12.5px; color: #475569; line-height: 1.5; margin-bottom: 0;",
                                ),
                                style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 14px 16px; margin-top: 8px;",
                            ),
                        ),
                        style="margin-top: 4px;",
                    ),
                    class_="skill-promo-banner",
                ),
            ),
            # --- 3. Features Grid: Statistical Analysis Modules ---
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

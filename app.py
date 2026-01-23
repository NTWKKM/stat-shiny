from __future__ import annotations

from pathlib import Path
from typing import Any

from shiny import App, Inputs, Outputs, Session, reactive, ui

# Import Config/Logger
from config import CONFIG
from logger import LoggerFactory, get_logger

# Import Tabs Modules (Verified)
from tabs import (
    tab_advanced_inference,
    tab_agreement,  # 🟢 Agreement Module
    tab_baseline_matching,
    tab_causal_inference,
    tab_core_regression,
    tab_corr,
    tab_data,  # 🟢 Data Module
    tab_diag,
    tab_home,  # 🏠 Home Module
    tab_sample_size,  # 🟢 Sample Size Module
    tab_settings,
    tab_survival,
)
from tabs._common import wrap_with_container
from utils.logic import HAS_FIRTH

# ==========================================
# 0. CONSTANTS & CONFIG
# ==========================================


class TabNames:
    HOME = "home"
    DATA = "data"
    BASELINE_MATCHING = "bm"
    DIAGNOSTIC = "Diagnostic Tests"
    CORRELATION = "Correlation Analysis"
    AGREEMENT = "Agreement & Reliability"
    REGRESSION = "Regression Analysis"
    SURVIVAL = "Survival Analysis"
    ADVANCED_REGRESSION = "Advanced Regression"
    CAUSAL = "Causal Methods"
    SAMPLE_SIZE = "Sample Size Calculator"
    SETTINGS = "settings"


# Initialize Logger
LoggerFactory.configure()
logger = get_logger(__name__)

# ==========================================
# 1. UI DEFINITION
# ==========================================

# 🟢 Footer Definition
footer_ui = ui.tags.div(
    ui.HTML(
        """&copy; 2025 <a href="https://github.com/NTWKKM/" target="_blank">NTWKKM n donate</a> | Powered by GitHub, Antigravity, Shiny"""
    ),
    class_="report-footer",
    style="text-align: center; padding: 20px 0; border-top: 1px solid #e5e5e5; margin-top: 40px; color: #666;",
)

app_ui = ui.page_fluid(
    # ♿ Accessibility: Skip Links
    ui.div(
        ui.a("Skip to main content", href="#main-content", class_="skip-link"),
        ui.a("Skip to navigation", href="#main_nav", class_="skip-link"),
        ui.a("Skip to footer", href="#footer", class_="skip-link"),
        class_="skip-links",
    ),
    # ⬇⬇⬇ inject theme CSS (EXTERNAL - Optimized for performance)
    ui.tags.head(
        ui.tags.meta(charset="utf-8"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1.0"),
        # ✅ Google Fonts: Inter
        ui.tags.link(
            href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
            rel="stylesheet",
        ),
        # ✅ Preload CSS for faster loading
        ui.tags.link(rel="preload", href="static/styles.css", as_="style"),
        # ✅ Link to external CSS file
        ui.tags.link(rel="stylesheet", href="static/styles.css"),
        # ✅ Custom JS Handlers
        ui.tags.script(src="static/js/custom_handlers.js"),
        ui.tags.style(".navbar-brand { font-size: 1.5rem !important; }"),
    ),
    ui.page_navbar(
        # ========================================
        # 🏠 TAB 0: HOME
        # ========================================
        ui.nav_panel(
            "🏠 Home",
            wrap_with_container(tab_home.home_ui("home")),
            value=TabNames.HOME,
        ),
        # ========================================
        # 📁 TAB 1: DATA MANAGEMENT (Standalone)
        # ========================================
        ui.nav_panel(
            "📁 Data Management",
            wrap_with_container(tab_data.data_ui("data")),
            value=TabNames.DATA,
        ),
        # ========================================
        # 📋 TAB 2: TABLE 1 & MATCHING (Standalone)
        # ========================================
        ui.nav_panel(
            "📋 Table 1 & Matching",
            wrap_with_container(tab_baseline_matching.baseline_matching_ui("bm")),
            value=TabNames.BASELINE_MATCHING,
        ),
        # ========================================
        # 📊 TAB 3: GENERAL STATISTICS (Dropdown)
        # ========================================
        ui.nav_menu(
            "📊 General Statistics",
            ui.nav_panel(
                "Diagnostic Tests",
                wrap_with_container(tab_diag.diag_ui("diag")),
                value=TabNames.DIAGNOSTIC,
            ),
            ui.nav_panel(
                "Correlation Analysis",
                wrap_with_container(tab_corr.corr_ui("corr")),
                value=TabNames.CORRELATION,
            ),
            ui.nav_panel(
                "Agreement & Reliability",
                wrap_with_container(tab_agreement.agreement_ui("agreement")),
                value=TabNames.AGREEMENT,
            ),
        ),
        # ========================================
        # 🔬 TAB 4: ADVANCED MODELING (Dropdown)
        # ========================================
        ui.nav_menu(
            "🔬 Advanced Statistics",
            ui.nav_panel(
                "Regression Analysis",
                wrap_with_container(tab_core_regression.core_regression_ui("core_reg")),
                value=TabNames.REGRESSION,
            ),
            ui.nav_panel(
                "Survival Analysis",
                wrap_with_container(tab_survival.survival_ui("survival")),
                value=TabNames.SURVIVAL,
            ),
            ui.nav_panel(
                "Advanced Regression",
                wrap_with_container(
                    tab_advanced_inference.advanced_inference_ui("adv_inf")
                ),
                value=TabNames.ADVANCED_REGRESSION,
            ),
        ),
        # ========================================
        # 🏥 TAB 5: CLINICAL TOOLS (Dropdown)
        # ========================================
        ui.nav_menu(
            "🏥 Clinical Research Tools",
            ui.nav_panel(
                "Sample Size Calculator",
                wrap_with_container(tab_sample_size.sample_size_ui("sample_size")),
                value=TabNames.SAMPLE_SIZE,
            ),
            ui.nav_panel(
                "Causal Methods",
                wrap_with_container(tab_causal_inference.causal_inference_ui("causal")),
                value=TabNames.CAUSAL,
            ),
        ),
        # ========================================
        # ⚙️ TAB 6: SETTINGS (Standalone)
        # ========================================
        ui.nav_panel(
            "⚙️ System Settings",
            wrap_with_container(tab_settings.settings_ui("settings")),
            value=TabNames.SETTINGS,
        ),
        title=ui.tags.a(
            "🏥 Medical Stat Tool",
            href="#",
            id="navbar_brand_home",
            style="color: var(--color-primary); font-weight: 700; text-decoration: none;",
        ),
        id="main_nav",
        header=ui.tags.div(
            ui.tags.div(id="main-content"),
        ),
        footer=ui.div(
            footer_ui,
            class_="main-footer",
            id="footer",
            role="contentinfo",
        ),
    ),
    title=CONFIG.get("ui.page_title", "Medical Stat Tool"),
)


# ==========================================
# 2. SERVER LOGIC
# ==========================================
def server(input: Inputs, output: Outputs, session: Session) -> None:
    logger.info("📱 Shiny app session started")

    # --- Reactive State (Global) ---

    # Type hints added for better clarity, though specific DataFrame type isn't enforceble at runtime
    df: reactive.Value[Any | None] = reactive.Value(None)
    var_meta: reactive.Value[dict[str, Any]] = reactive.Value({})
    uploaded_file_info: reactive.Value[dict[str, Any] | None] = reactive.Value(None)

    # Matched data state (Shared across tabs)
    df_matched: reactive.Value[Any | None] = reactive.Value(None)
    is_matched: reactive.Value[bool] = reactive.Value(False)
    matched_treatment_col: reactive.Value[str | None] = reactive.Value(None)
    matched_covariates: reactive.Value[list[str]] = reactive.Value([])

    # --- Helper: Check Dependencies ---
    def check_optional_deps() -> None:
        # Check from HAS_FIRTH variable prepared in logic.py
        if HAS_FIRTH:
            logger.info("Optional dependencies: firth=True")
        else:
            logger.warning("Optional dependencies: firth=False")
            ui.notification_show("⚠️ Firth regression unavailable", type="warning")

    check_optional_deps()

    # ==========================================
    # 3. CALL MODULES SERVER
    # ==========================================

    # ==========================================
    # 3. CALL MODULES SERVER (EAGER LOADING)
    # ==========================================

    # --- Eager Loading (Core & Statistical Tabs) ---
    tab_home.home_server("home")

    tab_data.data_server(
        "data",
        df,
        var_meta,
        uploaded_file_info,
        df_matched,
        is_matched,
        matched_treatment_col,
        matched_covariates,
    )

    tab_baseline_matching.baseline_matching_server(
        "bm",
        df,
        var_meta,
        df_matched,
        is_matched,
        matched_treatment_col,
        matched_covariates,
    )

    tab_settings.settings_server("settings", CONFIG)

    # Statistical & Clinical Modules (Eager)
    tab_diag.diag_server("diag", df, var_meta, df_matched, is_matched)
    tab_corr.corr_server("corr", df, var_meta, df_matched, is_matched)
    tab_agreement.agreement_server("agreement", df, var_meta, df_matched, is_matched)
    tab_core_regression.core_regression_server(
        "core_reg", df, var_meta, df_matched, is_matched
    )
    tab_survival.survival_server("survival", df, var_meta, df_matched, is_matched)
    tab_advanced_inference.advanced_inference_server(
        "adv_inf", df, var_meta, df_matched, is_matched
    )
    tab_causal_inference.causal_inference_server(
        "causal", df, var_meta, df_matched, is_matched
    )
    tab_sample_size.sample_size_server("sample_size")


# ==========================================
# 4. APP LAUNCHER
# ==========================================

# Define explicit path to static folder
static_assets_path = Path(__file__).parent / "static"

# ✅ Create Shiny app instance
# Specify static_assets so /static/styles.css works on Cloud (Posit Connect)
app = App(
    app_ui,
    server,
    # 🟢 Mount static directory to /static path
    # This ensures that when we link to "static/styles.css", it resolves correctly
    # in both local/Posit (where it's handled here) and Docker/ASGI (where Starlette handles it, but this doesn't hurt)
    static_assets={"/static": str(static_assets_path)},
)

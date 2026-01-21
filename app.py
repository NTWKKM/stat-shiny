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
        ui.a("Skip to sidebar navigation", href="#main-nav", class_="skip-link"),
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
    ),
    # Header with Title
    ui.row(
        ui.column(
            12,
            ui.h2(
                "🏥 Medical Stat Tool",
                class_="app-title mt-3 mb-4 ps-3",
                style="color: var(--primary); font-weight: 700;",
            ),
        )
    ),
    # Mobile Menu Toggle (Visible only on mobile)
    ui.div(
        ui.input_action_button(
            "mobile_menu_btn", "☰ Menu", class_="btn-sm w-100 mb-2"
        ),
        class_="d-md-none px-3 pt-2",
    ),
    # Main Navigation (Sticky Sidebar) wrapped in ID for skip link
    ui.div(
        ui.navset_pill_list(
            # ========================================
            # 🏠 TAB 0: HOME
            # ========================================
            ui.nav_panel(
                "🏠 Home",
                wrap_with_container(tab_home.home_ui("home")),
            ),
            # ========================================
            # 📁 TAB 1: DATA MANAGEMENT (Standalone)
            # ========================================
            ui.nav_panel(
                "📁 Data",
                wrap_with_container(tab_data.data_ui("data")),
            ),
            # ========================================
            # 📋 TAB 2: TABLE 1 & MATCHING (Standalone)
            # ========================================
            ui.nav_panel(
                "📋 Table 1",
                wrap_with_container(tab_baseline_matching.baseline_matching_ui("bm")),
            ),
            # ========================================
            # 📊 TAB 3: GENERAL STATISTICS (Dropdown)
            # ========================================
            ui.nav_menu(
                "📊 General Stats",
                ui.nav_panel(
                    "Diagnostic Tests",
                    wrap_with_container(tab_diag.diag_ui("diag")),
                ),
                ui.nav_panel(
                    "Correlation Analysis",
                    wrap_with_container(tab_corr.corr_ui("corr")),
                ),
                ui.nav_panel(
                    "Agreement & Reliability",
                    wrap_with_container(tab_agreement.agreement_ui("agreement")),
                ),
            ),
            # ========================================
            # 🔬 TAB 4: ADVANCED MODELING (Dropdown)
            # ========================================
            ui.nav_menu(
                "🔬 Modeling",
                ui.nav_panel(
                    "Regression Analysis",
                    wrap_with_container(
                        tab_core_regression.core_regression_ui("core_reg")
                    ),
                ),
                ui.nav_panel(
                    "Survival Analysis",
                    wrap_with_container(tab_survival.survival_ui("survival")),
                ),
                ui.nav_panel(
                    "Advanced Regression",
                    wrap_with_container(
                        tab_advanced_inference.advanced_inference_ui("adv_inf")
                    ),
                ),
            ),
            # ========================================
            # 🏥 TAB 5: CLINICAL TOOLS (Dropdown)
            # ========================================
            ui.nav_menu(
                "🏥 Clinical",
                ui.nav_panel(
                    "Sample Size Calculator",
                    wrap_with_container(tab_sample_size.sample_size_ui("sample_size")),
                ),
                ui.nav_panel(
                    "Causal Methods",
                    wrap_with_container(
                        tab_causal_inference.causal_inference_ui("causal")
                    ),
                ),
            ),
            # ========================================
            # ⚙️ TAB 6: SETTINGS (Standalone)
            # ========================================
            ui.nav_panel(
                "⚙️ Settings",
                wrap_with_container(tab_settings.settings_ui("settings")),
            ),
            id="main_nav",
            widths=(2, 10),
            well=False,
        ),
        id="main-content",
        role="main",
    ),
    # Footer
    ui.div(
        footer_ui,
        class_="main-footer",
        style="margin-left: 16.66666667%;",  # Offset for sidebar (2/12 = 16.67%)
        id="footer",
        role="contentinfo",
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
    # 3. CALL MODULES SERVER (LAZY LOADING)
    # ==========================================

    # --- Eager Loading (Core Tabs) ---
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

    # --- Lazy Loading (Heavy/Secondary Tabs) ---
    # Store loaded state in session to prevent re-initialization
    if "loaded_modules" not in session.user_data:
        session.user_data["loaded_modules"] = set()

    @reactive.Effect
    def _lazy_load_tabs():
        current_tab = input.main_nav()
        loaded = session.user_data["loaded_modules"]

        # Helper to load module once
        def load_module(name, loader_func):
            if name not in loaded:
                with ui.Progress(min=0, max=1) as p:
                    p.set(message=f"Loading {name} module...", detail="This may take a moment")
                    logger.info(f"⚡ Lazy Loading Module: {name}")
                    loader_func()
                    loaded.add(name)

        # Map tab names (from UI) to module loaders
        # Note: Tab names must match ui.nav_panel titles EXACTLY
        
        # General Stats
        if current_tab == "Diagnostic Tests":
            load_module("diag", lambda: tab_diag.diag_server("diag", df, var_meta, df_matched, is_matched))
            
        elif current_tab == "Correlation Analysis":
            load_module("corr", lambda: tab_corr.corr_server("corr", df, var_meta, df_matched, is_matched))
            
        elif current_tab == "Agreement & Reliability":
            load_module("agreement", lambda: tab_agreement.agreement_server("agreement", df, var_meta, df_matched, is_matched))

        # Advanced Modeling
        elif current_tab == "Regression Analysis":
            load_module("core_reg", lambda: tab_core_regression.core_regression_server("core_reg", df, var_meta, df_matched, is_matched))
            
        elif current_tab == "Survival Analysis":
            load_module("survival", lambda: tab_survival.survival_server("survival", df, var_meta, df_matched, is_matched))
            
        elif current_tab == "Advanced Regression":
            load_module("adv_inf", lambda: tab_advanced_inference.advanced_inference_server("adv_inf", df, var_meta, df_matched, is_matched))
            
        # Clinical
        elif current_tab == "Causal Methods":
            load_module("causal", lambda: tab_causal_inference.causal_inference_server("causal", df, var_meta, df_matched, is_matched))
            
        elif current_tab == "Sample Size Calculator":
            load_module("sample_size", lambda: tab_sample_size.sample_size_server("sample_size"))


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

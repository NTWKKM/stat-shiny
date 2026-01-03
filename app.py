from shiny import App, ui, reactive, Session, render
import html

# Import Config/Logger
from config import CONFIG
from logger import get_logger, LoggerFactory

# Import Tabs Modules
from tabs import tab_data            # 🟢 Data Module
from tabs import tab_baseline_matching
from tabs import tab_diag
from tabs import tab_logit
from tabs import tab_corr
from tabs import tab_survival
from tabs import tab_settings

from tabs._styling import get_shiny_css
from tabs._common import get_color_palette

# === LAYER 1, 2, 3: Import optimization managers ===
from utils.cache_manager import COMPUTATION_CACHE
from utils.memory_manager import MEMORY_MANAGER
from utils.connection_handler import CONNECTION_HANDLER

# Initialize Logger
LoggerFactory.configure()
logger = get_logger(__name__)

# === LAYER 2 & 3: Initialize optimization systems ===
logger.info("🚀 Initializing HF optimization layers...")
logger.info("  %s", COMPUTATION_CACHE)      # Layer 1: Caching
logger.info("  %s", MEMORY_MANAGER)         # Layer 2: Memory Mgmt
logger.info("  %s", CONNECTION_HANDLER)     # Layer 3: Connection Resilience

# Get color palette for navbar styling
colors = get_color_palette()

# ==========================================
# 1. UI DEFINITION
# ==========================================
app_ui = ui.page_navbar(
    # --- 1. Data Management Module ---
    # ✅ เรียกใช้ tab_data.data_ui("data") โดยตรง (มี nav_panel อยู่ใน module แล้ว)
    tab_data.data_ui("data"),
    
    # --- 2. Table 1 & Matching Module ---
    ui.nav_panel(
        "📋 Table 1 & Matching", 
        ui.div(tab_baseline_matching.baseline_matching_ui("bm"), class_="app-container")
    ),

    # --- 3. Diagnostic Tests Module ---
    ui.nav_panel(
        "🧪 Diagnostic Tests", 
        ui.div(tab_diag.diag_ui("diag"), class_="app-container")
    ),

    # --- 4. Logistic Regression Module ---
    ui.nav_panel(
        "📊 Risk Factors", 
        ui.div(tab_logit.logit_ui("logit"), class_="app-container")
    ),

    # --- 5. Correlation & ICC Module ---
    ui.nav_panel(
        "📈 Correlation & ICC", 
        ui.div(tab_corr.corr_ui("corr"), class_="app-container")
    ),

    # --- 6. Survival Analysis Module ---
    ui.nav_panel(
        "⏳ Survival Analysis", 
        ui.div(tab_survival.survival_ui("survival"), class_="app-container")
    ),

    # --- 7. Settings Module ---
    ui.nav_panel(
        "⚙️ Settings", 
        ui.div(tab_settings.settings_ui("settings"), class_="app-container")
    ),

    # === LAYER 2 & 3: Dynamic Status Badge ===
    footer=ui.div(
        ui.output_ui("optimization_status"),
        style="padding: 10px; border-top: 1px solid #eee; margin-top: 20px;"
    ),

    title=CONFIG.get('ui.page_title', 'Medical Stat Tool'),
    id="main_navbar",
    window_title="Medical Stat Tool",

    header=ui.tags.head(
        ui.HTML(get_shiny_css())
    ),
)

# ==========================================
# 2. SERVER LOGIC
# ==========================================
def server(input, output, session: Session):
    logger.info("📱 Shiny app session started")
    logger.info("💾 Cache stats: %s", COMPUTATION_CACHE.get_stats())
    logger.info("🧠 Memory status: %s", MEMORY_MANAGER.get_memory_status())

    # --- Reactive State (Global) ---
    df = reactive.value(None)
    var_meta = reactive.value({})
    uploaded_file_info = reactive.value(None)
    
    # Matched data state (Shared across tabs)
    df_matched = reactive.value(None)
    is_matched = reactive.value(False)
    matched_treatment_col = reactive.value(None)
    matched_covariates = reactive.value([])

    # === 🚀 DYNAMIC STATUS BADGE LOGIC ===
    @render.ui
    def optimization_status():
        """แสดงสถานะ optimization layers (ไม่ต้อง refresh อัตโนมัติ)"""

        # 1. Cache status (L1)
        cache_stats = COMPUTATION_CACHE.get_stats()
        cache_icon = "🟢" if cache_stats.get("cached_items", 0) > 0 else "⚪"

        # 2. Memory status (L2)
        mem_status = MEMORY_MANAGER.get_memory_status()
        mem_icon = "💗"
        if mem_status.get("status") == "WARNING":
            mem_icon = "💛"
        elif mem_status.get("status") == "CRITICAL":
            mem_icon = "🔴"
        elif mem_status.get("status") == "UNKNOWN":
            mem_icon = "⚪"

        # 3. Connection status (L3)
        conn_stats = CONNECTION_HANDLER.get_stats()
        try:
            rate_str = str(conn_stats.get("success_rate", "0"))
            success_val = float(rate_str.replace("%", ""))
        except (ValueError, TypeError, AttributeError):
            success_val = 0.0
        conn_icon = "🟢" if success_val >= 90 else "🟠" if success_val >= 70 else "🔴"

        import html as _html

        cache_title = _html.escape(
            f"Cache: {cache_stats.get('cached_items', 0)}/{cache_stats.get('max_size', 0)} items "
            f"(Hit rate: {cache_stats.get('hit_rate', '0%')})"
        )

        usage_pct = mem_status.get("usage_pct")
        current_mb = mem_status.get("current_mb")
        max_mb = mem_status.get("max_mb", "N/A")
        if usage_pct is not None and current_mb is not None:
            mem_title = _html.escape(
                f"Memory: {usage_pct:.1f}% ({current_mb}MB / {max_mb}MB)"
            )
        else:
            mem_title = _html.escape(f"Memory: Unknown ({max_mb}MB max)")

        conn_title = _html.escape(
            f"Resilience: {conn_stats.get('success_rate', 'N/A')} success rate "
            f"({conn_stats.get('failed_attempts', 0)} failures)"
        )

        return ui.HTML(
            f"""
            <div style="display:flex; gap:16px; align-items:center; font-size:13px;">
              <span title="{cache_title}">{cache_icon} Cache</span>
              <span title="{mem_title}">{mem_icon} Memory</span>
              <span title="{conn_title}">{conn_icon} Resilience</span>
            </div>
            """
        )

    # --- Helper: Check Dependencies ---
    def check_optional_deps() -> None:
        """ตรวจสอบ optional dependencies"""
        deps_status = {}
        try:
            import firthlogist
            deps_status['firth'] = {'installed': True, 'msg': '✅ Firth regression enabled'}
        except ImportError:
            deps_status['firth'] = {'installed': False, 'msg': '⚠️ Firth regression unavailable'}
        
        if not deps_status['firth']['installed']:
            ui.notification_show(deps_status['firth']['msg'], type="warning", duration=5)
            
    check_optional_deps()

    # ==========================================
    # 3. CALL MODULES SERVER
    # ==========================================
    
    # --- 1. Data Management ---
    logger.info("🔧 Initializing Data Management Module...")
    try:
        tab_data.data_server(
            "data",
            df,                   # 1
            var_meta,             # 2
            uploaded_file_info,   # 3
            df_matched,           # 4
            is_matched,           # 5
            matched_treatment_col,# 6
            matched_covariates    # 7
        )
        logger.info("✅ Data Management Module initialized successfully")
    except Exception as e:
        logger.exception("❌ Critical Error in Data Module")
        ui.notification_show(
            f"Critical Error in Data Module: {str(e)[:100]}", 
            type="error", 
            duration=10
        )

    # --- 2. Table 1 & Matching ---
    logger.info("🔧 Initializing Table 1 & Matching Module...")
    try:
        tab_baseline_matching.baseline_matching_server(
            "bm", 
            df, 
            var_meta, 
            df_matched, 
            is_matched, 
            matched_treatment_col, 
            matched_covariates
        )
        logger.info("✅ Table 1 & Matching Module initialized")
    except Exception as e:
        logger.exception("❌ Error in Table 1 & Matching Module")

    # --- 3. Diagnostic Tests ---
    logger.info("🔧 Initializing Diagnostic Tests Module...")
    try:
        tab_diag.diag_server(
            "diag", 
            df, 
            var_meta, 
            df_matched, 
            is_matched
        )
        logger.info("✅ Diagnostic Tests Module initialized")
    except Exception as e:
        logger.exception("❌ Error in Diagnostic Tests Module")

    # --- 4. Logistic Regression ---
    logger.info("🔧 Initializing Logistic Regression Module...")
    try:
        tab_logit.logit_server(
            "logit",
            df, 
            var_meta, 
            df_matched, 
            is_matched
        )
        logger.info("✅ Logistic Regression Module initialized")
    except Exception as e:
        logger.exception("❌ Error in Logistic Regression Module")

    # --- 5. Correlation & ICC ---
    logger.info("🔧 Initializing Correlation & ICC Module...")
    try:
        tab_corr.corr_server(
            "corr",
            df, 
            var_meta, 
            df_matched, 
            is_matched
        )
        logger.info("✅ Correlation & ICC Module initialized")
    except Exception as e:
        logger.exception("❌ Error in Correlation & ICC Module")

    # --- 6. Survival Analysis Module ---
    logger.info("🔧 Initializing Survival Analysis Module...")
    try:
        tab_survival.survival_server(
            "survival",
            df, 
            var_meta, 
            df_matched, 
            is_matched
        )
        logger.info("✅ Survival Analysis Module initialized")
    except Exception as e:
        logger.exception("❌ Error in Survival Analysis Module")

    # --- 7. Settings Module ---
    logger.info("🔧 Initializing Settings Module...")
    try:
        tab_settings.settings_server("settings", CONFIG)
        logger.info("✅ Settings Module initialized")
    except Exception as e:
        logger.exception("❌ Error in Settings Module")

    logger.info("🎉 All modules initialized")

# ==========================================
# 4. APP LAUNCHER
# ==========================================
app = App(app_ui, server)

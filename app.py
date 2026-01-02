from shiny import App, ui, reactive, Session, render
import html

# Import Config/Logger
from config import CONFIG
from logger import get_logger, LoggerFactory

# Import Tabs Modules
from tabs import tab_data           # 🟢 Data Module
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
logger.info("  %s", COMPUTATION_CACHE)     # Layer 1: Caching
logger.info("  %s", MEMORY_MANAGER)        # Layer 2: Memory Mgmt
logger.info("  %s", CONNECTION_HANDLER)    # Layer 3: Connection Resilience

# Get color palette for navbar styling
colors = get_color_palette()

# ==========================================
# 1. UI DEFINITION
# ==========================================
app_ui = ui.page_navbar(
    # --- 1. Data Management Module ---
    ui.nav_panel(
        "📁 Data Management",
        ui.div(tab_data.data_ui("data"), class_="app-container")
    ),
    
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

    # === LAYER 2 & 3: ปรับเป็น Dynamic Status Badge ===
    footer=ui.div(
        ui.output_ui("optimization_status"), # ✅ แสดงสถานะแบบ Real-time
        style="padding: 10px; border-top: 1px solid #eee; margin-top: 20px;"
    ),

    title=CONFIG.get('ui.page_title', 'Medical Stat Tool'),
    id="main_navbar",
    window_title="Medical Stat Tool",

    # ⬇⬇⬇ inject theme CSS
    header=ui.tags.head(
        ui.HTML(get_shiny_css())
    ),
)

# ==========================================
# 2. SERVER LOGIC
# ==========================================
def server(input, output, session: Session):
    """
    Initialize the Shiny server: create shared reactive state, register dynamic optimization-status UI, verify optional dependencies, and wire tab module servers.
    
    This function sets up global reactive values used across tabs (data, metadata, uploaded file info, and matching state), defines the `optimization_status` UI output which refreshes every 5 seconds to display cache, memory, and connection subsystem status, performs an optional-dependency check (notifying the user if Firth regression is unavailable), and attaches each tab's server logic to the app with the shared reactive state.
    """
    logger.info("📱 Shiny app session started")
    logger.info("💾 Cache stats: %s", COMPUTATION_CACHE.get_stats())
    logger.info("🧠 Memory status: %s", MEMORY_MANAGER.get_memory_status())

    # --- Reactive State (Global) ---
    df = reactive.Value(None)
    var_meta = reactive.Value({})
    uploaded_file_info = reactive.Value(None)
    
    # Matched data state (Shared across tabs)
    df_matched = reactive.Value(None)
    is_matched = reactive.Value(False)
    matched_treatment_col = reactive.Value(None)
    matched_covariates = reactive.Value([])

    # === 🚀 DYNAMIC STATUS BADGE LOGIC ===
    @output
    @render.ui
    def optimization_status():
        # ตั้งค่าให้ Refresh ตัวเองทุก 5 วินาที
        """
        Render a compact HTML status badge summarizing cache, memory, and connection subsystem health.
        
        The badge displays an icon and tooltip for each subsystem:
        - L1 Cache: number of cached items, max size, and hit rate.
        - L2 Memory: usage percentage and current/max MB with severity coloring.
        - L3 Resilience: connection success rate and failure count with severity coloring.
        
        This UI auto-refreshes every 5 seconds.
        
        Returns:
            ui_element (shiny.ui): An HTML UI element containing the three status indicators and copyright.
        """
        reactive.invalidate_later(5)

        # 1. เช็คสถานะ Cache (L1)
        cache_stats = COMPUTATION_CACHE.get_stats()
        # ถ้ามีของใน Cache ให้เป็นเขียว 🟢 ถ้าว่างให้เป็นเทา ⚪
        cache_icon = "🟢" if cache_stats['cached_items'] > 0 else "⚪"
        cache_title = f"Cache: {cache_stats['cached_items']}/{cache_stats['max_size']} items (Hit rate: {cache_stats['hit_rate']})"

        # 2. เช็คสถานะ Memory (L2)
        mem_status = MEMORY_MANAGER.get_memory_status()
        mem_icon = "💗"  # Normal
        if mem_status['status'] == 'WARNING':
            mem_icon = "💛"  # Approaching limit
        elif mem_status['status'] == 'CRITICAL':
            mem_icon = "🔴"  # Critical
        mem_title = f"Memory: {mem_status['usage_pct']} ({mem_status['current_mb']}MB / {mem_status['max_mb']}MB)"

        # 3. เช็คสถานะ Connection (L3)
        conn_stats = CONNECTION_HANDLER.get_stats()
        success_val = float(conn_stats['success_rate'].replace('%',''))
        conn_icon = "🟢" if success_val >= 90 else "🟠" if success_val >= 70 else "🔴"
        conn_title = f"Resilience: {conn_stats['success_rate']} success rate ({conn_stats['failed_attempts']} failures)"

        cache_title = html.escape(f"Cache: {cache_stats['cached_items']}/{cache_stats['max_size']} items (Hit rate: {cache_stats['hit_rate']})")
        mem_title = html.escape(f"Memory: {mem_status['usage_pct']} ({mem_status['current_mb']}MB / {mem_status['max_mb']}MB)")
        conn_title = html.escape(f"Resilience: {conn_stats['success_rate']} success rate ({conn_stats['failed_attempts']} failures)")
        
        return ui.HTML(f"""
        <div style='text-align: right; font-size: 0.75em; color: #999;'>
            <span title='{cache_title}' style='cursor: help;'>{cache_icon} L1 Cache</span> | 
            <span title='{mem_title}' style='cursor: help;'>{mem_icon} L2 Memory</span> | 
            <span title='{conn_title}' style='cursor: help;'>{conn_icon} L3 Resilience</span> |
            &copy; 2025 Medical Stat Tool
        </div>
        """)

    # --- Helper: Check Dependencies ---
    def check_optional_deps():
        """
        Check for the optional firthlogist dependency and show a warning notification if it is not available.
        
        This function verifies whether the `firthlogist` package is installed and, when it is missing, displays a user-facing warning via the app notification system.
        """
        deps_status = {}
        try:
            import firthlogist
            deps_status['firth'] = {'installed': True, 'msg': '✅ Firth regression enabled'}
        except ImportError:
            deps_status['firth'] = {'installed': False, 'msg': '⚠️ Firth regression unavailable'}
        
        if not deps_status['firth']['installed']:
            ui.notification_show(deps_status['firth']['msg'], type="warning")
            
    check_optional_deps()

    # ==========================================
    # 3. CALL MODULES SERVER
    # ==========================================
    
    # --- 1. Data Management ---
    tab_data.data_server("data",
        df, var_meta, uploaded_file_info,
        df_matched, is_matched, matched_treatment_col, matched_covariates
    )

    # --- 2. Table 1 & Matching ---
    tab_baseline_matching.baseline_matching_server("bm", 
        df, var_meta, df_matched, is_matched, 
        matched_treatment_col, matched_covariates
    )

    # --- 3. Diagnostic Tests ---
    tab_diag.diag_server("diag", 
        df, var_meta, df_matched, is_matched
    )

    # --- 4. Logistic Regression ---
    tab_logit.logit_server("logit",
        df, var_meta, df_matched, is_matched
    )

    # --- 5. Correlation & ICC ---
    tab_corr.corr_server("corr",
        df, var_meta, df_matched, is_matched
    )

    # --- 6. Survival Analysis Module ---
    tab_survival.survival_server("survival",
        df, var_meta, df_matched, is_matched
    )

    # --- 7. Settings Module ---
    tab_settings.settings_server("settings", CONFIG)

# ==========================================
# 4. APP LAUNCHER
# ==========================================
app = App(app_ui, server)
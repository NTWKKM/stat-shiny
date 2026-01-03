from shiny import App, ui, reactive, Session

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
logger.info(f"🚀 Initializing HF optimization layers...")
logger.info(f"  {COMPUTATION_CACHE}")     # Layer 1: Caching
logger.info(f"  {MEMORY_MANAGER}")        # Layer 2: Memory Mgmt
logger.info(f"  {CONNECTION_HANDLER}")    # Layer 3: Connection Resilience

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

    # === LAYER 2 & 3: Add optimization status badge to footer ===
    # ปรับปรุง: ใช้ ui.output_ui เพื่อรองรับการแสดงผลแบบ Dynamic ในอนาคตแต่ไม่มี invalidate_later ที่นี่
    footer=ui.div( 
        ui.output_ui("optimization_status_footer"),
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
    logger.info("📱 Shiny app session started")
    
    # --- Reactive State (Global) ---
    df = reactive.Value(None)
    var_meta = reactive.Value({})
    uploaded_file_info = reactive.Value(None)
    
    # Matched data state (Shared across tabs)
    df_matched = reactive.Value(None)
    is_matched = reactive.Value(False)
    matched_treatment_col = reactive.Value(None)
    matched_covariates = reactive.Value([])

    # --- Optimization Status (Footer) ---
    @render.ui
    def optimization_status_footer():
        # ใช้ isolate เพื่อป้องกันไม่ให้การเปลี่ยนหน้าหรือเปลี่ยน data มา trigger ส่วนนี้
        # และเอา reactive.invalidate_later ออกถาวรเพื่อแก้ปัญหา Spinner ค้าง
        with reactive.isolate():
            return ui.HTML("""
                <div style='text-align: right; font-size: 0.75em; color: #999;'>
                    <span title='Cache enabled'>🟢 L1 Cache</span> | 
                    <span title='Memory monitoring'>💗 L2 Memory</span> | 
                    <span title='Connection resilience'>🟠 L3 Resilience</span> |
                    &copy; 2025 Medical Stat Tool
                </div>
            """)

    # --- Helper: Check Dependencies ---
    def check_optional_deps():
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

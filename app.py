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
from tabs._common import wrap_with_container

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

# ==========================================
# 1. UI DEFINITION
# ==========================================
app_ui = ui.page_navbar(
    # --- 1. Data Management Module ---
    ui.nav_panel(
        "📁 Data Management",
        wrap_with_container(
            tab_data.data_ui("data")
        )
    ),
    
    # --- 2. Table 1 & Matching Module ---
    ui.nav_panel("📋 Table 1 & Matching", 
        wrap_with_container(
            tab_baseline_matching.baseline_matching_ui("bm")
        )
    ),

    # --- 3. Diagnostic Tests Module ---
    ui.nav_panel("🧪 Diagnostic Tests", 
        wrap_with_container(
            tab_diag.diag_ui("diag")
        )
    ),

    # --- 4. Logistic Regression Module ---
    ui.nav_panel("📊 Risk Factors", 
        wrap_with_container(
            tab_logit.logit_ui("logit")
        )
    ),

    # --- 5. Correlation & ICC Module ---
    ui.nav_panel("📈 Correlation & ICC", 
        wrap_with_container(
            tab_corr.corr_ui("corr")
        )
    ),

    # --- 6. Survival Analysis Module ---
    ui.nav_panel("⏳ Survival Analysis", 
        wrap_with_container(
            # ✅ เรียกใช้ survival_ui โดยระบุแค่ ID (Namespace) เท่านั้น
            tab_survival.survival_ui("survival")
        )
    ),

    # --- 7. Settings Module ---
    ui.nav_panel("⚙️ Settings", 
        wrap_with_container(
            tab_settings.settings_ui("settings")
        )
    ),

    # === LAYER 2 & 3: Add optimization status badge to footer ===
    ui.tags.footer(
        ui.HTML("""
        <div style='text-align: right; font-size: 0.75em; color: #999; padding: 10px; border-top: 1px solid #eee; margin-top: 20px;'>
            <span title='Cache enabled'>🟢 L1 Cache</span> | 
            <span title='Memory monitoring'>💗 L2 Memory</span> | 
            <span title='Connection resilience'>🟠 L3 Resilience</span> |
            &copy; 2025 Medical Stat Tool
        </div>
        """)
    ),

    title=CONFIG.get('ui.page_title', 'Medical Stat Tool'),
    id="main_navbar",
    window_title="Medical Stat Tool",
    
    # 🟢 ย้าย inverse=True ไปไว้ใน navbar_options
    navbar_options=ui.navbar_options(inverse=True),

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
    logger.info(f"💾 Cache stats: {COMPUTATION_CACHE.get_stats()}")
    logger.info(f"🧠 Memory status: {MEMORY_MANAGER.get_memory_status()}")

    # --- Reactive State (Global) ---
    df = reactive.Value(None)
    var_meta = reactive.Value({})
    uploaded_file_info = reactive.Value(None)
    
    # Matched data state (Shared across tabs)
    df_matched = reactive.Value(None)
    is_matched = reactive.Value(False)
    matched_treatment_col = reactive.Value(None)
    matched_covariates = reactive.Value([])

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
    # ✅ แก้ไขตรงนี้: ไม่ต้องส่ง input, output, session เข้าไปเองแล้ว
    # เพราะ @module.server จะดึงค่าเหล่านั้นจาก ID "survival" ให้โดยอัตโนมัติ
    tab_survival.survival_server("survival",
        df, var_meta, df_matched, is_matched
    )

    # --- 7. Settings Module ---
    tab_settings.settings_server("settings", CONFIG)

# ==========================================
# 4. APP LAUNCHER
# ==========================================
app = App(app_ui, server)

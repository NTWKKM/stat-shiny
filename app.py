from shiny import App, ui, reactive, Session

# Import Config/Logger
from config import CONFIG
from logger import get_logger, LoggerFactory

# Import Tabs Modules
from tabs import tab_data           # 🟢 Data Module (NEW)
from tabs import tab_baseline_matching
from tabs import tab_diag
from tabs import tab_logit
from tabs import tab_corr
from tabs import tab_survival
from tabs import tab_settings

from tabs._styling import get_shiny_css

# Initialize Logger
LoggerFactory.configure()
logger = get_logger(__name__)

# ==========================================
# 1. UI DEFINITION
# ==========================================
app_ui = ui.page_navbar(
    # --- 1. Data Management Module ---
    tab_data.data_ui("data"),
    
    # --- 2. Table 1 & Matching Module ---
    ui.nav_panel("📋 Table 1 & Matching", 
        tab_baseline_matching.baseline_matching_ui("bm")
    ),

    # --- 3. Diagnostic Tests Module ---
    ui.nav_panel("🧪 Diagnostic Tests", 
        tab_diag.diag_ui("diag")
    ),

    # --- 4. Logistic Regression Module ---
    ui.nav_panel("📊 Risk Factors", 
        tab_logit.logit_ui("logit")
    ),

    # --- 5. Correlation & ICC Module ---
    ui.nav_panel("📈 Correlation & ICC", 
        tab_corr.corr_ui("corr")
    ),

    # --- 6. Survival Analysis Module ---
    ui.nav_panel("⏳ Survival Analysis", 
        tab_survival.survival_ui("survival")
    ),

    # --- 7. Settings Module ---
    ui.nav_panel("⚙️ Settings", 
        tab_settings.settings_ui("settings")
    ),

    title=CONFIG.get('ui.page_title', 'Medical Stat Tool'),
    id="main_navbar",
    window_title="Medical Stat Tool",
    
    # ✅ ปรับปรุง: เพิ่ม inverse=True เพื่อให้ Text และปุ่มเมนูเป็นสีขาวสำหรับพื้นหลังเข้ม
    inverse=True,  

    # ⬇⬇⬇ inject teal theme CSS
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
    # These values are shared across all tabs
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
        
        logger.info("Optional dependencies: firth=%s", deps_status['firth']['installed'])
        if not deps_status['firth']['installed']:
            ui.notification_show(deps_status['firth']['msg'], type="warning")
            
    # Run check on start
    check_optional_deps()

    # ==========================================
    # 3. CALL MODULES SERVER
    # ==========================================
    
    # --- 1. Data Management Module ---
    # ส่ง Global Reactive Values เข้าไปจัดการข้างใน
    tab_data.data_server("data",
        df, var_meta, uploaded_file_info,
        df_matched, is_matched, matched_treatment_col, matched_covariates
    )

    # --- 2. Table 1 & Matching Module ---
    tab_baseline_matching.baseline_matching_server("bm", 
        df, var_meta, df_matched, is_matched, 
        matched_treatment_col, matched_covariates
    )

    # --- 3. Diagnostic Tests Module ---
    tab_diag.diag_server("diag", 
        df, var_meta, df_matched, is_matched
    )

    # --- 4. Logistic Regression Module ---
    tab_logit.logit_server("logit",
        df, var_meta, df_matched, is_matched
    )

    # --- 5. Correlation & ICC Module ---
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

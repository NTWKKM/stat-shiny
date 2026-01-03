from shiny import App, ui, reactive, Session

# Import Config/Logger
from config import CONFIG
from logger import get_logger, LoggerFactory
from logic import HAS_FIRTH

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

# Initialize Logger
LoggerFactory.configure()
logger = get_logger(__name__)

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
        # เช็คจากตัวแปร HAS_FIRTH ที่ logic.py เตรียมไว้ให้แล้ว
        if HAS_FIRTH:
            logger.info("Optional dependencies: firth=True")
        else:
            logger.warning("Optional dependencies: firth=False")
            ui.notification_show("⚠️ Firth regression unavailable", type="warning")


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

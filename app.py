from shiny import App, ui, reactive, render, Session
from shiny.types import FileInfo
import pandas as pd
import numpy as np
import io

# Import Config/Logger
from config import CONFIG
from logger import get_logger, LoggerFactory

# Import Tabs Modules
from tabs import tab_baseline_matching
from tabs import tab_diag  # <--- 1. เพิ่มการ Import Module Diagnostic ที่นี่

# Initialize Logger
LoggerFactory.configure()
logger = get_logger(__name__)

# ==========================================
# 1. UI DEFINITION
# ==========================================
app_ui = ui.page_navbar(
    # --- Global Sidebar (Data Management) ---
    ui.nav_panel("📁 Data Management",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("MENU"),
                ui.h5("1. Data Management"),
                
                ui.input_action_button("btn_load_example", "📄 Load Example Data", class_="btn-secondary"),
                ui.br(), ui.br(),
                
                ui.input_file("file_upload", "Upload CSV/Excel", accept=[".csv", ".xlsx"], multiple=False),
                
                ui.hr(),
                
                ui.output_ui("ui_btn_clear_match"),
                ui.input_action_button("btn_reset_all", "⚠️ Reset All Data", class_="btn-danger"),
                
                ui.hr(),
                
                ui.h5("2. Variable Metadata"),
                ui.input_select("sel_var_edit", "Edit Var:", choices=["Select..."]),
                ui.panel_conditional(
                    "input.sel_var_edit != 'Select...'",
                    ui.input_radio_buttons("radio_var_type", "Type:", 
                                         choices={"Categorical": "Categorical", "Continuous": "Continuous"}),
                    ui.input_text_area("txt_var_map", "Labels (Format: 0=No)", height="80px"),
                    ui.input_action_button("btn_save_meta", "💾 Save")
                ),
                width=350,
                bg="#f8f9fa"
            ),
            
            ui.card(
                ui.card_header("📁 Raw Data Preview"),
                ui.output_data_frame("out_df_preview"),
                full_screen=True
            )
        )
    ),
    
    # --- 1. Table 1 & Matching Module ---
    ui.nav_panel("📋 Table 1 & Matching", 
        tab_baseline_matching.baseline_matching_ui("bm")
    ),

    # --- 2. Diagnostic Tests Module ---
    ui.nav_panel("🧪 Diagnostic Tests", 
        # <--- 2. แทนที่ Placeholder ด้วย UI ของ Module tab_diag
        tab_diag.diag_ui("diag")
    ),

    # --- Placeholders for other tabs (To be implemented) ---
    ui.nav_panel("📈 Correlation & ICC", 
        ui.card(ui.p("🚧 Please convert 'tabs/tab_corr.py' to Shiny module."))
    ),
    ui.nav_panel("📊 Risk Factors", 
        ui.card(ui.p("🚧 Please convert 'tabs/tab_logit.py' to Shiny module."))
    ),
    ui.nav_panel("⏳ Survival Analysis", 
        ui.card(ui.p("🚧 Please convert 'tabs/tab_survival.py' to Shiny module."))
    ),
    ui.nav_panel("⚙️ Settings", 
        ui.card(ui.p("🚧 Settings UI"))
    ),

    title=CONFIG.get('ui.page_title', 'Medical Stat Tool'),
    id="main_navbar",
    window_title="Medical Stat Tool"
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
        try:
            import firthlogist
            logger.info("Firth regression enabled")
        except ImportError:
            ui.notification_show("⚠️ Firth regression unavailable", type="warning")
            
    check_optional_deps()

    # --- 1. Data Loading Logic (Example Data & File Upload) ---
    # (โค้ดส่วนนี้คงเดิมเหมือนที่คุณมีอยู่...)
    @reactive.Effect
    @reactive.event(input.btn_load_example)
    def _():
        id_notify = ui.notification_show("Generating simulation...", duration=None)
        try:
            # ... (สร้าง simulation data เหมือนเดิม)
            np.random.seed(42)
            n = 600
            age = np.random.normal(60, 12, n).astype(int).clip(30, 95)
            # (ตัดส่วนสร้างข้อมูลออกเพื่อความกระชับ แต่ให้ใช้ของเดิมของคุณ)
            # สร้าง dummy data สำหรับทดสอบ diagnostic
            gold_std = np.random.binomial(1, 0.3, n)
            rapid_score = np.where(gold_std==0, np.random.normal(20, 10, n), np.random.normal(50, 15, n))
            
            data = {'Age': age, 'Gold_Standard': gold_std, 'Test_Score': rapid_score}
            new_df = pd.DataFrame(data)
            df.set(new_df)
            ui.notification_remove(id_notify)
        except Exception as e:
            ui.notification_remove(id_notify)
            ui.notification_show(f"Error: {e}", type="error")

    # (File Upload / Reset / Metadata Logic คงเดิม...)
    @reactive.Effect
    @reactive.event(input.file_upload)
    def _():
        file_infos = input.file_upload()
        if file_infos:
            f = file_infos[0]
            if f['name'].lower().endswith('.csv'):
                df.set(pd.read_csv(f['datapath']))
            ui.notification_show("File Uploaded Successfully!", type="message")

    # ==========================================
    # 3. CALL MODULES SERVER
    # ==========================================
    
    # --- 1. Table 1 & Matching Module ---
    tab_baseline_matching.baseline_matching_server("bm", 
        df, var_meta, df_matched, is_matched, 
        matched_treatment_col, matched_covariates
    )

    # --- 2. Diagnostic Tests Module ---
    # <--- 3. เรียกใช้งาน Server ของ tab_diag พร้อมส่งค่า State ที่จำเป็นเข้าไป
    tab_diag.diag_server("diag", 
        df, var_meta, df_matched, is_matched
    )

# ==========================================
# 4. APP LAUNCHER
# ==========================================
app = App(app_ui, server)

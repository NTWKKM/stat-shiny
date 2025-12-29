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
    matched_treatment_col = reactive.Value(None) # เพิ่ม state นี้
    matched_covariates = reactive.Value([])      # เพิ่ม state นี้

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

    # --- 1. Data Loading Logic ---
    
    @reactive.Effect
    @reactive.event(input.btn_load_example)
    def _():
        """Logic for generating example data"""
        logger.info("Generating example data...")
        
        id_notify = ui.notification_show("Generating simulation...", duration=None)
            
        try:
            np.random.seed(42)
            n = 600
            
            # --- Simulation Logic (Same as before) ---
            age = np.random.normal(60, 12, n).astype(int).clip(30, 95)
            sex = np.random.binomial(1, 0.5, n)
            bmi = np.random.normal(25, 5, n).round(1).clip(15, 50)
            
            logit_treat = -4.5 + (0.05 * age) + (0.08 * bmi) + (0.2 * sex)
            p_treat = 1 / (1 + np.exp(-logit_treat))
            group = np.random.binomial(1, p_treat, n)
            
            logit_dm = -5 + (0.04 * age) + (0.1 * bmi)
            p_dm = 1 / (1 + np.exp(-logit_dm))
            diabetes = np.random.binomial(1, p_dm, n)
            
            logit_ht = -4 + (0.06 * age) + (0.05 * bmi)
            p_ht = 1 / (1 + np.exp(-logit_ht))
            hypertension = np.random.binomial(1, p_ht, n)

            lambda_base = 0.002 
            linear_predictor = 0.03 * age + 0.4 * diabetes + 0.3 * hypertension - 0.6 * group
            hazard = lambda_base * np.exp(linear_predictor)
            surv_time = np.random.exponential(1/hazard, n)
            censor_time = np.random.uniform(0, 100, n)
            time_obs = np.minimum(surv_time, censor_time).round(1)
            time_obs = np.maximum(time_obs, 0.5)
            status_death = (surv_time <= censor_time).astype(int)
            
            logit_cure = 0.5 + 1.2 * group - 0.04 * age - 0.5 * diabetes
            p_cure = 1 / (1 + np.exp(-logit_cure))
            outcome_cured = np.random.binomial(1, p_cure, n)

            gold_std = np.random.binomial(1, 0.3, n)
            rapid_score = np.where(gold_std==0, 
                                   np.random.normal(20, 10, n), 
                                   np.random.normal(50, 15, n))
            rapid_score = np.clip(rapid_score, 0, 100).round(1)
            
            rater_a = np.where(gold_std==1, 
                               np.random.binomial(1, 0.85, n), 
                               np.random.binomial(1, 0.10, n))
            
            agree_prob = 0.85
            rater_b = np.where(np.random.binomial(1, agree_prob, n)==1, 
                               rater_a, 
                               1 - rater_a)

            hba1c = np.random.normal(6.5, 1.5, n).clip(4, 14).round(1)
            glucose = (hba1c * 15) + np.random.normal(0, 15, n)
            glucose = glucose.round(0)

            icc_rater1 = np.random.normal(120, 15, n).round(1)
            icc_rater2 = icc_rater1 + 5 + np.random.normal(0, 4, n)
            icc_rater2 = icc_rater2.round(1)

            data = {
                'ID': range(1, n+1),
                'Treatment_Group': group,  
                'Age_Years': age,
                'Sex_Male': sex,
                'BMI_kgm2': bmi,
                'Comorb_Diabetes': diabetes,
                'Comorb_Hypertension': hypertension,
                'Outcome_Cured': outcome_cured,
                'Time_Months': time_obs,
                'Status_Death': status_death,
                'Gold_Standard_Disease': gold_std,
                'Test_Score_Rapid': rapid_score, 
                'Diagnosis_Dr_A': rater_a,
                'Diagnosis_Dr_B': rater_b,
                'Lab_HbA1c': hba1c,
                'Lab_Glucose': glucose,
                'ICC_SysBP_Rater1': icc_rater1,
                'ICC_SysBP_Rater2': icc_rater2,
            }
            
            new_df = pd.DataFrame(data)
            
            meta = {
                'Treatment_Group': {'type':'Categorical', 'map':{0:'Standard Care', 1:'New Drug'}, 'label': 'Treatment Group'},
                'Sex_Male': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}, 'label': 'Sex'},
                'Comorb_Diabetes': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}, 'label': 'Diabetes'},
                'Comorb_Hypertension': {'type':'Categorical', 'map':{0:'No', 1:'Yes'}, 'label': 'Hypertension'},
                'Outcome_Cured': {'type':'Categorical', 'map':{0:'Not Cured', 1:'Cured'}, 'label': 'Outcome (Cured)'},
                'Status_Death': {'type':'Categorical', 'map':{0:'Censored/Alive', 1:'Dead'}, 'label': 'Status (Death)'},
                'Gold_Standard_Disease': {'type':'Categorical', 'map':{0:'Healthy', 1:'Disease'}, 'label': 'Gold Standard'},
                'Diagnosis_Dr_A': {'type':'Categorical', 'map':{0:'Normal', 1:'Abnormal'}, 'label': 'Diagnosis (Dr. A)'},
                'Diagnosis_Dr_B': {'type':'Categorical', 'map':{0:'Normal', 1:'Abnormal'}, 'label': 'Diagnosis (Dr. B)'},
                
                'Age_Years': {'type': 'Continuous', 'label': 'Age (Years)', 'map': {}},
                'BMI_kgm2': {'type': 'Continuous', 'label': 'BMI (kg/m²)', 'map': {}},
                'Time_Months': {'type': 'Continuous', 'label': 'Time (Months)', 'map': {}},
                'Test_Score_Rapid': {'type': 'Continuous', 'label': 'Rapid Test Score (0-100)', 'map': {}},
                'Lab_HbA1c': {'type': 'Continuous', 'label': 'HbA1c (%)', 'map': {}},
                'Lab_Glucose': {'type': 'Continuous', 'label': 'Fasting Glucose (mg/dL)', 'map': {}},
                'ICC_SysBP_Rater1': {'type': 'Continuous', 'label': 'Sys BP (Rater 1)', 'map': {}},
                'ICC_SysBP_Rater2': {'type': 'Continuous', 'label': 'Sys BP (Rater 2)', 'map': {}},
            }
            
            df.set(new_df)
            var_meta.set(meta)
            uploaded_file_info.set({"name": "Example Clinical Data"})
            
            ui.notification_remove(id_notify)
            ui.notification_show(f"✅ Loaded {n} Clinical Records (Simulated)", type="message")

        except Exception as e:
            ui.notification_remove(id_notify)
            ui.notification_show(f"Error: {e}", type="error")

    @reactive.Effect
    @reactive.event(input.file_upload)
    def _():
        """Logic for File Upload"""
        file_infos: list[FileInfo] = input.file_upload()
        if not file_infos:
            return
            
        f = file_infos[0]
        logger.info(f"File uploaded: {f['name']}")
        
        try:
            if f['name'].lower().endswith('.csv'):
                new_df = pd.read_csv(f['datapath'])
            else:
                new_df = pd.read_excel(f['datapath'])
            
            df.set(new_df)
            uploaded_file_info.set({"name": f['name']})
            
            # Preserve/Init Metadata
            current_meta = var_meta.get().copy()
            for col in new_df.columns:
                if col not in current_meta:
                    unique_vals = new_df[col].dropna().unique()
                    unique_count = len(unique_vals)
                    is_numeric = pd.api.types.is_numeric_dtype(new_df[col])
                    
                    if is_numeric and unique_count > 10:
                         current_meta[col] = {'type': 'Continuous', 'map': {}, 'label': col}
                    else:
                         current_meta[col] = {'type': 'Categorical', 'map': {}, 'label': col}
            
            var_meta.set(current_meta)
            ui.notification_show("File Uploaded Successfully!", type="message")
            
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            ui.notification_show(f"Error: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.btn_reset_all)
    def _():
        """Reset Logic"""
        df.set(None)
        var_meta.set({})
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])
        ui.notification_show("All data reset", type="warning")

    # --- 2. Sidebar Metadata Logic ---
    
    @reactive.Effect
    def _update_var_select():
        """Update dropdown choices when df changes"""
        data = df.get()
        if data is not None:
            cols = ["Select..."] + data.columns.tolist()
            ui.update_select("sel_var_edit", choices=cols)

    @reactive.Effect
    @reactive.event(input.sel_var_edit)
    def _load_meta_to_ui():
        """Load metadata into inputs when variable is selected"""
        var_name = input.sel_var_edit()
        meta = var_meta.get()
        
        if var_name != "Select..." and var_name in meta:
            m = meta[var_name]
            ui.update_radio_buttons("radio_var_type", selected=m.get('type', 'Categorical'))
            
            map_str = "\n".join([f"{k}={v}" for k,v in m.get('map', {}).items()])
            ui.update_text_area("txt_var_map", value=map_str)

    @reactive.Effect
    @reactive.event(input.btn_save_meta)
    def _save_metadata():
        """Save logic for metadata editor"""
        var_name = input.sel_var_edit()
        if var_name == "Select...":
            return
            
        new_map = {}
        for line in input.txt_var_map().split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                try:
                    k = k.strip()
                    try:
                        k_num = float(k)
                        k = int(k_num) if k_num.is_integer() else k_num
                    except ValueError:
                        pass
                    new_map[k] = v.strip()
                except Exception:
                    pass

        current_meta = var_meta.get().copy()
        current_meta[var_name] = {
            'type': input.radio_var_type(),
            'map': new_map,
            'label': var_name
        }
        var_meta.set(current_meta)
        ui.notification_show(f"Saved metadata for {var_name}", type="message")

    # --- 3. Render Outputs ---

    @render.data_frame
    def out_df_preview():
        """Render the Data Grid"""
        d = df.get()
        if d is not None:
            return render.DataGrid(d, filters=True, height="500px")
        return None

    @render.ui
    def ui_btn_clear_match():
        """Show clear match button only if matched data exists"""
        if is_matched.get():
             return ui.input_action_button("btn_clear_match", "🔄 Clear Matched Data")
        return None
    
    @reactive.Effect
    @reactive.event(input.btn_clear_match)
    def _():
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])

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

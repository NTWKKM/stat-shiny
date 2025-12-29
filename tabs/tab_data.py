from shiny import ui, module, reactive, render, session as shiny_session
from shiny.types import FileInfo
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

# --- 1. UI Definition ---
def data_ui(id):
    # ✅ ใช้ Underscore (_) แทน Hyphen (-) เพื่อผ่านกฎ Validation
    ns = lambda x: f"{id}_{x}"
    
    return ui.nav_panel("📁 Data Management",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("MENU"),
                ui.h5("1. Data Management"),
                
                ui.input_action_button(ns("btn_load_example"), "📄 Load Example Data", class_="btn-secondary"),
                ui.br(), ui.br(),
                
                ui.input_file(ns("file_upload"), "Upload CSV/Excel", accept=[".csv", ".xlsx"], multiple=False),
                
                ui.hr(),
                
                ui.output_ui(ns("ui_btn_clear_match")),
                ui.input_action_button(ns("btn_reset_all"), "⚠️ Reset All Data", class_="btn-danger"),
                
                width=300,
                bg="#f8f9fa"
            ),
            
            # --- ส่วนการตั้งค่าตัวแปรแบบ Accordion (เหมือน st.expander) ---
            ui.accordion(
                ui.accordion_panel(
                    "🛠️ Variable Settings & Value Labels",
                    ui.layout_columns(
                        ui.div(
                            ui.input_select(ns("sel_var_edit"), "เลือกตัวแปรเพื่อตั้งค่า:", choices=["Select..."]),
                        ),
                        ui.div(
                            ui.panel_conditional(
                                f"input['{ns('sel_var_edit')}'] != 'Select...'",
                                ui.input_radio_buttons(
                                    ns("radio_var_type"), 
                                    "ประเภทตัวแปร:", 
                                    choices={"Continuous": "Continuous", "Categorical": "Categorical"},
                                    inline=True
                                ),
                                ui.input_text_area(ns("txt_var_map"), "Value Labels (Format: 0=No, 1=Yes)", height="100px"),
                                ui.input_action_button(ns("btn_save_meta"), "💾 Save Settings", class_="btn-primary")
                            )
                        ),
                        col_widths=(4, 8)
                    ),
                ),
                id=ns("acc_settings"),
                open=True
            ),

            ui.br(),
            
            # --- ส่วนแสดงผลข้อมูล ---
            ui.card(
                ui.card_header("📄 Raw Data Preview (Top 10 rows)"),
                ui.output_data_frame(ns("data_out_df_preview")),
                full_screen=True
            )
        )
    )

# --- 2. Server Logic ---
def data_server(id, df, var_meta, uploaded_file_info, 
                df_matched, is_matched, matched_treatment_col, matched_covariates):
    
    session = shiny_session.get_current_session()
    input = session.input
    ns = lambda x: f"{id}_{x}"

    # --- 1. Data Loading Logic ---
    @reactive.Effect
    @reactive.event(lambda: input[ns("btn_load_example")]())
    def _():
        logger.info("Generating example data...")
        id_notify = ui.notification_show("Generating simulation...", duration=None)
        try:
            np.random.seed(42)
            n = 600
            
            # --- Simulation Logic ---
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
    @reactive.event(lambda: input[ns("file_upload")]())
    def _():
        file_infos: list[FileInfo] = input[ns("file_upload")]()
        if not file_infos: return
        f = file_infos[0]
        try:
            if f['name'].lower().endswith('.csv'):
                new_df = pd.read_csv(f['datapath'])
            else:
                new_df = pd.read_excel(f['datapath'])
            df.set(new_df)
            uploaded_file_info.set({"name": f['name']})
            current_meta = var_meta.get().copy()
            for col in new_df.columns:
                if col not in current_meta:
                    unique_vals = new_df[col].dropna().unique()
                    is_numeric = pd.api.types.is_numeric_dtype(new_df[col])
                    if is_numeric and len(unique_vals) > 10:
                         current_meta[col] = {'type': 'Continuous', 'map': {}, 'label': col}
                    else:
                         current_meta[col] = {'type': 'Categorical', 'map': {}, 'label': col}
            var_meta.set(current_meta)
            ui.notification_show("File Uploaded Successfully!", type="message")
        except Exception as e:
            ui.notification_show(f"Error: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(lambda: input[ns("btn_reset_all")]())
    def _():
        df.set(None)
        var_meta.set({})
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])
        ui.notification_show("All data reset", type="warning")

    # --- ส่วนการจัดการ Metadata (Variable Selection + Bullets) ---

    @reactive.Effect
    def _update_var_select():
        data = df.get()
        if data is not None:
            cols = ["Select..."] + data.columns.tolist()
            ui.update_select(ns("sel_var_edit"), choices=cols)

    @reactive.Effect
    @reactive.event(lambda: input[ns("sel_var_edit")]())
    def _load_meta_to_ui():
        var_name = input[ns("sel_var_edit")]()
        meta = var_meta.get()
        if var_name != "Select..." and var_name in meta:
            m = meta[var_name]
            # โหลดประเภทตัวแปรลง Radio Buttons
            ui.update_radio_buttons(ns("radio_var_type"), selected=m.get('type', 'Continuous'))
            # โหลด Mapping ลง Text Area
            map_str = "\n".join([f"{k}={v}" for k,v in m.get('map', {}).items()])
            ui.update_text_area(ns("txt_var_map"), value=map_str)

    @reactive.Effect
    @reactive.event(lambda: input[ns("btn_save_meta")]())
    def _save_metadata():
        var_name = input[ns("sel_var_edit")]()
        if var_name == "Select...": return
        
        # 1. จัดการ Value Mapping
        new_map = {}
        for line in input[ns("txt_var_map")]().split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                try:
                    k = k.strip()
                    try:
                        k_num = float(k)
                        k = int(k_num) if k_num.is_integer() else k_num
                    except ValueError: pass
                    new_map[k] = v.strip()
                except Exception: pass
        
        # 2. อัปเดต Metadata ทั้งหมด (Type + Map)
        current_meta = var_meta.get().copy()
        current_meta[var_name] = {
            'type': input[ns("radio_var_type")](),
            'map': new_map,
            'label': var_name
        }
        var_meta.set(current_meta)
        ui.notification_show(f"✅ Saved settings for {var_name}", type="message")

    # --- 3. Render Outputs ---

    @render.data_frame
    def data_out_df_preview():
        d = df.get()
        if d is not None:
            # ✅ แสดงผลเพียง 10 แถวแรก และปิด Filter
            return render.DataGrid(d.head(10), filters=False, height="400px")
        return None

    @render.ui
    def data_ui_btn_clear_match():
        if is_matched.get():
             return ui.input_action_button(ns("btn_clear_match"), "🔄 Clear Matched Data")
        return None
    
    @reactive.Effect
    @reactive.event(lambda: input[ns("btn_clear_match")]())
    def _():
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])

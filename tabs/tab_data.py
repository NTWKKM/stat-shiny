from shiny import ui, module, reactive, render
from shiny.types import FileInfo
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

# --- 1. UI Definition ---
@module.ui
def data_ui():
    # ใช้ layout_sidebar ภายใต้ nav_panel ของ app.py
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("MENU"),
            ui.h5("1. Data Management"),
            
            ui.input_action_button("btn_load_example", "📄 Load Example Data", class_="btn-secondary"),
            ui.br(), ui.br(),
            
            ui.input_file("file_upload", "Upload CSV/Excel", accept=[".csv", ".xlsx"], multiple=False),
            
            ui.hr(),
            
            ui.output_ui("ui_btn_clear_match"),
            ui.input_action_button("btn_reset_all", "⚠️ Reset All Data", class_="btn-danger"),
            
            width=300,
            bg="#f8f9fa"
        ),
        
        # ส่วน Main Content
        ui.accordion(
            ui.accordion_panel(
                "🛠️ 1. Variable Settings & Labels",
                ui.layout_columns(
                    ui.div(
                        ui.input_select("sel_var_edit", "Select Variable to Configure:", choices=["Select..."]),
                    ),
                    ui.div(
                        ui.output_ui("ui_var_settings")
                    ),
                    col_widths=(4, 8)
                ),
            ),
            id="acc_settings",
            open=True
        ),

        ui.br(),
        
        ui.card(
            ui.card_header("📄 2. Raw Data Preview"),
            # เปลี่ยนจาก output_table เป็น output_data_frame
            ui.output_data_frame("out_df_preview"), 
            height="600px",
            full_screen=True
        )
    )

# --- 2. Server Logic ---
@module.server
def data_server(input, output, session, df, var_meta, uploaded_file_info, 
                df_matched, is_matched, matched_treatment_col, matched_covariates):
    
    logger.debug("data_server started successfully")

    is_loading_data = reactive.Value(False)

    # --- 1. Data Loading Logic ---
    @reactive.Effect
    @reactive.event(input.btn_load_example) 
    def _():
        logger.info("Generating example data...")
        is_loading_data.set(True)
        id_notify = ui.notification_show("🔄 Generating simulation...", duration=None)
        
        try:
            np.random.seed(42)
            n = 1500  
            
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
            rapid_score = np.where(
                gold_std == 0, 
                np.random.normal(20, 10, n), 
                np.random.normal(50, 15, n)
            )
            rapid_score = np.clip(rapid_score, 0, 100).round(1)
            
            rater_a = np.where(
                gold_std == 1, 
                np.random.binomial(1, 0.85, n), 
                np.random.binomial(1, 0.10, n)
            )
            
            agree_prob = 0.85
            rater_b = np.where(
                np.random.binomial(1, agree_prob, n) == 1, 
                rater_a, 
                1 - rater_a
            )

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
            
            logger.info(f"✅ Successfully generated {n} records")
            ui.notification_remove(id_notify)
            ui.notification_show(f"✅ Loaded {n} Clinical Records (Simulated)", type="message")

        except Exception as e:
            logger.error(f"Error generating example data: {e}")
            ui.notification_remove(id_notify)
            ui.notification_show(f"❌ Error: {e}", type="error")
        
        finally:
            is_loading_data.set(False)

    @reactive.Effect
    @reactive.event(input.file_upload) 
    def _():
        is_loading_data.set(True)
        file_infos: list[FileInfo] = input.file_upload()
        
        if not file_infos:
            is_loading_data.set(False)
            return
        
        f = file_infos[0]
        try:
            if f['name'].lower().endswith('.csv'):
                new_df = pd.read_csv(f['datapath'])
            else:
                new_df = pd.read_excel(f['datapath'])
            
            if len(new_df) > 100000:
                new_df = new_df.head(100000)
                ui.notification_show("⚠️ Large file: showing first 100,000 rows", type="warning")
            
            df.set(new_df)
            uploaded_file_info.set({"name": f['name']})
            
            current_meta = var_meta.get() or {}
            
            for col in new_df.columns:
                if col not in current_meta:
                    unique_vals = new_df[col].dropna().unique()
                    is_numeric = pd.api.types.is_numeric_dtype(new_df[col])
                    if is_numeric and len(unique_vals) > 10:
                        current_meta[col] = {'type': 'Continuous', 'map': {}, 'label': col}
                    else:
                        current_meta[col] = {'type': 'Categorical', 'map': {}, 'label': col}
            
            var_meta.set(current_meta)
            ui.notification_show(f"✅ Loaded {len(new_df)} rows", type="message")
            logger.info("✅ File uploaded: %s with %s rows", f['name'], len(new_df))
            
        except Exception as e:
            logger.error(f"Error: {e}")
            ui.notification_show(f"❌ Error: {str(e)}", type="error")
        finally:
            is_loading_data.set(False)

    @reactive.Effect
    @reactive.event(input.btn_reset_all)
    def _():
        df.set(None)
        var_meta.set({})
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])
        is_loading_data.set(False)
        ui.notification_show("All data reset", type="warning")

    # --- 2. Metadata Logic ---
    
    @reactive.Effect
    def _update_var_select():
        data = df.get()
        if data is not None:
            cols = ["Select..."] + data.columns.tolist()
            ui.update_select("sel_var_edit", choices=cols)
        else:
            ui.update_select("sel_var_edit", choices=["Select..."])

    @render.ui
    def ui_var_settings():
        var_name = input.sel_var_edit()
        
        if not var_name or var_name == "Select...":
            return None
            
        meta = var_meta.get()
        current_type = 'Continuous'
        map_str = ""
        
        if meta and var_name in meta:
            m = meta[var_name]
            current_type = m.get('type', 'Continuous')
            map_str = "\n".join([f"{k}={v}" for k, v in m.get('map', {}).items()])
            
        return ui.TagList(
            ui.input_radio_buttons(
                "radio_var_type", 
                "Variable Type:", 
                choices={"Continuous": "Continuous", "Categorical": "Categorical"},
                selected=current_type,
                inline=True
            ),
            ui.input_text_area(
                "txt_var_map", 
                "Value Labels (Format: 0=No, 1=Yes)", 
                value=map_str,
                height="100px"
            ),
            ui.input_action_button("btn_save_meta", "💾 Save Settings", class_="btn-primary")
        )

    @reactive.Effect
    @reactive.event(input.btn_save_meta)
    def _save_metadata():
        var_name = input.sel_var_edit()
        if var_name == "Select...":
            return
        
        new_map = {}
        map_input = input.txt_var_map()
        if map_input:
            for line in map_input.split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    try:
                        k_clean = k.strip()
                        try:
                            k_num = float(k_clean)
                            k_val = int(k_num) if k_num.is_integer() else k_num
                        except (ValueError, TypeError):
                            k_val = k_clean
                        new_map[k_val] = v.strip()
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Skipping malformed mapping line: {line} - {e}")

        current_meta = var_meta.get() or {}
        current_meta[var_name] = {
            'type': input.radio_var_type(), 
            'map': new_map, 
            'label': var_name
        }
        var_meta.set(current_meta)
        ui.notification_show(f"✅ Saved settings for {var_name}", type="message")

    # --- 3. Render Outputs ---
    @render.data_frame  # เปลี่ยนเป็น render.data_frame เพื่อใช้ร่วมกับ DataGrid
    def out_df_preview():
        try:
            d = df.get()

            if d is None:
                # กรณีไม่มีข้อมูล ให้ส่ง DataFrame เปล่าที่มีคำแนะนำ
                return render.DataGrid(
                    pd.DataFrame({'Status': ['No data loaded. Please load example data or upload a file.']}),
                    width="100%"
                )

            if isinstance(d, pd.DataFrame) and len(d) == 0:
                return render.DataGrid(pd.DataFrame({'Status': ['Dataset is empty.']}), width="100%")

            # ✅ ใช้ DataGrid: รองรับข้อมูลจำนวนมากได้ดี (Virtual Scrolling) 
            # Browser จะไม่ค้างเพราะโหลดเฉพาะแถวที่แสดงผล
            return render.DataGrid(
                d,
                width="100%",
                height="100%", # ให้ความสูงปรับตาม Card
                filters=True   # แถม: เพิ่มช่อง Filter หัวคอลัมน์ให้ด้วย
            )

        except Exception as e:
            logger.exception("Error rendering preview: %s", e)
            return render.DataGrid(pd.DataFrame({'Error': [f'Rendering Error: {e!s}']}))
            
    @render.ui
    def ui_btn_clear_match():
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
        ui.notification_show("✅ Matched data cleared", type="message")

from shiny import ui, module, reactive, render, session as shiny_session
from shiny.types import FileInfo
import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

# --- 1. UI Definition ---
def data_ui(id):
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
            
            # แก้ไขส่วนที่ 1: เปลี่ยนเป็น Dropdown + Radio ตามที่เคยตกลงกัน เพื่อลดภาระตาราง
            ui.accordion(
                ui.accordion_panel(
                    "🛠️ Variable Settings & Labels",
                    ui.layout_columns(
                        ui.div(
                            ui.input_select(ns("sel_var_edit"), "เลือกตัวแปร:", choices=["Select..."]),
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
                                ui.input_text_area(ns("txt_var_map"), "Value Labels (0=No, 1=Yes):", height="80px"),
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
            
            # แก้ไขส่วนที่ 2: ตาราง Preview
            ui.card(
                ui.card_header("📄 2. Raw Data Preview (Top 10 rows)"),
                # ตรวจสอบว่าชื่อ ID ตรงกับฟังก์ชันใน server (data_out_df_preview)
                ui.output_data_frame(ns("out_df_preview")),
                height="450px"
            )
        )
    )

# --- 2. Server Logic ---
def data_server(id, df, var_meta, uploaded_file_info, 
                df_matched, is_matched, matched_treatment_col, matched_covariates):
    
    session = shiny_session.get_current_session()
    input = session.input
    ns = lambda x: f"{id}_{x}"

    # --- 1. Data Loading Logic (Example Data) ---
    @reactive.Effect
    @reactive.event(lambda: input[ns("btn_load_example")]())
    def _():
        logger.info("Generating example data...")
        id_notify = ui.notification_show("Generating simulation...", duration=None)
        try:
            np.random.seed(42)
            n = 600
            age = np.random.normal(60, 12, n).astype(int).clip(30, 95)
            sex = np.random.binomial(1, 0.5, n)
            bmi = np.random.normal(25, 5, n).round(1).clip(15, 50)
            
            # สร้าง Dataframe
            new_df = pd.DataFrame({
                'ID': range(1, n+1),
                'Age_Years': age,
                'Sex_Male': sex,
                'BMI_kgm2': bmi,
                'Treatment': np.random.binomial(1, 0.5, n)
            })
            
            # ตั้งค่า Meta เบื้องต้น
            meta = {
                'Sex_Male': {'type':'Categorical', 'map':{0:'Female', 1:'Male'}, 'label': 'Sex'},
                'Age_Years': {'type': 'Continuous', 'label': 'Age', 'map': {}},
                'BMI_kgm2': {'type': 'Continuous', 'label': 'BMI', 'map': {}},
                'Treatment': {'type':'Categorical', 'map':{0:'Control', 1:'Treated'}, 'label': 'Group'}
            }
            
            df.set(new_df)
            var_meta.set(meta)
            ui.notification_remove(id_notify)
            ui.notification_show("✅ Loaded Example Data", type="message")
        except Exception as e:
            ui.notification_remove(id_notify)
            ui.notification_show(f"Error: {e}", type="error")

    # --- 2. File Upload Logic ---
    @reactive.Effect
    @reactive.event(lambda: input[ns("file_upload")]())
    def _():
        file_infos = input[ns("file_upload")]()
        if not file_infos: return
        f = file_infos[0]
        try:
            new_df = pd.read_csv(f['datapath']) if f['name'].endswith('.csv') else pd.read_excel(f['datapath'])
            df.set(new_df)
            current_meta = var_meta.get().copy()
            for col in new_df.columns:
                if col not in current_meta:
                    is_numeric = pd.api.types.is_numeric_dtype(new_df[col])
                    current_meta[col] = {'type': 'Continuous' if is_numeric else 'Categorical', 'map': {}, 'label': col}
            var_meta.set(current_meta)
        except Exception as e:
            ui.notification_show(f"Error: {e}", type="error")

    # --- 3. Settings Logic (Dropdown + Bullets) ---
    @reactive.Effect
    def _update_var_select():
        d = df.get()
        if d is not None:
            ui.update_select(ns("sel_var_edit"), choices=["Select..."] + d.columns.tolist())

    @reactive.Effect
    @reactive.event(lambda: input[ns("sel_var_edit")]())
    def _load_settings():
        var_name = input[ns("sel_var_edit")]()
        meta = var_meta.get()
        if var_name != "Select..." and var_name in meta:
            m = meta[var_name]
            ui.update_radio_buttons(ns("radio_var_type"), selected=m.get('type', 'Continuous'))
            map_str = "\n".join([f"{k}={v}" for k, v in m.get('map', {}).items()])
            ui.update_text_area(ns("txt_var_map"), value=map_str)

    @reactive.Effect
    @reactive.event(lambda: input[ns("btn_save_meta")]())
    def _save_settings():
        var_name = input[ns("sel_var_edit")]()
        if var_name == "Select...": return
        
        # Parse Map
        new_map = {}
        for line in input[ns("txt_var_map")]().split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                try:
                    new_map[k.strip()] = v.strip()
                except: pass

        current_meta = var_meta.get().copy()
        current_meta[var_name] = {
            'type': input[ns("radio_var_type")](),
            'map': new_map,
            'label': var_name
        }
        var_meta.set(current_meta)
        ui.notification_show(f"✅ Saved settings for {var_name}", type="message")

    # --- 4. Render Data Preview (ชื่อต้องตรงกับ UI: out_df_preview) ---
    @render.data_frame
    def data_out_df_preview():
        d = df.get()
        if d is not None:
            # ใช้ .head(10) เพื่อให้โหลดไว และปิด filter ตามคำแนะนำ UX เดิม
            return render.DataGrid(d.head(10), filters=False)
        return None

    @render.ui
    def data_ui_btn_clear_match():
        if is_matched.get():
             return ui.input_action_button(ns("btn_clear_match"), "🔄 Clear Matched Data")
        return None

    @reactive.Effect
    @reactive.event(lambda: input[ns("btn_reset_all")]())
    def _():
        df.set(None)
        var_meta.set({})
        ui.notification_show("All data reset", type="warning")

import html as _html
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from shiny import module, reactive, render, ui
from shiny.types import FileInfo

from logger import get_logger
from tabs._common import get_color_palette, wrap_with_container

logger = get_logger(__name__)
COLORS = get_color_palette()  # เรียกใช้ Palette กลาง

# --- 1. UI Definition ---
@module.ui
def data_ui() -> ui.TagChild:
    """
    UI for the Data Management tab.
    Refactored for UI consistency using ui.card and theme-aligned styling.
    Now includes Data Health Report section.
    """
    return ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                ui.h4("⚙️ Data Controls", class_="mb-3 text-primary"),
                ui.input_action_button(
                    "btn_load_example", 
                    "📄 Load Example Data", 
                    class_="btn-outline-primary w-100 mb-2 shadow-sm"
                ),
                ui.input_file(
                    "file_upload", 
                    "📂 Upload CSV/Excel", 
                    accept=[".csv", ".xlsx"], 
                    multiple=False,
                    width="100%"
                ),
                ui.hr(),
                ui.div(
                    ui.output_ui("ui_btn_clear_match"),
                    ui.input_action_button(
                        "btn_reset_all", 
                        "⚠️ Reset Workspace", 
                        class_="btn-outline-danger w-100 shadow-sm"
                    ),
                    class_="d-grid gap-2"
                ),
                class_="p-2"
            ),
            width=320,
            bg=COLORS['smoke_white'],
            title="Data Management"
        ),
        
        ui.div(
            # New: Data Health Report Section (Visible only when issues exist)
            ui.output_ui("ui_data_report_card"),

            # 1. Variable Settings Card (3-column layout with Missing Data Config)
            ui.card(
                ui.card_header(ui.tags.span("🛠️ Variable Configuration", class_="fw-bold")),
                ui.layout_columns(
                    # LEFT COLUMN: Variable Selection
                    ui.div(
                        ui.input_select(
                            "sel_var_edit", 
                            "Select Variable to Edit:", 
                            choices=["Select..."],
                            width="100%"
                        ),
                        ui.markdown(
                            """
                            > [!NOTE]
                            > **Categorical Mapping**: 
                            > Format as `0=Control, 1=Treat`.
                            """
                        ),
                        class_="p-2"
                    ),
                    # MIDDLE COLUMN: Variable Settings
                    ui.div(
                        ui.output_ui("ui_var_settings"),
                        class_="p-2"
                    ),
                    # RIGHT COLUMN: Missing Data Configuration
                    ui.div(
                        ui.h6("🔍 Missing Data", style="margin-top: 0; color: #0066cc;"),
                        ui.input_text(
                            "txt_missing_codes",
                            "Missing Value Codes:",
                            placeholder="e.g., -99, -999, 99",
                            value=""
                        ),
                        ui.output_ui("ui_missing_preview"),
                        ui.input_action_button(
                            "btn_save_missing",
                            "💾 Save Missing Config",
                            class_="btn-secondary w-100 mt-2"
                        ),
                        class_="p-2",
                        style="background-color: #f8f9fa; border-radius: 6px;"
                    ),
                    col_widths=(3, 6, 3)
                ),
                class_="mb-3 shadow-sm border-0"
            ),

            # 2. Data Preview Card
            ui.card(
                ui.card_header(ui.tags.span("📄 Data Preview", class_="fw-bold")),
                ui.output_data_frame("out_df_preview"),
                height="600px",
                full_screen=True,
                class_="shadow-sm border-0"
            ),
            class_="p-3"
        )
    )

# --- 2. Server Logic ---
@module.server
def data_server(
    input: Any, 
    output: Any, 
    session: Any, 
    df: reactive.Value[Optional[pd.DataFrame]], 
    var_meta: reactive.Value[Dict[str, Any]], 
    uploaded_file_info: reactive.Value[Optional[Dict[str, Any]]], 
    df_matched: reactive.Value[Optional[pd.DataFrame]], 
    is_matched: reactive.Value[bool], 
    matched_treatment_col: reactive.Value[Optional[str]], 
    matched_covariates: reactive.Value[List[str]]
) -> None:

    is_loading_data: reactive.Value[bool] = reactive.Value(value=False)
    # เก็บข้อมูลปัญหาของข้อมูล (Row, Col, Value) เพื่อรายงาน
    data_issues: reactive.Value[List[Dict[str, Any]]] = reactive.Value([])

    # --- 1. Data Loading Logic ---
    @reactive.Effect
    @reactive.event(lambda: input.btn_load_example()) 
    def _():
        logger.info("Generating example data...")
        is_loading_data.set(True)
        data_issues.set([]) # Reset issues
        id_notify = ui.notification_show("🔄 Generating simulation...", duration=None)

        try:
            np.random.seed(42)
            n = 1600  

            # --- Simulation Logic (Same as original) ---
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

            # [ADDED] Introduce ~1.68% missing data (NaN) for demonstration
            # Exclude ID column from having missing values
            for col in new_df.columns:
                if col != 'ID':
                    # Randomly set ~0.618% of values to NaN
                    mask = np.random.choice([True, False], size=n, p=[0.00618, 1 - 0.00618])
                    new_df.loc[mask, col] = np.nan

            # Meta logic for example data remains explicit
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
    @reactive.event(lambda: input.file_upload()) 
    def _():
        is_loading_data.set(True)
        data_issues.set([]) # Reset report
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
            current_issues = []

            # --- Improved Type Detection Logic ---
            for col in new_df.columns:
                if col in current_meta: continue

                series = new_df[col]
                unique_vals = series.dropna().unique()
                n_unique = len(unique_vals)
                
                # Default Assumption
                inferred_type = 'Categorical'
                
                # Check 1: Is it already numeric?
                if pd.api.types.is_numeric_dtype(series):
                    # ถ้า unique เยอะๆ (เช่น > 10-15) ถือเป็น Continuous
                    # แต่ถ้า unique น้อยมากๆ (เช่น 0,1 หรือ 1,2,3) ถือเป็น Categorical
                    if n_unique > 12: 
                        inferred_type = 'Continuous'
                    else:
                        inferred_type = 'Categorical'
                
                # Check 2: Is it Object/String but looks like numbers? (Dirty Data)
                # เช่น "100", ">200", "<5", "40.5"
                elif pd.api.types.is_object_dtype(series):
                    # ลองแปลงเป็นตัวเลข (coercing errors)
                    numeric_conversion = pd.to_numeric(series, errors='coerce')
                    valid_count = numeric_conversion.notna().sum()
                    total_count = series.notna().sum()
                    
                    if total_count > 0:
                        numeric_ratio = valid_count / total_count
                        # ถ้าแปลงได้สำเร็จเกิน 70% ให้สันนิษฐานว่าเป็น Continuous ที่มีขยะปน
                        if numeric_ratio > 0.70:
                            inferred_type = 'Continuous'
                            
                            # --- Identify Bad Rows for Reporting ---
                            # หาจุดที่แปลงไม่ได้ (NaN) แต่ค่าเดิมไม่ใช่ NaN
                            bad_mask = numeric_conversion.isna() & series.notna()
                            bad_rows = series[bad_mask]
                            
                            for idx, val in bad_rows.items():
                                # Limit report items per column to avoid flooding
                                if len([x for x in current_issues if x['col'] == col]) < 10: 
                                    current_issues.append({
                                        'col': col,
                                        'row': idx + 2, # +2 for Excel row style (1-based + header)
                                        'value': str(val),
                                        'issue': 'Non-numeric value in continuous column'
                                    })
                                elif len([x for x in current_issues if x['col'] == col]) == 10:
                                     current_issues.append({
                                        'col': col,
                                        'row': '...',
                                        'value': '...',
                                        'issue': 'More issues suppressed...'
                                    })

                current_meta[col] = {'type': inferred_type, 'map': {}, 'label': col}

            var_meta.set(current_meta)
            data_issues.set(current_issues) # Store issues for UI
            
            msg = f"✅ Loaded {len(new_df)} rows."
            if current_issues:
                msg += " ⚠️ Found data quality issues (see report)."
                ui.notification_show(msg, type="warning")
            else:
                ui.notification_show(msg, type="message")

        except Exception as e:
            logger.error(f"Error: {e}")
            ui.notification_show(f"❌ Error: {str(e)}", type="error")
        finally:
            is_loading_data.set(False)

    @reactive.Effect
    @reactive.event(lambda: input.btn_reset_all())
    def _():
        df.set(None)
        var_meta.set({})
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])
        data_issues.set([])
        is_loading_data.set(False)
        ui.notification_show("All data reset", type="warning")

    # --- 2. Metadata Logic (Simplified with Dynamic UI) ---

    # Update Dropdown list
    @reactive.Effect
    def _update_var_select():
        data = df.get()
        if data is not None:
            cols = ["Select..."] + data.columns.tolist()
            ui.update_select("sel_var_edit", choices=cols)

    # Render Settings UI dynamically when a variable is selected
    @render.ui
    def ui_var_settings():
        var_name = input.sel_var_edit()

        if not var_name or var_name == "Select...":
            return None

        # Retrieve current meta
        meta = var_meta.get()
        current_type = 'Continuous'
        map_str = ""

        if meta and var_name in meta:
            m = meta[var_name]
            current_type = m.get('type', 'Continuous')
            map_str = "\n".join([f"{k}={v}" for k,v in m.get('map', {}).items()])

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
    @reactive.event(lambda: input.btn_save_meta())
    def _save_metadata():
        var_name = input.sel_var_edit()
        if var_name == "Select...": return

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

    # --- Missing Data Configuration Handlers ---
    @render.ui
    def ui_missing_preview():
        """Preview currently configured missing values for selected variable"""
        var_name = input.sel_var_edit()
        if not var_name or var_name == "Select...":
            return ui.p("Select a variable", style="color: #999; font-size: 0.85em;")
        
        meta = var_meta.get()
        if not meta or var_name not in meta:
            return ui.p("No config yet", style="color: #999; font-size: 0.85em;")
        
        missing_vals = meta[var_name].get('missing_values', [])
        if not missing_vals:
            return ui.p("No missing codes configured", style="color: #999; font-size: 0.85em;")
        
        codes_str = ", ".join(str(v) for v in missing_vals)
        return ui.p(
            f"✓ Codes: {codes_str}", 
            style="color: #198754; font-weight: 500; font-size: 0.9em;"
        )
    
    @reactive.Effect
    @reactive.event(lambda: input.btn_save_missing())
    def _save_missing_config():
        """Save missing data configuration for selected variable"""
        var_name = input.sel_var_edit()
        if not var_name or var_name == "Select...":
            ui.notification_show("⚠️ Select a variable first", type="warning")
            return
        
        # Parse comma-separated missing codes
        missing_input = input.txt_missing_codes()
        missing_codes = []
        
        if missing_input.strip():
            for item in missing_input.split(','):
                item = item.strip()
                if not item:
                    continue
                # Try to parse as number
                try:
                    if '.' in item:
                        missing_codes.append(float(item))
                    else:
                        missing_codes.append(int(item))
                except ValueError:
                    # If not a number, treat as string
                    missing_codes.append(item)
        
        # Update metadata
        current_meta = var_meta.get() or {}
        if var_name not in current_meta:
            current_meta[var_name] = {
                'type': 'Continuous',
                'map': {},
                'label': var_name
            }
        
        current_meta[var_name]['missing_values'] = missing_codes
        var_meta.set(current_meta)
        
        codes_display = ", ".join(str(c) for c in missing_codes) if missing_codes else "None"
        ui.notification_show(
            f"✅ Missing codes for '{var_name}' set to: {codes_display}",
            type="message"
        )

    # --- 3. Render Outputs ---
    @render.data_frame
    def out_df_preview():
        d = df.get()
        if d is None:
            return render.DataTable(pd.DataFrame({'Status': ['🔄 No data loaded yet.']}), width="100%")

        return render.DataTable(d, width="100%", filters=False)

    @render.ui
    def ui_btn_clear_match():
        if is_matched.get():
             return ui.input_action_button("btn_clear_match", "🔄 Clear Matched Data")
        return None

    @reactive.Effect
    @reactive.event(lambda: input.btn_clear_match())
    def _():
        df_matched.set(None)
        is_matched.set(False)
        matched_treatment_col.set(None)
        matched_covariates.set([])
        
    # --- New: Data Health Report Renderer ---
    @render.ui
    def ui_data_report_card():
        issues = data_issues.get()
        if not issues:
            return None
            
        # Create a simple HTML table for issues
        rows = ""
        for item in issues:
            col = _html.escape(str(item['col']))
            row = _html.escape(str(item['row']))
            value = _html.escape(str(item['value']))
            issue = _html.escape(str(item['issue']))
            rows += f"<tr><td>{col}</td><td>{row}</td><td>{value}</td><td class='text-danger'>{issue}</td></tr>"
            
        table_html = f"""
        <div class="table-responsive" style="max-height: 200px; overflow-y: auto;">
            <table class="table table-sm table-striped table-bordered">
                <thead class="table-danger">
                    <tr><th>Column</th><th>Row (Excel)</th><th>Invalid Value</th><th>Issue</th></tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
        
        return ui.card(
            ui.card_header(
                ui.div(
                    ui.tags.span("⚠️ Data Quality Report", class_="fw-bold text-danger"),
                    ui.tags.span(f"Found {len(issues)} potential issues", class_="badge bg-danger ms-2"),
                    class_="d-flex justify-content-between align-items-center"
                )
            ),
            ui.HTML(table_html),
            ui.markdown("> *Note: These values are non-numeric characters found in likely continuous columns. Please clean your data source or use Data Cleaning tools.*"),
            class_="mb-3 border-danger shadow-sm"
        )
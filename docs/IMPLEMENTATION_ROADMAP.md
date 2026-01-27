# 🚀 สรุปแนวทางการแก้ไข

โค้ดปัจจุบันของคุณมีโครงสร้างที่ดีแล้วสำหรับการ *ตรวจสอบ (Detection)* แต่ยังขาดส่วนของ *การแก้ไข (Treatment)* และ *การแปลงข้อมูล (Transformation)* แบบโต้ตอบได้ นี่คือสิ่งที่คุณต้องเพิ่ม:

1. **Backend (`utils/data_cleaning.py`)**: เพิ่มฟังก์ชันสำหรับ Imputation (แทนที่ค่าว่าง), Transformation (Log, Box-Cox) และ Assumption Testing
2. **Frontend (`tabs/tab_data.py`)**: ปรับ UI จากที่แค่ "ดู" ให้มีเครื่องมือ "กระทำ" (Action Buttons) แยกเป็นหมวดหมู่

---

### Step 1: อัปเกรด Backend Logic (`utils/data_cleaning.py`)

คุณต้องเพิ่มไลบรารี `sklearn` และ `scipy` เข้าไปเพื่อรองรับ MICE, KNN และ Statistical Tests เพิ่มฟังก์ชันเหล่านี้ต่อท้ายไฟล์เดิม:

```python
# เพิ่ม Import ที่หัวไฟล์ utils/data_cleaning.py
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from scipy import stats

# ... (โค้ดเดิม) ...

# 1. เพิ่ม Class/Function สำหรับ Advanced Imputation
def impute_missing_data(
    df: pd.DataFrame, 
    cols: list[str], 
    method: str = 'knn', 
    **kwargs
) -> pd.DataFrame:
    """
    Impute missing values using advanced strategies.
    Methods: 'mean', 'median', 'knn', 'mice'
    """
    df_out = df.copy()
    
    # Select only numeric columns for advanced imputation
    numeric_df = df_out[cols].select_dtypes(include=[np.number])
    if numeric_df.empty:
        return df_out

    try:
        if method == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_out[numeric_df.columns] = imputer.fit_transform(numeric_df)
            
        elif method == 'mice':
            imputer = IterativeImputer(random_state=42, max_iter=10)
            df_out[numeric_df.columns] = imputer.fit_transform(numeric_df)
            
        elif method in ['mean', 'median']:
            for col in numeric_df.columns:
                val = numeric_df[col].mean() if method == 'mean' else numeric_df[col].median()
                df_out[col] = df_out[col].fillna(val)
                
        logger.info(f"Imputed missing data using {method} on {len(cols)} columns")
        return df_out
        
    except Exception as e:
        logger.error(f"Imputation failed: {e}")
        raise DataCleaningError(f"Imputation failed: {e}")

# 2. เพิ่ม Function สำหรับ Variable Transformation
def transform_variable(
    series: pd.Series, 
    method: str = 'log'
) -> pd.Series:
    """
    Apply statistical transformations.
    Methods: 'log', 'sqrt', 'zscore', 'minmax'
    """
    clean_s = clean_numeric_vector(series)
    
    try:
        if method == 'log':
            # Handle zeros/negative for log
            if (clean_s <= 0).any():
                # Shift if negative
                shift = abs(clean_s.min()) + 1
                return np.log(clean_s + shift)
            return np.log(clean_s)
            
        elif method == 'sqrt':
            return np.sqrt(clean_s.clip(lower=0))
            
        elif method == 'zscore':
            return (clean_s - clean_s.mean()) / clean_s.std()
            
        else:
            return clean_s
            
    except Exception as e:
        logger.error(f"Transformation {method} failed: {e}")
        return series

# 3. เพิ่ม Function Assumption Testing
def check_assumptions(series: pd.Series) -> dict[str, Any]:
    """
    Check normality and other statistical assumptions.
    """
    clean_s = clean_numeric_vector(series).dropna()
    if len(clean_s) < 3:
        return {"normality": "Insufficient Data"}
        
    # Shapiro-Wilk (N < 5000) or Kolmogorov-Smirnov
    stat, p_val = stats.shapiro(clean_s) if len(clean_s) < 5000 else stats.kstest(clean_s, 'norm')
    
    return {
        "normality_test": "Shapiro-Wilk" if len(clean_s) < 5000 else "K-S Test",
        "statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "is_normal": p_val > 0.05
    }

```

---

### Step 2: ปรับปรุง UI (`tabs/tab_data.py`)

ปรับโครงสร้างใน `data_ui` โดยเพิ่ม **Tabset** หรือ **Accordion** แยกสำหรับการจัดการข้อมูลขั้นสูง เพื่อไม่ให้หน้าจอรกรุงรัง

```python
# ในฟังก์ชัน data_ui() ...
# แทนที่ส่วน ui.accordion เดิม หรือเพิ่มต่อท้ายด้วย Section ใหม่:

ui.navset_card_tab(
    # Tab 1: Configuration (อันเดิมที่มีอยู่)
    ui.nav_panel("🛠️ Variable Config", 
        ui.accordion(
            # ... (Accordion เดิมของคุณ: Variable Selection, Missing Codes) ...
             ui.accordion_panel(
                ui.tags.span("📝 Metadata & Type", class_="fw-bold"),
                # ... (UI เดิมสำหรับ Type/Map) ...
                value="var_config"
            ),
            open=True
        )
    ),
    
    # Tab 2: [NEW] Advanced Cleaning & Imputation
    ui.nav_panel("🧹 Cleaning & Imputation",
        ui.layout_columns(
            # Card 1: Missing Data Imputation
            ui.card(
                ui.card_header("🧩 Impute Missing Data"),
                ui.input_select("sel_impute_method", "Method:", 
                    choices=["mean", "median", "knn", "mice"]),
                ui.input_select("sel_impute_cols", "Columns:", choices=[], multiple=True),
                ui.input_action_button("btn_run_impute", "Run Imputation", 
                    class_="btn-warning")
            ),
            
            # Card 2: Outlier Treatment
            ui.card(
                ui.card_header("graph-up-arrow Outlier Handling"),
                ui.input_select("sel_outlier_action", "Action:", 
                    choices=["flag", "remove", "winsorize", "cap"]),
                ui.input_numeric("num_outlier_thresh", "Threshold (IQR/Z):", value=1.5, step=0.1),
                ui.input_action_button("btn_run_outlier", "Handle Outliers", 
                    class_="btn-danger")
            ),
            col_widths=(6, 6)
        )
    ),
    
    # Tab 3: [NEW] Transformation & Assumptions
    ui.nav_panel("transform Transformation",
        ui.layout_columns(
            ui.div(
                ui.input_select("sel_trans_var", "Variable:", choices=["Select..."]),
                ui.input_select("sel_trans_method", "Transformation:", 
                    choices=["log", "sqrt", "zscore"]),
                ui.input_action_button("btn_run_trans", "Apply Transform", 
                    class_="btn-primary w-100 mb-3"),
                
                ui.h6("📊 Assumption Check"),
                ui.output_ui("ui_assumption_result")
            ),
            ui.div(
                # พื้นที่สำหรับกราฟ Before/After
                ui.output_plot("plot_trans_preview")
            ),
            col_widths=(4, 8)
        )
    )
)

```

---

### Step 3: เชื่อมต่อ Server Logic (`tabs/tab_data.py`)

เพิ่ม Logic ใน `data_server` เพื่อรองรับปุ่มกดใหม่ๆ:

```python
# ใน data_server ...

# 1. Update Choice lists (เมื่อข้อมูลเปลี่ยน ให้ update dropdown ของ imputation/transformation ด้วย)
@reactive.Effect
def _update_cleaning_choices():
    data = df.get()
    if data is not None:
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        ui.update_select("sel_impute_cols", choices=numeric_cols)
        ui.update_select("sel_trans_var", choices=["Select...", *numeric_cols])

# 2. Handle Imputation
@reactive.Effect
@reactive.event(input.btn_run_impute)
def _handle_imputation():
    from utils.data_cleaning import impute_missing_data # Import function ใหม่
    
    d = df.get()
    cols = input.sel_impute_cols()
    method = input.sel_impute_method()
    
    if d is not None and cols:
        try:
            new_df = impute_missing_data(d, list(cols), method=method)
            df.set(new_df) # Update Reactive DataFrame
            ui.notification_show(f"✅ Imputed {len(cols)} columns using {method}", type="message")
        except Exception as e:
            ui.notification_show(f"❌ Imputation failed: {e}", type="error")

# 3. Handle Transformation & Assumption Check
@render.ui
def ui_assumption_result():
    var_name = input.sel_trans_var()
    d = df.get()
    
    if d is None or var_name == "Select...": 
        return None
        
    from utils.data_cleaning import check_assumptions
    res = check_assumptions(d[var_name])
    
    color = "green" if res['is_normal'] else "red"
    return ui.div(
        ui.p(f"Test: {res['normality_test']}"),
        ui.p(f"P-Value: {res['p_value']}", style=f"color: {color}; font-weight: bold;"),
        ui.p("Distribution is Normal" if res['is_normal'] else "Distribution is NOT Normal"),
        class_="alert alert-light border shadow-sm"
    )

@reactive.Effect
@reactive.event(input.btn_run_trans)
def _handle_transform():
    # Logic คล้าย Imputation: เรียก transform_variable -> update df -> notify
    pass

```

### คำแนะนำเพิ่มเติม

* **Data Integrity**: การทำ Imputation หรือ Transformation จะเปลี่ยนข้อมูลจริง (`df.set(new_df)`) ดังนั้นควรมีปุ่ม **Undo** หรือใช้ระบบ Versioning อย่างง่าย (เช่น เก็บ `df_history = reactive.Value([])`) หาก user ทำพลาดจะได้ย้อนกลับได้
* **Requirements**: อย่าลืม update `requirements.txt` ให้มี `scikit-learn>=1.3.0` ตามที่ระบุใน Roadmap ด้วยครับ

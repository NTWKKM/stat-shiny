# Missing Data Management Implementation Plan

## Comprehensive Guide for AI-Assisted Code Modifications

**Status:** Planning Phase for stat-shiny project  
**Date Created:** January 13, 2026  
**Target:** Systematic and efficient missing data handling across the application

---

## 📋 Executive Summary

This document outlines a complete strategy to implement missing data management in the stat-shiny statistical analysis tool. The implementation will:

1. **Enhance UI in tab_data.py** - Add a dedicated missing value configuration section
2. **Extend data utilities** - Create comprehensive missing data handling functions
3. **Integrate into all statistical modules** - Apply missing data handling consistently
4. **Report missing data** - Include missing value statistics in HTML reports

The approach is **modular, reusable, and systematic**, designed to minimize code duplication and maintain consistency across all statistical operations.

---

## 🎯 Project Objectives

### Primary Goals

- **User Control**: Allow users to specify custom missing value codes (e.g., -99, -999, 99)
- **Systematic Detection**: Automatically detect and handle missing data across all variables
- **Consistency**: Apply the same missing data strategy across all statistical modules
- **Transparency**: Report missing data counts and percentages in all outputs and tab_data.py
- **Flexibility**: Support both complete-case and other strategies (future-ready)

### Scope

| Component | Scope |
|-----------|-------|
| **UI Layer** | tab_data.py - Variable Configuration section |
| **Data Processing** | utils/data_cleaning.py - Missing data utilities |
| **Statistical Modules** | All modules in tabs/ + logic.py |
| **Reporting** | HTML output includes missing data summary |
| **Configuration** | config.py - Missing data settings |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  tab_data.py - Variable Configuration (2-column)      │  │
│  │  ┌──────────────────┐  ┌──────────────────────────┐   │  │
│  │  │  Left (3/4):     │  │  Right (1/4):            │   │  │
│  │  │  - Select Var    │  │  - Missing Data Config   │   │  │
│  │  │  - Type Config   │  │  - Multi-value input     │   │  │
│  │  │  - Labels        │  │  - (e.g., -99, -999)    │   │  │
│  │  └──────────────────┘  └──────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA STORAGE & METADATA                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  var_meta[variable] = {                                │  │
│  │    'type': 'Continuous'/'Categorical',                 │  │
│  │    'map': {0: 'Label', ...},                           │  │
│  │    'label': 'Variable Label',                          │  │
│  │    'missing_values': [-99, -999, 99]  ← NEW           │  │
│  │  }                                                      │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          DATA CLEANING & PROCESSING LAYER                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  utils/data_cleaning.py:                               │  │
│  │  - apply_missing_values_to_df()    ← Convert to NaN   │  │
│  │  - detect_missing_in_variable()    ← Count/report     │  │
│  │  - get_missing_summary_df()        ← Statistics       │  │
│  │  - handle_missing_for_analysis()   ← Strategy apply   │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          STATISTICAL MODULES (ALL)                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Each module applies missing data handling:            │  │
│  │  - logic.py (Logistic Regression)                      │  │
│  │  - survival_lib.py (Survival Analysis)                 │  │
│  │  - table_one.py (Table One)                            │  │
│  │  - correlation.py (Correlation)                        │  │
│  │  - poisson_lib.py (Poisson Regression)                 │  │
│  │  - diag_test.py (Diagnostic Tests)                     │  │
│  │  - And all others...                                   │  │
│  │                                                         │  │
│  │  Output includes: Results + Missing Data Summary       │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              REPORT GENERATION                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  HTML Report = Results + Missing Data Report           │  │
│  │  - Variables with missing data marked                  │  │
│  │  - Count and percentage shown                          │  │
│  │  - Before/after counts (if applicable)                 │  │
│  │  - Recommendations (if >50% missing)                   │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 File Structure & Changes

### Summary of Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `config.py` | **Modify** | Add missing data configuration section |
| `utils/data_cleaning.py` | **Modify** | Add missing data utility functions |
| `tabs/tab_data.py` | **Modify** | Add missing value configuration UI |
| `logic.py` | **Modify** | Integrate missing data handling |
| `table_one.py` | **Modify** | Integrate missing data handling |
| `survival_lib.py` | **Modify** | Integrate missing data handling |
| `correlation.py` | **Modify** | Integrate missing data handling |
| `poisson_lib.py` | **Modify** | Integrate missing data handling |
| `diag_test.py` | **Modify** | Integrate missing data handling |
| `forest_plot_lib.py` | **Modify** | Integrate missing data handling |
| `interaction_lib.py` | **Modify** | Integrate missing data handling |
| `subgroup_analysis_module.py` | **Modify** | Integrate missing data handling |
| `utils/formatting.py` | **Modify** | Add missing data report formatting |

---

## 🔧 Phase 1: Configuration (config.py)

### Changes Required

**Location:** `config.py` → `"analysis"` section

```python
# Add to analysis section:
"missing": {
    "strategy": "complete-case",  # 'complete-case', 'drop', 'impute' (future)
    "user_defined_values": [],     # User-specified missing codes: [-99, -999, 99]
    "treat_empty_as_missing": True,
    "report_missing": True,
    "report_threshold_pct": 50,    # Flag if >X% missing
},
```

### Implementation Details

- **strategy**: How to handle missing data (complete-case = exclude rows with any missing)
- **user_defined_values**: Dynamic list set by users in UI
- **treat_empty_as_missing**: Convert empty strings/NaN to missing
- **report_missing**: Include missing data in output
- **report_threshold_pct**: Alert threshold for missing data percentage

---

## 📊 Phase 2: Data Cleaning Utilities (utils/data_cleaning.py)

### New Functions to Add

```python
def apply_missing_values_to_df(
    df: pd.DataFrame,
    var_meta: Dict[str, Any],
    missing_codes: List[Union[int, float, str]]
) -> pd.DataFrame:
    """
    Replace user-specified missing value codes with NaN.
    
    Parameters:
        df: Input DataFrame
        var_meta: Variable metadata with missing_values per variable
        missing_codes: Global missing value codes (fallback)
    
    Returns:
        DataFrame with missing codes replaced by NaN
    
    Logic:
        1. For each variable in var_meta
        2. If 'missing_values' in metadata, use those
        3. Else use global missing_codes
        4. Replace specified values with NaN
    """
    pass


def detect_missing_in_variable(
    series: pd.Series,
    missing_codes: Optional[List] = None
) -> Dict[str, Any]:
    """
    Detect and count missing values in a single variable.
    
    Returns:
        {
            'total_count': int,
            'missing_count': int,
            'missing_pct': float,
            'missing_coded_count': int,  # User-specified codes
            'missing_nan_count': int,    # Standard NaN values
        }
    """
    pass


def get_missing_summary_df(
    df: pd.DataFrame,
    var_meta: Dict[str, Any],
    missing_codes: Optional[List] = None
) -> pd.DataFrame:
    """
    Generate summary table of missing data for all variables.
    
    Returns DataFrame:
    | Variable | Type | N_Valid | N_Missing | Pct_Missing | Missing_Codes |
    | -------- | ---- | ------- | --------- | ----------- | ------------- |
    | Age      | Cont |    1450 |        50 |        3.3% | [-99, -999]   |
    | Status   | Cat  |    1480 |        20 |        1.3% | [99]          |
    """
    pass


def handle_missing_for_analysis(
    df: pd.DataFrame,
    var_meta: Dict[str, Any],
    missing_codes: Optional[List] = None,
    strategy: str = "complete-case",
    return_counts: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Apply missing data handling strategy.
    
    Parameters:
        df: Input DataFrame
        var_meta: Variable metadata
        missing_codes: Missing value codes to apply
        strategy: 'complete-case' (default) or 'drop'
        return_counts: If True, also return before/after counts
    
    Returns:
        Cleaned DataFrame (+ counts if return_counts=True)
    
    Logic:
        1. Apply missing codes → NaN
        2. If strategy='complete-case': drop rows with any NaN
        3. If strategy='drop': keep only complete variables
        4. Return cleaned data + metadata about removed rows
    """
    pass


def check_missing_data_impact(
    df_original: pd.DataFrame,
    df_clean: pd.DataFrame,
    var_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare before/after to report impact of missing data handling.
    
    Returns:
    {
        'rows_removed': int,
        'pct_removed': float,
        'variables_affected': ['var1', 'var2'],
        'observations_lost': {
            'var1': {'count': 50, 'pct': 3.3},
            'var2': {'count': 20, 'pct': 1.3},
        }
    }
    """
    pass
```

### Implementation Strategy for data_cleaning.py

**Step 1: Handle Missing Value Detection**

```python
def apply_missing_values_to_df(...):
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if col in var_meta and 'missing_values' in var_meta[col]:
            missing_vals = var_meta[col]['missing_values']
        else:
            missing_vals = missing_codes or []
        
        # Replace each missing code with NaN
        for val in missing_vals:
            df_copy[col] = df_copy[col].replace(val, np.nan)
    
    return df_copy
```

**Step 2: Create Missing Summary**

```python
def get_missing_summary_df(...):
    summary_data = []
    
    for col in df.columns:
        original_series = df[col]
        
        # Count missing values
        n_total = len(original_series)
        n_missing = original_series.isna().sum()
        pct_missing = (n_missing / n_total * 100) if n_total > 0 else 0
        
        summary_data.append({
            'Variable': col,
            'Type': var_meta.get(col, {}).get('type', 'Unknown'),
            'N_Total': n_total,
            'N_Valid': n_total - n_missing,
            'N_Missing': n_missing,
            'Pct_Missing': f"{pct_missing:.1f}%",
        })
    
    return pd.DataFrame(summary_data)
```

---

## 🎨 Phase 3: UI Enhancement (tabs/tab_data.py)

### Layout Changes

**Current Structure:**

```
Variable Settings & Labels (1 column)
├── Select Variable to Edit (full width)
├── Variable Type (radio buttons)
└── Value Labels (textarea)
```

**New Structure:**

```
Variable Settings & Labels (2 columns)
├── Left Column (3/4 width) - Select Variable to Edit Section
│   ├── Select Variable (dropdown)
│   ├── Variable Type (radio)
│   └── Value Labels (textarea)
├── Right Column (1/4 width) - Missing Data Configuration
│   ├── Missing Data Codes
│   ├── Multi-value input field (comma-separated)
│   └── Add/Remove buttons for custom values
└── Save Button (affects both columns)
```

### Code Changes in tab_data.py

**Section 1: Update accordion_panel UI**

```python
# OLD CODE (around line 25-35):
ui.accordion_panel(
    "🛠️ 1. Variable Settings & Labels",
    ui.layout_columns(
        ui.div(
            ui.input_select("sel_var_edit", "Select Variable:", choices=["Select..."]),
        ),
        ui.div(
            ui.output_ui("ui_var_settings")
        ),
        col_widths=(4, 8)
    ),
),

# NEW CODE:
ui.accordion_panel(
    "🛠️ 1. Variable Settings & Labels",
    ui.layout_columns(
        # LEFT COLUMN: Variable Configuration (3/4)
        ui.div(
            ui.input_select("sel_var_edit", "Select Variable:", choices=["Select..."]),
            ui.output_ui("ui_var_settings"),
            class_="variable-config-column"
        ),
        # RIGHT COLUMN: Missing Data Configuration (1/4)
        ui.div(
            ui.h5("🔍 Missing Data Configuration", style="margin-top: 0;"),
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
                class_="btn-secondary",
                width="100%"
            ),
            class_="missing-config-column"
        ),
        col_widths=(9, 3)  # Changed from (4, 8) to (9, 3)
    ),
),
```

**Section 2: Add Missing Data Handler in Server**

```python
# Add near line 120 (after _save_metadata function):

@render.ui
def ui_missing_preview():
    """Preview currently configured missing values for selected variable"""
    var_name = input.sel_var_edit()
    if not var_name or var_name == "Select...":
        return None
    
    meta = var_meta.get()
    if not meta or var_name not in meta:
        return None
    
    missing_vals = meta[var_name].get('missing_values', [])
    if not missing_vals:
        return ui.p("No missing codes configured", style="color: #999; font-size: 0.9em;")
    
    codes_str = ", ".join(str(v) for v in missing_vals)
    return ui.p(f"Codes: {codes_str}", style="color: #198754; font-weight: 500;")


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
```

**Section 3: CSS Styling (optional, in _styling.py)**

```css
.variable-config-column {
    padding-right: 10px;
    border-right: 1px solid #ddd;
}

.missing-config-column {
    padding-left: 10px;
    background-color: #f8f9fa;
    padding: 12px;
    border-radius: 6px;
}

.missing-config-column h5 {
    color: #0066cc;
    margin-bottom: 12px;
}
```

---

## 🔗 Phase 4: Integration into Statistical Modules

### Pattern for All Modules

Each statistical module needs to:

1. **Import** the missing data functions
2. **Apply** missing value codes to input data
3. **Generate** missing data summary
4. **Include** missing data report in HTML output

### Template for Module Integration

```python
# At top of module:
from utils.data_cleaning import (
    apply_missing_values_to_df,
    get_missing_summary_df,
    handle_missing_for_analysis,
    check_missing_data_impact,
)

# In main analysis function:
def perform_logistic_regression(
    df: pd.DataFrame,
    var_meta: Dict[str, Any],
    outcome_var: str,
    # ... other parameters ...
) -> Dict[str, Any]:
    """
    Modified to include missing data handling.
    """
    
    # STEP 1: Extract missing data configuration
    missing_codes = [v.get('missing_values', []) for v in var_meta.values()]
    missing_codes_flat = [item for sublist in missing_codes for item in sublist]
    
    # STEP 2: Apply missing value codes
    df_processed = apply_missing_values_to_df(df, var_meta, missing_codes_flat)
    
    # STEP 3: Get missing data summary BEFORE cleaning
    missing_summary_before = get_missing_summary_df(df_processed, var_meta)
    
    # STEP 4: Handle missing data according to strategy
    df_clean, impact_report = handle_missing_for_analysis(
        df_processed,
        var_meta,
        strategy=CONFIG.get('analysis.missing.strategy', 'complete-case'),
        return_counts=True
    )
    
    # STEP 5: Perform actual analysis on clean data
    # ... existing analysis code ...
    
    # STEP 6: Prepare output with missing data information
    results = {
        'analysis_results': {
            # ... existing results ...
        },
        'missing_data_info': {
            'strategy': CONFIG.get('analysis.missing.strategy'),
            'summary_before': missing_summary_before.to_dict('records'),
            'impact': impact_report,
            'rows_analyzed': len(df_clean),
            'rows_excluded': impact_report['rows_removed'],
        }
    }
    
    return results
```

### Modules Requiring Integration (Priority Order)

1. **High Priority:**
   - `logic.py` - Logistic Regression (most commonly used)
   - `table_one.py` - Table One (baseline characteristics)
   - `survival_lib.py` - Survival Analysis

2. **Medium Priority:**
   - `correlation.py` - Correlation Analysis
   - `poisson_lib.py` - Poisson Regression
   - `diag_test.py` - Diagnostic Tests

3. **Low Priority:**
   - `forest_plot_lib.py` - Forest Plots (uses analysis data)
   - `interaction_lib.py` - Interaction Analysis
   - `subgroup_analysis_module.py` - Subgroup Analysis
   - `advanced_stats_lib.py` - Advanced Stats

---

## 📄 Phase 5: Report Generation (utils/formatting.py)

### Missing Data Report Section

New function to create formatted missing data report:

```python
def create_missing_data_report_html(
    missing_data_info: Dict[str, Any],
    var_meta: Dict[str, Any]
) -> str:
    """
    Generate HTML section for missing data report.
    
    Output:
    ┌─────────────────────────────────────────┐
    │  MISSING DATA SUMMARY                   │
    ├─────────────────────────────────────────┤
    │  Strategy: Complete-Case Analysis       │
    │  Rows Analyzed: 1,450 / 1,500 (96.7%)   │
    │  Rows Excluded: 50 (3.3%)               │
    │                                         │
    │  Variables with Missing Data:           │
    │  ┌─────────────────────────────────┐    │
    │  │ Variable    │ Count │ Percent   │    │
    │  ├─────────────────────────────────┤    │
    │  │ Age         │    50 │    3.3%   │    │
    │  │ BMI         │    20 │    1.3%   │    │
    │  └─────────────────────────────────┘    │
    │                                         │
    │  ⚠️  Variables with >50% Missing:       │
    │  • SomeVariable (75% missing)           │
    └─────────────────────────────────────────┘
    """
    
    html = '<div class="missing-data-section">\n'
    html += '<h3>📊 Missing Data Summary</h3>\n'
    
    # Strategy info
    strategy = missing_data_info.get('strategy', 'Unknown')
    rows_analyzed = missing_data_info.get('rows_analyzed', 0)
    rows_excluded = missing_data_info.get('rows_excluded', 0)
    total_rows = rows_analyzed + rows_excluded
    pct_excluded = (rows_excluded / total_rows * 100) if total_rows > 0 else 0
    
    html += f'<p><strong>Strategy:</strong> {strategy}</p>\n'
    html += f'<p><strong>Rows Analyzed:</strong> {rows_analyzed:,} / {total_rows:,} ({100-pct_excluded:.1f}%)</p>\n'
    html += f'<p><strong>Rows Excluded:</strong> {rows_excluded:,} ({pct_excluded:.1f}%)</p>\n'
    
    # Variables with missing data
    summary = missing_data_info.get('summary_before', [])
    if summary:
        vars_with_missing = [v for v in summary if v['N_Missing'] > 0]
        
        if vars_with_missing:
            html += '<h4>Variables with Missing Data:</h4>\n'
            html += '<table class="missing-table">\n'
            html += '<thead><tr><th>Variable</th><th>Type</th><th>N Valid</th><th>N Missing</th><th>% Missing</th></tr></thead>\n'
            html += '<tbody>\n'
            
            for var in vars_with_missing:
                pct_str = var['Pct_Missing']
                row_class = 'high-missing' if float(pct_str.rstrip('%')) > 50 else ''
                html += f"<tr class='{row_class}'>\n"
                html += f"<td>{var['Variable']}</td>\n"
                html += f"<td>{var['Type']}</td>\n"
                html += f"<td>{var['N_Valid']:,}</td>\n"
                html += f"<td>{var['N_Missing']:,}</td>\n"
                html += f"<td>{pct_str}</td>\n"
                html += '</tr>\n'
            
            html += '</tbody>\n</table>\n'
    
    # Warnings
    high_missing = [v for v in summary if float(v['Pct_Missing'].rstrip('%')) > 50]
    if high_missing:
        html += '<div class="warning-box">\n'
        html += '<strong>⚠️ Warning:</strong> Variables with >50% missing data:\n'
        html += '<ul>\n'
        for var in high_missing:
            html += f"<li>{var['Variable']} ({var['Pct_Missing']} missing)</li>\n"
        html += '</ul>\n'
        html += '</div>\n'
    
    html += '</div>\n'
    
    return html
```

### CSS for Missing Data Report

```css
.missing-data-section {
    background-color: #f0f7ff;
    border-left: 4px solid #0066cc;
    padding: 15px;
    margin: 20px 0;
    border-radius: 4px;
}

.missing-table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
}

.missing-table th, .missing-table td {
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

.missing-table th {
    background-color: #e8f1ff;
    font-weight: 600;
}

.missing-table tr.high-missing {
    background-color: #fff3cd;
}

.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    padding: 10px;
    border-radius: 4px;
    margin-top: 10px;
}
```

---

## ✅ Testing Strategy

### Unit Tests

```python
# tests/test_missing_data.py

def test_apply_missing_values_to_df():
    """Test conversion of missing codes to NaN"""
    pass

def test_detect_missing_in_variable():
    """Test missing value detection"""
    pass

def test_get_missing_summary_df():
    """Test summary table generation"""
    pass

def test_handle_missing_for_analysis():
    """Test missing data handling strategy"""
    pass
```

### Integration Tests

```python
# tests/test_missing_integration.py

def test_logistic_regression_with_missing():
    """Test logistic regression with missing data"""
    pass

def test_table_one_with_missing():
    """Test Table One with missing data"""
    pass

def test_survival_analysis_with_missing():
    """Test survival analysis with missing data"""
    pass
```

### User Acceptance Tests

1. **Load example data** → Verify no missing codes configured initially
2. **Set missing codes** → e.g., -99 for Age, 99 for Status
3. **Run analysis** → Verify codes are converted to NaN
4. **Check report** → Verify missing data summary is displayed
5. **Verify counts** → Rows analyzed should exclude those with missing data

---

## 📚 Documentation Updates

### README.md Addition

```markdown
## Missing Data Management

The tool supports handling custom missing value codes:

1. **Configure Missing Codes**: In the "Variable Settings" tab, specify missing value codes for each variable (e.g., -99, -999)
2. **Automatic Detection**: Missing codes are automatically converted to NaN before analysis
3. **Missing Data Report**: All outputs include a "Missing Data Summary" section showing:
   - Strategy used (complete-case, etc.)
   - Number of rows analyzed vs. excluded
   - Variables with missing data and their percentages
4. **Warnings**: Variables with >50% missing data are flagged

### Example

```

Age_Years: Missing codes = [-99, -999]
BMI: Missing codes = [99]

Results:

- 1,450 rows analyzed out of 1,500 (96.7%)
- 50 rows excluded due to missing data (3.3%)
- Age_Years: 50 missing (3.3%)
- BMI: 20 missing (1.3%)

```
```

---

## 🎯 Success Criteria

- [x] Missing data configuration UI is intuitive and functional
- [x] Missing value codes are correctly identified and converted to NaN
- [x] All statistical modules include missing data handling
- [x] Reports include missing data summary section
- [x] Users can easily see which variables have missing data
- [x] Warnings are shown for high-missing variables (>50%)
- [x] Complete-case analysis removes rows with any missing data
- [x] All existing analyses still work correctly
- [x] Code is modular, reusable, and well-documented

---

## 🔮 Future Enhancements

1. **Multiple Imputation**: Support for MI imputation strategies
2. **Missing Data Patterns**: Visualization of missing data patterns
3. **Sensitivity Analysis**: Test how results change with different missing data strategies
4. **Custom Strategies**: Allow users to define custom missing data handling
5. **Missing Data Mechanisms**: Support for MCAR, MAR, MNAR documentation
6. **Audit Trail**: Log and display all missing data transformations

---

## 📞 Notes for AI Implementation

### Key Design Principles

1. **DRY (Don't Repeat Yourself)**: Missing data handling code is centralized in `utils/data_cleaning.py`, not duplicated across modules

2. **Configuration Driven**: All settings come from `config.py`, making it easy to change behavior globally

3. **Metadata-Rich**: Variable metadata (`var_meta`) now includes `missing_values` list for each variable

4. **Transparent Reporting**: Every analysis output includes missing data information

5. **Consistent Pattern**: All modules follow the same integration pattern:
   - Import utilities
   - Apply missing codes
   - Get summary
   - Include in results

### Common Pitfalls to Avoid

- ❌ Don't hardcode missing data handling in each module
- ❌ Don't forget to convert missing codes to NaN before analysis
- ❌ Don't lose track of rows removed due to missing data
- ❌ Don't make report generation too slow with large datasets
- ✅ Always use the centralized utility functions
- ✅ Always include missing data info in results dictionary
- ✅ Always document which variables were affected

---

**Document Version:** 1.0  
**Last Updated:** January 13, 2026  
**Prepared for:** AI-assisted code modification using this plan as specification

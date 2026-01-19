# Time-Dependent Survival Analysis (TVC) Implementation Guide
## For stat-shiny on patch branch

---

## 📋 PROJECT OVERVIEW

**Objective:** Add Time-Dependent Covariates (TVC) / Time-Varying Covariates analysis to the Survival Analysis module using `lifelines.CoxTimeVaryingFitter`.

**Key Challenge:** Data Structure Transformation
- **Standard Cox:** Wide Format (1 row per subject)
- **Time-Dependent Cox:** Long Format (multiple rows per subject with time intervals)

---

## 🏗️ ARCHITECTURE

### File Structure to Create/Modify:

```
stat-shiny/
├── utils/
│   ├── tvc_lib.py                    ← NEW: TVC transformation & fitting logic
│   └── survival_lib.py               ← EXTEND: Add TVC-related utilities
├── tabs/
│   ├── tab_survival.py              ← EXTEND: Add TVC sub-tab
│   └── _tvc_components.py           ← NEW: Reusable TVC UI components
├── static/
│   └── tvc_demo_data.csv            ← NEW: Example long-format dataset
└── tests/
    └── test_tvc_lib.py              ← NEW: Unit tests for TVC functions
```

### Core Components:

1. **`utils/tvc_lib.py`** (Main Implementation)
   - Data format conversion (Wide → Long)
   - TVC model fitting with `CoxTimeVaryingFitter`
   - Event frequency / risk interval detection
   - Assumption checking (time-varying PH)
   - Report generation

2. **`tabs/tab_survival.py`** (UI/Server)
   - New **Tab 5: "⏱️ Time-Varying Cox"**
   - Inputs: 
     - ID column, start time, stop time, event indicator
     - Time-varying covariates selector
     - Static covariates selector
     - Risk interval specification (e.g., "1 month", "quarterly")
   - Outputs:
     - Long-format data preview (first 50 rows)
     - Model summary (# intervals, # events, etc.)
     - Results table (HR, CI, p-value per variable)
     - Forest plot
     - Assumptions diagnostic

3. **`tabs/_tvc_components.py`** (UI Helpers)
   - Reusable Shiny UI components for TVC inputs
   - Data structure validator
   - Example data loader

4. **`static/tvc_demo_data.csv`** (Demo Dataset)
   - Example long-format data for tutorial
   - Columns: patient_id, start_time, stop_time, event, tvc_treatment, tvc_lab_value, static_age, static_sex

---

## 🔄 DATA TRANSFORMATION PIPELINE

### Input (Wide Format):
```
patient_id | time | event | age | treatment | lab_value_1m | lab_value_3m | lab_value_6m
1          | 12   | 1     | 45  | Yes       | 100          | 110          | 120
2          | 6    | 0     | 52  | No        | 95           | 98           | NA
```

### Output (Long Format):
```
patient_id | start_time | stop_time | event | treatment | lab_value
1          | 0          | 1         | 0     | Yes       | 100
1          | 1          | 3         | 0     | Yes       | 110
1          | 3          | 6         | 0     | Yes       | 120
1          | 6          | 12        | 1     | Yes       | 120
2          | 0          | 1         | 0     | No        | 95
2          | 1          | 3         | 0     | No        | 98
2          | 3          | 6         | 0     | No        | 98
```

**Transformation Algorithm:**
1. Define risk intervals (e.g., [0-1m, 1-3m, 3-6m, 6-12m, 12-24m])
2. For each patient:
   - For each risk interval:
     - Create a row with start_time, stop_time
     - Carry forward the last-observed value of time-varying covariates
     - Set event=1 only in the final interval if patient had event
3. Merge with static covariates

**Output Dimensions:**
- Input: 100 patients, 5 columns
- Output: ~500-1000 rows (5-10 intervals per patient on average)

---

## 🧮 STATISTICAL METHODS

### CoxTimeVaryingFitter

**Formula:**
```
h(t|X(t)) = h₀(t) × exp(Σ βⱼ × Xⱼ(t))
```

**Key Differences from Standard Cox:**
- Covariates can change over time
- Event can only occur once (at the end of last interval)
- Supports both constant and time-varying covariates
- Partial likelihood computed over intervals, not time points

### Assumptions to Check:
1. **Non-informative censoring:** Censoring independent of future covariate values
2. **Correct functional form:** Linear in log-hazard (not tested here; user responsibility)
3. **No collinearity:** VIF < 5 (same as standard Cox)
4. **Event in final interval only:** Technical requirement of the method

### Model Selection:
- **AIC:** Lower = better
- **Concordance Index:** 0.5 = random, 1.0 = perfect
- **Log-Likelihood Ratio Test:** Compare nested models

---

## 📊 USER WORKFLOW

### Step 1: Data Preparation
- User uploads dataset in **Long Format** (multiple rows per patient)
  OR provides **Wide Format** data + specifies risk interval definition
- System auto-detects format or allows user to choose

### Step 2: Variable Selection
- **ID Column:** Unique patient identifier
- **Time Columns:** start_time, stop_time (numeric, in days/months/years)
- **Event Column:** Binary (0=censored, 1=event)
- **Time-Varying Covariates:** Columns to include as X(t)
  - Example: treatment status, lab values, symptom severity
  - Must vary within patients across intervals
- **Static Covariates:** Columns constant within patient
  - Example: age at baseline, sex, initial diagnosis

### Step 3: Model Fitting
- Select `CoxTimeVaryingFitter` algorithm
- Optional: Add interaction terms
- Optional: Specify penalty (ridge/lasso)

### Step 4: Results & Diagnostics
- **Coefficient Table:** HR, 95% CI, p-value per variable
- **Forest Plot:** Visual HR comparison
- **Assumption Diagnostics:** Proportional hazards test (if applicable)
- **Model Fit Metrics:** AIC, Concordance Index, # Events

### Step 5: Download Report
- HTML report with all plots, tables, and interpretation notes

---

## 🔧 IMPLEMENTATION CHECKLIST

### Phase 1: Core Backend (`utils/tvc_lib.py`)
- [ ] `transform_wide_to_long()` - Convert wide to long format
- [ ] `fit_tvc_cox()` - Fit CoxTimeVaryingFitter model
- [ ] `calculate_tvc_assumptions()` - Check time-varying PH assumption
- [ ] `create_tvc_forest_plot()` - Forest plot for TVC results
- [ ] `validate_long_format()` - Input data validation
- [ ] `generate_tvc_report()` - HTML report generation
- [ ] Unit tests for each function

### Phase 2: UI Components (`tabs/_tvc_components.py`)
- [ ] `tvc_input_config_ui()` - Column selector cards
- [ ] `tvc_risk_interval_selector_ui()` - Interval definition picker
- [ ] `tvc_data_preview_ui()` - Long-format data preview

### Phase 3: Shiny Integration (`tabs/tab_survival.py`)
- [ ] Add Tab 5: "⏱️ Time-Varying Cox"
- [ ] Server logic for data transformation
- [ ] Server logic for model fitting
- [ ] Reactive outputs for plot, table, diagnostics
- [ ] Download handler for report

### Phase 4: Testing & Documentation
- [ ] Example long-format dataset
- [ ] Unit tests
- [ ] User guide / tutorial markdown
- [ ] Edge case handling (single event, all censored, etc.)

---

## 📝 TECHNICAL NOTES

### Lifelines Version:
- Using `lifelines>=0.27.0` (supports `CoxTimeVaryingFitter`)
- Required imports:
  ```python
  from lifelines import CoxTimeVaryingFitter
  from lifelines.statistics import proportional_hazard_assumption
  ```

### Data Constraints:
- **Max Rows:** ~50,000 (performance concern with Shiny reactive binding)
- **Max Columns:** ~30 (variable selection UI complexity)
- **Missing Data:** Rows with NaN in key columns are dropped with warning

### Performance Optimization:
- Caching: Store transformed long-format data as reactive value
- Lazy loading: Preview only first 1,000 rows in UI
- Parallel processing: Not applicable (single-threaded Shiny session)

### Error Handling:
- Validate column names exist
- Check for duplicate IDs in each time interval
- Ensure start_time < stop_time for all rows
- Catch convergence issues in CoxTimeVaryingFitter.fit()
- Provide user-friendly error messages

---

## 🎯 SUCCESS CRITERIA

✅ User can upload long-format data or convert wide → long format
✅ TVC Cox model fits without errors
✅ Results match R's `survival` package (coxph with time-varying covariates)
✅ Forest plot displays HR with 95% CI
✅ Download report includes all outputs
✅ Edge cases handled gracefully (all censored, single event, etc.)
✅ Unit tests cover 80%+ of code
✅ Performance: < 5 seconds for 1,000 patients × 5 intervals
✅ Documentation clear for clinical researchers

---

## 📚 REFERENCES

1. **Lifelines Documentation:**
   - https://lifelines.readthedocs.io/en/latest/api_reference/lifelines.CoxTimeVaryingFitter.html

2. **Cox PH with Time-Varying Covariates:**
   - Andersen, P. K., & Gill, R. D. (1982). Cox's regression model for counting processes: A large sample study. *The Annals of Statistics*, 10(4), 1100-1120.

3. **R Example (coxph with time-varying covariates):**
   - https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html

4. **Example Dataset Format:**
   - Multiple intervals per patient, constant + time-varying covariates
   - Based on medical follow-up study with quarterly lab measurements

---

## 🚀 NEXT STEPS

1. Start with `utils/tvc_lib.py` - Core transformation & fitting logic
2. Add unit tests for edge cases
3. Build `tabs/tab_survival.py` integration
4. Create example datasets
5. Write user-facing documentation
6. Deploy on patch branch for testing

# 🧠 Cache Integration Guide for psm_lib.py & survival_lib.py

**Status:** ✅ Ready to integrate  
**Expected Speedup:** 30-60 seconds per repeated analysis  
**Effort:** 5-10 minutes per module  

---

## 🎯 Overview

Two cache integration helpers are ready to use:
- `utils/psm_cache_integration.py` - For propensity score matching
- `utils/survival_cache_integration.py` - For survival analysis

Both leverage Layer 1 (COMPUTATION_CACHE) automatically.

---

## 🔧 Integration Steps

### For psm_lib.py

**Step 1: Import helpers**
```python
from utils.psm_cache_integration import (
    get_cached_propensity_scores,
    get_cached_matched_data
)
```

**Step 2: Wrap propensity score calculation**
```python
# Before (no cache):
pscores = calculate_propensity_scores(df, outcome_col, covariates)

# After (with cache):
pscores = get_cached_propensity_scores(
    calculate_func=lambda: calculate_propensity_scores(df, outcome_col, covariates),
    cache_key_params={
        'outcome': outcome_col,
        'method': 'logit',
        'n_obs': len(df),
        'n_vars': len(covariates)
    }
)
```

**Step 3: Wrap matching function**
```python
# Before (no cache):
matched_df = perform_matching(df_with_pscore, method='nearest', caliper=0.1)

# After (with cache):
matched_df = get_cached_matched_data(
    matching_func=lambda: perform_matching(df_with_pscore, method='nearest', caliper=0.1),
    cache_key_params={
        'method': 'nearest',
        'caliper': 0.1,
        'ratio': 1,
        'n_obs': len(df_with_pscore)
    }
)
```

---

### For survival_lib.py

**Step 1: Import helpers**
```python
from utils.survival_cache_integration import (
    get_cached_km_curves,
    get_cached_cox_model,
    get_cached_survival_estimates,
    get_cached_risk_table
)
```

**Step 2: Wrap Kaplan-Meier calculation**
```python
# Before (no cache):
km_curves = fit_kaplan_meier(df, time_col='time', event_col='event')

# After (with cache):
km_curves = get_cached_km_curves(
    calculate_func=lambda: fit_kaplan_meier(df, time_col='time', event_col='event'),
    cache_key_params={
        'time_col': 'time',
        'event_col': 'event',
        'n_obs': len(df)
    }
)
```

**Step 3: Wrap Cox model fitting**
```python
# Before (no cache):
cox_model = fit_cox_model(df, formula='time ~ age + sex')

# After (with cache):
cox_model = get_cached_cox_model(
    calculate_func=lambda: fit_cox_model(df, formula='time ~ age + sex'),
    cache_key_params={
        'formula': 'time ~ age + sex',
        'n_obs': len(df)
    }
)
```

**Step 4: Wrap survival estimates**
```python
# Before (no cache):
estimates = predict_survival(cox_model, newdata, horizon=5)

# After (with cache):
estimates = get_cached_survival_estimates(
    calculate_func=lambda: predict_survival(cox_model, newdata, horizon=5),
    cache_key_params={
        'model_type': 'cox',
        'horizon': 5,
        'n_predict': len(newdata)
    }
)
```

---

## 📊 Cache Key Parameters

### For PSM Cache Keys
```python
{
    'outcome': outcome_column_name,           # Which outcome to match on
    'method': 'logit' or 'probit',           # Propensity score method
    'n_obs': number_of_observations,         # Data size
    'n_vars': number_of_covariates,          # Number of predictors
    'treatment': treatment_column_name        # Treatment variable
}
```

### For Survival Cache Keys
```python
{
    'time_col': time_column_name,            # Time-to-event column
    'event_col': event_column_name,          # Event indicator column
    'formula': model_formula_string,         # Cox regression formula
    'n_obs': number_of_observations,         # Data size
    'horizon': follow_up_time,               # Prediction horizon (years)
    'n_predict': n_prediction_rows           # Rows to predict for
}
```

---

## ✅ Best Practices

### 1. Use Stable Parameters
```python
# ✅ GOOD - Same parameters = cache hit
cache_key_params = {
    'outcome': 'diabetes',
    'method': 'logit'
}

# ❌ BAD - Different parameters = cache miss
cache_key_params = {
    'outcome': outcome_var,  # Changes with user input
    'method': method         # Changes dynamically
}
```

### 2. Include Data Characteristics
```python
# ✅ GOOD - Cache invalidates when data changes
cache_key_params = {
    'n_obs': len(df),        # If df changes, cache misses
    'n_vars': len(covariates)
}

# ✅ BETTER - Hash the actual data
cache_key_params = {
    'data_hash': hash(pd.util.hash_pandas_object(df).values.tobytes())
}
```

### 3. Log Cache Hits/Misses
```python
# Already done automatically!
# Look for these in logs:
# ✅ "Cache HIT" - Using cached result (2-3 seconds)
# 🔄 "Cache MISS" - Computing fresh result (30+ seconds)
```

---

## 🧪 Testing

### Test PSM Caching
```python
# In Shiny app:
# 1. Select covariates and outcome
# 2. Click "Calculate PSM" → Wait ~30s
# 3. Change back to same covariates and outcome
# 4. Click "Calculate PSM" → Should be ~2-3s (cache hit!)
# 5. Check logs for "✅ PSM Cache HIT"
```

### Test Survival Caching
```python
# In Shiny app:
# 1. Select time, event, and formula
# 2. Click "Fit Cox Model" → Wait ~20s
# 3. Change plot options (doesn't change formula)
# 4. Re-render plot → Should be instant (same cache)
# 5. Check logs for "✅ Survival Cox Cache HIT"
```

---

## 📈 Expected Performance

### PSM Module
```
Before cache:
- Propensity scores: 30 seconds
- Matching: 10 seconds
- Total: 40 seconds (every time)

After cache:
- First run: 40 seconds (cache miss)
- Repeat with same params: 2-3 seconds (cache hit!)
- Savings per repeat: ~37 seconds
```

### Survival Module
```
Before cache:
- KM curves: 10 seconds
- Cox model: 15 seconds
- Estimates: 10 seconds
- Total: 35 seconds (every time)

After cache:
- First run: 35 seconds (cache miss)
- Repeat with same params: 2-3 seconds (cache hit!)
- Savings per repeat: ~32 seconds
```

---

## 🔍 Debugging

### Cache Not Working?

**Check 1: Are parameters identical?**
```python
# If parameters change → cache miss (expected)
# Make sure you're comparing same analysis
```

**Check 2: Check logs**
```python
# ✅ Look for: "✅ PSM Cache HIT" or "✅ Survival Cox Cache HIT"
# ❌ If seeing: "🔄 Cache MISS" repeatedly, check parameters
```

**Check 3: Verify cache_key_params**
```python
print(cache_key_params)
# Should show same values when repeating same analysis
```

### Memory Usage High?

**Solution:** Layer 2 handles auto-cleanup
- Automatic at 80% of 280MB threshold
- Expired cache items removed automatically
- No manual action needed

---

## 🚀 Deployment

### Before Pushing
- [ ] Test PSM caching locally (30s → 2-3s)
- [ ] Test Survival caching locally (35s → 2-3s)
- [ ] Verify logs show "Cache HIT" messages
- [ ] Verify memory < 200MB
- [ ] No new errors in console

### After Pushing to HF
- [ ] Monitor PSM analysis times (should be 2-3s on repeat)
- [ ] Monitor Survival analysis times (should be 2-3s on repeat)
- [ ] Check cache hit rate (target > 50%)
- [ ] Verify memory stays < 200MB
- [ ] Collect user feedback

---

## 📖 Reference

### Files
- `utils/psm_cache_integration.py` - PSM caching helpers
- `utils/survival_cache_integration.py` - Survival caching helpers
- `utils/cache_manager.py` - Core cache system (Layer 1)
- `OPTIMIZATION.md` - Technical details

### Key Functions

**PSM Module:**
- `get_cached_propensity_scores()` - Cache propensity scores
- `get_cached_matched_data()` - Cache matched datasets

**Survival Module:**
- `get_cached_km_curves()` - Cache Kaplan-Meier curves
- `get_cached_cox_model()` - Cache Cox models
- `get_cached_survival_estimates()` - Cache survival estimates
- `get_cached_risk_table()` - Cache risk tables

---

## ✨ Summary

✅ **PSM Integration:** Copy `get_cached_propensity_scores()` usage  
✅ **Survival Integration:** Copy `get_cached_km_curves()` usage  
✅ **Expected Speedup:** 30-60 seconds per repeat  
✅ **Effort:** 5-10 minutes per module  
✅ **Risk:** Zero (can disable instantly)  

**Ready to integrate!**

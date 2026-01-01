"""
🧮 Logistic Regression Core Logic (Shiny Compatible) - Optimized & VIF Check

Features:
- Binary Logistic Regression (BFGS / Firth)
- Automatic Mode Detection (Categorical/Linear)
- Variance Inflation Factor (VIF) Check for Multicollinearity
- Optimized Data Processing
- LAYER 1 CACHING: Computation results cached for 30 minutes
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import html
from logger import get_logger
from forest_plot_lib import create_forest_plot

# === LAYER 1: Import Cache Manager ===
# from utils.cache_manager import COMPUTATION_CACHE # <-- ไม่ใช้ตรงๆ แล้ว
from utils.memory_manager import MEMORY_MANAGER
from utils.connection_handler import CONNECTION_HANDLER

# === INTEGRATION: Import Cache Wrappers ===
from utils.logit_cache_integration import get_cached_logistic_analysis

logger = get_logger(__name__)
COLORS = {
    'primary': '#2180BE',
    'primary_dark': '#1a5a8a',
    'danger': '#d32f2f',
    'warning': '#f57c00',
    'text_secondary': '#666',
    'border': '#e0e0e0'
}

# ... (ส่วน Import Firth และ Helper functions: clean_numeric_series, _robust_sort_key, calculate_vif, run_binary_logit, get_label, fmt_p_with_styling เหมือนเดิม ไม่ต้องแก้) ...

# Try to import Firth regression
try:
    from firthlogist import FirthLogisticRegression
    
    if not hasattr(FirthLogisticRegression, "_validate_data"):
        from sklearn.utils.validation import check_X_y, check_array
        
        logger.info("Applying sklearn compatibility patch to FirthLogisticRegression")
        
        def _validate_data_patch(self, X, y=None, reset=True, validate_separately=False, **check_params):
            """Compatibility shim for sklearn >= 1.6."""
            if y is None:
                return check_array(X, **check_params)
            else:
                return check_X_y(X, y, **check_params)
        
        FirthLogisticRegression._validate_data = _validate_data_patch
        logger.info("Patch applied successfully")
    
    HAS_FIRTH = True
except (ImportError, AttributeError):
    HAS_FIRTH = False
    logger.warning("firthlogist not available")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*")


def clean_numeric_series(series):
    """Vectorized cleanup of numeric series."""
    return pd.to_numeric(series, errors='coerce')


def _robust_sort_key(x):
    """Sort key placing numeric values first."""
    try:
        if pd.isna(x):
            return (2, "")
        val = float(x)
        return (0, val)
    except (ValueError, TypeError):
        return (1, str(x))


def calculate_vif(X):
    """Calculate Variance Inflation Factor (VIF)."""
    try:
        if 'const' not in X.columns:
            X_vif = sm.add_constant(X)
        else:
            X_vif = X.copy()
            
        X_vif = X_vif.loc[:, (X_vif != X_vif.iloc[0]).any()]
        
        vif_data = {}
        for i, col in enumerate(X_vif.columns):
            if col == 'const':
                continue
            try:
                val = variance_inflation_factor(X_vif.values, i)
                vif_data[col] = val
            except Exception:
                vif_data[col] = np.inf
                
        return vif_data
    except Exception as e:
        logger.warning(f"Could not calculate VIF: {e}")
        return {}


def run_binary_logit(y, X, method='default'):
    """Fit binary logistic regression."""
    stats_metrics = {"mcfadden": np.nan, "nagelkerke": np.nan}
    
    try:
        X = X.astype(float)
        y = y.astype(int)
        
        X_const = sm.add_constant(X, has_constant='add')
        
        if method == 'firth':
            if not HAS_FIRTH:
                return None, None, None, "firthlogist not installed", stats_metrics
            
            fl = FirthLogisticRegression(fit_intercept=False)
            fl.fit(X_const, y)
            
            coef = np.asarray(fl.coef_).reshape(-1)
            if coef.shape[0] != len(X_const.columns):
                return None, None, None, "Firth output shape mismatch", stats_metrics
            
            params = pd.Series(coef, index=X_const.columns)
            pvalues = pd.Series(getattr(fl, "pvals_", np.full(len(X_const.columns), np.nan)), index=X_const.columns)
            ci = getattr(fl, "ci_", None)
            conf_int = (
                pd.DataFrame(ci, index=X_const.columns, columns=[0, 1])
                if ci is not None
                else pd.DataFrame(np.nan, index=X_const.columns, columns=[0, 1])
            )
            return params, conf_int, pvalues, "OK", stats_metrics
        
        elif method == 'bfgs':
            model = sm.Logit(y, X_const)
            result = model.fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const)
            try:
                result = model.fit(disp=0)
            except np.linalg.LinAlgError:
                return None, None, None, "Singular Matrix (Perfect Collinearity)", stats_metrics
        
        try:
            llf = result.llf
            llnull = result.llnull
            nobs = result.nobs
            mcfadden = 1 - (llf / llnull) if llnull != 0 else np.nan
            cox_snell = 1 - np.exp((2/nobs) * (llnull - llf))
            max_r2 = 1 - np.exp((2/nobs) * llnull)
            nagelkerke = cox_snell / max_r2 if max_r2 > 1e-9 else np.nan
            stats_metrics = {"mcfadden": mcfadden, "nagelkerke": nagelkerke}
        except Exception as e:
            logger.debug(f"Failed to calculate R2: {e}")
        
        return result.params, result.conf_int(), result.pvalues, "OK", stats_metrics
    
    except Exception as e:
        logger.error(f"Logistic regression failed: {e}")
        return None, None, None, str(e), stats_metrics


def get_label(col_name, var_meta):
    """Create formatted label for column."""
    display_name = col_name
    secondary_label = ""
    
    if var_meta:
        if col_name in var_meta and 'label' in var_meta[col_name]:
            secondary_label = var_meta[col_name]['label']
        elif '_' in col_name:
            parts = col_name.split('_', 1)
            if len(parts) > 1:
                short_name = parts[1]
                if short_name in var_meta and 'label' in var_meta[short_name]:
                    secondary_label = var_meta[short_name]['label']
    
    safe_name = html.escape(str(display_name))
    if secondary_label:
        safe_label = html.escape(str(secondary_label))
        return f"<b>{safe_name}</b><br><span style='color:#666; font-size:0.9em'>{safe_label}</span>"
    else:
        return f"<b>{safe_name}</b>"


def fmt_p_with_styling(val):
    """Format p-value with red highlighting if significant (p < 0.05)."""
    if pd.isna(val):
        return "-"
    try:
        val = float(val)
        val = max(0, min(1, val))
        if val < 0.001:
            p_str = "<0.999"
        elif val > 0.999:
            p_str = ">0.999"
        else:
            p_str = f"{val:.3f}"
        
        if val < 0.05:
            return f"<span class='sig-p'>{p_str}</span>"
        else:
            return p_str
    except (ValueError, TypeError):
        return "-"


def analyze_outcome(outcome_name, df, var_meta=None, method='auto'):
    """
    Perform logistic regression analysis for binary outcome.
    Includes VIF check for multivariate analysis.
    
    🟢 LAYER 1 OPTIMIZATION: Results cached for 30 minutes
    """
    # === MEMORY CHECK ===
    if not MEMORY_MANAGER.check_and_cleanup():
        logger.warning("Memory critical - proceeding with caution")
    
    # === CACHE KEY PREPARATION ===
    cache_key_params = {
        'outcome': outcome_name,
        'df_shape': df.shape,
        'df_hash': hash(pd.util.hash_pandas_object(df, index=True).values.tobytes()),
        'var_meta': str(var_meta),
        'method': method
    }
    
    # === INNER LOGIC FUNCTION ===
    def _perform_analysis():
        logger.info(f"📊 Starting logistic analysis logic for outcome: {outcome_name}")
        
        if outcome_name not in df.columns:
            return (f"<div class='alert'>Outcome '{outcome_name}' not found</div>", {}, {})
        
        y_raw = df[outcome_name].dropna()
        unique_outcomes = set(y_raw.unique())
        
        if len(unique_outcomes) != 2:
            return (f"<div class='alert'>Invalid outcome: expected 2 values, found {len(unique_outcomes)}</div>", {}, {})
        
        if not unique_outcomes.issubset({0, 1}):
            sorted_outcomes = sorted(unique_outcomes, key=str)
            outcome_map = {sorted_outcomes[0]: 0, sorted_outcomes[1]: 1}
            y = y_raw.map(outcome_map).astype(int)
        else:
            y = y_raw.astype(int)
        
        df_aligned = df.loc[y.index]
        total_n = len(y)
        candidates = []
        results_db = {}
        sorted_cols = sorted(df.columns.astype(str))
        mode_map = {}
        cat_levels_map = {}
        
        # Check for perfect separation
        has_perfect_separation = False
        if method == 'auto':
            for col in sorted_cols:
                if col == outcome_name or col not in df_aligned.columns: continue
                try:
                    X_num = clean_numeric_series(df_aligned[col])
                    if X_num.nunique() > 1 and (pd.crosstab(X_num, y) == 0).any().any():
                        has_perfect_separation = True
                        break
                except Exception:
                    continue
        
        # Select fitting method
        preferred_method = 'bfgs'
        if method == 'auto' and HAS_FIRTH and (has_perfect_separation or len(df) < 50 or (y == 1).sum() < 20):
            preferred_method = 'firth'
        elif method == 'firth':
            preferred_method = 'firth' if HAS_FIRTH else 'bfgs'
        
        or_results = {}
        
        # ==========================
        # 1. Univariate Analysis
        # ==========================
        logger.info(f"Starting univariate analysis for {len(sorted_cols)-1} variables")
        
        for col in sorted_cols:
            if col == outcome_name or col not in df_aligned.columns or df_aligned[col].isnull().all():
                continue
            
            res = {'var': col}
            X_raw = df_aligned[col]
            X_num = clean_numeric_series(X_raw)
            
            X_neg = X_raw[y == 0]
            X_pos = X_raw[y == 1]
            
            unique_vals = X_num.dropna().unique()
            mode = 'linear'
            
            # Auto-detect mode
            if set(unique_vals).issubset({0, 1}):
                mode = 'categorical'
            elif len(unique_vals) < 10:
                is_int = np.all(np.mod(unique_vals, 1) == 0)
                if is_int:
                    mode = 'categorical'
            
            if var_meta:
                orig_name = col.split('_', 1)[1] if '_' in col else col
                key = col if col in var_meta else orig_name
                if key in var_meta:
                    user_mode = var_meta[key].get('type', '').lower()
                    if 'cat' in user_mode or 'simp' in user_mode:
                        mode = 'categorical'
                    elif 'lin' in user_mode or 'cont' in user_mode:
                        mode = 'linear'
            
            mode_map[col] = mode
            
            if mode == 'categorical':
                try:
                    levels = sorted(X_raw.dropna().unique(), key=_robust_sort_key)
                except (TypeError, ValueError):
                    levels = sorted(X_raw.astype(str).unique())
                cat_levels_map[col] = levels
                
                n_used = len(X_raw.dropna())
                desc_tot, desc_neg, desc_pos = [f"n={n_used}"], [f"n={len(X_neg.dropna())}"], [f"n={len(X_pos.dropna())}"]
                
                for lvl in levels:
                    lbl_txt = str(int(float(lvl))) if str(lvl).endswith('.0') else str(lvl)
                    mask_all = (X_raw == lvl)
                    c_all = mask_all.sum()
                    p_all = (c_all / n_used * 100) if n_used else 0
                    c_n = (X_neg == lvl).sum()
                    p_n = (c_n / len(X_neg.dropna()) * 100) if len(X_neg.dropna()) else 0
                    c_p = (X_pos == lvl).sum()
                    p_p = (c_p / len(X_pos.dropna()) * 100) if len(X_pos.dropna()) else 0
                    
                    desc_tot.append(f"{lbl_txt}: {c_all} ({p_all:.1f}%)")
                    desc_neg.append(f"{c_n} ({p_n:.1f}%)")
                    desc_pos.append(f"{c_p} ({p_p:.1f}%)")
                
                res.update({'desc_total': "<br>".join(desc_tot), 'desc_neg': "<br>".join(desc_neg), 'desc_pos': "<br>".join(desc_pos)})
                
                try:
                    ct = pd.crosstab(X_raw, y)
                    _, p, _, _ = stats.chi2_contingency(ct) if ct.size > 0 else (0, np.nan, 0, 0)
                    res.update({'p_comp': p, 'test_name': "Chi-square"})
                except Exception:
                    res.update({'p_comp': np.nan, 'test_name': "-"})
                
                if len(levels) > 1:
                    temp_df = pd.DataFrame({'y': y, 'raw': X_raw}).dropna()
                    dummy_cols = []
                    for lvl in levels[1:]:
                        d_name = f"{col}::{lvl}"
                        temp_df[d_name] = (temp_df['raw'].astype(str) == str(lvl)).astype(int)
                        dummy_cols.append(d_name)
                    
                    if dummy_cols and temp_df[dummy_cols].std().sum() > 0:
                        params, conf, pvals, status, _ = run_binary_logit(temp_df['y'], temp_df[dummy_cols], method=preferred_method)
                        
                        if status == "OK":
                            or_lines, coef_lines, p_lines = ["Ref."], ["-"], ["-"]
                            for lvl in levels[1:]:
                                d_name = f"{col}::{lvl}"
                                if d_name in params:
                                    coef = params[d_name]
                                    odd = np.exp(coef)
                                    ci_l, ci_h = np.exp(conf.loc[d_name][0]), np.exp(conf.loc[d_name][1])
                                    pv = pvals[d_name]
                                    
                                    coef_lines.append(f"{coef:.3f}")
                                    or_lines.append(f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})")
                                    p_lines.append(fmt_p_with_styling(pv))
                                    or_results[f"{col}: {lvl}"] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                                else:
                                    or_lines.append("-")
                                    p_lines.append("-")
                                    coef_lines.append("-")
                            
                            res.update({'or': "<br>".join(or_lines), 'coef': "<br>".join(coef_lines), 'p_or': "<br>".join(p_lines)})
                        else:
                            res.update({'or': "-", 'coef': "-"})
                    else:
                        res.update({'or': "-", 'coef': "-"})
                else:
                    res.update({'or': "-", 'coef': "-"})
            
            else:  # Linear mode
                n_used = len(X_num.dropna())
                m_t, s_t = X_num.mean(), X_num.std()
                X_num_neg = clean_numeric_series(X_neg)
                X_num_pos = clean_numeric_series(X_pos)
                m_n, s_n = X_num_neg.mean(), X_num_neg.std()
                m_p, s_p = X_num_pos.mean(), X_num_pos.std()
                
                res['desc_total'] = f"n={n_used}<br>Mean: {m_t:.2f} (SD {s_t:.2f})"
                res['desc_neg'] = f"{m_n:.2f} ({s_n:.2f})"
                res['desc_pos'] = f"{m_p:.2f} ({s_p:.2f})"
                
                try:
                    _, p = stats.mannwhitneyu(X_num_neg.dropna(), X_num_pos.dropna())
                    res.update({'p_comp': p, 'test_name': "Mann-Whitney"})
                except Exception:
                    res.update({'p_comp': np.nan, 'test_name': "-"})
                
                data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
                if not data_uni.empty and data_uni['x'].nunique() > 1:
                    params, hex_conf, pvals, status, _ = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
                    if status == "OK" and 'x' in params:
                        coef = params['x']
                        odd = np.exp(coef)
                        ci_l, ci_h = np.exp(hex_conf.loc['x'][0]), np.exp(hex_conf.loc['x'][1])
                        pv = pvals['x']
                        
                        res.update({
                            'coef': f"{coef:.3f}",
                            'or': f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})",
                            'p_or': pv
                        })
                        or_results[col] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                    else:
                        res.update({'or': "-", 'coef': "-"})
                else:
                    res.update({'or': "-", 'coef': "-"})
            
            results_db[col] = res
            
            p_screen = res.get('p_comp', np.nan)
            if isinstance(p_screen, (int, float)) and pd.notna(p_screen) and p_screen < 0.20:
                candidates.append(col)
        
        # ==========================
        # 2. Multivariate Analysis & VIF Check
        # ==========================
        aor_results = {}
        final_n_multi = 0
        mv_metrics_text = ""
        vif_warning_text = ""
        
        def _is_candidate_valid(col):
            mode = mode_map.get(col, "linear")
            series = df_aligned[col]
            if mode == "categorical":
                return series.notna().sum() > 5
            return clean_numeric_series(series).notna().sum() > 5
        
        cand_valid = [c for c in candidates if _is_candidate_valid(c)]
        
        if len(cand_valid) > 0:
            multi_df = pd.DataFrame({'y': y})
            
            for c in cand_valid:
                mode = mode_map.get(c, 'linear')
                if mode == 'categorical':
                    levels = cat_levels_map.get(c, [])
                    raw_vals = df_aligned[c]
                    if len(levels) > 1:
                        for lvl in levels[1:]:
                            d_name = f"{c}::{lvl}"
                            multi_df[d_name] = (raw_vals.astype(str) == str(lvl)).astype(int)
                else:
                    multi_df[c] = clean_numeric_series(df_aligned[c])
            
            multi_data = multi_df.dropna()
            final_n_multi = len(multi_data)
            predictors = [col for col in multi_data.columns if col != 'y']
            
            if not multi_data.empty and final_n_multi > 10 and len(predictors) > 0:
                # --- VIF CHECK ---
                vif_data = calculate_vif(multi_data[predictors])
                high_vif_vars = [k for k, v in vif_data.items() if v > 10]
                if high_vif_vars:
                    grouped_vifs = set()
                    for v in high_vif_vars:
                        orig_var = v.split("::")[0]
                        grouped_vifs.add(orig_var)
                    
                    vif_warning_text = (
                        f"<div class='alert-box'>"
                        f"⚠️ <b>High Multicollinearity (VIF > 10) detected in:</b> "
                        f"{', '.join(grouped_vifs)}<br>"
                        f"<small>Consider removing one of these variables or using regularization.</small>"
                        f"</div>"
                    )

                params, conf, pvals, status, mv_stats = run_binary_logit(multi_data['y'], multi_data[predictors], method=preferred_method)
                
                if status == "OK":
                    r2_parts = []
                    mcf = mv_stats.get('mcfadden')
                    nag = mv_stats.get('nagelkerke')
                    if pd.notna(mcf): r2_parts.append(f"McFadden R² = {mcf:.3f}")
                    if pd.notna(nag): r2_parts.append(f"Nagelkerke R² = {nag:.3f}")
                    if r2_parts: mv_metrics_text = " | ".join(r2_parts)
                    
                    for var in cand_valid:
                        mode = mode_map.get(var, 'linear')
                        if mode == 'categorical':
                            levels = cat_levels_map.get(var, [])
                            aor_entries = []
                            for lvl in levels[1:]:
                                d_name = f"{var}::{lvl}"
                                if d_name in params:
                                    coef = params[d_name]
                                    aor = np.exp(coef)
                                    ci_low, ci_high = np.exp(conf.loc[d_name][0]), np.exp(conf.loc[d_name][1])
                                    pv = pvals[d_name]
                                    aor_entries.append({'lvl': lvl, 'coef': coef, 'aor': aor, 'l': ci_low, 'h': ci_high, 'p': pv})
                                    aor_results[f"{var}: {lvl}"] = {'aor': aor, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': pv}
                            results_db[var]['multi_res'] = aor_entries
                        else:
                            if var in params:
                                coef = params[var]
                                aor = np.exp(coef)
                                ci_low, ci_high = np.exp(conf.loc[var][0]), np.exp(conf.loc[var][1])
                                pv = pvals[var]
                                results_db[var]['multi_res'] = {'coef': coef, 'aor': aor, 'l': ci_low, 'h': ci_high, 'p': pv}
                                aor_results[var] = {'aor': aor, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': pv}
        
        # ==========================
        # 3. Build HTML Report
        # ==========================
        html_rows = []
        current_sheet = ""
        valid_cols_for_html = [c for c in sorted_cols if c in results_db]
        grouped_cols = sorted(valid_cols_for_html, key=lambda x: (x.split('_')[0] if '_' in x else "Variables", x))
        
        for col in grouped_cols:
            if col == outcome_name: continue
            res = results_db[col]
            mode = mode_map.get(col, 'linear')
            sheet = col.split('_')[0] if '_' in col else "Variables"
            
            if sheet != current_sheet:
                html_rows.append(f"<tr class='sheet-header'><td colspan='11'>{sheet}</td></tr>")
                current_sheet = sheet
            
            lbl = get_label(col, var_meta)
            mode_badge = {'categorical': '📊 (All Levels)', 'linear': '📉 (Trend)'}
            if mode in mode_badge:
                lbl += f"<br><span style='font-size:0.8em; color:#888'>{mode_badge[mode]}</span>"
            
            or_s = res.get('or', '-')
            coef_s = res.get('coef', '-')
            
            if mode == 'categorical':
                p_col_display = res.get('p_or', '-')
            else:
                p_col_display = fmt_p_with_styling(res.get('p_comp', np.nan))
            
            aor_s, acoef_s, ap_s = "-", "-", "-"
            multi_res = res.get('multi_res')
            
            if multi_res:
                if isinstance(multi_res, list): # Categorical
                    aor_lines, acoef_lines, ap_lines = ["Ref."], ["-"], ["-"]
                    for item in multi_res:
                        p_txt = fmt_p_with_styling(item['p'])
                        acoef_lines.append(f"{item['coef']:.3f}")
                        aor_lines.append(f"{item['aor']:.2f} ({item['l']:.2f}-{item['h']:.2f})")
                        ap_lines.append(p_txt)
                    aor_s, acoef_s, ap_s = "<br>".join(aor_lines), "<br>".join(acoef_lines), "<br>".join(ap_lines)
                else: # Linear
                    if 'coef' in multi_res and pd.notna(multi_res['coef']):
                        acoef_s = f"{multi_res['coef']:.3f}"
                    else:
                        acoef_s = "-"
                    aor_s = f"{multi_res['aor']:.2f} ({multi_res['l']:.2f}-{multi_res['h']:.2f})"
                    ap_s = fmt_p_with_styling(multi_res['p'])
            
            html_rows.append(f"""<tr>
                <td>{lbl}</td>
                <td>{res.get('desc_total','')}</td>
                <td>{res.get('desc_neg','')}</td>
                <td>{res.get('desc_pos','')}</td>
                <td>{coef_s}</td>
                <td>{or_s}</td>
                <td>{res.get('test_name', '-')}</td>
                <td>{p_col_display}</td>
                <td>{acoef_s}</td>
                <td>{aor_s}</td>
                <td>{ap_s}</td>
            </tr>""")
        
        logger.info(f"Logistic analysis complete. Multivariate n={final_n_multi}")
        
        model_fit_html = ""
        if mv_metrics_text:
            model_fit_html = f"<div style='margin-top: 8px; padding-top: 8px; border-top: 1px dashed #ccc; color: {COLORS['primary_dark']};'><b>Model Fit:</b> {mv_metrics_text}</div>"
        
        if vif_warning_text:
            model_fit_html = vif_warning_text + model_fit_html

        html_table = f"""<div id='{outcome_name}' class='table-container'>
        <div class='outcome-title'>Outcome: {outcome_name} (n={total_n})</div>
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Total</th>
                    <th>Group 0</th>
                    <th>Group 1</th>
                    <th>Crude Coef.</th>
                    <th>Crude OR (95% CI)</th>
                    <th>Test</th>
                    <th>P-value</th>
                    <th>Adj. Coef.</th>
                    <th>aOR (95% CI)</th>
                    <th>aP-value</th>
                </tr>
            </thead>
            <tbody>{chr(10).join(html_rows)}</tbody>
        </table>
        <div class='summary-box'>
            <b>Method:</b> {preferred_method.capitalize()} Logit<br>
            <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; font-size: 0.9em; color: #666;'>
                <b>Selection:</b> Variables with Crude P < 0.20 (n={final_n_multi})<br>
                <b>Modes:</b> 📊 Categorical (vs Reference) | 📉 Linear (Per-unit)
                {model_fit_html}
            </div>
        </div>
        </div><br>"""
        
        return (html_table, or_results, aor_results)

    # === USE CACHE INTEGRATION WRAPPER ===
    return get_cached_logistic_analysis(
        calculate_func=_perform_analysis,
        cache_key_params=cache_key_params
    )

# ... (generate_forest_plot_html และ process_data_and_generate_html เหมือนเดิม) ...

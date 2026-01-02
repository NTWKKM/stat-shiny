"""
🧮 Logistic Regression Core Logic (Shiny Compatible) - Optimized & VIF Check

Features:
- Binary Logistic Regression (BFGS / Firth)
- Automatic Mode Detection (Categorical/Linear)
- Variance Inflation Factor (VIF) Check for Multicollinearity
- Optimized Data Processing
- LAYER 1 CACHING: Computation results cached for 30 minutes (via Integration Layer)
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

# === IMPORT UTILS ===
# ไม่ต้อง import COMPUTATION_CACHE โดยตรงแล้ว
from utils.memory_manager import MEMORY_MANAGER
from utils.connection_handler import CONNECTION_HANDLER

# === INTEGRATION: Import Cache Wrapper ===
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
            
        # Remove constant columns (zero variance)
        non_const_cols = X_vif.columns[(X_vif != X_vif.iloc[0]).any()]
        X_vif = X_vif[non_const_cols]
        
        vif_data = {}
        col_list = list(X_vif.columns)
        for i, col in enumerate(col_list):
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
            p_str = "<0.001"
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
    
    🟢 LAYER 1 OPTIMIZATION: Results cached for 30 minutes (via Wrapper)
    """
    # === MEMORY CHECK ===
    if not MEMORY_MANAGER.check_and_cleanup():
        logger.warning("Memory critical - proceeding with caution")
    
    # === CACHE KEY PREPARATION ===
    cache_key_params = {
        'outcome': outcome_name,
        'df_shape': df.shape,
        'df_hash': hash(pd.util.hash_pandas_object(df, index=True).values.tobytes()),
        'var_meta_hash': hash(tuple(sorted(var_meta.items()))) if var_meta else None,
        'method': method
    }
    
    # === INNER COMPUTATION LOGIC ===
    # ย้าย Logic การคำนวณทั้งหมดเข้ามาไว้ในฟังก์ชันย่อย
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
        
        # 1. Univariate Analysis
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
            
            if set(unique_vals).issubset({0, 1}):
                mode = 'categorical'
            elif len(unique_vals) < 10:
                is_int = np.all(np.mod(unique_vals, 1) == 0)
                if is_int: mode = 'categorical'
            
            if var_meta:
                orig_name = col.split('_', 1)[1] if '_' in col else col
                key = col if col in var_meta else orig_name
                if key in var_meta:
                    user_mode = var_meta[key].get('type', '').lower()
                    if 'cat' in user_mode or 'simp' in user_mode: mode = 'categorical'
                    elif 'lin' in user_mode or 'cont' in user_mode: mode = 'linear'
            
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
                    mask_all = (X_raw == lvl)
                    c_all = mask_all.sum()
                    p_all = (c_all/n_used*100) if n_used else 0
                    c_n = (X_neg == lvl).sum()
                    p_n = (c_n/len(X_neg.dropna())*100) if len(X_neg.dropna()) else 0
                    c_p = (X_pos == lvl).sum()
                    p_p = (c_p/len(X_pos.dropna())*100) if len(X_pos.dropna()) else 0
                    
                    lbl_txt = str(int(float(lvl))) if str(lvl).endswith('.0') else str(lvl)
                    desc_tot.append(f"{lbl_txt}: {c_all} ({p_all:.1f}%)")
                    desc_neg.append(f"{c_n} ({p_n:.1f}%)")
                    desc_pos.append(f"{c_p} ({p_p:.1f}%)")
                
                res.update({'desc_total': "<br>".join(desc_tot), 'desc_neg': "<br>".join(desc_neg), 'desc_pos': "<br>".join(desc_pos)})
                try:
                    ct = pd.crosstab(X_raw, y)
                    _, p, _, _ = stats.chi2_contingency(ct) if ct.size > 0 else (0, np.nan, 0, 0)
                    res.update({'p_comp': p, 'test_name': "Chi-square"})
                except (ValueError, ZeroDivisionError):
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
                                    or_lines.append("-"); p_lines.append("-"); coef_lines.append("-")
                            res.update({'or': "<br>".join(or_lines), 'coef': "<br>".join(coef_lines), 'p_or': "<br>".join(p_lines)})
                        else: res.update({'or': "-", 'coef': "-"})
                    else: res.update({'or': "-", 'coef': "-"})
                else: res.update({'or': "-", 'coef': "-"})
            
            else: # Linear
                n_used = len(X_num.dropna())
                m_t, s_t = X_num.mean(), X_num.std()
                X_num_neg, X_num_pos = clean_numeric_series(X_neg), clean_numeric_series(X_pos)
                m_n, s_n = X_num_neg.mean(), X_num_neg.std()
                m_p, s_p = X_num_pos.mean(), X_num_pos.std()
                
                res['desc_total'] = f"n={n_used}<br>Mean: {m_t:.2f} (SD {s_t:.2f})"
                res['desc_neg'] = f"{m_n:.2f} ({s_n:.2f})"
                res['desc_pos'] = f"{m_p:.2f} ({s_p:.2f})"
                try:
                    _, p = stats.mannwhitneyu(X_num_neg.dropna(), X_num_pos.dropna())
                    res.update({'p_comp': p, 'test_name': "Mann-Whitney"})
                except: res.update({'p_comp': np.nan, 'test_name': "-"})
                
                data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
                if not data_uni.empty and data_uni['x'].nunique() > 1:
                    params, hex_conf, pvals, status, _ = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
                    if status == "OK" and 'x' in params:
                        coef = params['x']
                        odd = np.exp(coef)
                        ci_l, ci_h = np.exp(hex_conf.loc['x'][0]), np.exp(hex_conf.loc['x'][1])
                        pv = pvals['x']
                        res.update({'coef': f"{coef:.3f}", 'or': f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})", 'p_or': pv})
                        or_results[col] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                    else: res.update({'or': "-", 'coef': "-"})
                else: res.update({'or': "-", 'coef': "-"})
            
            results_db[col] = res
            p_screen = res.get('p_comp', np.nan)
            if isinstance(p_screen, (int, float)) and pd.notna(p_screen) and p_screen < 0.20:
                candidates.append(col)
        
        # 2. Multivariate Analysis
        aor_results = {}
        final_n_multi = 0
        mv_metrics_text = ""
        vif_warning_text = ""
        
        def _is_candidate_valid(col):
            mode = mode_map.get(col, "linear")
            series = df_aligned[col]
            return series.notna().sum() > 5 if mode == "categorical" else clean_numeric_series(series).notna().sum() > 5
        
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
                            multi_df[f"{c}::{lvl}"] = (raw_vals.astype(str) == str(lvl)).astype(int)
                else:
                    multi_df[c] = clean_numeric_series(df_aligned[c])
            
            multi_data = multi_df.dropna()
            final_n_multi = len(multi_data)
            predictors = [col for col in multi_data.columns if col != 'y']
            
            if not multi_data.empty and final_n_multi > 10 and len(predictors) > 0:
                # VIF Check
                vif_data = calculate_vif(multi_data[predictors])
                high_vif = [k for k, v in vif_data.items() if v > 10]
                if high_vif:
                    grouped_vifs = {v.split("::")[0] for v in high_vif}
                    vif_warning_text = f"<div class='alert-box'>⚠️ <b>High Multicollinearity (VIF > 10):</b> {', '.join(grouped_vifs)}</div>"

                params, conf, pvals, status, mv_stats = run_binary_logit(multi_data['y'], multi_data[predictors], method=preferred_method)
                
                if status == "OK":
                    mcf, nag = mv_stats.get('mcfadden'), mv_stats.get('nagelkerke')
                    r2_parts = [f"McFadden R²={mcf:.3f}" if pd.notna(mcf) else "", f"Nagelkerke R²={nag:.3f}" if pd.notna(nag) else ""]
                    mv_metrics_text = " | ".join(filter(None, r2_parts))
                    
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
                                    ci_l, ci_h = np.exp(conf.loc[d_name][0]), np.exp(conf.loc[d_name][1])
                                    pv = pvals[d_name]
                                    aor_entries.append({'lvl': lvl, 'coef': coef, 'aor': aor, 'l': ci_l, 'h': ci_h, 'p': pv})
                                    aor_results[f"{var}: {lvl}"] = {'aor': aor, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                            results_db[var]['multi_res'] = aor_entries
                        else:
                            if var in params:
                                coef = params[var]
                                aor = np.exp(coef)
                                ci_l, ci_h = np.exp(conf.loc[var][0]), np.exp(conf.loc[var][1])
                                pv = pvals[var]
                                results_db[var]['multi_res'] = {'coef': coef, 'aor': aor, 'l': ci_l, 'h': ci_h, 'p': pv}
                                aor_results[var] = {'aor': aor, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
        
        # 3. Build HTML Report
        html_rows = []
        current_sheet = ""
        valid_cols = [c for c in sorted_cols if c in results_db]
        grouped_cols = sorted(valid_cols, key=lambda x: (x.split('_')[0] if '_' in x else "Variables", x))
        
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
            if mode in mode_badge: lbl += f"<br><span style='font-size:0.8em; color:#888'>{mode_badge[mode]}</span>"
            
            or_s, coef_s = res.get('or', '-'), res.get('coef', '-')
            p_disp = res.get('p_or', '-') if mode == 'categorical' else fmt_p_with_styling(res.get('p_comp', np.nan))
            
            aor_s, acoef_s, ap_s = "-", "-", "-"
            multi_res = res.get('multi_res')
            if multi_res:
                if isinstance(multi_res, list): # Cat
                    aor_l, acoef_l, ap_l = ["Ref."], ["-"], ["-"]
                    for it in multi_res:
                        acoef_l.append(f"{it['coef']:.3f}")
                        aor_l.append(f"{it['aor']:.2f} ({it['l']:.2f}-{it['h']:.2f})")
                        ap_l.append(fmt_p_with_styling(it['p']))
                    aor_s, acoef_s, ap_s = "<br>".join(aor_l), "<br>".join(acoef_l), "<br>".join(ap_l)
                else: # Linear
                    acoef_s = f"{multi_res['coef']:.3f}" if pd.notna(multi_res['coef']) else "-"
                    aor_s = f"{multi_res['aor']:.2f} ({multi_res['l']:.2f}-{multi_res['h']:.2f})"
                    ap_s = fmt_p_with_styling(multi_res['p'])
            
            html_rows.append(f"""<tr>
                <td>{lbl}</td><td>{res.get('desc_total','')}</td><td>{res.get('desc_neg','')}</td><td>{res.get('desc_pos','')}</td>
                <td>{coef_s}</td><td>{or_s}</td><td>{res.get('test_name', '-')}</td><td>{p_disp}</td>
                <td>{acoef_s}</td><td>{aor_s}</td><td>{ap_s}</td>
            </tr>""")
        
        logger.info(f"Logistic analysis complete. Multivariate n={final_n_multi}")
        model_fit_html = f"<div style='margin-top:8px;padding-top:8px;border-top:1px dashed #ccc;'><b>Model Fit:</b> {mv_metrics_text}</div>" if mv_metrics_text else ""
        if vif_warning_text: model_fit_html = vif_warning_text + model_fit_html

        html_table = f"""<div id='{outcome_name}' class='table-container'>
        <div class='outcome-title'>Outcome: {outcome_name} (n={total_n})</div>
        <table>
            <thead><tr>
                <th>Variable</th><th>Total</th><th>Group 0</th><th>Group 1</th>
                <th>Crude Coef.</th><th>Crude OR (95% CI)</th><th>Test</th><th>P-value</th>
                <th>Adj. Coef.</th><th>aOR (95% CI)</th><th>aP-value</th>
            </tr></thead>
            <tbody>{chr(10).join(html_rows)}</tbody>
        </table>
        <div class='summary-box'>
            <b>Method:</b> {preferred_method.capitalize()} Logit<br>
            <div style='margin-top:8px;padding-top:8px;border-top:1px solid #eee;color:#666;'>
                <b>Selection:</b> Variables with Crude P < 0.20 (n={final_n_multi})<br>
                <b>Modes:</b> 📊 Categorical | 📉 Linear
                {model_fit_html}
            </div>
        </div></div><br>"""
        
        return (html_table, or_results, aor_results)

    # === CALL WRAPPER ===
    # เรียกใช้ get_cached_logistic_analysis แทนการ cache เอง
    return get_cached_logistic_analysis(
        calculate_func=_perform_analysis,
        cache_key_params=cache_key_params
    )

def generate_forest_plot_html(or_results, aor_results, plot_title="Forest Plots: Odds Ratios"):
    """Generate forest plots from results."""
    html_parts = [f"<h2 style='margin-top:30px; color:{COLORS['primary']};'>{plot_title}</h2>"]
    has_plot = False
    
    if or_results:
        df_crude = pd.DataFrame([{'variable': k, **v} for k, v in or_results.items()])
        if not df_crude.empty:
            fig = create_forest_plot(df_crude, 'or', 'ci_low', 'ci_high', 'variable', 'p_value',
                                     "<b>Univariable: Crude OR</b>", "Odds Ratio", 1.0)
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=True))
            has_plot = True
    
    if aor_results:
        df_adj = pd.DataFrame([{'variable': k, **v} for k, v in aor_results.items()])
        if not df_adj.empty:
            fig = create_forest_plot(df_adj, 'aor', 'ci_low', 'ci_high', 'variable', 'p_value',
                                     "<b>Multivariable: Adjusted OR</b>", "Adjusted OR", 1.0)
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
            has_plot = True
    
    if not has_plot:
        html_parts.append("<p style='color:#999'>No results available.</p>")
    else:
        html_parts.append(f"""<div style='margin-top:20px; padding:15px; background:#f8f9fa; border-left:4px solid {COLORS['primary']};'>
            <b>Interpretation:</b> OR > 1 (Risk Factor), OR < 1 (Protective), CI crosses 1 (Not Significant)
        </div>""")
    
    return "".join(html_parts)


def process_data_and_generate_html(df, target_outcome, var_meta=None, method='auto'):
    """Generate complete HTML report with logistic regression results."""
    css = f"""<style>
        body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f8; }}
        .table-container {{ background: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); overflow-x: auto; }}
        table {{ width: 100%; border-collapse: separate; border-spacing: 0; }}
        th {{ background-color: {COLORS['primary_dark']}; color: #fff; padding: 12px; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .outcome-title {{ background-color: {COLORS['primary_dark']}; color: white; padding: 15px; font-weight: bold; }}
        .sig-p {{ color: #fff; background-color: {COLORS['danger']}; font-weight: bold; padding: 2px 6px; border-radius: 3px; }}
        .sheet-header td {{ background-color: #e8f4f8; color: {COLORS['primary']}; font-weight: bold; }}
        .alert-box {{ background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 10px; margin-top: 10px; border-radius: 4px; }}
        .summary-box {{ padding: 15px; background-color: #fcfcfc; border-top: 1px solid #eee; }}
    </style>"""
    
    html_table, or_res, aor_res = analyze_outcome(target_outcome, df, var_meta, method=method)
    plot_html = generate_forest_plot_html(or_res, aor_res)
    
    full_html = f"<!DOCTYPE html><html><head>{css}</head><body><h1>Logistic Regression Report</h1>{html_table}{plot_html}"
    full_html += "<div style='text-align: right; font-size: 0.75em; color: #999; margin-top: 20px;'>&copy; 2025 Medical Stat Tool</div>"
    
    return full_html, or_res, aor_res

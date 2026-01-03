"""
🧮 Logistic & Poisson Regression Core Logic (Shiny Compatible) - Optimized & VIF Check

Features:
- Binary Logistic Regression (BFGS / Firth)
- Poisson Regression for Count Data
- Interaction Terms Support
- Automatic Mode Detection (Categorical/Linear)
- Variance Inflation Factor (VIF) Check for Multicollinearity
- Data Leakage Prevention & Validation
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import html
import json
from logger import get_logger
from forest_plot_lib import create_forest_plot

logger = get_logger(__name__)

COLORS = {
    'primary': '#2180BE',
    'primary_dark': '#1a5a8a',
    'danger': '#d32f2f',
    'warning': '#f57c00',
    'text_secondary': '#666',
    'border': '#e0e0e0',
    'bg_light': '#f8f9fa'
}

# --- Check Firth Availability ONLY (No Patching) ---
try:
    from firthlogist import FirthLogisticRegression
    HAS_FIRTH = True
except (ImportError, AttributeError):
    HAS_FIRTH = False
    logger.warning("FirthLogisticRegression not found. Firth method will be disabled.")

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
    """Calculate Variance Inflation Factor (VIF) for multicollinearity check."""
    try:
        if 'const' not in X.columns:
            X_vif = sm.add_constant(X)
        else:
            X_vif = X.copy()
            
        non_const_cols = X_vif.columns[(X_vif != X_vif.iloc[0]).any()]
        X_vif = X_vif[non_const_cols]
        
        vif_data = {}
        col_list = list(X_vif.columns)
        for i, col in enumerate(col_list):
            if col == 'const': continue
            try:
                val = variance_inflation_factor(X_vif.values, i)
                vif_data[col] = val
            except:
                vif_data[col] = np.inf
        return vif_data
    except Exception as e:
        logger.warning(f"Could not calculate VIF: {e}")
        return {}

def run_regression_model(y, X, model_type='logistic', method='default'):
    """Unified function for Logistic and Poisson regression."""
    stats_metrics = {"mcfadden": np.nan, "nagelkerke": np.nan}
    try:
        X = X.astype(float)
        y = pd.to_numeric(y, errors='coerce')
        valid_mask = y.notna()
        y, X = y[valid_mask], X.loc[valid_mask]
        X_const = sm.add_constant(X, has_constant='add')

        if model_type == 'poisson':
            model = sm.GLM(y, X_const, family=sm.families.Poisson())
            result = model.fit()
        else:
            if method == 'firth':
                if not HAS_FIRTH: return None, None, None, "firthlogist not installed", stats_metrics
                fl = FirthLogisticRegression(fit_intercept=False)
                fl.fit(X_const, y)
                coef = np.asarray(fl.coef_).reshape(-1)
                params = pd.Series(coef, index=X_const.columns)
                pvalues = pd.Series(getattr(fl, "pvals_", np.full(len(X_const.columns), np.nan)), index=X_const.columns)
                ci = getattr(fl, "ci_", None)
                conf_int = pd.DataFrame(ci, index=X_const.columns, columns=[0, 1]) if ci is not None else pd.DataFrame(np.nan, index=X_const.columns, columns=[0, 1])
                return params, conf_int, pvalues, "OK", stats_metrics
            elif method == 'bfgs':
                model = sm.Logit(y, X_const)
                result = model.fit(method='bfgs', maxiter=100, disp=0)
            else:
                model = sm.Logit(y, X_const)
                result = model.fit(disp=0)

        # Process Results for statsmodels
        if result:
            try:
                llf, llnull, nobs = result.llf, result.llnull, result.nobs
                mcfadden = 1 - (llf / llnull) if llnull != 0 else np.nan
                cox_snell = 1 - np.exp((2/nobs) * (llnull - llf))
                max_r2 = 1 - np.exp((2/nobs) * llnull)
                nagelkerke = cox_snell / max_r2 if max_r2 > 1e-9 else np.nan
                stats_metrics = {"mcfadden": mcfadden, "nagelkerke": nagelkerke}
            except: pass
            return result.params, result.conf_int(), result.pvalues, "OK", stats_metrics
            
    except Exception as e:
        logger.error(f"Regression failed: {e}")
        return None, None, None, str(e), stats_metrics
    return None, None, None, "Unknown error", stats_metrics

def get_label(col_name, var_meta):
    """Create formatted label for column."""
    display_name = col_name
    secondary_label = ""
    if var_meta:
        if col_name in var_meta and 'label' in var_meta[col_name]:
            secondary_label = var_meta[col_name]['label']
        elif '_' in col_name:
            parts = col_name.split('_', 1)
            if len(parts) > 1 and parts[1] in var_meta and 'label' in var_meta[parts[1]]:
                secondary_label = var_meta[parts[1]]['label']
    
    safe_name = html.escape(str(display_name))
    if secondary_label:
        return f"<b>{safe_name}</b><br><span style='color:#666; font-size:0.9em'>{html.escape(str(secondary_label))}</span>"
    return f"<b>{safe_name}</b>"

def fmt_p_with_styling(val):
    """Format p-value with red highlighting if significant (p < 0.05)."""
    if pd.isna(val): return "-"
    try:
        val = float(val)
        p_str = "<0.001" if val < 0.001 else (">0.999" if val > 0.999 else f"{val:.3f}")
        return f"<span class='sig-p'>{p_str}</span>" if val < 0.05 else p_str
    except: return "-"

def analyze_outcome(outcome_name, df, var_meta=None, method='auto', model_type='logistic', interaction_terms=None):
    """Perform regression analysis with VIF, Leakage protection, and Interactions."""
    logger.info(f"📊 Starting analysis: {outcome_name}, Model: {model_type}")
    if outcome_name not in df.columns:
        return f"<div class='alert'>Outcome '{outcome_name}' not found</div>", {}, {}
    
    y_raw = df[outcome_name].dropna()
    if model_type == 'logistic':
        unique_outcomes = sorted(y_raw.unique(), key=str)
        if len(unique_outcomes) != 2:
            return f"<div class='alert'>Invalid outcome for Logistic: expected 2 values, found {len(unique_outcomes)}</div>", {}, {}
        y = (y_raw != unique_outcomes[0]).astype(int)
    else:
        if not pd.api.types.is_numeric_dtype(y_raw) or (y_raw < 0).any():
             return "<div class='alert'>Invalid outcome for Poisson: must be non-negative numeric</div>", {}, {}
        y = y_raw

    df_aligned = df.loc[y.index].copy()
    total_n = len(y)
    results_db, mode_map, cat_levels_map, candidates = {}, {}, {}, []
    interaction_cols, leakage_alerts = [], []

    # --- Interaction Terms & Leakage Check ---
    if interaction_terms:
        for term in interaction_terms:
            v1, v2 = (term.split(":") if isinstance(term, str) else term)
            if outcome_name in [v1, v2]:
                leakage_alerts.append(f"Excluded '<b>{v1}:{v2}</b>' (Outcome Leakage Risk)")
                continue
            if v1 in df_aligned.columns and v2 in df_aligned.columns:
                new_col = f"{v1}*{v2}"
                try:
                    df_aligned[new_col] = clean_numeric_series(df_aligned[v1]) * clean_numeric_series(df_aligned[v2])
                    interaction_cols.append(new_col)
                    mode_map[new_col] = 'linear'
                except: pass

    # Detect method
    preferred_method = 'bfgs'
    if model_type == 'logistic':
        if method == 'firth': preferred_method = 'firth' if HAS_FIRTH else 'bfgs'
        elif method == 'auto' and HAS_FIRTH and (len(df) < 50 or (y == 1).sum() < 20): preferred_method = 'firth'
    else: preferred_method = 'glm'

    # 1. Univariate Analysis
    all_cols = sorted(df.columns.astype(str)) + interaction_cols
    for col in all_cols:
        if col == outcome_name or col not in df_aligned.columns or df_aligned[col].isnull().all(): continue
        
        res = {'var': col}
        X_raw = df_aligned[col]
        X_num = clean_numeric_series(X_raw)
        
        if col not in mode_map:
            u = X_num.dropna().unique()
            mode = 'categorical' if set(u).issubset({0,1}) or (len(u) < 10 and np.all(np.mod(u, 1) == 0)) else 'linear'
            if var_meta:
                key = col if col in var_meta else (col.split('_', 1)[1] if '_' in col else col)
                if key in var_meta:
                    m = var_meta[key].get('type', '').lower()
                    if 'cat' in m: mode = 'categorical'
                    elif 'lin' in m: mode = 'linear'
            mode_map[col] = mode

        mode = mode_map[col]
        if mode == 'categorical':
            try: lvls = sorted(X_raw.dropna().unique(), key=_robust_sort_key)
            except: lvls = sorted(X_raw.astype(str).unique())
            cat_levels_map[col] = lvls
            
            n_used = len(X_raw.dropna())
            if model_type == 'logistic':
                X_neg, X_pos = X_raw[y==0], X_raw[y==1]
                desc = [f"n={n_used}"] + [f"{l}: {(X_raw==l).sum()} ({((X_raw==l).sum()/n_used*100):.1f}%)" for l in lvls]
                res.update({'desc_total': "<br>".join(desc), 'desc_neg': f"n={len(X_neg.dropna())}", 'desc_pos': f"n={len(X_pos.dropna())}"})
                try: res['p_comp'] = stats.chi2_contingency(pd.crosstab(X_raw, y))[1]; res['test_name'] = "Chi-square"
                except: res['p_comp'] = np.nan
            else:
                desc = [f"n={n_used}"] + [f"{l}: Mean={y.loc[X_raw==l].mean():.2f}" for l in lvls]
                res.update({'desc_total': "<br>".join(desc), 'desc_neg': "-", 'desc_pos': "-", 'p_comp': np.nan})

            if len(lvls) > 1:
                temp = pd.DataFrame({'y': y, 'raw': X_raw}).dropna()
                dummies = [f"{col}::{l}" for l in lvls[1:]]
                for l in lvls[1:]: temp[f"{col}::{l}"] = (temp['raw'].astype(str) == str(l)).astype(int)
                params, conf, pvals, status, _ = run_regression_model(temp['y'], temp[dummies], model_type, preferred_method)
                if status == "OK":
                    or_l, coef_l, p_l = ["Ref."], ["-"], ["-"]
                    for d in dummies:
                        if d in params:
                            coef, val_exp, pv = params[d], np.exp(params[d]), pvals[d]
                            ci = [np.exp(conf.loc[d][0]), np.exp(conf.loc[d][1])]
                            coef_l.append(f"{coef:.3f}"); or_l.append(f"{val_exp:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"); p_l.append(fmt_p_with_styling(pv))
                        else: or_l.append("-"); p_l.append("-")
                    res.update({'or': "<br>".join(or_l), 'coef': "<br>".join(coef_l), 'p_or': "<br>".join(p_l)})
        else:
            n_used = len(X_num.dropna())
            res['desc_total'] = f"n={n_used}<br>Mean: {X_num.mean():.2f}"
            if model_type == 'logistic':
                res['desc_neg'] = f"{X_num[y==0].mean():.2f}"; res['desc_pos'] = f"{X_num[y==1].mean():.2f}"
                try: res['p_comp'] = stats.mannwhitneyu(X_num[y==0].dropna(), X_num[y==1].dropna())[1]; res['test_name'] = "Mann-Whitney"
                except: res['p_comp'] = np.nan
            else:
                res.update({'desc_neg': "-", 'desc_pos': "-"}); try: res['p_comp'] = stats.spearmanr(X_num.dropna(), y.loc[X_num.dropna().index])[1]
                except: res['p_comp'] = np.nan
            
            params, conf, pvals, status, _ = run_regression_model(y.loc[X_num.dropna().index], X_num.dropna().to_frame(), model_type, preferred_method)
            if status == "OK" and X_num.name in params:
                coef, val_exp, pv = params[X_num.name], np.exp(params[X_num.name]), pvals[X_num.name]
                ci = [np.exp(conf.loc[X_num.name][0]), np.exp(conf.loc[X_num.name][1])]
                res.update({'coef': f"{coef:.3f}", 'or': f"{val_exp:.2f} ({ci[0]:.2f}-{ci[1]:.2f})", 'p_or': fmt_p_with_styling(pv)})

        results_db[col] = res
        p_val = res.get('p_comp', np.nan)
        if col in interaction_cols or (pd.notna(p_val) and p_val < 0.20): candidates.append(col)

    # 2. Multivariate Analysis
    aor_results, final_n_multi, mv_metrics, vif_alert = {}, 0, "", ""
    cand_v = [c for c in candidates if df_aligned[c].notna().sum() > 5]
    if cand_v:
        m_df = pd.DataFrame({'y': y})
        for c in cand_v:
            if mode_map.get(c) == 'categorical':
                for l in cat_levels_map.get(c, [])[1:]: m_df[f"{c}::{l}"] = (df_aligned[c].astype(str) == str(l)).astype(int)
            else: m_df[c] = clean_numeric_series(df_aligned[c])
        
        m_data = m_df.dropna(); final_n_multi = len(m_data)
        preds = [c for c in m_data.columns if c != 'y']
        if not m_data.empty and preds:
            vif_d = calculate_vif(m_data[preds])
            high_v = [k for k,v in vif_d.items() if v > 10]
            if high_v: vif_alert = f"<div class='alert-box'>⚠️ <b>High VIF (>10):</b> {', '.join({v.split('::')[0] for v in high_v})}</div>"
            
            params, conf, pvals, status, stats_d = run_regression_model(m_data['y'], m_data[preds], model_type, preferred_method)
            if status == "OK":
                mv_metrics = f"McFadden R²={stats_d.get('mcfadden',0):.3f}"
                for var in cand_v:
                    if mode_map.get(var) == 'categorical':
                        entries = []
                        for l in cat_levels_map.get(var, [])[1:]:
                            d = f"{var}::{l}"
                            if d in params:
                                aor, ci = np.exp(params[d]), [np.exp(conf.loc[d][0]), np.exp(conf.loc[d][1])]
                                entries.append({'lvl': l, 'coef': params[d], 'aor': aor, 'l': ci[0], 'h': ci[1], 'p': pvals[d]})
                                aor_results[f"{var}: {l}"] = {'aor': aor, 'ci_low': ci[0], 'ci_high': ci[1], 'p_value': pvals[d]}
                        results_db[var]['multi_res'] = entries
                    elif var in params:
                        aor, ci = np.exp(params[var]), [np.exp(conf.loc[var][0]), np.exp(conf.loc[var][1])]
                        results_db[var]['multi_res'] = {'coef': params[var], 'aor': aor, 'l': ci[0], 'h': ci[1], 'p': pvals[var]}
                        aor_results[var] = {'aor': aor, 'ci_low': ci[0], 'ci_high': ci[1], 'p_value': pvals[var]}

    # 3. HTML Builder
    eff = "IRR" if model_type == 'poisson' else "OR"
    html_rows = []
    curr_sheet = ""
    valid_cols = [c for c in (all_cols) if c in results_db]
    for col in valid_cols:
        res = results_db[col]
        sheet = "Interaction" if col in interaction_cols else (col.split('_')[0] if '_' in col else "Variables")
        if sheet != curr_sheet:
            html_rows.append(f"<tr class='sheet-header'><td colspan='11'>{sheet}</td></tr>")
            curr_sheet = sheet
        
        p_disp = res.get('p_or', '-') if mode_map.get(col)=='categorical' else fmt_p_with_styling(res.get('p_comp'))
        aor_s, acoef_s, ap_s = "-", "-", "-"
        multi = res.get('multi_res')
        if multi:
            if isinstance(multi, list):
                aor_l, acoef_l, ap_l = ["Ref."], ["-"], ["-"]
                for it in multi:
                    acoef_l.append(f"{it['coef']:.3f}"); aor_l.append(f"{it['aor']:.2f} ({it['l']:.2f}-{it['h']:.2f})"); ap_l.append(fmt_p_with_styling(it['p']))
                aor_s, acoef_s, ap_s = "<br>".join(aor_l), "<br>".join(acoef_l), "<br>".join(ap_l)
            else:
                acoef_s, aor_s, ap_s = f"{multi['coef']:.3f}", f"{multi['aor']:.2f} ({multi['l']:.2f}-{multi['h']:.2f})", fmt_p_with_styling(multi['p'])

        html_rows.append(f"<tr><td>{get_label(col, var_meta)}</td><td>{res.get('desc_total','')}</td><td>{res.get('desc_neg','')}</td><td>{res.get('desc_pos','')}</td><td>{res.get('coef','-')}</td><td>{res.get('or','-')}</td><td>{res.get('test_name','-')}</td><td>{p_disp}</td><td>{acoef_s}</td><td>{aor_s}</td><td>{ap_s}</td></tr>")

    table_html = f"""<div id='{outcome_name}' class='table-container'>
        <div class='outcome-title'>{model_type.capitalize()} Outcome: {outcome_name} (n={total_n})</div>
        <table><thead><tr><th>Variable</th><th>Total</th><th>Group 0</th><th>Group 1</th><th>Crude Coef.</th><th>Crude {eff}</th><th>Test</th><th>P-value</th><th>Adj. Coef.</th><th>Adj. {eff}</th><th>aP-value</th></tr></thead>
        <tbody>{''.join(html_rows)}</tbody></table>
        <div class='summary-box'><b>Method:</b> {preferred_method.capitalize()}<br>{vif_alert}{''.join(leakage_alerts)}<br>{mv_metrics}</div></div>"""
    
    return table_html, or_results, aor_results

def process_data_and_generate_html(df, target_outcome, var_meta=None, method='auto', model_type='logistic', interaction_terms=None):
    """Complete HTML generation."""
    css = f"<style>body {{ font-family: sans-serif; padding: 20px; }} .table-container {{ background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow-x: auto; margin-bottom: 20px; }} table {{ width: 100%; border-collapse: collapse; }} th {{ background: {COLORS['primary_dark']}; color: white; padding: 10px; }} td {{ padding: 10px; border-bottom: 1px solid #eee; }} .sig-p {{ background: {COLORS['danger']}; color: white; padding: 2px 5px; border-radius: 3px; }} .sheet-header td {{ background: #f0f7fa; color: {COLORS['primary']}; font-weight: bold; }} .alert-box {{ background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }} .outcome-title {{ padding: 15px; background: {COLORS['primary_dark']}; color: white; font-weight: bold; }}</style>"
    html_table, or_res, aor_res = analyze_outcome(target_outcome, df, var_meta, method, model_type, interaction_terms)
    
    eff = "IRR" if model_type == 'poisson' else "OR"
    plot_html = generate_forest_plot_html(or_res, aor_res, x_label=eff)
    full_html = f"<html><head>{css}</head><body><h1>{model_type.capitalize()} Regression Report</h1>{html_table}{plot_html}</body></html>"
    return full_html, or_res, aor_res

def generate_forest_plot_html(or_res, aor_res, x_label="OR"):
    """Forest plot HTML generation."""
    html_p = [f"<h2>Forest Plots ({x_label})</h2>"]
    if or_res:
        fig = create_forest_plot(pd.DataFrame([{'variable': k, **v} for k, v in or_res.items()]), 'or', 'ci_low', 'ci_high', 'variable', 'p_value', f"Crude {x_label}", x_label, 1.0)
        html_p.append(fig.to_html(full_html=False, include_plotlyjs=True))
    if aor_res:
        fig = create_forest_plot(pd.DataFrame([{'variable': k, **v} for k, v in aor_res.items()]), 'aor', 'ci_low', 'ci_high', 'variable', 'p_value', f"Adjusted {x_label}", x_label, 1.0)
        html_p.append(fig.to_html(full_html=False, include_plotlyjs=False))
    return "".join(html_p)

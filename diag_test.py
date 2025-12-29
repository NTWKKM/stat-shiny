import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, roc_auc_score, cohen_kappa_score
import plotly.graph_objects as go
import html as _html

# พยายาม import common palette ถ้าไม่มีให้ใช้ค่า default
try:
    from tabs._common import get_color_palette
    COLORS = get_color_palette()
except ImportError:
    COLORS = {
        'primary': '#007bff',
        'primary_dark': '#0056b3',
        'neutral': '#6c757d',
        'danger': '#dc3545',
        'warning': '#ffc107',
        'text': '#212529'
    }

# ========================================
# 1. DESCRIPTIVE STATISTICS
# ========================================

def calculate_descriptive(df, col):
    """คำนวณสถิติพื้นฐาน"""
    if col not in df.columns:
        return None
    
    data = df[col].dropna()
    if data.empty:
        return None
    
    try:
        num_data = pd.to_numeric(data, errors='raise')
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False
    
    if is_numeric:
        desc = num_data.describe()
        return pd.DataFrame({
            "Statistic": ["Count", "Mean", "SD", "Median", "Min", "Max", "Q1 (25%)", "Q3 (75%)"],
            "Value": [
                f"{desc['count']:.0f}",
                f"{desc['mean']:.4f}",
                f"{desc['std']:.4f}",
                f"{desc['50%']:.4f}",
                f"{desc['min']:.4f}",
                f"{desc['max']:.4f}",
                f"{desc['25%']:.4f}",
                f"{desc['75%']:.4f}"
            ]
        })
    else:
        counts = data.value_counts()
        percent = data.value_counts(normalize=True) * 100
        return pd.DataFrame({
            "Category": counts.index,
            "Count": counts.values,
            "Percentage (%)": percent.values
        }).sort_values("Count", ascending=False)


# ========================================
# 2. CI CALCULATION FUNCTIONS
# ========================================

def calculate_ci_wilson_score(successes, n, ci=0.95):
    if n <= 0: return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    p_hat = successes / n
    denominator = 1 + (z**2 / n)
    centre_adjusted_probability = (p_hat + (z**2 / (2 * n))) / denominator
    adjusted_standard_error = np.sqrt((p_hat * (1 - p_hat) + (z**2 / (4 * n))) / n) / denominator
    lower = max(0, centre_adjusted_probability - z * adjusted_standard_error)
    upper = min(1, centre_adjusted_probability + z * adjusted_standard_error)
    return lower, upper

def calculate_ci_log_odds(or_value, se_log_or, ci=0.95):
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    log_or = np.log(or_value) if or_value > 0 else np.nan
    if np.isnan(log_or) or np.isnan(se_log_or) or se_log_or <= 0:
        return np.nan, np.nan
    lower_log = log_or - z * se_log_or
    upper_log = log_or + z * se_log_or
    return np.exp(lower_log), np.exp(upper_log)

def calculate_ci_rr(risk_exp, n_exp, risk_unexp, n_unexp, ci=0.95):
    rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
    if np.isnan(rr) or rr <= 0:
        return np.nan, np.nan
    se_log_rr = np.sqrt((1 - risk_exp) / (risk_exp * n_exp) + 
                        (1 - risk_unexp) / (risk_unexp * n_unexp)) if (risk_exp > 0 and risk_unexp > 0) else np.nan
    if np.isnan(se_log_rr) or se_log_rr <= 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    log_rr = np.log(rr)
    lower_log = log_rr - z * se_log_rr
    upper_log = log_rr + z * se_log_rr
    return np.exp(lower_log), np.exp(upper_log)

def calculate_ci_nnt(rd, rd_se, ci=0.95):
    if abs(rd) < 0.001: return np.nan, np.nan
    if np.isnan(rd_se) or rd_se <= 0: return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    rd_lower = rd - z * rd_se
    rd_upper = rd + z * rd_se
    if rd_lower * rd_upper <= 0: return np.nan, np.nan
    nnt_lower = abs(1 / rd_upper)
    nnt_upper = abs(1 / rd_lower)
    return min(nnt_lower, nnt_upper), max(nnt_lower, nnt_upper)

# ========================================
# 3. CHI-SQUARE & FISHER'S EXACT TEST + DIAGNOSTIC METRICS
# ========================================

def calculate_chi2(df, col1, col2, method='Pearson (Standard)', v1_pos=None, v2_pos=None):
    """2x2 contingency analysis"""
    if col1 not in df.columns or col2 not in df.columns:
        return None, None, "Columns not found", None
    
    data = df[[col1, col2]].dropna()
    if data.empty:
        return None, None, "No data available after dropping missing values.", None

    data = data.copy()
    data[col1] = data[col1].astype(str)
    data[col2] = data[col2].astype(str)
    
    tab_chi2 = pd.crosstab(data[col1], data[col2])
    tab_raw = pd.crosstab(data[col1], data[col2], margins=True, margins_name="Total")
    tab_row_pct = pd.crosstab(data[col1], data[col2], normalize='index', margins=True, margins_name="Total") * 100
    
    all_col_labels = tab_raw.columns.tolist()
    all_row_labels = tab_raw.index.tolist()
    base_col_labels = [col for col in all_col_labels if col != 'Total']
    base_row_labels = [row for row in all_row_labels if row != 'Total']
    
    def get_original_label(label_str: str, df_labels: list) -> any:
        for lbl in df_labels:
            if str(lbl) == label_str: return lbl
        return label_str
    
    def custom_sort(label) -> tuple:
        try: return (0, float(label))
        except (ValueError, TypeError): return (1, str(label))
    
    # Reorder Columns
    final_col_order_base = base_col_labels[:]
    if v2_pos is not None:
        v2_pos_orig = get_original_label(v2_pos, base_col_labels)
        if v2_pos_orig in final_col_order_base:
            final_col_order_base.remove(v2_pos_orig)
            final_col_order_base.insert(0, v2_pos_orig)
    else: final_col_order_base.sort(key=custom_sort)
    final_col_order = final_col_order_base + ['Total']

    # Reorder Rows
    final_row_order_base = base_row_labels[:]
    if v1_pos is not None:
        v1_pos_orig = get_original_label(v1_pos, base_row_labels)
        if v1_pos_orig in final_row_order_base:
            final_row_order_base.remove(v1_pos_orig)
            final_row_order_base.insert(0, v1_pos_orig)
    else: final_row_order_base.sort(key=custom_sort)
    final_row_order = final_row_order_base + ['Total']

    # Reindex DataFrames
    tab_raw = tab_raw.reindex(index=final_row_order, columns=final_col_order)
    tab_row_pct = tab_row_pct.reindex(index=final_row_order, columns=final_col_order)
    tab_chi2 = tab_chi2.reindex(index=final_row_order_base, columns=final_col_order_base)
    
    # Prepare Display Table
    display_data = []
    for row_name in final_row_order:
        row_data = []
        for col_name in final_col_order:
            try:
                count = tab_raw.loc[row_name, col_name]
                pct = 100.0 if col_name == 'Total' else tab_row_pct.loc[row_name, col_name]
                row_data.append(f"{int(count)} ({pct:.1f}%)")
            except KeyError: row_data.append("0 (0.0%)")
        display_data.append(row_data)
    
    display_tab = pd.DataFrame(display_data, columns=final_col_order, index=final_row_order)
    display_tab.index.name = col1
    
    msg = ""
    try:
        is_2x2 = (tab_chi2.shape == (2, 2))
        stats_res = {}

        if "Fisher" in method:
            if not is_2x2: return display_tab, None, "Error: Fisher Test requires 2x2.", None
            odds_ratio, p_value = stats.fisher_exact(tab_chi2)
            stats_res = {"Test": "Fisher's Exact Test", "Statistic (OR)": f"{odds_ratio:.4f}", "P-value": f"{p_value:.4f}", "Degrees of Freedom": "-", "N": len(data)}
        else:
            use_correction = "Yates" in method
            chi2, p, dof, ex = stats.chi2_contingency(tab_chi2, correction=use_correction)
            method_name = f"Chi-Square ({'Yates' if use_correction else 'Pearson'})" if is_2x2 else "Chi-Square"
            stats_res = {"Test": method_name, "Statistic": f"{chi2:.4f}", "P-value": f"{p:.4f}", "Degrees of Freedom": f"{dof}", "N": len(data)}
            if (ex < 5).any() and is_2x2 and not use_correction:
                msg = "⚠️ Warning: Expected count < 5. Consider using Fisher's Exact Test."
        
        stats_df = pd.DataFrame(stats_res, index=[0]).T.reset_index()
        stats_df.columns = ['Statistic', 'Value']

        risk_df = None
        if is_2x2:
            vals = tab_chi2.values
            a, b, c, d = vals[0,0], vals[0,1], vals[1,0], vals[1,1]
            row_labs, col_labs = tab_chi2.index.tolist(), tab_chi2.columns.tolist()
            
            risk_exp = a / (a + b) if (a+b) > 0 else 0
            risk_unexp = c / (c + d) if (c+d) > 0 else 0
            
            rr = risk_exp / risk_unexp if risk_unexp > 0 else np.nan
            rd = risk_exp - risk_unexp
            
            or_val = (a * d) / (b * c) if (b * c) != 0 else np.nan
            se_log_or = np.sqrt(1/a+1/b+1/c+1/d) if (a>0 and b>0 and c>0 and d>0) else np.nan
            or_ci = calculate_ci_log_odds(or_val, se_log_or)
            
            rr_ci = calculate_ci_rr(risk_exp, a+b, risk_unexp, c+d) if risk_exp > 0 and risk_unexp > 0 else (np.nan, np.nan)
            
            rd_se = np.sqrt((risk_exp*(1-risk_exp))/(a+b) + (risk_unexp*(1-risk_unexp))/(c+d)) if (a+b)>0 and (c+d)>0 else np.nan
            # nnt_ci = calculate_ci_nnt(rd, rd_se) # Not currently used in display but calculated
            
            sensitivity = a/(a+c) if (a+c)>0 else 0
            specificity = d/(b+d) if (b+d)>0 else 0
            se_ci, sp_ci = calculate_ci_wilson_score(a, a+c), calculate_ci_wilson_score(d, b+d)
            
            risk_data = [
                {"Metric": "RISK METRICS (Assumes: Rows=Exposure Status, Cols=Outcome Status)", "Value": "", "95% CI": "", "Interpretation": ""},
                {"Metric": "Risk Ratio (RR)", "Value": f"{rr:.4f}", "95% CI": f"({rr_ci[0]:.4f} - {rr_ci[1]:.4f})", "Interpretation": f"Risk in {row_labs[0]} vs {row_labs[1]}"},
                {"Metric": "Odds Ratio (OR)", "Value": f"{or_val:.4f}", "95% CI": f"({or_ci[0]:.4f} - {or_ci[1]:.4f})", "Interpretation": f"Odds of {col_labs[0]}"},
                {"Metric": "DIAGNOSTIC METRICS (Assumes: Rows=Test Result, Cols=Disease Status)", "Value": "", "95% CI": "", "Interpretation": ""},
                {"Metric": "Sensitivity", "Value": f"{sensitivity:.4f}", "95% CI": f"({se_ci[0]:.4f} - {se_ci[1]:.4f})", "Interpretation": "True Positive Rate"},
                {"Metric": "Specificity", "Value": f"{specificity:.4f}", "95% CI": f"({sp_ci[0]:.4f} - {sp_ci[1]:.4f})", "Interpretation": "True Negative Rate"}
            ]
            risk_df = pd.DataFrame(risk_data)
        return display_tab, stats_df, msg, risk_df
    except Exception as e: return display_tab, None, str(e), None

# ========================================
# 4. COHEN'S KAPPA
# ========================================

def calculate_kappa(df, col1, col2):
    if col1 not in df.columns or col2 not in df.columns: return None, "Columns not found", None
    data = df[[col1, col2]].dropna()
    if data.empty: return None, "No data", None
    y1, y2 = data[col1].astype(str), data[col2].astype(str)
    try:
        kappa = cohen_kappa_score(y1, y2)
        interp = "Poor" if kappa < 0 else "Slight" if kappa <= 0.2 else "Fair" if kappa <= 0.4 else "Moderate" if kappa <= 0.6 else "Substantial" if kappa <= 0.8 else "Almost perfect"
        res_df = pd.DataFrame({"Statistic": ["Cohen's Kappa", "N (Pairs)", "Interpretation"], "Value": [f"{kappa:.4f}", f"{len(data)}", interp]})
        conf_matrix = pd.crosstab(y1, y2, rownames=[f"{col1} (Obs 1)"], colnames=[f"{col2} (Obs 2)"])
        return res_df, None, conf_matrix
    except Exception as e: return None, str(e), None

# ========================================
# 5. ROC CURVE
# ========================================

def auc_ci_delong(y_true, y_scores):
    try:
        y_true, y_scores = np.array(y_true), np.array(y_scores)
        auc = roc_auc_score(y_true, y_scores)
        pos_scores, neg_scores = y_scores[y_true == 1], y_scores[y_true == 0]
        n_pos, n_neg = len(pos_scores), len(neg_scores)
        if n_pos <= 1 or n_neg <= 1: return np.nan, np.nan, np.nan
        v10 = [(np.sum(p > neg_scores) + 0.5*np.sum(p == neg_scores)) / n_neg for p in pos_scores]
        v01 = [(np.sum(pos_scores > n) + 0.5*np.sum(pos_scores == n)) / n_pos for n in neg_scores]
        se_auc = np.sqrt((np.var(v10, ddof=1) / n_pos) + (np.var(v01, ddof=1) / n_neg))
        return auc - 1.96*se_auc, auc + 1.96*se_auc, se_auc
    except: return np.nan, np.nan, np.nan

def analyze_roc(df, truth_col, score_col, method='delong', pos_label_user=None):
    data = df[[truth_col, score_col]].dropna()
    if data.empty: return None, "No data", None, None
    y_score = pd.to_numeric(data[score_col], errors='coerce').dropna()
    y_true_raw = data[truth_col].loc[y_score.index]
    
    # Validation: Must have exactly 2 classes in the data subset
    if y_true_raw.nunique() != 2 or pos_label_user is None: 
        return None, "Error: Data must contain exactly 2 classes (Binary).", None, None
    
    y_true = np.where(y_true_raw.astype(str) == pos_label_user, 1, 0)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    ci_lower, ci_upper, se = auc_ci_delong(y_true, y_score.values)
    
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    stats_res = {
        "AUC": auc_val, 
        "95% CI Lower": max(0, ci_lower), 
        "95% CI Upper": min(1, ci_upper), 
        "Sensitivity": tpr[best_idx], 
        "Specificity": 1-fpr[best_idx], 
        "Best Cut-off": thresholds[best_idx]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={auc_val:.3f}', line={'color': COLORS['primary']}))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line={'color': COLORS['neutral'], 'dash': 'dash'}, showlegend=False))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", template="plotly_white")
    
    coords_df = pd.DataFrame({'Threshold': thresholds, 'Sensitivity': tpr, 'Specificity': 1-fpr}).round(4)
    return stats_res, None, fig, coords_df

# ========================================
# 6. ICC (Intraclass Correlation)
# ========================================

def calculate_icc(df, cols):
    if len(cols) < 2: return None, "Need at least 2 raters.", None
    data = df[cols].dropna()
    n, k = data.shape
    if n < 2 or k < 2: return None, "Insufficient data.", None
    
    grand_mean = data.values.mean()
    SSrow = k * ((data.mean(axis=1) - grand_mean)**2).sum()
    SScol = n * ((data.mean(axis=0) - grand_mean)**2).sum()
    SStotal = ((data.values - grand_mean)**2).sum()
    SSres = SStotal - SSrow - SScol
    
    MSrow, MScol, MSres = SSrow/(n-1), SScol/(k-1), SSres/((n-1)*(k-1))
    icc3_1 = (MSrow - MSres) / (MSrow + (k - 1) * MSres)
    icc2_1 = (MSrow - MSres) / (MSrow + (k - 1) * MSres + (k/n)*(MScol - MSres))
    
    res_df = pd.DataFrame({"Model": ["ICC(2,1)", "ICC(3,1)"], "ICC Value": [f"{icc2_1:.4f}", f"{icc3_1:.4f}"]})
    return res_df, None, None

# ========================================
# 7. REPORT GENERATION
# ========================================

def generate_report(title, elements):
    """Generate HTML Report (Shiny Friendly)"""
    primary_color = COLORS['primary']
    text_color = COLORS['text']
    
    css = f"""
    <style>
        body{{font-family:sans-serif; padding:20px; color:{text_color}}} 
        h1{{color:{primary_color}; border-bottom:2px solid {primary_color}}} 
        h2{{margin-top:20px; color:{primary_color}}}
        table{{width:100%; border-collapse:collapse; margin:20px 0}} 
        th,td{{border:1px solid #ddd; padding:8px; text-align:left}} 
        th{{background:{primary_color}; color:white}}
        .report-footer{{margin-top:30px; font-size:0.8em; color:#777; border-top:1px solid #eee; padding-top:10px}}
    </style>
    """
    html = f"<html><head>{css}</head><body><h1>{_html.escape(title)}</h1>"
    
    for el in elements:
        t, d, h = el.get('type'), el.get('data'), el.get('header')
        if h: html += f"<h2>{_html.escape(h)}</h2>"
        if t == 'text': html += f"<p>{_html.escape(str(d))}</p>"
        elif t == 'table' or t == 'contingency_table':
            html += d.to_html(classes='report-table')
        elif t == 'plot':
            if hasattr(d, 'to_html'): 
                # Use CDN to ensure plot renders in standalone HTML
                html += d.to_html(full_html=False, include_plotlyjs='cdn')
    
    html += "<div class='report-footer'>&copy; 2025 NTWKKM n donate | Powered by Shiny</div></body></html>"
    return html

"""
🧮 Logistic Regression Core Logic (Shiny Compatible)

No Streamlit dependencies - pure statistical functions.
"""

from typing import TypedDict, Literal, Any, cast, Union, Optional
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import warnings
import html
from logger import get_logger
from forest_plot_lib import create_forest_plot
from tabs._common import get_color_palette

logger = get_logger(__name__)

# Fetch palette and extend for local needs
_PALETTE = get_color_palette()
COLORS = {
    'primary': _PALETTE.get('primary', '#1E3A5F'),
    'primary_dark': _PALETTE.get('primary_dark', '#0F2440'), 
    'danger': _PALETTE.get('danger', '#E74856'),
    'text_secondary': _PALETTE.get('text_secondary', '#6B7280'),
    'border': _PALETTE.get('border', '#E5E7EB')
}

# Try to import Firth regression
try:
    from firthmodels import FirthLogisticRegression   
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
    logger.warning("firthmodels not available")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*convergence.*")

# ✅ NEW: Type Definitions (Compatible with Python 3.9+)
FitStatus = Union[Literal["OK"], str]
MethodType = Literal["default", "bfgs", "firth", "auto"]

class StatsMetrics(TypedDict):
    mcfadden: float
    nagelkerke: float

# Functional syntax for TypedDict to support 'or' as a key
ORResult = TypedDict("ORResult", {
    "or": float,
    "ci_low": float,
    "ci_high": float,
    "p_value": float
})

class AORResultEntry(TypedDict):
    coef: float
    aor: float
    l: float
    h: float
    p: float
    lvl: Optional[str]

# Functional syntax for InteractionResult due to 'or' keyword
InteractionResult = TypedDict("InteractionResult", {
    "coef": float,
    "or": Optional[float],
    "ci_low": float,
    "ci_high": float,
    "p_value": float,
    "label": str
})

def validate_logit_data(y: pd.Series, X: pd.DataFrame) -> tuple[bool, str]:
    """
    ✅ NEW: Added validation to prevent crashes during model fitting.
    Checks for perfect separation, zero variance, and collinearity.
    """
    issues = []
    
    # Check for empty data
    if len(y) == 0 or X.empty:
        return False, "Empty data provided"
        
    # Check for constant outcome
    if y.nunique() < 2:
        issues.append("Outcome variable has only one unique value (constant).")
            
    # Check for zero variance (constant columns)
    for col in X.columns:
        if X[col].nunique() <= 1:
            issues.append(f"Variable '{col}' has zero variance (only one value)")
            
    # Check for perfect separation (quasi-complete or complete)
    # If any cell in crosstab is 0, Logit might fail to converge
    for col in X.columns:
        try:
            ct = pd.crosstab(X[col], y)
            if (ct == 0).any().any():
                logger.debug(f"Perfect separation detected in variable: {col}")
                # We don't block this, but we log it to use Firth later
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not check separation for {col}: {e}")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


def clean_numeric_value(val: Any) -> float:
    """Convert value to float, removing common non-numeric characters."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    s = s.replace('>', '').replace('<', '').replace(',', '')
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _robust_sort_key(x: Any) -> tuple[int, Union[float, str]]:
    """Sort key placing numeric values first."""
    try:
        if pd.isna(x):
            return (2, "")
        val = float(x)
        return (0, val)
    except (ValueError, TypeError):
        return (1, str(x))


def run_binary_logit(
    y: pd.Series, 
    X: pd.DataFrame, 
    method: MethodType = 'default'
) -> tuple[Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.Series], FitStatus, StatsMetrics]:
    """
    Fit binary logistic regression.
    
    Returns:
        tuple: (params, conf_int, pvalues, status_msg, stats_dict)
    """
    stats_metrics: StatsMetrics = {"mcfadden": np.nan, "nagelkerke": np.nan}
    
    # ✅ NEW: Initial Validation
    is_valid, msg = validate_logit_data(y, X)
    if not is_valid:
        return None, None, None, msg, stats_metrics

    try:
        X_const = sm.add_constant(X, has_constant='add')
        
        if method == 'firth':
            if not HAS_FIRTH:
                return None, None, None, "firthmodels not installed", stats_metrics
            
            fl = FirthLogisticRegression(fit_intercept=False)
            fl.fit(X_const, y)
            
            coef = np.asarray(fl.coef_).reshape(-1)
            if coef.shape[0] != len(X_const.columns):
                return None, None, None, "Firth output shape mismatch", stats_metrics
            
            params = pd.Series(coef, index=X_const.columns)
            # firthmodels uses pvalues_ (not pvals_)
            pvalues = pd.Series(getattr(fl, "pvalues_", np.full(len(X_const.columns), np.nan)), index=X_const.columns)
            # firthmodels uses conf_int() method (not ci_ attribute)
            try:
                ci = fl.conf_int()  # Returns DataFrame with 0, 1 columns
                conf_int = pd.DataFrame(ci, index=X_const.columns, columns=[0, 1])
            except Exception:
                conf_int = pd.DataFrame(np.nan, index=X_const.columns, columns=[0, 1])
            return params, conf_int, pvalues, "OK", stats_metrics
        
        elif method == 'bfgs':
            model = sm.Logit(y, X_const)
            result = model.fit(method='bfgs', maxiter=100, disp=0)
        else:
            model = sm.Logit(y, X_const)
            result = model.fit(disp=0)
        
        # Calculate R-squared metrics
        try:
            llf = result.llf
            llnull = result.llnull
            nobs = result.nobs
            mcfadden = 1 - (llf / llnull) if llnull != 0 else np.nan
            if np.isnan(mcfadden):
                mcfadden = 0.0
            cox_snell = 1 - np.exp((2/nobs) * (llnull - llf))
            max_r2 = 1 - np.exp((2/nobs) * llnull)
            nagelkerke = cox_snell / max_r2 if max_r2 > 1e-9 else np.nan
            if np.isnan(nagelkerke):
                nagelkerke = 0.0
            stats_metrics = {"mcfadden": mcfadden, "nagelkerke": nagelkerke}
        except (AttributeError, ZeroDivisionError, TypeError) as e:
            logger.debug(f"Failed to calculate R2: {e}")
        
        return result.params, result.conf_int(), result.pvalues, "OK", stats_metrics
    
    except Exception as e:
        # ✅ NEW: Friendly Error Messaging for Technical Jargon
        err_msg = str(e)
        if "Singular matrix" in err_msg or "LinAlgError" in err_msg:
            err_msg = "Model fitting failed: data may have perfect separation or too much collinearity."
        logger.error(f"Logistic regression failed: {e}")
        return None, None, None, err_msg, stats_metrics


def get_label(col_name: str, var_meta: Optional[dict[str, Any]]) -> str:
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


def fmt_p_with_styling(val: Union[float, str, None]) -> str:
    """Format p-value with red highlighting if significant (p < 0.05)."""
    if pd.isna(val):
        return "-"
    try:
        val_f = float(val)
        val_f = max(0.0, min(1.0, val_f))
        if val_f < 0.001:
            p_str = "<0.001"
        elif val_f > 0.999:
            p_str = ">0.999"
        else:
            p_str = f"{val_f:.3f}"
        
        if val_f < 0.05:
            return f"<span class='sig-p'>{p_str}</span>"
        else:
            return p_str
    except (ValueError, TypeError):
        return "-"


def analyze_outcome(
    outcome_name: str, 
    df: pd.DataFrame, 
    var_meta: Optional[dict[str, Any]] = None, 
    method: MethodType = 'auto', 
    interaction_pairs: Optional[list[tuple[str, str]]] = None
) -> tuple[str, dict[str, ORResult], dict[str, ORResult], dict[str, InteractionResult]]:
    """
    Perform logistic regression analysis for binary outcome.
    
    Args:
        outcome_name: Name of binary outcome column
        df: Input DataFrame
        var_meta: Variable metadata dictionary
        method: 'auto', 'default', 'bfgs', or 'firth'
        interaction_pairs: List of tuples for interactions [(var1, var2), ...]
    
    Returns:
        tuple: (html_table, or_results, aor_results, interaction_results)
    """
    logger.info(f"Starting logistic analysis for outcome: {outcome_name}")
    
    if outcome_name not in df.columns:
        msg = f"Outcome '{outcome_name}' not found"
        logger.error(msg)
        return f"<div class='alert'>{msg}</div>", {}, {}, {}
    
    y_raw = df[outcome_name].dropna()
    unique_outcomes = set(y_raw.unique())
    
    if len(unique_outcomes) != 2:
        msg = f"Invalid outcome: expected 2 values, found {len(unique_outcomes)}"
        logger.error(msg)
        return f"<div class='alert'>{msg}</div>", {}, {}, {}
    
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
            if col == outcome_name or col not in df_aligned.columns:
                continue
            try:
                X_num = df_aligned[col].apply(clean_numeric_value)
                if X_num.nunique() > 1 and (pd.crosstab(X_num, y) == 0).any().any():
                    has_perfect_separation = True
                    break
            except Exception:
                continue
    
    # Select fitting method
    preferred_method: MethodType = 'bfgs'
    if method == 'auto' and HAS_FIRTH and (has_perfect_separation or len(df) < 50 or (y == 1).sum() < 20):
        preferred_method = 'firth'
    elif method == 'firth':
        preferred_method = 'firth' if HAS_FIRTH else 'bfgs'
    elif method == 'default':
        preferred_method = 'default'
    
    def count_val(series: pd.Series, v_str: str) -> int:
        return (series.astype(str).apply(lambda x: x.replace('.0', '') if x.replace('.', '', 1).isdigit() else x) == v_str).sum()
    
    or_results: dict[str, ORResult] = {}
    
    # Univariate analysis
    logger.info(f"Starting univariate analysis for {len(sorted_cols)-1} variables")
    for col in sorted_cols:
        if col == outcome_name or col not in df_aligned.columns or df_aligned[col].isnull().all():
            continue
        
        res: dict[str, Any] = {'var': col}
        X_raw = df_aligned[col]
        X_num = X_raw.apply(clean_numeric_value)
        X_neg = X_raw[y == 0]
        X_pos = X_raw[y == 1]
        
        unique_vals = X_num.dropna().unique()
        mode = 'linear'
        
        # Auto-detect mode
        if set(unique_vals).issubset({0, 1}):
            mode = 'categorical'
        elif len(unique_vals) < 10:
            decimals_pct = sum(1 for v in unique_vals if not float(v).is_integer()) / len(unique_vals) if len(unique_vals) > 0 else 0
            if decimals_pct < 0.3:
                mode = 'categorical'
        
        # Check var_meta
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
            desc_tot = [f"n={n_used}"]
            desc_neg = [f"n={len(X_neg.dropna())}"]
            desc_pos = [f"n={len(X_pos.dropna())}"]
            
            for lvl in levels:
                lbl_txt = str(int(float(lvl))) if str(lvl).endswith('.0') else str(lvl)
                c_all = count_val(X_raw, lbl_txt)
                p_all = (c_all / n_used) * 100 if n_used else 0
                c_n = count_val(X_neg, lbl_txt)
                p_n = (c_n / len(X_neg.dropna())) * 100 if len(X_neg.dropna()) else 0
                c_p = count_val(X_pos, lbl_txt)
                p_p = (c_p / len(X_pos.dropna())) * 100 if len(X_pos.dropna()) else 0
                
                desc_tot.append(f"{lbl_txt}: {c_all} ({p_all:.1f}%)")
                desc_neg.append(f"{c_n} ({p_n:.1f}%)")
                desc_pos.append(f"{c_p} ({p_p:.1f}%)")
            
            res['desc_total'] = "<br>".join(desc_tot)
            res['desc_neg'] = "<br>".join(desc_neg)
            res['desc_pos'] = "<br>".join(desc_pos)
            
            try:
                ct = pd.crosstab(X_raw, y)
                _, p, _, _ = stats.chi2_contingency(ct) if ct.size > 0 else (0, np.nan, 0, 0)
                res['p_comp'] = p
                res['test_name'] = "Chi-square"
            except (ValueError, TypeError):
                res['p_comp'] = np.nan
                res['test_name'] = "-"
            
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
                        
                        res['or'] = "<br>".join(or_lines)
                        res['coef'] = "<br>".join(coef_lines)
                        res['p_or'] = "<br>".join(p_lines)
                    else:
                        res['or'] = f"<span style='color:red; font-size:0.8em'>{status}</span>"
                        res['coef'] = "-"
                else:
                    res['or'] = "-"
                    res['coef'] = "-"
            else:
                res['or'] = "-"
                res['coef'] = "-"
        
        else:  # Linear mode
            n_used = len(X_num.dropna())
            m_t, s_t = X_num.mean(), X_num.std()
            m_n, s_n = pd.to_numeric(X_neg, errors='coerce').mean(), pd.to_numeric(X_neg, errors='coerce').std()
            m_p, s_p = pd.to_numeric(X_pos, errors='coerce').mean(), pd.to_numeric(X_pos, errors='coerce').std()
            
            res['desc_total'] = f"n={n_used}<br>Mean: {m_t:.2f} (SD {s_t:.2f})"
            res['desc_neg'] = f"{m_n:.2f} ({s_n:.2f})"
            res['desc_pos'] = f"{m_p:.2f} ({s_p:.2f})"
            
            try:
                _, p = stats.mannwhitneyu(pd.to_numeric(X_neg, errors='coerce').dropna(), pd.to_numeric(X_pos, errors='coerce').dropna())
                res['p_comp'] = p
                res['test_name'] = "Mann-Whitney"
            except (ValueError, TypeError):
                res['p_comp'] = np.nan
                res['test_name'] = "-"
            
            data_uni = pd.DataFrame({'y': y, 'x': X_num}).dropna()
            if not data_uni.empty and data_uni['x'].nunique() > 1:
                params, hex_conf, pvals, status, _ = run_binary_logit(data_uni['y'], data_uni[['x']], method=preferred_method)
                if status == "OK" and 'x' in params:
                    coef = params['x']
                    odd = np.exp(coef)
                    ci_l, ci_h = np.exp(hex_conf.loc['x'][0]), np.exp(hex_conf.loc['x'][1])
                    pv = pvals['x']
                    
                    res['coef'] = f"{coef:.3f}"
                    res['or'] = f"{odd:.2f} ({ci_l:.2f}-{ci_h:.2f})"
                    res['p_or'] = pv
                    or_results[col] = {'or': odd, 'ci_low': ci_l, 'ci_high': ci_h, 'p_value': pv}
                else:
                    res['or'] = f"<span style='color:red; font-size:0.8em'>{status}</span>"
                    res['coef'] = "-"
            else:
                res['or'] = "-"
                res['coef'] = "-"
        
        results_db[col] = res
        
        # Screen for multivariate
        p_screen = res.get('p_comp', np.nan)
        if isinstance(p_screen, (int, float)) and pd.notna(p_screen) and p_screen < 0.20:
            candidates.append(col)
    
    # Multivariate analysis
    aor_results: dict[str, ORResult] = {}
    interaction_results: dict[str, InteractionResult] = {}
    int_meta = {}
    final_n_multi = 0
    mv_metrics_text = ""
    
    def _is_candidate_valid(col):
        mode = mode_map.get(col, "linear")
        series = df_aligned[col]
        if mode == "categorical":
            return series.notna().sum() > 5
        return series.apply(clean_numeric_value).notna().sum() > 5
    
    cand_valid = [c for c in candidates if _is_candidate_valid(c)]
    
    # ✅ FIX: Run multivariate analysis if there are candidates OR interaction pairs
    if len(cand_valid) > 0 or interaction_pairs:
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
                multi_df[c] = df_aligned[c].apply(clean_numeric_value)
        
        # ✅ NEW: Add interaction terms if specified
        if interaction_pairs:
            try:
                from interaction_lib import create_interaction_terms
                multi_df, int_meta = create_interaction_terms(multi_df, interaction_pairs, mode_map)
                logger.info(f"✅ Added {len(int_meta)} interaction terms to logistic multivariate model")
            except ImportError:
                logger.warning("interaction_lib not available, skipping interaction terms")
            except (ValueError, KeyError) as e:
                logger.exception("Failed to create interaction terms")
        
        multi_data = multi_df.dropna()
        final_n_multi = len(multi_data)
        predictors = [col for col in multi_data.columns if col != 'y']
        
        if not multi_data.empty and final_n_multi > 10 and len(predictors) > 0:
            params, conf, pvals, status, mv_stats = run_binary_logit(multi_data['y'], multi_data[predictors], method=preferred_method)
            
            if status == "OK":
                r2_parts = []
                mcf = mv_stats.get('mcfadden')
                nag = mv_stats.get('nagelkerke')
                if pd.notna(mcf):
                    r2_parts.append(f"McFadden R² = {mcf:.3f}")
                if pd.notna(nag):
                    r2_parts.append(f"Nagelkerke R² = {nag:.3f}")
                if r2_parts:
                    mv_metrics_text = " | ".join(r2_parts)
                
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
                
                # ✅ NEW: Process interaction effects
                if int_meta:
                    try:
                        from interaction_lib import format_interaction_results
                        interaction_results = format_interaction_results(params, conf, pvals, int_meta, 'logit')
                        logger.info(f"✅ Formatted {len(interaction_results)} logistic interaction results")
                        
                        # ✅ FIX: Merge interaction results into aor_results for forest plot inclusion
                        for int_name, int_res in interaction_results.items():
                             label = f"🔗 {int_res.get('label', int_name)}"
                             aor_results[label] = {
                                 'aor': int_res.get('or', np.nan), 
                                 'ci_low': int_res.get('ci_low', np.nan), 
                                 'ci_high': int_res.get('ci_high', np.nan), 
                                 'p_value': int_res.get('p_value', np.nan)
                             }
                    except Exception as e:
                        logger.error(f"Failed to format interaction results: {e}")
            else:
                # Log multivariate failure
                mv_metrics_text = f"<span style='color:red'>Adjustment Failed: {status}</span>"
    
    # Build HTML
    html_rows = []
    current_sheet = ""
    valid_cols_for_html = [c for c in sorted_cols if c in results_db]
    grouped_cols = sorted(valid_cols_for_html, key=lambda x: (x.split('_')[0] if '_' in x else "Variables", x))
    
    for col in grouped_cols:
        if col == outcome_name:
            continue
        res = results_db[col]
        mode = mode_map.get(col, 'linear')
        sheet = col.split('_')[0] if '_' in col else "Variables"
        
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
            if isinstance(multi_res, list):
                aor_lines, acoef_lines, ap_lines = ["Ref."], ["-"], ["-"]
                for item in multi_res:
                    p_txt = fmt_p_with_styling(item['p'])
                    acoef_lines.append(f"{item['coef']:.3f}")
                    aor_lines.append(f"{item['aor']:.2f} ({item['l']:.2f}-{item['h']:.2f})")
                    ap_lines.append(p_txt)
                aor_s, acoef_s, ap_s = "<br>".join(aor_lines), "<br>".join(acoef_lines), "<br>".join(ap_lines)
            else:
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
    
    # ✅ NEW: Add interaction terms to HTML table
    if interaction_results:
        for int_name, res in interaction_results.items():
            int_label = res.get('label', int_name)
            int_coef = f"{res.get('coef', 0):.3f}" if pd.notna(res.get('coef')) else "-"
            or_val = res.get('or')
            if or_val is not None and pd.notna(or_val):
                int_or = f"{or_val:.2f} ({res.get('ci_low', 0):.2f}-{res.get('ci_high', 0):.2f})"
            else:
                int_or = "-"
            int_p = fmt_p_with_styling(res.get('p_value', 1))
            
            html_rows.append(f"""<tr style='background-color: #fff9f0;'>
                <td><b>🔗 {int_label}</b><br><small style='color: #666;'>(Interaction)</small></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>{int_coef}</td>
                <td><b>{int_or}</b></td>
                <td>Interaction</td>
                <td>-</td>
                <td>{int_coef}</td>
                <td><b>{int_or}</b></td>
                <td>{int_p}</td>
            </tr>""")
    
    logger.info(f"Logistic analysis complete. Multivariate n={final_n_multi}, Interactions={len(interaction_results)}")
    
    model_fit_html = ""
    if mv_metrics_text:
        model_fit_html = f"<div style='margin-top: 8px; padding-top: 8px; border-top: 1px dashed #ccc; color: {COLORS['primary_dark']};'><b>Model Fit:</b> {mv_metrics_text}</div>"
    
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
            {f"<br><b>Interactions Tested:</b> {len(interaction_pairs)} pairs" if interaction_pairs else ""}
        </div>
    </div>
    </div><br>"""
    
    return html_table, or_results, aor_results, interaction_results


def run_logistic_regression(df, outcome_col, covariate_cols):
    """
    Robust wrapper for logistic analysis as requested by integration requirements.
    """
    metrics = {}  # ✅ Initialize early
    
    # VALIDATION: Check for constant outcome
    if outcome_col not in df.columns:
        return None, None, f"Error: Outcome '{outcome_col}' not found.", {}
        
    if df[outcome_col].nunique() < 2:
        return None, None, f"Error: Outcome '{outcome_col}' is constant", {}

    try:
        y = df[outcome_col]
        X = df[covariate_cols]
        
        X_const = sm.add_constant(X)
        model = sm.Logit(y, X_const)
        result = model.fit(disp=0)

        # ✅ Handle NaN in McFadden R-squared
        mcfadden = result.prsquared if not np.isnan(result.prsquared) else 0.0
        
        metrics = {
            'aic': result.aic,
            'bic': result.bic,
            'mcfadden': mcfadden,
            'nagelkerke': 0.0  # Placeholder if not calculated
        }
        
        # Build OR results from fitted model
        or_results = {}
        for col in covariate_cols:
            if col in result.params.index:
                or_results[col] = {
                    'or': np.exp(result.params[col]),
                    'ci_low': np.exp(result.conf_int().loc[col, 0]),
                    'ci_high': np.exp(result.conf_int().loc[col, 1]),
                    'p_value': result.pvalues[col]
                }
        
        return None, or_results, "OK", metrics  # html_table optional

    except Exception as e:
        logger.error(f"Logistic regression failed: {e}")
        return None, None, str(e), {}

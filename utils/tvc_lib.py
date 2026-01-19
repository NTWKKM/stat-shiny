"""
⏱️ Time-Varying Covariates (TVC) Module - Time-Dependent Survival Analysis

Provides comprehensive utilities for Cox regression with time-varying covariates using lifelines.CoxTimeVaryingFitter.

Key Features:
- Data format transformation (Wide ↔ Long)
- CoxTimeVaryingFitter model fitting with auto-detection
- Proportional hazards assumption testing (time-varying)
- Forest plot generation
- Report generation with diagnostic plots
- Input validation & error handling

Usage:
    >>> from utils.tvc_lib import transform_wide_to_long, fit_tvc_cox
    >>> 
    >>> # Transform wide → long format
    >>> long_data = transform_wide_to_long(
    ...     wide_df, id_col='patient_id', time_col='followup_time',
    ...     event_col='event', risk_intervals=[0, 1, 3, 6, 12]
    ... )
    >>> 
    >>> # Fit TVC Cox model
    >>> cph, results_df, clean_data, err, stats, missing_info = fit_tvc_cox(
    ...     long_data, start_col='start_time', stop_col='stop_time',
    ...     event_col='event', tvc_cols=['treatment', 'lab_value'],
    ...     static_cols=['age', 'sex']
    ... )

References:
    - Andersen, P. K., & Gill, R. D. (1982). Cox's regression model for counting processes.
    - lifelines docs: https://lifelines.readthedocs.io/en/latest/api_reference/lifelines.CoxTimeVaryingFitter.html
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lifelines import CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats as sp_stats

from logger import get_logger
from utils.data_cleaning import handle_missing_for_analysis

logger = get_logger(__name__)


# ==============================================================================
# DATA TRANSFORMATION FUNCTIONS
# ==============================================================================

def validate_long_format(df: pd.DataFrame, 
                          id_col: str, 
                          start_col: str, 
                          stop_col: str, 
                          event_col: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a dataset is in proper long format for TVC Cox analysis.
    """
    
    # 1. Check column existence
    required_cols = {id_col, start_col, stop_col, event_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return False, f"❌ Missing columns: {', '.join(missing_cols)}"
    
    # 2. Check for NaN in required columns
    nan_cols = [c for c in required_cols if df[c].isna().any()]
    if nan_cols:
        return False, f"❌ NaN values found in: {', '.join(nan_cols)}"
    
    # 3. Check start_time < stop_time
    if (df[start_col] >= df[stop_col]).any():
        bad_rows = (df[start_col] >= df[stop_col]).sum()
        return False, f"❌ {bad_rows} rows have start_time >= stop_time"
    
    # 4. Check duplicate intervals per patient
    interval_cols = [id_col, start_col, stop_col]
    if df.duplicated(subset=interval_cols).any():
        n_dup = df.duplicated(subset=interval_cols).sum()
        return False, f"❌ {n_dup} duplicate time intervals per patient"
    
    # 5. Check event values are binary
    if not set(df[event_col].unique()).issubset({0, 1}):
        return False, f"❌ Event column must contain only 0 and 1. Found: {df[event_col].unique()}"
    
    # 6. Check event only in final interval per patient
    df_sorted = df.sort_values([id_col, stop_col])
    grouped = df_sorted.groupby(id_col)
    
    for patient_id, group in grouped:
        # Event should only be 1 in last row, and all events should be in that row
        event_mask = group[event_col] == 1
        if event_mask.any():
            event_rows = group[event_mask].index.tolist()
            last_row = group.index[-1]
            
            if not all(idx == last_row for idx in event_rows):
                return False, f"❌ Patient {patient_id}: Event occurs in non-final interval"
    
    return True, None


def transform_wide_to_long(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    event_col: str,
    tvc_cols: Optional[List[str]] = None,
    static_cols: Optional[List[str]] = None,
    risk_intervals: Optional[List[float]] = None,
    interval_method: str = "quantile"
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Transform wide-format survival data to long format for time-varying covariate analysis.
    """
    
    try:
        # Validate inputs
        if id_col not in df.columns:
            return None, f"❌ ID column '{id_col}' not found"
        if time_col not in df.columns:
            return None, f"❌ Time column '{time_col}' not found"
        if event_col not in df.columns:
            return None, f"❌ Event column '{event_col}' not found"
        
        # Validate TVC columns
        if not tvc_cols:
            tvc_cols = []
        else:
            missing_tvc = set(tvc_cols) - set(df.columns)
            if missing_tvc:
                return None, f"❌ Missing TVC columns: {', '.join(missing_tvc)}"
        
        # Validate static columns
        if not static_cols:
            static_cols = []
        else:
            missing_static = set(static_cols) - set(df.columns)
            if missing_static:
                return None, f"❌ Missing static columns: {', '.join(missing_static)}"
        
        # Auto-detect risk intervals if not provided
        if not risk_intervals:
            if interval_method == "quantile" and len(tvc_cols) > 1:
                # Extract time points from TVC column names (e.g., 'tvc_3m' → 3)
                import re
                extracted_times = []
                pattern = r'(\d+)'
                
                for col in tvc_cols:
                    matches = re.findall(pattern, col)
                    if matches:
                        extracted_times.append(int(matches[0]))
                
                if extracted_times:
                    risk_intervals = sorted(set([0] + extracted_times))
                    logger.info(f"Auto-detected risk intervals: {risk_intervals}")
                else:
                    # Fallback: use quantiles of follow-up time
                    risk_intervals = [0] + sorted(df[time_col].quantile([0.25, 0.5, 0.75]).tolist())
                    logger.warning(f"Extracted times not found; using quantiles: {risk_intervals}")
            else:
                # Default: equal intervals
                max_time = df[time_col].max()
                risk_intervals = [0, max_time * 0.25, max_time * 0.5, max_time * 0.75, max_time]
                logger.warning(f"Using default risk intervals: {risk_intervals}")
        
        risk_intervals = sorted(set(risk_intervals))
        logger.info(f"Risk intervals: {risk_intervals}")
        
        # --- Handle Missing Data ---
        # Clean the input dataframe before processing to avoid NaN errors during iteration
        # Critical columns that must be present: ID, Time, Event
        critical_cols = [id_col, time_col, event_col]
        # Also include covariates to ensure consistent cleaning (optional but recommended)
        processing_cols = critical_cols + (tvc_cols or []) + (static_cols or [])
        
        # We use 'complete-case' on critical columns essentially, but for TVC we might want to be careful.
        # Let's clean rows that have missing values in CRITICAL columns first.
        df_clean = df.dropna(subset=critical_cols)
        
        if len(df_clean) < len(df):
            logger.warning(f"Transform: Dropped {len(df) - len(df_clean)} rows with missing ID/Time/Event")
        
        # Build long-format data
        long_rows = []
        
        for idx, row in df_clean.iterrows():
            patient_id = row[id_col]
            followup_time = row[time_col]
            event = row[event_col]
            
            # Create intervals for this patient
            patient_intervals = []
            for i in range(len(risk_intervals) - 1):
                start = risk_intervals[i]
                stop = risk_intervals[i + 1]
                
                # Skip intervals beyond follow-up time
                if start >= followup_time:
                    break
                
                # Truncate final interval at follow-up time
                if stop > followup_time:
                    stop = followup_time
                
                # Determine if event occurs in this interval
                interval_event = 1 if (event == 1 and stop >= followup_time) else 0
                
                # Create interval row
                interval_row = {
                    id_col: patient_id,
                    'start': start,
                    'stop': stop,
                    event_col: interval_event
                }
                
                # Add last-observed TVC values (carry forward)
                for tvc_col in tvc_cols:
                    # Find the closest TVC column with time ≤ stop
                    best_value = None
                    for prev_tvc_col in tvc_cols:
                        if pd.notna(row[prev_tvc_col]):
                            # Extract time from column name
                            import re
                            match = re.search(r'(\d+)', prev_tvc_col)
                            if match:
                                col_time = int(match.group(1))
                                if col_time <= stop:
                                    best_value = row[prev_tvc_col]
                    
                    if best_value is not None:
                        interval_row[tvc_col] = best_value
                    else:
                        interval_row[tvc_col] = row[tvc_cols[0]] if tvc_cols else np.nan
                
                # Add static covariates
                for static_col in static_cols:
                    interval_row[static_col] = row[static_col]
                
                patient_intervals.append(interval_row)
            
            long_rows.extend(patient_intervals)
        
        long_df = pd.DataFrame(long_rows)
        
        # Validate transformed data
        is_valid, val_err = validate_long_format(
            long_df, id_col=id_col, 
            start_col='start', 
            stop_col='stop', 
            event_col=event_col
        )
        
        if not is_valid:
            return None, val_err
        
        logger.info(f"✅ Wide→Long: {len(df)} patients → {len(long_df)} intervals")
        
        return long_df, None
        
    except Exception as e:
        logger.exception("Transform error")
        return None, f"❌ Transformation error: {str(e)}"


# ==============================================================================
# MODEL FITTING FUNCTIONS
# ==============================================================================

def fit_tvc_cox(
    df: pd.DataFrame,
    start_col: str,
    stop_col: str,
    event_col: str,
    tvc_cols: List[str],
    static_cols: Optional[List[str]] = None,
    penalizer: float = 0.0,
    var_meta: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[CoxTimeVaryingFitter], 
           Optional[pd.DataFrame], 
           Optional[pd.DataFrame], 
           Optional[str],
           Dict[str, Any],
           Optional[Dict[str, Any]]]:
    """
    Fit Cox proportional hazards model with time-varying covariates.
    """
    
    try:
        # --- 1. Column Resolution & Robust Detection ---
        # Handle cases where user-selected columns (from UI) differ from actual DataFrame columns
        # (e.g. standard 'start'/'stop' generated by transform_wide_to_long)
        
        real_start_col = start_col
        real_stop_col = stop_col
        
        # If specified columns don't exist, check for standard names
        if start_col not in df.columns and 'start' in df.columns:
            logger.info(f"TVC Fit: Start column '{start_col}' missing. Found 'start', using it.")
            real_start_col = 'start'
            
        if stop_col not in df.columns and 'stop' in df.columns:
            logger.info(f"TVC Fit: Stop column '{stop_col}' missing. Found 'stop', using it.")
            real_stop_col = 'stop'

        # Validate existence of resolved columns
        required_cols = [real_start_col, real_stop_col, event_col]
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            msg = f"❌ Missing columns: {', '.join(missing_required)}. Available: {sorted(df.columns.tolist())}"
            return None, None, None, msg, {}, {}

        # Combine all needed columns for cleaning
        all_covariates = (tvc_cols or []) + (static_cols or [])
        if not all_covariates:
            return None, None, None, "❌ No covariates specified", {}, {}
        
        missing_covars = set(all_covariates) - set(df.columns)
        if missing_covars:
            return None, None, None, f"❌ Missing covariates: {', '.join(missing_covars)}", {}, {}

        # --- 2. Data Cleaning ---
        id_col = df.columns[0] # Use first column as ID by convention
        
        # Prepare subset for cleaning
        key_cols = [id_col, real_start_col, real_stop_col, event_col] + all_covariates
        key_cols = list(dict.fromkeys(key_cols)) # Remove duplicates, preserve order
        
        clean_data, missing_info = handle_missing_for_analysis(
            df[key_cols],
            var_meta=var_meta or {},
            strategy="complete-case",
            return_counts=True
        )
        
        if len(clean_data) == 0:
            return None, None, None, "❌ All data dropped due to missing values", {}, missing_info

        # --- 2.5 Safety Restoration ---
        # [CRITICAL FIX] Ensure critical columns exist (restore from df if dropped by cleaner)
        # This fixes the issue where clean_data loses the start/stop/event columns
        for col in [real_start_col, real_stop_col, event_col]:
            if col not in clean_data.columns and col in df.columns:
                clean_data[col] = df.loc[clean_data.index, col]
                logger.warning(f"Restored column '{col}' from original data after cleaning.")

        # --- 3. Safety Check & Standardization ---
        standard_start = 'start'
        standard_stop = 'stop'
        rename_map = {}

        # If we found the columns but they aren't named 'start'/'stop', prepare rename
        if real_start_col != standard_start:
            rename_map[real_start_col] = standard_start
        if real_stop_col != standard_stop:
            rename_map[real_stop_col] = standard_stop

        # Apply renaming
        if rename_map:
            clean_data = clean_data.rename(columns=rename_map)
            logger.info(f"TVC Fit: Renamed columns for lifelines: {rename_map}")

        # Final check before fitting
        if standard_start not in clean_data.columns or standard_stop not in clean_data.columns:
             # Should be impossible if logic above holds, but safety first
             return None, None, None, f"❌ Internal Error: 'start' or 'stop' columns missing after standardization.", {}, {}

        # --- 4. Fitting ---
        cph = CoxTimeVaryingFitter(penalizer=penalizer)
        
        logger.info(f"TVC Fit - Final Columns: {clean_data.columns.tolist()}")
        logger.info(f"TVC Fit - Args: event='{event_col}', start='{standard_start}', stop='{standard_stop}'")
        
        cph.fit(
            df=clean_data,
            event_col=event_col,
            start_col=standard_start,
            stop_col=standard_stop
        )
        
        # --- 5. Post-Fit Validation ---
        is_valid, val_err = validate_long_format(
            clean_data, id_col=clean_data.columns[0],
            start_col=standard_start, stop_col=standard_stop, event_col=event_col
        )
        if not is_valid:
             return None, None, clean_data, f"❌ Data validation failed after cleaning: {val_err}", {}, missing_info
        
        # --- 6. Results Extraction ---
        summary_df = cph.summary.copy()
        
        # Apply labels
        label_map = {}
        if var_meta:
            for key, item in var_meta.items():
                if isinstance(item, dict):
                    label_map[key] = item.get('label', key)
        
        results_data = []
        for covar in all_covariates:
            if covar not in summary_df.index:
                logger.warning(f"Covariate {covar} not in model results")
                continue
            
            row = summary_df.loc[covar]
            label = label_map.get(covar, covar)
            
            results_data.append({
                'Variable': label or covar,
                'Coef': row['coef'],
                'HR': row['exp(coef)'],
                'HR_Lower': row['exp(coef) lower 95%'],
                'HR_Upper': row['exp(coef) upper 95%'],
                'p-value': row['p'],
                'Significant': '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Stats
        c_index = getattr(cph, 'concordance_index_', np.nan)
        aic = getattr(cph, 'AIC_partial_', getattr(cph, 'AIC_', np.nan))
        log_lik = getattr(cph, 'log_likelihood_', np.nan)
        
        stats = {
            'Concordance Index': round(c_index, 4) if pd.notna(c_index) else "N/A",
            'AIC': round(aic, 1) if pd.notna(aic) else "N/A",
            'Log-Likelihood': round(log_lik, 2) if pd.notna(log_lik) else "N/A",
            'N Events': clean_data[event_col].sum(),
            'N Observations': len(clean_data),
            'N Intervals': len(clean_data),
            'Penalizer': penalizer
        }
        
        logger.info(f"✅ TVC Cox fitted successfully.")
        
        return cph, results_df, clean_data, None, stats, missing_info
        
    except np.linalg.LinAlgError:
        logger.warning("TVC Cox: Singular matrix detected")
        return None, None, None, "❌ Singular Matrix Error: Variables are likely highly correlated or collinear.", {}, {}
    except Exception as e:
        logger.exception("TVC Cox fitting error")
        return None, None, None, f"❌ Model fitting error: {str(e)}", {}, {}


# ==============================================================================
# DIAGNOSTIC & VALIDATION FUNCTIONS
# ==============================================================================

def check_tvc_assumptions(
    cph: CoxTimeVaryingFitter,
    df: pd.DataFrame,
    start_col: str,
    stop_col: str,
    event_col: str
) -> Tuple[str, List[go.Figure]]:
    """
    Check proportional hazards assumptions for TVC Cox model.
    """
    try:
        # Standardize column names based on model attributes if available
        # This aligns with the standardized names used during fit
        if hasattr(cph, 'start_col') and cph.start_col in df.columns:
            start_col = cph.start_col
        elif 'start' in df.columns:
            start_col = 'start'

        if hasattr(cph, 'stop_col') and cph.stop_col in df.columns:
            stop_col = cph.stop_col
        elif 'stop' in df.columns:
            stop_col = 'stop'
            
        # Get covariates from model
        covariates = cph.params_.index.tolist()
        
        # --- Diagnostic 1: Coefficient Stability Over Time ---
        df_time_sorted = df.sort_values(stop_col)
        quartile_size = len(df_time_sorted) // 4
        
        quartile_coefs = []
        quartile_labels = []
        
        for q in range(4):
            start_idx = q * quartile_size
            end_idx = (q + 1) * quartile_size if q < 3 else len(df_time_sorted)
            df_quartile = df_time_sorted.iloc[start_idx:end_idx]
            
            if len(df_quartile) > 10 and df_quartile[event_col].sum() > 0:
                try:
                    cph_q = CoxTimeVaryingFitter()
                    cph_q.fit(df_quartile, stop_col=stop_col, 
                              event_col=event_col, start_col=start_col)
                    
                    quartile_coefs.append(cph_q.params_.to_dict())
                    quartile_labels.append(f"Q{q+1}")
                except:
                    pass
        
        # --- Diagnostic 2: Partial Likelihood Residuals ---
        residuals = cph.compute_partial_hazard(df)
        
        # --- Build Interpretation Text ---
        interpretation = """
        ### ⚠️ Time-Varying Covariate Diagnostics
        
        **Important Note:** Standard Schoenfeld residual tests are not directly applicable 
        to time-varying covariate models. Instead, we recommend:
        
        1. **Coefficient Stability Check (see plots below):**
           - Refit the model separately in time quartiles
           - Large changes in HR across quartiles suggest time-varying effects
           - If coefficients are stable, PH assumption is likely reasonable
        
        2. **Domain Knowledge:**
           - Consult clinical literature for expected covariate effects over time
           - Consider whether covariates should change their effect dynamically
        """
        
        # --- Build Diagnostic Plots ---
        plots = []
        
        # Plot 1: Coefficient Trends
        if len(quartile_coefs) > 1:
            coef_data = {covar: [] for covar in covariates}
            
            for q, coef_dict in enumerate(quartile_coefs):
                for covar in covariates:
                    coef_data[covar].append(coef_dict.get(covar, np.nan))
            
            fig = go.Figure()
            for covar in covariates[:min(5, len(covariates))]:
                fig.add_trace(go.Scatter(
                    x=quartile_labels,
                    y=coef_data[covar],
                    mode='lines+markers',
                    name=covar,
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Coefficient Stability Across Time Quartiles",
                xaxis_title="Time Period",
                yaxis_title="Coefficient (log-HR)",
                hovermode='x unified',
                height=400
            )
            plots.append(fig)
        
        # Plot 2: Residuals
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[stop_col],
            y=np.log(residuals + 1e-8),
            mode='markers',
            marker=dict(size=4, opacity=0.5, color='steelblue'),
            name='Log Partial Hazard'
        ))
        
        fig.update_layout(
            title="Partial Hazard Over Follow-up Time",
            xaxis_title="Stop Time",
            yaxis_title="Log(Partial Hazard)",
            height=400
        )
        plots.append(fig)
        
        return interpretation, plots
        
    except Exception as e:
        logger.exception("Assumption check error")
        return f"⚠️ Assumption diagnostics unavailable: {str(e)}", []


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_tvc_forest_plot(results_df: pd.DataFrame) -> go.Figure:
    """
    Create a forest plot for TVC Cox model results.
    """
    
    results_sorted = results_df.sort_values('HR', ascending=False)
    
    fig = go.Figure()
    
    # Add HR points and CI error bars
    for idx, row in results_sorted.iterrows():
        ci_text = f"{row['HR']:.2f} (95% CI: {row['HR_Lower']:.2f}-{row['HR_Upper']:.2f})"
        sig_marker = "***" if row['p-value'] < 0.001 else ("**" if row['p-value'] < 0.01 else ("*" if row['p-value'] < 0.05 else ""))
        
        fig.add_trace(go.Scatter(
            x=[row['HR_Lower'], row['HR_Upper']],
            y=[row['Variable'], row['Variable']],
            mode='lines',
            line=dict(width=2, color='steelblue'),
            name='',
            hoverinfo='skip',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[row['HR']],
            y=[row['Variable']],
            mode='markers',
            marker=dict(size=10, color='darkblue'),
            name='',
            text=f"{ci_text} {sig_marker}<br>p={row['p-value']:.4f}",
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
    
    # Add reference line at HR=1
    fig.add_vline(x=1, line_dash="dash", line_color="red", 
                  annotation_text="No Effect (HR=1)", 
                  annotation_position="top")
    
    fig.update_layout(
        title="Forest Plot: Time-Varying Cox Model",
        xaxis_title="Hazard Ratio (log scale)",
        yaxis_title="Variable",
        height=400 + len(results_sorted) * 30,
        xaxis_type="log",
        hovermode='closest'
    )
    
    return fig


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_tvc_report(
    title: str,
    elements: List[Dict[str, Any]],
    stats: Dict[str, Any],
    missing_data_info: Dict[str, Any],
    var_meta: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate HTML report for TVC Cox analysis.
    """
    from utils.formatting import create_missing_data_report_html
    from config import CONFIG
    import html as _html

    primary_color = CONFIG.get('ui.colors.primary', '#2180BE')
    primary_dark = CONFIG.get('ui.colors.primary_dark', '#1a5a8a')
    text_color = '#333'
    
    css_style = f"""<style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 20px;
            background-color: #f4f6f8;
            color: {text_color};
            line-height: 1.6;
        }}
        h1 {{
            color: {primary_dark};
            border-bottom: 3px solid {primary_color};
            padding-bottom: 12px;
            font-size: 2em;
            margin-bottom: 20px;
        }}
        h2 {{
            color: {primary_dark};
            border-left: 5px solid {primary_color};
            padding-left: 12px;
            margin: 25px 0 15px 0;
        }}
        .stats-box {{
            background-color: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border-left: 5px solid {primary_color};
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
            background-color: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: {primary_dark};
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #f1f3f5;
        }}
        .plot {{ 
            margin: 20px 0; 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .report-footer {{
            text-align: center;
            font-size: 0.75em;
            color: #666;
            margin-top: 40px;
            border-top: 1px dashed #ccc;
            padding-top: 10px;
        }}
        .sig-p {{
            font-weight: bold;
            color: #d63384;
        }}
    </style>"""

    safe_title = _html.escape(str(title))
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{safe_title}</title>",
        css_style,
        "</head>",
        "<body>"
    ]

    # Title
    html_parts.append(f"<h1>{safe_title}</h1>")

    # Generate timestamp
    from datetime import datetime
    html_parts.append(f"<p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")

    # Model statistics
    if stats:
        html_parts.append("<h2>📊 Model Summary</h2>")
        html_parts.append("<div class='stats-box'>")
        html_parts.append("<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;'>")
        for key, val in stats.items():
            html_parts.append(f"<div><strong>{key}:</strong> {val}</div>")
        html_parts.append("</div>")
        html_parts.append("</div>")

    # Missing data info
    if missing_data_info:
        html_parts.append(create_missing_data_report_html(missing_data_info, var_meta or {}))

    # Add elements
    for elem in elements:
        elem_type = elem.get('type', '')
        data = elem.get('data')

        if elem_type == 'header':
            html_parts.append(f"<h2>{_html.escape(str(data))}</h2>")
        elif elem_type == 'text':
            html_parts.append(f"<div style='background:#f8f9fa; padding:15px; border-radius:5px;'>{str(data)}</div>")
        elif elem_type == 'table' and isinstance(data, pd.DataFrame):
            d_styled = data.copy()
            if 'p-value' in d_styled.columns:
                 p_vals = pd.to_numeric(d_styled['p-value'], errors='coerce')
                 d_styled['p-value'] = [
                     f'<span class="sig-p">{val:.4f}</span>' if not pd.isna(pv) and pv < 0.05 else f'{val:.4f}' if isinstance(val, float) else str(val)
                     for val, pv in zip(d_styled['p-value'], p_vals)
                 ]
            html_parts.append(d_styled.to_html(classes='table', index=False, escape=False))
        elif elem_type == 'plot' and hasattr(data, 'to_html'):
            html_parts.append(f"<div class='plot'>{data.to_html(include_plotlyjs='cdn', full_html=False)}</div>")

    html_parts.append("""<div class='report-footer'>
    © 2026 Powered by stat-shiny
    </div>""")
    
    html_parts.append("</body></html>")

    return "\n".join(html_parts)
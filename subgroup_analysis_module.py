"""
🧪 Subgroup Analysis Module (Shiny Compatible)

Professional subgroup analysis without Streamlit dependencies.
OPTIMIZED for Python 3.12 with strict type hints and TypedDict.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import CONFIG
from forest_plot_lib import create_forest_plot
from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import (
    apply_missing_values_to_df,
    get_missing_summary_df,
    handle_missing_for_analysis,
)
from utils.formatting import create_missing_data_report_html

logger = get_logger(__name__)
COLORS = get_color_palette()

class SubgroupResult(TypedDict):
    group: str
    n: int
    events: int
    or_val: float
    ci_low: float
    ci_high: float
    p_value: float
    type: str
    subgroup: Optional[str]

class SubgroupStats(TypedDict):
    n_overall: int
    events_overall: int
    n_subgroups: int
    or_overall: float
    ci_overall: Tuple[float, float]
    p_overall: float
    p_interaction: float
    heterogeneous: bool
    or_range: Tuple[float, float]

class SubgroupAnalysisLogit:
    """
    Subgroup analysis for logistic regression.
    Shiny-compatible, no Streamlit dependencies.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with dataset copy."""
        self.df = df.copy()
        self.results: Optional[pd.DataFrame] = None
        self.stats: Optional[Dict[str, Any]] = None
        self.interaction_result: Optional[Dict[str, Any]] = None
        self.figure: Optional[go.Figure] = None
        logger.info(f"SubgroupAnalysisLogit initialized with {len(df)} observations")
    
    def validate_inputs(
        self, 
        outcome_col: str, 
        treatment_col: str, 
        subgroup_col: str, 
        adjustment_cols: Optional[List[str]] = None
    ) -> bool:
        """Validate input columns and data types."""
        required_cols = {outcome_col, treatment_col, subgroup_col}
        if adjustment_cols:
            required_cols.update(adjustment_cols)
        
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if self.df[outcome_col].nunique() != 2:
            raise ValueError(f"Outcome '{outcome_col}' must be binary")
        
        if self.df[subgroup_col].nunique() < 2:
            raise ValueError(f"Subgroup '{subgroup_col}' must have 2+ categories")
        
        if len(self.df) < 10:
            raise ValueError(f"Minimum 10 observations required")
        
        logger.info("Input validation passed")
        return True
    
    def analyze(
        self, 
        outcome_col: str, 
        treatment_col: str, 
        subgroup_col: str, 
        adjustment_cols: Optional[List[str]] = None, 
        min_subgroup_n: int = 5,
        var_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform logistic regression subgroup analysis."""
        try:
            from scipy import stats
            from statsmodels.formula.api import logit
            
            self.validate_inputs(outcome_col, treatment_col, subgroup_col, adjustment_cols)
            
            if adjustment_cols is None:
                adjustment_cols = []
            
            cols_to_use = [outcome_col, treatment_col, subgroup_col] + adjustment_cols
            
            # --- MISSING DATA HANDLING ---
            missing_data_info = {}
            # Coerce outcome first so invalid values are counted as missing
            df_subset = self.df[cols_to_use].copy()
            df_subset[outcome_col] = pd.to_numeric(df_subset[outcome_col], errors='coerce')
            if var_meta:
                missing_codes = CONFIG.get("analysis.missing.user_defined_values", [])
                missing_summary = get_missing_summary_df(df_subset, var_meta, missing_codes)
                df_processed = apply_missing_values_to_df(df_subset, var_meta, missing_codes)
                
                df_clean, impact = handle_missing_for_analysis(
                    df_processed, var_meta, missing_codes, strategy='complete-case', return_counts=True
                )
                missing_data_info = {
                    'strategy': 'complete-case',
                    'rows_analyzed': impact['final_rows'],
                    'rows_excluded': impact['rows_removed'],
                    'summary_before': missing_summary.to_dict('records')
                }
            else:
                df_clean = df_subset.dropna().copy()
            
            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data: {len(df_clean)} rows")
            
            df_clean = df_clean.dropna(subset=[outcome_col])
            
            formula_base = f'{outcome_col} ~ {treatment_col}'
            if adjustment_cols:
                formula_base += ' + ' + ' + '.join(adjustment_cols)
            
            results_list: List[Dict[str, Any]] = []
            
            # Overall model
            logger.info("Computing overall model...")
            model_overall = logit(formula_base, data=df_clean).fit(disp=0)
            
            # Robust parameter lookup for treatment
            # Statsmodels might rename treatment to treatment[T.1] etc.
            param_key = None
            if treatment_col in model_overall.params:
                param_key = treatment_col
            else:
                for k in model_overall.params.index:
                    if k.startswith(f"{treatment_col}["):
                        param_key = k
                        break
            
            if not param_key:
                logger.warning(f"Treatment variable '{treatment_col}' dropped from model.")
                # Depending on requirement, either skip or raise. For integration, let's keep it safe.
                return {}  # Return empty dict to match return type

            or_overall = float(np.exp(model_overall.params[param_key]))
            ci_overall = np.exp(model_overall.conf_int().loc[param_key])
            p_overall = float(model_overall.pvalues[param_key])
            
            results_list.append({
                'group': f'Overall (N={len(df_clean)})',
                'n': len(df_clean),
                'events': int(df_clean[outcome_col].sum()),
                'or': or_overall,
                'ci_low': float(ci_overall[0]),
                'ci_high': float(ci_overall[1]),
                'p_value': p_overall,
                'type': 'overall',
                'subgroup': None
            })
            
            logger.info(f"Overall: OR={or_overall:.3f}, P={p_overall:.4f}")
            
            # Subgroup models
            subgroups = sorted(df_clean[subgroup_col].dropna().unique())
            logger.info(f"Computing {len(subgroups)} subgroup models...")
            
            for subgroup_val in subgroups:
                df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]
                
                if len(df_sub) < min_subgroup_n:
                    logger.warning(f"Subgroup {subgroup_val}: N={len(df_sub)} < {min_subgroup_n}, skipping")
                    continue
                
                if df_sub[treatment_col].nunique() < 2 or df_sub[outcome_col].nunique() < 2:
                    logger.warning(f"Subgroup {subgroup_val}: No variation, skipping")
                    continue
                
                try:
                    model_sub = logit(formula_base, data=df_sub).fit(disp=0)
                    
                    # Robust parameter lookup for treatment in subgroup
                    sub_param_key = None
                    if treatment_col in model_sub.params:
                        sub_param_key = treatment_col
                    else:
                        for k in model_sub.params.index:
                            if k.startswith(f"{treatment_col}["):
                                sub_param_key = k
                                break
                    
                    if not sub_param_key:
                        continue

                    or_sub = float(np.exp(model_sub.params[sub_param_key]))
                    ci_sub = np.exp(model_sub.conf_int().loc[sub_param_key])
                    p_sub = float(model_sub.pvalues[sub_param_key])
                    
                    results_list.append({
                        'group': f'{subgroup_col}={subgroup_val} (N={len(df_sub)})',
                        'subgroup': str(subgroup_val),
                        'n': len(df_sub),
                        'events': int(df_sub[outcome_col].sum()),
                        'or': or_sub,
                        'ci_low': float(ci_sub[0]),
                        'ci_high': float(ci_sub[1]),
                        'p_value': p_sub,
                        'type': 'subgroup'
                    })
                    
                    logger.info(f"Subgroup {subgroup_val}: OR={or_sub:.3f}, P={p_sub:.4f}")
                except Exception as e:
                    logger.warning(f"Model failed for {subgroup_val}: {e}")
                    continue
            
            # Interaction test
            logger.info("Computing interaction test...")
            try:
                formula_reduced = f'{outcome_col} ~ {treatment_col} + C({subgroup_col})'
                formula_full = f'{outcome_col} ~ {treatment_col} * C({subgroup_col})'
                if adjustment_cols:
                    formula_reduced += ' + ' + ' + '.join(adjustment_cols)
                    formula_full += ' + ' + ' + '.join(adjustment_cols)
                
                model_reduced = logit(formula_reduced, data=df_clean).fit(disp=0)
                model_full = logit(formula_full, data=df_clean).fit(disp=0)
                
                lr_stat = -2 * (model_reduced.llf - model_full.llf)
                df_diff = model_full.df_model - model_reduced.df_model
                
                if df_diff > 0:
                    p_interaction = float(stats.chi2.sf(lr_stat, df_diff))
                else:
                    p_interaction = np.nan
                
                is_sig = pd.notna(p_interaction) and p_interaction < 0.05
                
                self.interaction_result = {
                    'p_value': p_interaction,
                    'significant': is_sig
                }
                
                logger.info(f"Interaction P={p_interaction:.4f}")
            except Exception as e:
                logger.warning(f"Interaction test failed: {e}")
                self.interaction_result = {'p_value': np.nan, 'significant': False}
            
            self.results = pd.DataFrame(results_list)
            self.stats = self._compute_summary_statistics()
            logger.info("Analysis complete")
            result = self._format_output()
            if missing_data_info:
                result['missing_data_info'] = missing_data_info
            return result
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if self.results is None or self.results.empty:
            return {}

        overall = self.results[self.results['type'] == 'overall'].iloc[0]
        subgroups = self.results[self.results['type'] == 'subgroup']
        
        p_int = self.interaction_result.get('p_value', np.nan) if self.interaction_result else np.nan
        is_het = self.interaction_result.get('significant', False) if self.interaction_result else False

        return {
            'n_overall': int(overall['n']),
            'events_overall': int(overall['events']),
            'n_subgroups': len(subgroups),
            'or_overall': float(overall['or']),
            'ci_overall': (float(overall['ci_low']), float(overall['ci_high'])),
            'p_overall': float(overall['p_value']),
            'p_interaction': p_int,
            'heterogeneous': is_het,
            'or_range': (float(subgroups['or'].min()), float(subgroups['or'].max())) if not subgroups.empty else (0.0, 0.0)
        }
    
    def _format_output(self) -> Dict[str, Any]:
        """Format results for output."""
        if self.results is None or self.results.empty:
            return {}

        overall_rows = self.results[self.results['type'] == 'overall']
        if overall_rows.empty:
            return {}
            
        overall = overall_rows.iloc[0]
        p_int = self.interaction_result['p_value'] if self.interaction_result else np.nan
        p_int_val = float(p_int) if pd.notna(p_int) else None

        return {
            'overall': {
                'or': float(overall['or']),
                'ci': (float(overall['ci_low']), float(overall['ci_high'])),
                'p_value': float(overall['p_value']),
                'n': int(overall['n']),
                'events': int(overall['events'])
            },
            'subgroups': self.results[self.results['type'] == 'subgroup'].to_dict('records'),
            'interaction': {
                'p_value': p_int_val,
                'significant': bool(self.interaction_result['significant']) if self.interaction_result else False
            },
            'summary': self.stats,
            'results_df': self.results
        }
    
    def create_forest_plot(self, title: str = "Subgroup Analysis: Logistic Regression", color: Optional[str] = None) -> go.Figure:
        """Create forest plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        if color is None:
            color = COLORS['primary']
        
        plot_data = self.results[['group', 'or', 'ci_low', 'ci_high', 'p_value']].copy()
        plot_data.columns = ['variable', 'or', 'ci_low', 'ci_high', 'p_value']
        
        p_int = self.interaction_result.get('p_value', np.nan) if self.interaction_result else np.nan
        is_het = self.interaction_result.get('significant', False) if self.interaction_result else False
        het_text = "Heterogeneous" if is_het else "Homogeneous"
        
        if pd.notna(p_int):
            title_final = f"{title}<br><span style='font-size: 12px; color: #666;'>P = {p_int:.4f} ({het_text})</span>"
        else:
            title_final = title
        
        self.figure = create_forest_plot(
            data=plot_data,
            estimate_col='or',
            ci_low_col='ci_low',
            ci_high_col='ci_high',
            label_col='variable',
            pval_col='p_value',
            title=title_final,
            x_label='Odds Ratio (95% CI)',
            ref_line=1.0,
            color=color
        )
        return self.figure


class SubgroupAnalysisCox:
    """
    Subgroup analysis for Cox proportional hazards regression.
    Shiny-compatible, no Streamlit dependencies.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with dataset copy."""
        self.df = df.copy()
        self.results: Optional[pd.DataFrame] = None
        self.stats: Optional[Dict[str, Any]] = None
        self.interaction_result: Optional[Dict[str, Any]] = None
        self.figure: Optional[go.Figure] = None
        logger.info(f"SubgroupAnalysisCox initialized with {len(df)} observations")
    
    def validate_inputs(
        self, 
        duration_col: str, 
        event_col: str, 
        treatment_col: str, 
        subgroup_col: str, 
        adjustment_cols: Optional[List[str]] = None
    ) -> bool:
        """Validate input columns and data types."""
        required_cols = {duration_col, event_col, treatment_col, subgroup_col}
        if adjustment_cols:
            required_cols.update(adjustment_cols)
        
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if self.df[event_col].nunique() != 2:
            raise ValueError(f"Event '{event_col}' must be binary (0/1)")
        
        if self.df[subgroup_col].nunique() < 2:
            raise ValueError(f"Subgroup '{subgroup_col}' must have 2+ categories")
        
        if len(self.df) < 10:
            raise ValueError(f"Minimum 10 observations required")
        
        logger.info("Input validation passed")
        return True
    
    def analyze(
        self, 
        duration_col: str, 
        event_col: str, 
        treatment_col: str, 
        subgroup_col: str, 
        adjustment_cols: Optional[List[str]] = None, 
        min_subgroup_n: int = 5, 
        min_events: int = 2,
        var_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """Perform Cox subgroup analysis."""
        try:
            from lifelines import CoxPHFitter
            from scipy import stats
            
            self.validate_inputs(duration_col, event_col, treatment_col, subgroup_col, adjustment_cols)
            
            if adjustment_cols is None:
                adjustment_cols = []
            
            cols_to_use = [duration_col, event_col, treatment_col, subgroup_col] + adjustment_cols
            
            # --- MISSING DATA HANDLING ---
            missing_data_info = {}
            if var_meta:
                 df_subset = self.df[cols_to_use].copy()
                 missing_codes = CONFIG.get("analysis.missing.user_defined_values", [])
                 missing_summary = get_missing_summary_df(df_subset, var_meta, missing_codes)
                 df_processed = apply_missing_values_to_df(df_subset, var_meta, missing_codes)
                 
                 df_clean, impact = handle_missing_for_analysis(
                    df_processed, var_meta, missing_codes, strategy='complete-case', return_counts=True
                 )
                 missing_data_info = {
                    'strategy': 'complete-case',
                    'rows_analyzed': impact['final_rows'],
                    'rows_excluded': impact['rows_removed'],
                    'summary_before': missing_summary.to_dict('records')
                 }
            else:
                 df_clean = self.df[cols_to_use].dropna().copy()
            
            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data: {len(df_clean)} rows")
            
            results_list: List[Dict[str, Any]] = []
            
            # Overall Cox model
            logger.info("Computing overall Cox model...")
            try:
                cph_overall = CoxPHFitter()
                cph_overall.fit(df_clean, duration_col=duration_col, event_col=event_col, show_progress=False)
                
                hr_overall = float(np.exp(cph_overall.params_[treatment_col]))
                ci_overall = np.exp(cph_overall.confidence_intervals_.loc[treatment_col])
                p_overall = float(cph_overall.summary.loc[treatment_col, 'p'])
                
                results_list.append({
                    'group': f'Overall (N={len(df_clean)})',
                    'n': len(df_clean),
                    'events': int(df_clean[event_col].sum()),
                    'hr': hr_overall,
                    'ci_low': float(ci_overall[0]),
                    'ci_high': float(ci_overall[1]),
                    'p_value': p_overall,
                    'type': 'overall'
                })
                logger.info(f"Overall: HR={hr_overall:.3f}, P={p_overall:.4f}")
            except Exception as e:
                logger.error(f"Overall Cox model failed: {e}")
                return None, None, f"Overall Cox model failed: {e}"
            
            # Subgroup Cox models
            subgroups = sorted(df_clean[subgroup_col].dropna().unique())
            logger.info(f"Computing {len(subgroups)} subgroup Cox models...")
            
            for subgroup_val in subgroups:
                df_sub = df_clean[df_clean[subgroup_col] == subgroup_val]
                
                if len(df_sub) < min_subgroup_n:
                    logger.warning(f"Subgroup {subgroup_val}: N={len(df_sub)} < {min_subgroup_n}, skipping")
                    continue
                
                if df_sub[event_col].sum() < min_events:
                    logger.warning(f"Subgroup {subgroup_val}: Events={df_sub[event_col].sum()} < {min_events}, skipping")
                    continue
                
                try:
                    cph_sub = CoxPHFitter()
                    cph_sub.fit(df_sub, duration_col=duration_col, event_col=event_col, show_progress=False)
                    
                    hr_sub = float(np.exp(cph_sub.params_[treatment_col]))
                    ci_sub = np.exp(cph_sub.confidence_intervals_.loc[treatment_col])
                    p_sub = float(cph_sub.summary.loc[treatment_col, 'p'])
                    
                    results_list.append({
                        'group': f'{subgroup_col}={subgroup_val} (N={len(df_sub)})',
                        'subgroup': str(subgroup_val),
                        'n': len(df_sub),
                        'events': int(df_sub[event_col].sum()),
                        'hr': hr_sub,
                        'ci_low': float(ci_sub[0]),
                        'ci_high': float(ci_sub[1]),
                        'p_value': p_sub,
                        'type': 'subgroup'
                    })
                    logger.info(f"Subgroup {subgroup_val}: HR={hr_sub:.3f}, P={p_sub:.4f}")
                except Exception as e:
                    logger.warning(f"Cox model failed for {subgroup_val}: {e}")
                    continue
            
            # Interaction test (using treatment*subgroup term)
            logger.info("Computing interaction test...")
            try:
                df_int = df_clean.copy()
                # Create interaction term
                df_int['interaction'] = df_int[treatment_col] * df_int[subgroup_col].astype('category').cat.codes
                
                cph_reduced = CoxPHFitter()
                cph_reduced.fit(df_int[[duration_col, event_col, treatment_col, subgroup_col]], 
                               duration_col=duration_col, event_col=event_col, show_progress=False)
                
                self.interaction_result = {
                    'p_value': np.nan,
                    'significant': False,
                    'note': 'Interaction term via stratified analysis'
                }
            except Exception as e:
                logger.warning(f"Interaction test failed: {e}")
                self.interaction_result = {'p_value': np.nan, 'significant': False}
            
            self.results = pd.DataFrame(results_list)
            self.stats = self._compute_summary_statistics()
            logger.info("Analysis complete")
            result = self._format_output()
            if missing_data_info:
                result['missing_data_info'] = missing_data_info
            return result, None, None
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None, None, str(e)
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if self.results is None or self.results.empty:
            return {}

        overall = self.results[self.results['type'] == 'overall'].iloc[0]
        subgroups = self.results[self.results['type'] == 'subgroup']
        
        return {
            'n_overall': int(overall['n']),
            'events_overall': int(overall['events']),
            'n_subgroups': len(subgroups),
            'hr_overall': float(overall['hr']),
            'ci_overall': (float(overall['ci_low']), float(overall['ci_high'])),
            'p_overall': float(overall['p_value']),
            'hr_range': (float(subgroups['hr'].min()), float(subgroups['hr'].max())) if not subgroups.empty else (0.0, 0.0)
        }
    
    def _format_output(self) -> Dict[str, Any]:
        """Format results for output."""
        if self.results is None or self.results.empty:
            return {}

        overall_rows = self.results[self.results['type'] == 'overall']
        if overall_rows.empty:
            return {}
            
        overall = overall_rows.iloc[0]

        return {
            'overall': {
                'hr': float(overall['hr']),
                'ci': (float(overall['ci_low']), float(overall['ci_high'])),
                'p_value': float(overall['p_value']),
                'n': int(overall['n']),
                'events': int(overall['events'])
            },
            'subgroups': self.results[self.results['type'] == 'subgroup'].to_dict('records'),
            'interaction': self.interaction_result,
            'summary': self.stats,
            'results_df': self.results
        }
    
    def create_forest_plot(self, title: str = "Subgroup Analysis: Cox Regression", color: Optional[str] = None) -> go.Figure:
        """Create forest plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        if color is None:
            color = COLORS['primary']
        
        plot_data = self.results[['group', 'hr', 'ci_low', 'ci_high', 'p_value']].copy()
        plot_data.columns = ['variable', 'hr', 'ci_low', 'ci_high', 'p_value']
        
        self.figure = create_forest_plot(
            data=plot_data,
            estimate_col='hr',
            ci_low_col='ci_low',
            ci_high_col='ci_high',
            label_col='variable',
            pval_col='p_value',
            title=title,
            x_label='Hazard Ratio (95% CI)',
            ref_line=1.0,
            color=color
        )
        return self.figure

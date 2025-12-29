"""
🧪 Subgroup Analysis Module (Shiny Compatible)

Professional subgroup analysis without Streamlit dependencies.
"""

import pandas as pd
import numpy as np
from logger import get_logger
from forest_plot_lib import create_forest_plot
import warnings

logger = get_logger(__name__)


class SubgroupAnalysisLogit:
    """
    Subgroup analysis for logistic regression.
    Shiny-compatible, no Streamlit dependencies.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with dataset copy."""
        self.df = df.copy()
        self.results = None
        self.stats = None
        self.interaction_result = None
        self.figure = None
        logger.info(f"SubgroupAnalysisLogit initialized with {len(df)} observations")
    
    def validate_inputs(self, outcome_col, treatment_col, subgroup_col, adjustment_cols=None):
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
    
    def analyze(self, outcome_col, treatment_col, subgroup_col, adjustment_cols=None, min_subgroup_n=5):
        """Perform logistic regression subgroup analysis."""
        try:
            from statsmodels.formula.api import logit
            from scipy import stats
            
            self.validate_inputs(outcome_col, treatment_col, subgroup_col, adjustment_cols)
            
            if adjustment_cols is None:
                adjustment_cols = []
            
            cols_to_use = [outcome_col, treatment_col, subgroup_col] + adjustment_cols
            df_clean = self.df[cols_to_use].dropna().copy()
            
            if len(df_clean) < 10:
                raise ValueError(f"Insufficient data: {len(df_clean)} rows")
            
            formula_base = f'{outcome_col} ~ {treatment_col}'
            if adjustment_cols:
                formula_base += ' + ' + ' + '.join(adjustment_cols)
            
            results_list = []
            
            # Overall model
            logger.info("Computing overall model...")
            model_overall = logit(formula_base, data=df_clean).fit(disp=0)
            
            or_overall = np.exp(model_overall.params[treatment_col])
            ci_overall = np.exp(model_overall.conf_int().loc[treatment_col])
            p_overall = model_overall.pvalues[treatment_col]
            
            results_list.append({
                'group': f'Overall (N={len(df_clean)})',
                'n': len(df_clean),
                'events': int(df_clean[outcome_col].sum()),
                'or': or_overall,
                'ci_low': ci_overall[0],
                'ci_high': ci_overall[1],
                'p_value': p_overall,
                'type': 'overall'
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
                    
                    or_sub = np.exp(model_sub.params[treatment_col])
                    ci_sub = np.exp(model_sub.conf_int().loc[treatment_col])
                    p_sub = model_sub.pvalues[treatment_col]
                    
                    results_list.append({
                        'group': f'{subgroup_col}={subgroup_val} (N={len(df_sub)})',
                        'subgroup': subgroup_val,
                        'n': len(df_sub),
                        'events': int(df_sub[outcome_col].sum()),
                        'or': or_sub,
                        'ci_low': ci_sub[0],
                        'ci_high': ci_sub[1],
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
                    p_interaction = stats.chi2.sf(lr_stat, df_diff)
                else:
                    p_interaction = np.nan
                
                is_sig = isinstance(p_interaction, (int, float)) and pd.notna(p_interaction) and p_interaction < 0.05
                
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
            return self._format_output()
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _compute_summary_statistics(self):
        """Compute summary statistics."""
        if self.results is None or self.results.empty:
            return {}

        overall = self.results[self.results['type'] == 'overall'].iloc[0]
        subgroups = self.results[self.results['type'] == 'subgroup']
        
        return {
            'n_overall': int(overall['n']),
            'events_overall': int(overall['events']),
            'n_subgroups': len(subgroups),
            'or_overall': overall['or'],
            'ci_overall': (overall['ci_low'], overall['ci_high']),
            'p_overall': overall['p_value'],
            'p_interaction': self.interaction_result.get('p_value', np.nan),
            'heterogeneous': self.interaction_result.get('significant', False),
            'or_range': (subgroups['or'].min(), subgroups['or'].max()) if not subgroups.empty else (0, 0)
        }
    
    def _format_output(self):
        """Format results for output."""
        if self.results is None or self.results.empty:
            return {}

        overall_rows = self.results[self.results['type'] == 'overall']
        if overall_rows.empty:
            return {}
            
        overall = overall_rows.iloc[0]
        p_int = self.interaction_result['p_value']
        p_int_val = float(p_int) if isinstance(p_int, (int, float)) and pd.notna(p_int) else None

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
                'significant': bool(self.interaction_result['significant'])
            },
            'summary': self.stats,
            'results_df': self.results
        }
    
    def create_forest_plot(self, title="Subgroup Analysis: Logistic Regression", color="#2180BE"):
        """Create forest plot."""
        if self.results is None:
            raise ValueError("Run analyze() first")
        
        plot_data = self.results[['group', 'or', 'ci_low', 'ci_high', 'p_value']].copy()
        plot_data.columns = ['variable', 'or', 'ci_low', 'ci_high', 'p_value']
        
        p_int = self.interaction_result['p_value']
        is_het = self.interaction_result.get('significant', False)
        het_text = "Heterogeneous" if is_het else "Homogeneous"
        
        if isinstance(p_int, (int, float)) and pd.notna(p_int):
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

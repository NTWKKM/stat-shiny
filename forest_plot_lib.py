"""
📈 Forest Plot Visualization Module (Shiny Compatible)

For displaying effect sizes (OR, HR, RR) with confidence intervals.
Supports logistic regression, survival analysis, and epidemiological studies.
Shiny-compatible - no Streamlit dependencies.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from logger import get_logger
from tabs._common import get_color_palette
import warnings

logger = get_logger(__name__)
COLORS = get_color_palette()


class ForestPlot:
    """
    Interactive forest plot generator for statistical results.
    Shiny-compatible, no Streamlit dependencies.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        estimate_col: str,
        ci_low_col: str,
        ci_high_col: str,
        label_col: str,
        pval_col: str = None,
    ):
        """
        Initialize ForestPlot with validated data.
        """
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        
        required_cols = {estimate_col, ci_low_col, ci_high_col, label_col}
        if pval_col:
            required_cols.add(pval_col)

        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.data = data.copy()

        numeric_cols = [estimate_col, ci_low_col, ci_high_col]
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data = self.data.dropna(subset=numeric_cols)
        
        if self.data.empty:
            raise ValueError("No valid data after removing NaN values")
        
        self.data = self.data.iloc[::-1].reset_index(drop=True)
        
        self.estimate_col = estimate_col
        self.ci_low_col = ci_low_col
        self.ci_high_col = ci_high_col
        self.label_col = label_col
        self.pval_col = pval_col
        
        try:
            est_min = self.data[estimate_col].min()
            est_max = self.data[estimate_col].max()
            logger.info(f"ForestPlot initialized: {len(self.data)} variables, estimate range [{est_min:.3f}, {est_max:.3f}]")
        except Exception as e:
            logger.warning(f"Could not log estimate range: {e}")
    
    def _add_significance_stars(self, p):
        """
        Convert p-value to significance stars.
        """
        try:
            if pd.isna(p):
                return ""
            
            p_val = p
            if isinstance(p, str):
                clean_p = p.replace('<', '').replace('>', '').strip()
                p_val = float(clean_p)
            
            if p_val < 0.001:
                return "***"
            if p_val < 0.01:
                return "**"
            if p_val < 0.05:
                return "*"
        except (ValueError, TypeError):
            return ""
            
        return ""
    
    def _get_ci_width_colors(self, base_color: str) -> list:
        """
        Generate marker colors with opacity scaled by CI width.
        """
        ci_high = self.data[self.ci_high_col]
        ci_low = self.data[self.ci_low_col]
        
        ci_width = ci_high - ci_low
        
        is_finite = np.isfinite(ci_width)
        finite_widths = ci_width[is_finite]
        
        ci_normalized = pd.Series(np.nan, index=ci_width.index)
        
        if not finite_widths.empty:
            ci_min, ci_max = finite_widths.min(), finite_widths.max()
            
            if ci_max > ci_min:
                ci_normalized[is_finite] = (finite_widths - ci_min) / (ci_max - ci_min)
            else:
                ci_normalized[is_finite] = 0.5
        
        ci_normalized = ci_normalized.fillna(1.0)
        
        hex_color = base_color.lstrip('#')
        try:
            if len(hex_color) == 6:
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            else:
                rgb = (33, 128, 141)
        except ValueError:
            rgb = (33, 128, 141)
        
        marker_colors = [
            f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {1.0 - 0.5*float(norm_val):.2f})"
            for norm_val in ci_normalized
        ]
        
        return marker_colors, ci_normalized.values
    
    def get_summary_stats(self, ref_line: float = 1.0):
        """
        Compute summary statistics for the dataset.
        """
        n_sig = 0
        pct_sig = 0

        if self.pval_col and self.pval_col in self.data.columns:
            p_numeric = pd.to_numeric(
                self.data[self.pval_col].astype(str).str.replace('<', '').str.replace('>', ''), 
                errors='coerce'
            )
            n_sig = (p_numeric < 0.05).sum()
            pct_sig = 100 * n_sig / len(self.data) if len(self.data) > 0 else 0
        else:
            n_sig = pct_sig = None
        
        ci_low = self.data[self.ci_low_col]
        ci_high = self.data[self.ci_high_col]

        ci_sig = (
            ((ci_low > ref_line) | (ci_high < ref_line))
            if ref_line > 0
            else ((ci_low * ci_high) > 0)
        )
        n_ci_sig = ci_sig.sum()
        
        return {
            'n_variables': len(self.data),
            'median_est': self.data[self.estimate_col].median(),
            'min_est': self.data[self.estimate_col].min(),
            'max_est': self.data[self.estimate_col].max(),
            'n_significant': n_sig,
            'pct_significant': pct_sig,
            'n_ci_significant': n_ci_sig,
        }
    
    def create(
        self,
        title: str = "Forest Plot",
        x_label: str = "Effect Size (95% CI)",
        ref_line: float = 1.0,
        show_ref_line: bool = True,
        show_sig_stars: bool = True,
        show_ci_width_colors: bool = True,
        show_sig_divider: bool = True,
        height: int = None,
        color: str = None,
    ) -> go.Figure:
        """
        Build interactive forest plot.
        """
        if color is None:
            color = COLORS['primary']
        
        def fmt_est_ci(x) -> str:
            try:
                est = x[self.estimate_col]
                low = x[self.ci_low_col]
                high = x[self.ci_high_col]
                
                est_str = f"{est:.2f}" if np.isfinite(est) else "Inf"
                low_str = f"{low:.2f}" if np.isfinite(low) else "Inf"
                high_str = f"{high:.2f}" if np.isfinite(high) else "Inf"
            except (KeyError, TypeError, ValueError):
                return "Error"
            else:
                return f"{est_str} ({low_str}-{high_str})"
        
        self.data['__display_est'] = self.data.apply(fmt_est_ci, axis=1)

        if self.pval_col:
            def fmt_p(p):
                try:
                    p_str = str(p).replace('<', '').replace('>', '').strip()
                    p_float = float(p_str)
                    if p_float < 0.001: 
                        return "<0.001"
                    return f"{p_float:.3f}"
                except (ValueError, TypeError): 
                    return str(p)
            
            self.data['__display_p'] = self.data[self.pval_col].apply(fmt_p)
            
            def get_p_color(p):
                try:
                    if isinstance(p, str) and '<' in p: 
                        val = float(p.replace('<','').strip())
                        return "red" if val < 0.05 else "black"
                    
                    val = float(p)
                    return "red" if val < 0.05 else "black"
                except (ValueError, TypeError):
                    return "black"

            p_text_colors = self.data[self.pval_col].apply(get_p_color).tolist()
        else:
            self.data['__display_p'] = ""
            p_text_colors = ["black"] * len(self.data)
        
        if show_sig_stars and self.pval_col:
            self.data['__sig_stars'] = self.data[self.pval_col].apply(self._add_significance_stars)
            self.data['__display_label'] = (
                self.data[self.label_col].astype(str) + " " + self.data['__sig_stars']
            ).str.rstrip()
        else:
            self.data['__display_label'] = self.data[self.label_col]
        
        marker_colors, _ = self._get_ci_width_colors(color) if show_ci_width_colors else ([color] * len(self.data), None)

        has_pval = self.pval_col is not None and not self.data[self.pval_col].isna().all()
        column_widths = [0.25, 0.20, 0.10, 0.45] if has_pval else [0.25, 0.20, 0.55]
        num_cols = 4 if has_pval else 3
        plot_col = 4 if has_pval else 3
        
        fig = make_subplots(
            rows=1, cols=num_cols,
            shared_yaxes=True,
            horizontal_spacing=0.02,
            column_widths=column_widths,
            specs=[[{"type": "scatter"} for _ in range(num_cols)]]
        )

        y_pos = list(range(len(self.data)))
        
        fig.add_trace(go.Scatter(x=[0]*len(y_pos), y=y_pos, text=self.data['__display_label'], mode="text", textposition="middle right", textfont=dict(size=13, color="black"), hoverinfo="none", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0]*len(y_pos), y=y_pos, text=self.data['__display_est'], mode="text", textposition="middle center", textfont=dict(size=13, color="black"), hoverinfo="none", showlegend=False), row=1, col=2)

        if has_pval:
            fig.add_trace(go.Scatter(x=[0]*len(y_pos), y=y_pos, text=self.data['__display_p'], mode="text", textposition="middle center", textfont=dict(size=13, color=p_text_colors), hoverinfo="none", showlegend=False), row=1, col=3)

        finite_data = self.data[np.isfinite(self.data[self.estimate_col])]
        if not finite_data.empty:
            est_min = finite_data[self.estimate_col].min()
            est_max = finite_data[self.estimate_col].max()
            use_log_scale = (est_min > 0) and ((est_max / est_min) > 5)
        else:
            use_log_scale = False

        if show_ref_line:
            fig.add_vline(x=ref_line, line_dash='dash', line_color='rgba(192, 21, 47, 0.6)', line_width=2, annotation_text=f'No Effect ({ref_line})', annotation_position='top', row=1, col=plot_col)
        
        if show_sig_divider:
            ci_low = self.data[self.ci_low_col]
            ci_high = self.data[self.ci_high_col]
            
            ci_sig = ((ci_low > ref_line) | (ci_high < ref_line)) if ref_line > 0 else ((ci_low * ci_high) > 0)
            
            if ci_sig.any() and (~ci_sig).any():
                divider_y = ci_sig.idxmin() - 0.5
                fig.add_hline(y=divider_y, line_dash='dot', line_color='rgba(100, 100, 100, 0.3)', line_width=1.5, row=1, col=plot_col)

        hover_parts = ["<b>%{text}</b><br>", f"<b>{self.estimate_col}:</b> %{{x:.3f}}<br>", "<b>95% CI:</b> %{customdata[0]:.3f} - %{customdata[1]:.3f}<br>"]
        if has_pval: hover_parts.append("<b>P-value:</b> %{customdata[2]}<br>")
        if show_ci_width_colors: hover_parts.append("<b>CI Width:</b> %{customdata[3]:.3f}<br>")
        hover_parts.append("<extra></extra>")
        hovertemplate = "".join(hover_parts)
        
        customdata = np.stack((
            self.data[self.ci_low_col], 
            self.data[self.ci_high_col], 
            self.data['__display_p'] if has_pval else [None]*len(self.data),
            self.data[self.ci_high_col] - self.data[self.ci_low_col]
        ), axis=-1)

        fig.add_trace(go.Scatter(
            x=self.data[self.estimate_col], y=y_pos,
            error_x=dict(type='data', symmetric=False, array=self.data[self.ci_high_col] - self.data[self.estimate_col], arrayminus=self.data[self.estimate_col] - self.data[self.ci_low_col], color='rgba(100,100,100,0.5)', thickness=2, width=4),
            mode='markers', marker=dict(size=10, color=marker_colors, symbol='square', line=dict(width=1.5, color='white')),
            text=self.data['__display_label'], customdata=customdata, hovertemplate=hovertemplate, showlegend=False
        ), row=1, col=plot_col)

        if height is None: 
            height = max(400, len(self.data) * 35 + 120)
        
        summary = self.get_summary_stats(ref_line)
        summary_text = f"N={summary['n_variables']}, Median={summary['median_est']:.2f}"
        if summary['pct_significant'] is not None: 
            summary_text += f", Sig={summary['pct_significant']:.0f}%"
        
        title_with_summary = f"<b>{title}</b><br><span style='font-size: 13px; color: rgba(100,100,100,0.9);'>{summary_text}</span>"

        fig.update_layout(
            title=dict(text=title_with_summary, x=0.01, xanchor='left', font=dict(size=18)),
            height=height, showlegend=False, template='plotly_white',
            margin=dict(l=10, r=20, t=120, b=40),
            plot_bgcolor='white', autosize=True
        )

        for c in range(1, plot_col):
            fig.update_xaxes(visible=False, showgrid=False, zeroline=False, row=1, col=c)
            fig.update_yaxes(visible=False, showgrid=False, zeroline=False, row=1, col=c)

        fig.update_yaxes(visible=False, range=[-0.5, len(self.data)-0.5], row=1, col=plot_col)
        fig.update_xaxes(title_text=x_label, type='log' if use_log_scale else 'linear', row=1, col=plot_col, gridcolor='rgba(200, 200, 200, 0.2)')

        headers = ["Variable", "Estimate (95% CI)"] + (["P-value", f"{x_label} Plot"] if has_pval else [f"{x_label} Plot"])
        
        for i, h in enumerate(headers, 1):
            xref_val = f"x{i} domain" if i > 1 else "x domain"
            fig.add_annotation(x=0.5 if i != 1 else 1.0, y=1.0, xref=xref_val, yref="paper", text=f"<b>{h}</b>", showarrow=False, yanchor="bottom", font=dict(size=14, color="black"))

        logger.info(f"Forest plot generated: {title}, {len(self.data)} variables")
        return fig


def create_forest_plot(
    data: pd.DataFrame, estimate_col: str, ci_low_col: str, ci_high_col: str, label_col: str,
    pval_col: str = None, title: str = "Forest Plot", x_label: str = "Effect Size (95% CI)",
    ref_line: float = 1.0, height: int = None, **kwargs
) -> go.Figure:
    """
    Create forest plot from DataFrame.
    
    Returns:
        plotly.graph_objects.Figure
    """
    try:
        fp = ForestPlot(data, estimate_col, ci_low_col, ci_high_col, label_col, pval_col)
        return fp.create(title=title, x_label=x_label, ref_line=ref_line, height=height, **kwargs)
    except ValueError as e:
        logger.error(f"Forest plot creation failed: {e}")
        return go.Figure()

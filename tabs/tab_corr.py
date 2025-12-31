"""
📈 Correlation & ICC Analysis Module (Shiny)

Provides UI and server logic for:
- Pearson/Spearman correlation analysis with scatter plots
- Intraclass correlation (ICC) for reliability/agreement
- Interactive reporting and HTML export
"""

from shiny import ui, reactive, render, Session
import pandas as pd
import numpy as np
import correlation  # Import from root
import diag_test  # Import for ICC calculation
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)


def _get_dataset_for_correlation(df: pd.DataFrame, df_matched: reactive.Value, is_matched: reactive.Value) -> tuple:
    """
    Choose between original and matched datasets for correlation analysis.
    
    Args:
        df: Original dataset
        df_matched: Reactive value containing matched data
        is_matched: Reactive value indicating if matching has been done
        
    Returns:
        Tuple of (selected_dataframe, label_string)
    """
    if is_matched.get() and df_matched.get() is not None:
        return df_matched.get().copy(), f"✅ Matched Data ({len(df_matched.get())} rows)"
    else:
        return df, f"📊 Original Data ({len(df)} rows)"


def _auto_detect_icc_vars(cols: list) -> list:
    """
    Auto-detect ICC/Rater variables based on column name patterns.
    Looks for patterns like 'ICC_', 'Rater', 'Rater1', 'Rater2', etc.
    
    Args:
        cols: List of column names
        
    Returns:
        List of auto-detected ICC/Rater variable names
    """
    icc_patterns = ['icc', 'rater', 'method', 'observer', 'judge']
    detected = []
    
    for col in cols:
        col_lower = col.lower()
        for pattern in icc_patterns:
            if pattern in col_lower:
                detected.append(col)
                break
    
    return detected


def corr_ui(namespace: str) -> ui.TagChild:
    """
    Create the UI for correlation analysis tab.
    
    Args:
        namespace: Shiny module namespace ID
        
    Returns:
        UI elements for the correlation tab
    """
    return ui.navset_tab(
        # TAB 1: Pearson/Spearman Correlation
        ui.nav_panel(
            "📈 Pearson/Spearman",
            ui.card(
                ui.card_header("📈 Continuous Correlation Analysis"),
                
                ui.layout_columns(
                    ui.input_select(
                        f"{namespace}_coeff_type",
                        "Correlation Coefficient:",
                        choices={"Pearson": "Pearson", "Spearman": "Spearman"},
                        selected="Pearson"
                    ),
                    ui.input_select(
                        f"{namespace}_cv1",
                        "Variable 1 (X-axis):",
                        choices=["Select..."]
                    ),
                    ui.input_select(
                        f"{namespace}_cv2",
                        "Variable 2 (Y-axis):",
                        choices=["Select..."]
                    ),
                    col_widths=[3, 4, 4]
                ),
                
                ui.layout_columns(
                    ui.input_action_button(
                        f"{namespace}_btn_run_corr",
                        "📈 Analyze Correlation",
                        class_="btn-primary",
                        width="100%"
                    ),
                    ui.input_action_button(
                        f"{namespace}_btn_dl_corr",
                        "📥 Download Report",
                        class_="btn-secondary",
                        width="100%"
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.output_ui(f"{namespace}_out_corr_result"),
                
                full_screen=True
            )
        ),
        
        # TAB 2: ICC (Reliability)
        ui.nav_panel(
            "📍 Reliability (ICC)",
            ui.card(
                ui.card_header("📍 Intraclass Correlation Coefficient"),
                
                ui.output_ui(f"{namespace}_out_icc_note"),
                
                ui.input_selectize(
                    f"{namespace}_icc_vars",
                    "Select Variables (Raters/Methods) - Select 2+:",
                    choices=["Select..."],
                    multiple=True,
                    selected=[]
                ),
                
                ui.layout_columns(
                    ui.input_action_button(
                        f"{namespace}_btn_run_icc",
                        "📍 Calculate ICC",
                        class_="btn-primary",
                        width="100%"
                    ),
                    ui.input_action_button(
                        f"{namespace}_btn_dl_icc",
                        "📥 Download Report",
                        class_="btn-secondary",
                        width="100%"
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.output_ui(f"{namespace}_out_icc_result"),
                
                full_screen=True
            )
        ),
        
        # TAB 3: Reference & Interpretation
        ui.nav_panel(
            "ℹ️ Reference",
            ui.card(
                ui.card_header("📚 Reference & Interpretation Guide"),
                
                ui.layout_columns(
                    ui.card(
                        ui.card_header("📈 Correlation (Relationship)"),
                        ui.markdown("""
**Concept:** Measures the strength and direction of the relationship between 
**two continuous variables**.

**1. Pearson (r):**
* **Best for:** Linear relationships (straight line), normally distributed data.
* **Sensitive to:** Outliers.

**2. Spearman (rho):**
* **Best for:** Monotonic relationships, non-normal data, or ranks.
* **Robust to:** Outliers.

**Interpretation of Coefficient (r or rho):**
* **+1.0:** Perfect Positive (As X goes up, Y goes up).
* **-1.0:** Perfect Negative (As X goes up, Y goes down).
* **0.0:** No relationship.

**Strength Guidelines:**
* **0.7 - 1.0:** Strong 📈
* **0.4 - 0.7:** Moderate 📉
* **0.2 - 0.4:** Weak 📊
* **< 0.2:** Negligible
                        """)
                    ),
                    ui.card(
                        ui.card_header("📍 ICC (Reliability)"),
                        ui.markdown("""
**Concept:** Measures the reliability or agreement between **two or more 
raters/methods** measuring the same thing.

**Common Types:**
* **ICC(2,1) Absolute Agreement:** Use when exact scores must match.
* **ICC(3,1) Consistency:** Use when ranking consistency matters.

**Interpretation of ICC Value:**
* **> 0.90:** Excellent Reliability ✅
* **0.75 - 0.90:** Good Reliability
* **0.50 - 0.75:** Moderate Reliability ⚠️
* **< 0.50:** Poor Reliability ❌
                        """)
                    ),
                    col_widths=[6, 6]
                ),
                
                ui.card(
                    ui.card_header("📝 Common Questions"),
                    ui.markdown("""
**Q: Why use ICC instead of Pearson for reliability?**
* **A:** Pearson only measures linearity. If Rater A always gives exactly 10 points 
higher than Rater B, Pearson = 1.0 but they don't agree! ICC accounts for this.

**Q: What if p-value is significant but r is low (0.1)?**
* **A:** P-value means it's likely not zero. With large samples, tiny correlations 
can be "significant". **Focus on r-value magnitude** for clinical relevance.

**Q: How many variables do I need for ICC?**
* **A:** At least 2 (to compare two raters/methods). More raters = more reliable ICC.
                    """)
                ),
                
                full_screen=True
            )
        )
    )


def corr_server(namespace: str, df: reactive.Value, var_meta: reactive.Value, 
                df_matched: reactive.Value, is_matched: reactive.Value):
    """
    Server logic for correlation analysis module.
    
    Args:
        namespace: Module namespace ID
        df: Reactive value with original dataframe
        var_meta: Reactive value with variable metadata
        df_matched: Reactive value with matched dataframe
        is_matched: Reactive value indicating if matching completed
    """
    
    COLORS = get_color_palette()
    
    # ==================== REACTIVE STATES ====================
    
    corr_result = reactive.Value(None)  # Stores result from correlation analysis
    icc_result = reactive.Value(None)   # Stores result from ICC analysis
    numeric_cols_list = reactive.Value([])  # List of numeric columns
    
    # ==================== UPDATE NUMERIC COLUMNS ====================
    
    @reactive.Effect
    def _update_numeric_cols():
        """Update list of numeric columns when data changes."""
        data = df.get()
        if data is not None:
            # Simple numeric check; in production, use safer dtype checks
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols_list.set(cols)
            
            # Update UI selects
            if cols:
                ui.update_select(f"{namespace}_cv1", choices=cols, selected=cols[0])
                ui.update_select(f"{namespace}_cv2", 
                               choices=cols, 
                               selected=cols[1] if len(cols) > 1 else cols[0])
                
                # Auto-detect ICC variables
                icc_vars = _auto_detect_icc_vars(cols)
                
                # Update selectize with all numeric cols
                ui.update_selectize(
                    f"{namespace}_icc_vars",
                    choices=cols,
                    selected=icc_vars  # Pre-select auto-detected ones
                )
                
                logger.info(f"Auto-detected ICC/Rater variables: {icc_vars}")
    
    # ==================== ICC SELECTION INFO ====================
    
    @render.ui
    def ui_icc_note():
        """Display info about auto-detected ICC variables."""
        data = df.get()
        if data is None:
            return None
        
        cols = data.select_dtypes(include=[np.number]).columns.tolist()
        icc_vars = _auto_detect_icc_vars(cols)
        
        if icc_vars:
            return ui.div(
                ui.p(
                    f"👋 Auto-detected variables: {', '.join(icc_vars)}",
                    style=f"background-color: {COLORS['primary_light']}; padding: 10px; border-radius: 5px; border: 1px solid {COLORS['primary']}; margin-bottom: 15px; font-size: 0.9em;"
                )
            )
        return None
    
    # ==================== CORRELATION ANALYSIS ====================
    
    @reactive.Effect
    @reactive.event(lambda: getattr(ui.input_action_button, f"{namespace}_btn_run_corr", None)) # Placeholder fix
    def _run_correlation():
        """Run correlation analysis when button clicked."""
        # TODO: Implement correlation logic connecting to correlation.py
        pass
    
    @reactive.Effect
    def handle_corr_button():
         # TODO: Implement button handling logic
         pass
    
    # Using input directly in render function (better pattern)
    
    @render.ui
    def out_corr_result():
        """Display correlation analysis results."""
        # Check if button was clicked - simplified logic
        # In a real app, you might check input[f"{namespace}_btn_run_corr"]
        
        result = corr_result.get()
        if result is None:
            return ui.markdown("*Results will appear here after clicking '📈 Analyze Correlation'*")
        
        return ui.card(
            ui.card_header("Results"),
            ui.markdown(f"**Method:** {result['method']}"),
            ui.markdown(f"**Variables:** {result['var1']} vs {result['var2']}"),
            
            ui.output_data_frame("out_corr_table"),
            
            ui.card_header("Scatter Plot"),
            ui.output_plot("out_corr_plot"),
        )
    
    @render.data_frame
    def out_corr_table():
        """Render correlation results table."""
        result = corr_result.get()
        if result is None:
            return None
        
        df_result = pd.DataFrame([result['stats']])
        return render.DataGrid(df_result)
    
    @render.plot
    def out_corr_plot():
        """Render scatter plot."""
        result = corr_result.get()
        if result is None or result['figure'] is None:
            return None
        return result['figure']
    
    # ==================== ICC ANALYSIS ====================
    
    @render.ui
    def out_icc_result():
        """Display ICC analysis results."""
        result = icc_result.get()
        if result is None:
            return ui.markdown("*Results will appear here after clicking '📍 Calculate ICC'*")
        
        return ui.card(
            ui.card_header("ICC Results"),
            
            ui.card_header("Single Measures ICC"),
            ui.output_data_frame("out_icc_table"),
            
            ui.card_header("ANOVA Table (Reference)"),
            ui.output_data_frame("out_icc_anova_table"),
        )
    
    @render.data_frame
    def out_icc_table():
        """Render ICC results table."""
        result = icc_result.get()
        if result is None:
            return None
        return render.DataGrid(result['results_df'])
    
    @render.data_frame
    def out_icc_anova_table():
        """Render ANOVA table."""
        result = icc_result.get()
        if result is None:
            return None
        return render.DataGrid(result['anova_df'])


# ==================== MODULE EXPORT ====================

def correlation_ui(namespace: str) -> ui.TagChild:
    """Wrapper for compatibility."""
    return corr_ui(namespace)


def correlation_server(namespace: str, df: reactive.Value, var_meta: reactive.Value,
                      df_matched: reactive.Value, is_matched: reactive.Value):
    """Wrapper for compatibility."""
    return corr_server(namespace, df, var_meta, df_matched, is_matched)

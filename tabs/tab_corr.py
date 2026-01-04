"""
📈 Correlation & ICC Analysis Module (Shiny) - WORKING VERSION

Provides UI and server logic for:
- Pearson/Spearman correlation analysis with scatter plots
- Intraclass correlation (ICC) for reliability/agreement
- Interactive reporting and HTML export

Fixed: Uses @module.ui and @module.server decorators properly
"""

from shiny import ui, reactive, render, req, module
import pandas as pd
import numpy as np
import correlation  # Import from root
import diag_test  # Import for ICC calculation
from logger import get_logger
from tabs._common import get_color_palette

logger = get_logger(__name__)

def _auto_detect_icc_vars(cols: list) -> list:
    """
    Auto-detect ICC/Rater variables based on column name patterns.
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

# ✅ FIX: Use @module.ui decorator like tab_diag.py
@module.ui
def corr_ui():
    """
    Create the UI for correlation analysis tab.
    """
    return ui.navset_tab(
        # TAB 1: Pearson/Spearman Correlation
        ui.nav_panel(
            "📈 Pearson/Spearman",
            ui.card(
                ui.card_header("📈 Continuous Correlation Analysis"),

                # --- ส่วนที่เพิ่มใหม่: ตัวเลือก Dataset สำหรับสลับไปใช้ข้อมูลที่ผ่านการ Match ---
                ui.output_ui("ui_matched_status_banner_corr"),
                ui.output_ui("ui_dataset_selector_corr"),
                ui.hr(),

                ui.layout_columns(
                    ui.input_select(
                        "coeff_type",
                        "Correlation Coefficient:",
                        choices={"pearson": "Pearson", "spearman": "Spearman"},
                        selected="pearson"
                    ),
                    ui.input_select(
                        "cv1",
                        "Variable 1 (X-axis):",
                        choices=["Select..."]
                    ),
                    ui.input_select(
                        "cv2",
                        "Variable 2 (Y-axis):",
                        choices=["Select..."]
                    ),
                    col_widths=[3, 4, 4]
                ),

                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run_corr",
                        "📈 Analyze Correlation",
                        class_="btn-primary",
                        width="100%"
                    ),
                    ui.input_action_button(
                        "btn_dl_corr",
                        "📥 Download Report",
                        class_="btn-secondary",
                        width="100%"
                    ),
                    col_widths=[6, 6]
                ),

                ui.output_ui("out_corr_result"),

                full_screen=True
            )
        ),

        # TAB 2: ICC (Reliability)
        ui.nav_panel(
            "🔍 Reliability (ICC)",
            ui.card(
                ui.card_header("🔍 Intraclass Correlation Coefficient"),

                ui.output_ui("out_icc_note"),

                ui.input_selectize(
                    "icc_vars",
                    "Select Variables (Raters/Methods) - Select 2+:",
                    choices=["Select..."],
                    multiple=True,
                    selected=[]
                ),

                ui.layout_columns(
                    ui.input_action_button(
                        "btn_run_icc",
                        "🔍 Calculate ICC",
                        class_="btn-primary",
                        width="100%"
                    ),
                    ui.input_action_button(
                        "btn_dl_icc",
                        "📥 Download Report",
                        class_="btn-secondary",
                        width="100%"
                    ),
                    col_widths=[6, 6]
                ),

                ui.output_ui("out_icc_result"),

                full_screen=True
            )
        ),

        # TAB 3: Reference & Interpretation
        ui.nav_panel(
            "📖 Reference",
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
* **+1.0:** Perfect Positive (As X goes up, Y goes down).
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
                        ui.card_header("🔍 ICC (Reliability)"),
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
                    ui.card_header("💡 Common Questions"),
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


# ✅ FIX: Use @module.server decorator properly
@module.server
def corr_server(input, output, session, df: reactive.Value, var_meta: reactive.Value, 
                df_matched: reactive.Value, is_matched: reactive.Value):
    """
    Server logic for correlation analysis module.
    """
    COLORS = get_color_palette()
    
    # ==================== REACTIVE STATES ====================
    corr_result = reactive.Value(None)  # Stores result from correlation analysis
    icc_result = reactive.Value(None)   # Stores result from ICC analysis
    numeric_cols_list = reactive.Value([])  # List of numeric columns

    # ==================== DATASET SELECTOR LOGIC (เพิ่มใหม่) ====================
    
    @render.ui
    def ui_dataset_selector_corr():
        """Render radio buttons to select dataset if matched data is available."""
        if is_matched.get():
            return ui.input_radio_buttons(
                "radio_dataset_source",
                "📄 Select Dataset for Analysis:",
                choices={
                    "original": "📊 Original Data",
                    "matched": "✅ Matched Data (from PSM)"
                },
                selected="original",
                inline=True
            )
        return None

    @render.ui
    def ui_matched_status_banner_corr():
        """Show a success banner when matched data is ready."""
        if is_matched.get():
            return ui.div(
                ui.p(
                    ui.strong("✅ Matched Dataset Available"),
                    " - You can switch to matched data for correlation analysis",
                    style=f"color: {COLORS['success']}; margin-bottom: 0;"
                ),
                style=f"padding: 10px; border-radius: 6px; margin-bottom: 15px; background-color: {COLORS['success']}15; border: 1px solid {COLORS['success']};"
            )
        return None

    @reactive.Calc
    def get_current_data():
        """Helper to get the selected dataset based on user choice."""
        try:
            # ใช้ input.radio_dataset_source() ถ้ามี (กรณีมีการรัน PSM แล้ว)
            use_matched = input.radio_dataset_source() == "matched"
        except:
            use_matched = False

        if is_matched.get() and use_matched and df_matched.get() is not None:
            return df_matched.get().copy(), "✅ Matched Data"
        return df.get(), "📊 Original Data"

    # ==================== UPDATE NUMERIC COLUMNS ====================

    @reactive.Effect
    def _update_numeric_cols():
        """Update list of numeric columns when data OR dataset source changes."""
        # ดึงข้อมูลจาก Calc แทน df.get() โดยตรง เพื่อรองรับการสลับ Dataset
        data, _ = get_current_data()
        if data is not None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols_list.set(cols)

            if cols:
                ui.update_select("cv1", choices=cols, selected=cols[0])
                ui.update_select("cv2", 
                               choices=cols, 
                               selected=cols[1] if len(cols) > 1 else cols[0])

                icc_vars = _auto_detect_icc_vars(cols)

                ui.update_selectize(
                    "icc_vars",
                    choices=cols,
                    selected=icc_vars 
                )
                logger.info("Updated numeric columns for source: %s", cols)

    # ==================== ICC SELECTION INFO ====================

    @render.ui
    def out_icc_note():
        """Display info about auto-detected ICC variables."""
        data, _ = get_current_data()
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
    @reactive.event(input.btn_run_corr)
    def _run_correlation():
        """Run correlation analysis when button clicked."""
        data_source, label = get_current_data()

        if data_source is None:
            ui.notification_show("No data available", type="error")
            return

        col1 = input.cv1()
        col2 = input.cv2()
        method = input.coeff_type()

        if not col1 or not col2:
            ui.notification_show("Please select two variables", type="warning")
            return

        if col1 == col2:
            ui.notification_show("Please select different variables", type="warning")
            return

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Calculating correlation...", detail="This may take a moment")
            res_stats, err, fig = correlation.calculate_correlation(
                data_source, col1, col2, method=method
            )

        if err:
            ui.notification_show(f"Error: {err}", type="error")
            corr_result.set(None)
        else:
            corr_result.set({
                "stats": res_stats,
                "figure": fig,
                "method": method,
                "var1": col1,
                "var2": col2,
                "data_label": label
            })
            ui.notification_show(f"Correlation complete using {label}", type="default")

    @render.ui
    def out_corr_result():
        """Display correlation analysis results."""
        result = corr_result.get()
        if result is None:
            return ui.markdown("*Results will appear here after clicking '📈 Analyze Correlation'*")

        return ui.card(
            ui.card_header("Results"),
            ui.markdown(f"**Data Source:** {result['data_label']}"),
            ui.markdown(f"**Method:** {result['method'].title()}"),

            ui.output_data_frame("out_corr_table"),

            ui.card_header("Scatter Plot"),
            ui.output_ui("out_corr_plot_html"), 
        )

    @render.data_frame
    def out_corr_table():
        """Render correlation results table."""
        result = corr_result.get()
        if result is None:
            return None

        df_result = pd.DataFrame([result['stats']])
        return render.DataGrid(df_result)

    @render.ui
    def out_corr_plot_html():
        """Render scatter plot as HTML."""
        result = corr_result.get()
        if result is None or result['figure'] is None:
            return None

        fig = result['figure']
        html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
        return ui.HTML(html_str)

    # ==================== ICC ANALYSIS ====================

    @reactive.Effect
    @reactive.event(input.btn_run_icc)
    def _run_icc():
        """Run ICC analysis."""
        data_source, label = get_current_data()

        if data_source is None:
            ui.notification_show("No data available", type="error")
            return

        cols = input.icc_vars()

        if not cols or len(cols) < 2:
            ui.notification_show("Please select at least 2 variables for ICC", type="warning")
            return

        with ui.Progress(min=0, max=1) as p:
            p.set(message="Calculating ICC...", detail="Computing variance components")
            res_df, err, anova_df = diag_test.calculate_icc(data_source, list(cols))

        if err:
            ui.notification_show(f"Error: {err}", type="error")
            icc_result.set(None)
        else:
            icc_result.set({
                "results_df": res_df,
                "anova_df": anova_df,
                "data_label": label
            })
            ui.notification_show(f"ICC analysis complete using {label}", type="default")

    @render.ui
    def out_icc_result():
        """Display ICC analysis results."""
        result = icc_result.get()
        if result is None:
            return ui.markdown("*Results will appear here after clicking '🔍 Calculate ICC'*")

        return ui.card(
            ui.card_header("ICC Results"),
            ui.markdown(f"**Data Source:** {result['data_label']}"),

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

def correlation_ui(id: str) -> ui.TagChild:
    """Wrapper for module UI - called from app.py"""
    return corr_ui(id)


def correlation_server(id: str, df: reactive.Value, var_meta: reactive.Value,
                       df_matched: reactive.Value, is_matched: reactive.Value):
    """Wrapper for module server - called from app.py"""
    return corr_server(id, df, var_meta, df_matched, is_matched)

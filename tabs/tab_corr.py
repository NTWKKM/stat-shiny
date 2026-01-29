"""
📈 Correlation & ICC Analysis Module (Enhanced) - FIXED INTERPRETATION

Enhanced Features:
- Comprehensive statistics (CI, R², effect size)
- Matrix summary statistics
- HTML report download for all analyses
- Detailed interpretations (Fixed ICC display issue)

Updated: Uses dataset selector pattern like tab_diag.py
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

import numpy as np
import pandas as pd
from shiny import module, reactive, render, ui

from logger import get_logger
from tabs._common import (
    get_color_palette,
    select_variable_by_keyword,
)
from utils import (
    correlation,  # Import from utils
)
from utils.formatting import create_missing_data_report_html, format_p_value
from utils.plotly_html_renderer import plotly_figure_to_html


def _safe_filename_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    return s[:80] or "value"


logger = get_logger(__name__)


# ✅ Use @module.ui decorator
@module.ui
def corr_ui() -> ui.TagChild:
    """
    Create the UI for correlation analysis tab.
    NO manual namespace needed - Shiny handles it automatically.
    """
    return ui.div(
        # Title + Data Summary inline
        ui.output_ui("ui_title_with_summary"),
        # Dataset Info Box
        ui.output_ui("ui_matched_info"),
        ui.br(),
        # Dataset Selector
        ui.output_ui("ui_dataset_selector"),
        ui.br(),
        # Main Analysis Tabs
        ui.navset_tab(
            # TAB 1: Pearson/Spearman Correlation (Pairwise)
            ui.nav_panel(
                "📈 Pairwise Correlation",
                ui.card(
                    ui.card_header("📈 Continuous Correlation Analysis"),
                    ui.layout_columns(
                        ui.input_select(
                            "coeff_type",
                            "Correlation Coefficient:",
                            choices={
                                "pearson": "Pearson",
                                "spearman": "Spearman",
                                "kendall": "Kendall",
                            },
                            selected="pearson",
                        ),
                        ui.input_select(
                            "cv1", "Variable 1 (X-axis):", choices=["Select..."]
                        ),
                        ui.input_select(
                            "cv2", "Variable 2 (Y-axis):", choices=["Select..."]
                        ),
                        col_widths=[3, 4, 4],
                    ),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_corr",
                            "📈 Analyze Correlation",
                            class_="btn-primary",
                            width="100%",
                        ),
                        # ✅ CHANGED: Use download_button
                        ui.download_button(
                            "btn_dl_corr",
                            "📥 Download Report",
                            class_="btn-secondary",
                            width="100%",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_corr_result"),
                    full_screen=True,
                ),
            ),
            # TAB 2: Matrix/Heatmap (New!)
            ui.nav_panel(
                "📊 Matrix/Heatmap",
                ui.card(
                    ui.card_header("📊 Correlation Matrix & Heatmap"),
                    ui.input_selectize(
                        "matrix_vars",
                        "Select Variables (Multi-select):",
                        choices=["Select..."],
                        multiple=True,
                        selected=[],
                    ),
                    ui.input_select(
                        "matrix_method",
                        "Correlation Method:",
                        choices={
                            "pearson": "Pearson",
                            "spearman": "Spearman",
                            "kendall": "Kendall",
                        },
                        selected="pearson",
                    ),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_matrix",
                            "🎨 Generate Heatmap",
                            class_="btn-primary",
                            width="100%",
                        ),
                        # ✅ CHANGED: Use download_button
                        ui.download_button(
                            "btn_dl_matrix",
                            "📥 Download Report",
                            class_="btn-secondary",
                            width="100%",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.output_ui("out_matrix_result"),
                    full_screen=True,
                ),
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

                            <div class="alert alert-warning" role="alert">
                            <strong>Warning:</strong> Correlation does not imply causation. A strong relationship between two variables does not mean one causes the other.
                            </div>

                            **1. Pearson (r):**
                            * **Best for:** Linear relationships (straight line), normally distributed data.
                            * **Sensitive to:** Outliers.
                            * **Returns:** R-squared (R²) = proportion of variance explained

                            **2. Spearman (rho) & Kendall (tau):**
                            * **Best for:** Monotonic relationships, non-normal data, or ranks.
                            * **Robust to:** Outliers.
                            * **Kendall's Tau** is often preferred for small datasets with many tied ranks.

                            **Interpretation of Coefficient (r, rho, or tau):**
                            * **+1.0:** Perfect Positive (As X goes up, Y goes up).
                            * **-1.0:** Perfect Negative (As X goes up, Y goes down).
                            * **0.0:** No relationship.

                            **Strength Guidelines:**
                            * **0.9 - 1.0:** Very Strong 🔥
                            * **0.7 - 0.9:** Strong 📈
                            * **0.5 - 0.7:** Moderate 📊
                            * **0.3 - 0.5:** Weak 📉
                            * **< 0.3:** Very Weak/Negligible
                            
                            **Confidence Intervals (95% CI):**
                            * Shows the range where the true correlation likely falls
                            * Wider CI = less precise estimate (usually with small samples)
                            """),
                        ),
                        ui.card(
                            ui.card_header("💡 Common Questions"),
                            ui.markdown("""
                            **Q: What is R-squared (R²)?**
                            * **A:** R² tells you the proportion of variance in Y that is explained by X. 
                            For example, R² = 0.64 means 64% of the variation in Y is explained by X.

                            **Q: Why use ICC instead of Pearson for reliability?**
                            * **A:** Pearson only measures linearity. If Rater A always gives exactly 10 points 
                            higher than Rater B, Pearson = 1.0 but they don't agree! ICC accounts for this.

                            **Q: What if p-value is significant but r is low (0.1)?**
                            * **A:** P-value means it's likely not zero. With large samples, tiny correlations 
                            can be "significant". **Focus on r-value magnitude** for clinical relevance.

                            **Q: How to interpret confidence intervals?**
                            * **A:** If 95% CI includes 0, the correlation is not statistically significant. 
                            Narrow CI = more precise estimate, Wide CI = less precise (need more data).
                            
                            **Q: How many variables do I need for ICC?**
                            * **A:** At least 2 (to compare two raters/methods). More raters = more reliable ICC.
                            """),
                        ),
                        col_widths=[6, 6],
                    ),
                    full_screen=True,
                ),
            ),
        ),
    )


# ✅ Use @module.server decorator properly
@module.server
def corr_server(
    input: Any,
    output: Any,
    session: Any,
    df: reactive.Value[pd.DataFrame | None],
    var_meta: reactive.Value[dict[str, Any]],
    df_matched: reactive.Value[pd.DataFrame | None],
    is_matched: reactive.Value[bool],
) -> None:
    """
    Register server-side reactives, event handlers, and UI outputs for the Correlation & ICC Analysis tab.
    """
    COLORS = get_color_palette()

    # ==================== REACTIVE STATES ====================

    corr_result: reactive.Value[dict[str, Any] | None] = reactive.Value(
        None
    )  # Pairwise result
    matrix_result: reactive.Value[dict[str, Any] | None] = reactive.Value(
        None
    )  # Matrix result
    numeric_cols_list: reactive.Value[list[str]] = reactive.Value(
        []
    )  # List of numeric columns

    # ==================== DATASET SELECTION LOGIC ====================

    @reactive.Calc
    def current_df() -> pd.DataFrame | None:
        """Select between original and matched dataset based on user preference."""
        if is_matched.get() and input.radio_corr_source() == "matched":
            return df_matched.get()
        return df.get()

    @render.ui
    def ui_title_with_summary():
        """Display title with dataset summary."""
        d = current_df()
        if d is not None:
            return ui.div(
                ui.h3("📈 Correlation Analysis"),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3",
                ),
            )
        return ui.h3("📈 Correlation Analysis")

    @render.ui
    def ui_matched_info():
        """Display matched dataset availability info."""
        if is_matched.get():
            return ui.div(
                ui.tags.div(
                    "✅ **Matched Dataset Available** - You can select it below for analysis",
                    class_="alert alert-info",
                )
            )
        return None

    @render.ui
    def ui_dataset_selector():
        """Render dataset selector radio buttons."""
        if is_matched.get():
            original = df.get()
            matched = df_matched.get()
            original_len = len(original) if original is not None else 0
            matched_len = len(matched) if matched is not None else 0
            return ui.input_radio_buttons(
                "radio_corr_source",
                "📊 Select Dataset:",
                {
                    "original": f"📊 Original Data ({original_len:,} rows)",
                    "matched": f"✅ Matched Data ({matched_len:,} rows)",
                },
                selected="matched",
                inline=True,
            )
        return None

    # ==================== UPDATE NUMERIC COLUMNS ====================

    @reactive.Effect
    def _update_numeric_cols():
        """Update list of numeric columns when data changes."""
        data = current_df()
        if data is not None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols_list.set(cols)

            if cols:
                # ✅ FILTER: Filter columns starting with 'lab', 'value', 'values'
                filtered_cols = [
                    c for c in cols if c.lower().startswith(("lab", "value", "values"))
                ]

                # If no columns match, fallback to all numeric columns
                final_cols = filtered_cols if filtered_cols else cols

                # Pairwise selectors
                selected_v1 = select_variable_by_keyword(
                    final_cols, ["glucose", "lab_glucose"], default_to_first=True
                )
                ui.update_select("cv1", choices=final_cols, selected=selected_v1)

                remaining_cols = [c for c in final_cols if c != selected_v1]
                selected_v2 = select_variable_by_keyword(
                    remaining_cols, ["hba1c", "lab_hba1c"], default_to_first=True
                )
                ui.update_select("cv2", choices=final_cols, selected=selected_v2)

                # Matrix selector
                ui.update_selectize("matrix_vars", choices=cols, selected=cols[:5])

    # ==================== PAIRWISE CORRELATION ====================

    @reactive.Effect
    @reactive.event(input.btn_run_corr)
    def _run_correlation() -> None:
        """Run pairwise correlation analysis."""
        data = current_df()

        if data is None:
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
                data, col1, col2, method=method, var_meta=var_meta.get() or {}
            )

        if err:
            ui.notification_show(f"Error: {err}", type="error")
            corr_result.set(None)
        else:
            # Determine data label
            if is_matched.get() and input.radio_corr_source() == "matched":
                data_label = f"✅ Matched Data ({len(data)} rows)"
            else:
                data_label = f"📊 Original Data ({len(data)} rows)"

            corr_result.set(
                {
                    "stats": res_stats,
                    "figure": fig,
                    "method": method,
                    "var1": col1,
                    "var2": col2,
                    "data_label": data_label,
                }
            )
            ui.notification_show("✅ Correlation analysis complete", type="default")

    @render.ui
    def out_corr_result():
        """Display pairwise correlation results."""
        result = corr_result.get()
        if result is None:
            return ui.markdown(
                "*Results will appear here after clicking '📈 Analyze Correlation'*"
            )

        stats = result["stats"]

        # Format interpretation
        var1 = _html.escape(str(result["var1"]))
        var2 = _html.escape(str(result["var2"]))
        interpretation = _html.escape(str(stats.get("Interpretation", "")))
        sample_note = _html.escape(str(stats.get("Sample Note", "")))
        r2 = float(stats["R-squared (R²)"])
        interp_html = f"""
        <div style='background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%); 
                    border-left: 4px solid {COLORS["primary"]}; 
                    padding: 14px 15px; 
                    margin: 16px 0; 
                    border-radius: 5px;'>
            <strong>Interpretation:</strong> {interpretation}<br>
            <strong>R² = {r2:.3f}</strong> →
            {r2 * 100:.1f}% of variance in {var2} is explained by {var1}<br>
            <strong>Sample:</strong> {sample_note}
        </div>
        """

        return ui.div(
            ui.card(
                ui.card_header("Results"),
                ui.markdown(f"**Data Source:** {result['data_label']}"),
                ui.markdown(f"**Method:** {result['method'].title()}"),
                ui.output_data_frame("out_corr_table"),
                ui.HTML(interp_html),
                # Missing Data Report
                ui.HTML(
                    create_missing_data_report_html(
                        stats.get("missing_data_info", {}), var_meta.get() or {}
                    )
                ),
                ui.card_header("Scatter Plot"),
                ui.output_ui("out_corr_plot_widget"),
            ),
            class_="fade-in-entry",
        )

    @render.data_frame
    def out_corr_table():
        """
        Create a formatted table of the most relevant pairwise correlation statistics for the current result.
        """
        result = corr_result.get()
        if result is None:
            return None

        # Create formatted table
        stats = result["stats"]
        # Helper to get coefficient safely
        coef_key = (
            "Coefficient (r/rho/tau)"
            if "Coefficient (r/rho/tau)" in stats
            else "Coefficient (r)"
        )
        coef_val = stats.get(coef_key)
        coef_display = (
            f"{coef_val:.4f}"
            if isinstance(coef_val, (int, float)) and not pd.isna(coef_val)
            else "N/A"
        )

        display_data = {
            "Metric": [
                "Method",
                "Correlation Coefficient",
                "95% CI Lower",
                "95% CI Upper",
                "R-squared (R²)",
                "P-value",
                "Sample Size (N)",
                "Interpretation",
            ],
            "Value": [
                stats["Method"],
                coef_display,
                f"{stats.get('95% CI Lower', float('nan')):.4f}",
                f"{stats.get('95% CI Upper', float('nan')):.4f}",
                f"{stats.get('R-squared (R²)', float('nan')):.4f}",
                format_p_value(stats.get("P-value", float("nan")), use_style=False),
                str(stats.get("N", "N/A")),
                stats.get("Interpretation", "N/A"),
            ],
        }

        df_display = pd.DataFrame(display_data)
        return render.DataGrid(df_display, width="100%")

    @render.ui
    def out_corr_plot_widget():
        """Render the correlation scatter plot as an HTML UI element."""
        result = corr_result.get()
        if result is None or result["figure"] is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            result["figure"],
            div_id="plot_corr_scatter",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.download(
        filename=lambda: (
            (
                lambda r: f"correlation_{_safe_filename_part(r['var1'])}_{_safe_filename_part(r['var2'])}.html"
            )(corr_result.get())
            if corr_result.get() is not None
            else "correlation_report.html"
        ),
    )
    def btn_dl_corr():
        """Generate and download correlation report."""
        result = corr_result.get()
        if result is None:
            yield b"No results available"
            return

        stats = result["stats"]

        # Build report elements
        elements = [
            {"type": "text", "data": f"Data Source: {result['data_label']}"},
            {"type": "text", "data": f"Method: {result['method'].title()}"},
            {
                "type": "text",
                "data": f"Variables: {result['var1']} vs {result['var2']}",
            },
            {"type": "text", "header": "Statistical Results", "data": ""},
        ]

        # Add statistics
        # Helper for key lookup
        coef_key = (
            "Coefficient (r/rho/tau)"
            if "Coefficient (r/rho/tau)" in stats
            else "Coefficient (r)"
        )

        for key in [
            "Method",
            coef_key,
            "95% CI Lower",
            "95% CI Upper",
            "R-squared (R²)",
            "P-value",
            "N",
        ]:
            val = stats.get(key, "N/A")
            if key == "P-value" and isinstance(val, (int, float)):
                # Use standard P-value formatter with styling for the report
                elements.append(
                    {
                        "type": "html",
                        "data": f"<strong>P-value:</strong> {format_p_value(val, use_style=True)}",
                    }
                )
            # Format numeric if possible
            elif isinstance(val, (int, float)):
                elements.append(
                    {
                        "type": "text",
                        "data": (
                            f"{key if key != coef_key else 'Correlation Coefficient'}: {val:.4f}"
                            if isinstance(val, float)
                            else f"{key}: {val}"
                        ),
                    }
                )
            else:
                elements.append({"type": "text", "data": f"{key}: {val}"})

        # Add interpretation
        elements.append(
            {
                "type": "interpretation",
                "data": f"{stats['Interpretation']}. R² = {stats['R-squared (R²)']:.3f} means {stats['R-squared (R²)'] * 100:.1f}% of variance is explained.",
            }
        )

        elements.append({"type": "note", "data": stats["Sample Note"]})

        # Add plot
        elements.append(
            {"type": "plot", "header": "Scatter Plot", "data": result["figure"]}
        )

        # Missing Data Report (moved to end)
        if "missing_data_info" in stats:
            elements.append(
                {
                    "type": "html",
                    "data": create_missing_data_report_html(
                        stats["missing_data_info"], var_meta.get() or {}
                    ),
                }
            )

        # Generate HTML
        html_content = correlation.generate_report(
            title=f"Correlation Analysis: {result['var1']} vs {result['var2']}",
            elements=elements,
        )

        yield html_content.encode("utf-8")

    # ==================== CORRELATION MATRIX / HEATMAP ====================

    @reactive.Effect
    @reactive.event(input.btn_run_matrix)
    def _run_matrix() -> None:
        """Run correlation matrix and heatmap generation."""
        data = current_df()

        if data is None:
            ui.notification_show("No data available", type="error")
            return

        cols = input.matrix_vars()
        method = input.matrix_method()

        if not cols or len(cols) < 2:
            ui.notification_show("Please select at least 2 variables", type="warning")
            return

        with ui.Progress(min=0, max=1) as p:
            p.set(
                message="Generating Heatmap...",
                detail=f"Processing {len(cols)} variables",
            )
            corr_matrix, fig, summary = correlation.compute_correlation_matrix(
                data, list(cols), method=method, var_meta=var_meta.get() or {}
            )

        if corr_matrix is not None:
            # Determine data label
            if is_matched.get() and input.radio_corr_source() == "matched":
                data_label = f"✅ Matched Data ({len(data)} rows)"
            else:
                data_label = f"📊 Original Data ({len(data)} rows)"

            matrix_result.set(
                {
                    "matrix": corr_matrix,
                    "figure": fig,
                    "method": method,
                    "summary": summary,
                    "data_label": data_label,
                    "strategy": summary.get("missing_data_info", {}).get(
                        "strategy", "pairwise-complete"
                    ),
                }
            )
            ui.notification_show("✅ Heatmap generated!", type="default")
        else:
            matrix_result.set(None)
            ui.notification_show("Failed to generate matrix", type="error")

    @render.ui
    def out_matrix_result():
        """Render the matrix/heatmap results card for the current analysis."""
        result = matrix_result.get()
        if result is None:
            return ui.markdown(
                "*Results will appear here after clicking '🎨 Generate Heatmap'*"
            )

        summary = result["summary"]

        # Format summary statistics
        strongest_pos = _html.escape(str(summary["strongest_positive"]))
        strongest_neg = _html.escape(str(summary["strongest_negative"]))
        summary_html = f"""
        <div style='background: linear-gradient(135deg, #fff3e0 0%, #f8f9fa 100%); 
                    border: 2px solid #ff9800; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 20px 0;'>
            <h4 style='color: #e65100; margin-top: 0;'>📊 Matrix Summary</h4>
            <p><strong>Variables:</strong> {summary["n_variables"]}</p>
            <p><strong>Correlations Computed:</strong> {summary["n_correlations"]} (unique pairs)</p>
            <p><strong>Mean |Correlation|:</strong> {summary["mean_correlation"]:.3f}</p>
            <p><strong>Strongest Positive:</strong> {strongest_pos}</p>
            <p><strong>Strongest Negative:</strong> {strongest_neg}</p>
            <p><strong>Significant Correlations (p<0.05):</strong> {summary["n_significant"]} ({summary["pct_significant"]:.1f}%)</p>
        </div>
        """

        return ui.div(
            ui.card(
                ui.card_header("Matrix Results"),
                ui.markdown(f"**Data Source:** {result['data_label']}"),
                ui.markdown(f"**Method:** {result['method'].title()}"),
                ui.markdown(f"**Missing Data Strategy:** {result['strategy'].title()}"),
                ui.HTML(summary_html),
                # Missing Data Report
                ui.HTML(
                    create_missing_data_report_html(
                        summary.get("missing_data_info", {}), var_meta.get() or {}
                    )
                ),
                ui.card_header("Heatmap"),
                ui.output_ui("out_heatmap_widget"),
                ui.card_header("Correlation Table"),
                ui.markdown(
                    "*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001*"
                ),
                ui.output_data_frame("out_matrix_table"),
            ),
            class_="fade-in-entry",
        )

    @render.ui
    def out_heatmap_widget():
        """Render the correlation heatmap plot or a waiting placeholder as a Shiny UI element."""
        result = matrix_result.get()
        if result is None or result["figure"] is None:
            return ui.div(
                ui.markdown("⏳ *Waiting for results...*"),
                style="color: #999; text-align: center; padding: 20px;",
            )
        html_str = plotly_figure_to_html(
            result["figure"],
            div_id="plot_corr_heatmap",
            include_plotlyjs="cdn",
            responsive=True,
        )
        return ui.HTML(html_str)

    @render.data_frame
    def out_matrix_table():
        """Render the correlation matrix as a DataGrid suitable for display."""
        result = matrix_result.get()
        if result is None:
            return None
        # Add index as a column for better display in DataGrid
        df_display = (
            result["matrix"].reset_index().rename(columns={"index": "Variable"})
        )
        return render.DataGrid(df_display, width="100%")

    @render.download(
        filename=lambda: (
            (lambda r: f"correlation_matrix_{_safe_filename_part(r['method'])}.html")(
                matrix_result.get()
            )
            if matrix_result.get() is not None
            else "correlation_matrix.html"
        ),
    )
    def btn_dl_matrix():
        """Generate and download matrix report."""
        result = matrix_result.get()
        if result is None:
            yield b"No results available"
            return

        summary = result["summary"]

        # Build report elements
        elements = [
            {"type": "text", "data": f"Data Source: {result['data_label']}"},
            {"type": "text", "data": f"Method: {result['method'].title()}"},
            {"type": "text", "data": f"Number of Variables: {summary['n_variables']}"},
        ]

        # Add summary statistics
        summary_text = f"""
        <h3>Matrix Summary Statistics</h3>
        <p><strong>Correlations Computed:</strong> {_html.escape(str(summary["n_correlations"]))} unique pairs</p>
        <p><strong>Mean |Correlation|:</strong> {_html.escape(str(summary["mean_correlation"]))}</p>
        <p><strong>Maximum |Correlation|:</strong> {summary["max_correlation"]:.3f}</p>
        <p><strong>Minimum |Correlation|:</strong> {summary["min_correlation"]:.3f}</p>
        <p><strong>Significant Correlations (p<0.05):</strong> {summary["n_significant"]} out of {summary["n_correlations"]} ({summary["pct_significant"]:.1f}%)</p>
        <p><strong>Strongest Positive:</strong> {summary["strongest_positive"]}</p>
        <p><strong>Strongest Negative:</strong> {summary["strongest_negative"]}</p>
        """

        elements.append({"type": "summary", "data": summary_text})

        # Add heatmap
        elements.append(
            {"type": "plot", "header": "Correlation Heatmap", "data": result["figure"]}
        )

        # Add matrix table
        elements.append(
            {"type": "table", "header": "Correlation Matrix", "data": result["matrix"]}
        )

        elements.append(
            {
                "type": "note",
                "data": "Significance levels: * p<0.05, ** p<0.01, *** p<0.001",
            }
        )

        # Missing Data Report (moved to end)
        if "missing_data_info" in summary:
            elements.append(
                {
                    "type": "html",
                    "data": create_missing_data_report_html(
                        summary["missing_data_info"], var_meta.get() or {}
                    ),
                }
            )

        # Generate HTML
        html_content = correlation.generate_report(
            title=f"Correlation Matrix Analysis ({result['method'].title()})",
            elements=elements,
        )

        yield html_content.encode("utf-8")

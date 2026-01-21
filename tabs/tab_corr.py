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
from utils import correlation  # Import from utils
from utils.formatting import create_missing_data_report_html
from utils.plotly_html_renderer import plotly_figure_to_html
from utils.ui_helpers import (
    create_error_alert,
    create_input_group,
    create_loading_state,
    create_placeholder_state,
    create_results_container,
    create_tooltip_label,
)


def _safe_filename_part(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    return s[:80] or "value"


logger = get_logger(__name__)


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
                    ui.card_header("Pairwise Correlation Analysis"),
                    create_input_group(
                        "Variables & Method",
                        ui.input_select(
                            "coeff_type",
                            create_tooltip_label(
                                "Correlation Coefficient",
                                "Pearson for linear, Spearman for monotonic/ranked.",
                            ),
                            choices={"pearson": "Pearson", "spearman": "Spearman"},
                            selected="pearson",
                        ),
                        ui.input_select(
                            "cv1", "Variable 1 (X-axis):", choices=["Select..."]
                        ),
                        ui.input_select(
                            "cv2", "Variable 2 (Y-axis):", choices=["Select..."]
                        ),
                        type="required",
                    ),
                    ui.output_ui("out_corr_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_corr",
                            "📈 Analyze Correlation",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_corr",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container(
                    "Correlation Results", ui.output_ui("out_corr_result")
                ),
            ),
            # TAB 2: Matrix/Heatmap (New!)
            ui.nav_panel(
                "📊 Matrix/Heatmap",
                ui.card(
                    ui.card_header("Correlation Matrix & Heatmap"),
                    create_input_group(
                        "Variables & Method",
                        ui.input_selectize(
                            "matrix_vars",
                            create_tooltip_label(
                                "Select Variables (Multi-select)",
                                "Choose continuous variables for matrix.",
                            ),
                            choices=["Select..."],
                            multiple=True,
                            selected=[],
                        ),
                        ui.input_select(
                            "matrix_method",
                            "Correlation Method:",
                            choices={"pearson": "Pearson", "spearman": "Spearman"},
                            selected="pearson",
                        ),
                        type="required",
                    ),
                    ui.output_ui("out_matrix_validation"),
                    ui.hr(),
                    ui.layout_columns(
                        ui.input_action_button(
                            "btn_run_matrix",
                            "🎨 Generate Heatmap",
                            class_="btn-primary w-100",
                        ),
                        ui.download_button(
                            "btn_dl_matrix",
                            "📥 Download Report",
                            class_="btn-secondary w-100",
                        ),
                        col_widths=[6, 6],
                    ),
                ),
                create_results_container("Results", ui.output_ui("out_matrix_result")),
            ),
            # TAB 4: Reference & Interpretation
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
                            * **Returns:** R-squared (R²) = proportion of variance explained

                            **2. Spearman (rho):**
                            * **Best for:** Monotonic relationships, non-normal data, or ranks.
                            * **Robust to:** Outliers.

                            **Interpretation of Coefficient (r or rho):**
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
                        col_widths=[6, 6],
                    ),
                    ui.card(
                        ui.card_header("💡 Common Questions"),
                        ui.markdown("""
                        **Q: What is R-squared (R²)?**
                        * **A:** R² tells you the proportion of variance in Y that is explained by X. 
                        For example, R² = 0.64 means 64% of the variation in Y is explained by X.

                        **Q: What if p-value is significant but r is low (0.1)?**
                        * **A:** P-value means it's likely not zero. With large samples, tiny correlations 
                        can be "significant". **Focus on r-value magnitude** for clinical relevance.

                        **Q: How to interpret confidence intervals?**
                        * **A:** If 95% CI includes 0, the correlation is not statistically significant. 
                        Narrow CI = more precise estimate, Wide CI = less precise (need more data).
                        """),
                    ),
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

    Sets up and manages reactive state, dataset selection logic, numeric-column discovery, pairwise correlation (calculation, rendering, and download), correlation matrix/heatmap (calculation, rendering, and download), and ICC analysis (calculation, interpretation, rendering, and download). This function attaches renderers, effects, and download handlers used by the corresponding UI to present results and reports.

    Parameters:
        input: Shiny input object for accessing UI inputs and events.
        output: Shiny output object used by the render decorators.
        session: Shiny session object for the current user session.
        df (reactive.Value[pd.DataFrame | None]): Primary dataset reactive; used as the default data source.
        var_meta (reactive.Value[dict[str, Any]]): Reactive dictionary of variable metadata (column attributes and labels).
        df_matched (reactive.Value[pd.DataFrame | None]): Optional matched dataset reactive used when the user selects the matched data source.
        is_matched (reactive.Value[bool]): Reactive boolean flag indicating whether a matched dataset is available/selected.
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

    # Running States
    corr_is_running = reactive.Value(False)
    matrix_is_running = reactive.Value(False)

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
                ui.h3("📈 Correlation & ICC Analysis"),
                ui.p(
                    f"{len(d):,} rows | {len(d.columns)} columns",
                    class_="text-secondary mb-3",
                ),
            )
        return ui.h3("📈 Correlation & ICC Analysis")

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

                # Smart defaults using select_variable_by_keyword
                default_cv1 = select_variable_by_keyword(
                    final_cols, ["lab", "value", "score"], default_to_first=True
                )
                
                # For cv2, try to find a different variable
                remaining_cols = [c for c in final_cols if c != default_cv1]
                default_cv2 = select_variable_by_keyword(
                    remaining_cols, ["lab", "value", "score"], default_to_first=True
                )
                
                # If we couldn't find a different second variable (e.g. only 1 col), fallback
                if not default_cv2 and final_cols:
                     default_cv2 = final_cols[0]


                # Pairwise selectors
                ui.update_select("cv1", choices=final_cols, selected=default_cv1)
                ui.update_select(
                    "cv2",
                    choices=final_cols,
                    selected=default_cv2,
                )

                # Matrix selector (Use all cols or filtered? Usually matrix uses all, but let's default to filtered if available)
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

        try:
            corr_is_running.set(True)
            corr_result.set(None)

            # Using progress here is still fine, but our loading state in result area is better UX
            # We can keep the progress bar for additional feedback or remove it.
            # I will keep it for now as it doesn't hurt.
            with ui.Progress(min=0, max=1) as p:
                p.set(
                    message="Calculating correlation...",
                    detail="This may take a moment",
                )
                res_stats, err, fig = correlation.calculate_correlation(
                    data, col1, col2, method=method, var_meta=var_meta.get() or {}
                )

            if err:
                corr_result.set({"error": err})
                ui.notification_show("Analysis failed", type="error")
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
        except Exception as e:
            logger.exception("Correlation analysis failed")
            corr_result.set({"error": f"Analysis Error: {str(e)}"})
            ui.notification_show("Analysis failed", type="error")
        finally:
            corr_is_running.set(False)

    @render.ui
    def out_corr_result():
        """Display pairwise correlation results."""
        if corr_is_running.get():
            return create_loading_state("Calculating correlation...")

        result = corr_result.get()
        if result is None:
            return create_placeholder_state(
                "Select two variables and click 'Analyze Correlation'.", icon="📈"
            )

        if "error" in result:
            return create_error_alert(result["error"])

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
            ui.row(
                ui.column(6, ui.markdown(f"**Data Source:** {result['data_label']}")),
                ui.column(6, ui.markdown(f"**Method:** {result['method'].title()}")),
            ),
            ui.output_data_frame("out_corr_table"),
            ui.HTML(interp_html),
            # Missing Data Report
            ui.HTML(
                create_missing_data_report_html(
                    stats.get("missing_data_info", {}), var_meta.get() or {}
                )
            ),
            ui.h5("Scatter Plot", class_="mt-4 mb-3"),
            ui.output_ui("out_corr_plot_widget"),
        )

    @render.data_frame
    def out_corr_table():
        """
        Create a formatted table of the most relevant pairwise correlation statistics for the current result.

        Returns:
            A DataGrid showing metrics (Method; Correlation Coefficient (r); 95% CI Lower; 95% CI Upper; R-squared (R²); P-value; Sample Size (N); Interpretation) for the computed correlation, or `None` if no correlation result is available.
        """
        result = corr_result.get()
        if result is None or "error" in result:
            return None

        # Create formatted table
        stats = result["stats"]
        display_data = {
            "Metric": [
                "Method",
                "Correlation Coefficient (r)",
                "95% CI Lower",
                "95% CI Upper",
                "R-squared (R²)",
                "P-value",
                "Sample Size (N)",
                "Interpretation",
            ],
            "Value": [
                stats["Method"],
                f"{stats['Coefficient (r)']:.4f}",
                f"{stats['95% CI Lower']:.4f}",
                f"{stats['95% CI Upper']:.4f}",
                f"{stats['R-squared (R²)']:.4f}",
                f"{stats['P-value']:.4f}",
                str(stats["N"]),
                stats["Interpretation"],
            ],
        }

        df_display = pd.DataFrame(display_data)
        return render.DataGrid(df_display, width="100%")

    @render.ui
    def out_corr_plot_widget():
        """
        Render the correlation scatter plot as an HTML UI element.

        Returns:
            ui_element: A UI element containing the Plotly scatter plot HTML when a figure is available, or a centered waiting placeholder if no result/figure exists.
        """
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

    # ✅ CHANGED: Logic for downloading file
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
        if result is None or "error" in result:
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
        for key in [
            "Method",
            "Coefficient (r)",
            "95% CI Lower",
            "95% CI Upper",
            "R-squared (R²)",
            "P-value",
            "N",
        ]:
            val = stats[key]
            if isinstance(val, (int, float)):
                elements.append(
                    {
                        "type": "text",
                        "data": (
                            f"{key}: {val:.4f}"
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

        try:
            matrix_is_running.set(True)
            matrix_result.set(None)

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
                matrix_result.set({"error": "Failed to generate correlation matrix"})
                ui.notification_show("Analysis failed", type="error")
        except Exception as e:
            logger.exception("Matrix analysis failed")
            matrix_result.set({"error": f"Error: {str(e)}"})
            ui.notification_show("Analysis failed", type="error")
        finally:
            matrix_is_running.set(False)

    @render.ui
    def out_matrix_result():
        """
        Render the matrix/heatmap results card for the current analysis.
        """
        if matrix_is_running.get():
            return create_loading_state("Generating Correlation Matrix & Heatmap...")

        result = matrix_result.get()
        if result is None:
            return create_placeholder_state(
                "Select variables and click 'Generate Heatmap'.", icon="📊"
            )

        if "error" in result:
            return create_error_alert(result["error"])

        summary = result["summary"]

        # Format summary statistics
        # Escape summary strings that include column names
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
            ui.markdown(
                f"**Data Source:** {result['data_label']} | **Method:** {result['method'].title()} | **Missing Data Strategy:** {result['strategy'].title()}"
            ),
            ui.HTML(summary_html),
            # Missing Data Report
            ui.HTML(
                create_missing_data_report_html(
                    summary.get("missing_data_info", {}), var_meta.get() or {}
                )
            ),
            ui.h5("Heatmap", class_="mt-4 mb-3"),
            ui.output_ui("out_heatmap_widget"),
            ui.h5("Correlation Table", class_="mt-4 mb-3"),
            ui.markdown("*Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001*"),
            ui.output_data_frame("out_matrix_table"),
        )

    @render.ui
    def out_heatmap_widget():
        """
        Render the correlation heatmap plot or a waiting placeholder as a Shiny UI element.

        Returns:
            ui_element: A Shiny UI element containing the rendered Plotly heatmap when available, or a centered "Waiting for results..." placeholder otherwise.
        """
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
        """
        Render the correlation matrix as a DataGrid suitable for display.

        The matrix's row index is converted into a column named "Variable" to make variable names visible in the grid.

        Returns:
            ui_element (shiny.ui.output/DataGrid) or None: A DataGrid rendering of the matrix when results are available, otherwise None.
        """
        result = matrix_result.get()
        if result is None or "error" in result:
            return None
        # Add index as a column for better display in DataGrid
        df_display = (
            result["matrix"].reset_index().rename(columns={"index": "Variable"})
        )
        return render.DataGrid(df_display, width="100%")

    # ✅ CHANGED: Logic for downloading file
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
        if result is None or "error" in result:
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

        # Add matrix table
        elements.append(
            {"type": "table", "header": "Correlation Matrix", "data": result["matrix"]}
        )

        # Add heatmap
        elements.append(
            {"type": "plot", "header": "Correlation Heatmap", "data": result["figure"]}
        )

        # Missing Data Report
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
            title="Correlation Matrix Analysis", elements=elements
        )

        yield html_content.encode("utf-8")

    # ==================== VALIDATION LOGIC ====================
    @render.ui
    def out_corr_validation():
        d = current_df()
        cv1 = input.cv1()
        cv2 = input.cv2()

        if d is None or d.empty:
            return None
        alerts = []

        if not cv1 or not cv2:
            return None

        if cv1 == cv2:
            alerts.append(
                create_error_alert(
                    "Please select different variables.", title="Configuration Error"
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None

    @render.ui
    def out_matrix_validation():
        d = current_df()
        cols = input.matrix_vars()

        if d is None or d.empty:
            return None
        alerts = []

        if not cols:
            return None

        if len(cols) < 2:
            alerts.append(
                create_error_alert(
                    "Please select at least 2 variables for the matrix.",
                    title="Configuration Error",
                )
            )

        if alerts:
            return ui.div(*alerts)
        return None



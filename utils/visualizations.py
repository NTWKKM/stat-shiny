import pandas as pd
import plotly.graph_objects as go


def plot_missing_pattern(df: pd.DataFrame, max_cols: int = 50) -> go.Figure:
    """
    Create a heatmap visualization of missing data patterns.

    Parameters:
        df: Input DataFrame (will be sampled if too large)
        max_cols: Maximum number of columns to display (prioritizes those with missing data)

    Returns:
        Plotly Figure object
    """
    if df is None or df.empty:
        return go.Figure()

    # Handling large datasets: limit display
    # 1. Filter to columns with any missing values first, then other interesting ones
    missing_counts = df.isna().sum()
    cols_with_missing = (
        missing_counts[missing_counts > 0].sort_values(ascending=False).index.tolist()
    )

    # If we have too many columns with missing data, take top N
    if len(cols_with_missing) > max_cols:
        display_cols = cols_with_missing[:max_cols]
    else:
        # Fill remaining slots with complete columns
        remaining_slots = max_cols - len(cols_with_missing)
        cols_complete = missing_counts[missing_counts == 0].index.tolist()
        display_cols = cols_with_missing + cols_complete[:remaining_slots]

    # Subset data
    df_plot = df[display_cols].copy()

    # Create binary matrix: 0=Missing, 1=Present
    # We want Missing to be distinct.
    # Let's map: 0 = Present, 1 = Missing (for intuitive 'heat' on missing)
    z = df_plot.isna().astype(int).transpose().values

    x = df_plot.index
    y = df_plot.columns

    # Colors: 0 (Present) = Light Grey/White, 1 (Missing) = Red/Orange
    colorscale = [
        [0.0, "rgb(240, 240, 240)"],  # Present
        [1.0, "rgb(220, 53, 69)"],  # Missing (Bootstrap 'danger' red)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=x, y=y, colorscale=colorscale, showscale=False, xgap=1, ygap=1
        )
    )

    fig.update_layout(
        title="Missing Data Pattern (Red = Missing)",
        xaxis_title="Row Index",
        yaxis_title="Variable",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=40),
        height=min(800, max(400, len(y) * 25)),  # Dynamic height
    )

    return fig

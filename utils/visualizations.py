import pandas as pd
import plotly.graph_objects as go
from plotly import subplots


def plot_missing_pattern(
    df: pd.DataFrame, max_cols: int = 50, max_rows: int = 1000
) -> go.Figure:
    """
    Create a composite visualization of missing data:
    1. Top: Bar chart of % missing per column
    2. Bottom: Heatmap of missingness (Red=Missing), subsampled for performance

    Parameters:
        df: Input DataFrame
        max_cols: Maximum number of columns to display
        max_rows: Maximum rows to display in heatmap (randomly sampled)
    """
    if df is None or df.empty:
        return go.Figure()

    # --- 1. Column Selection ---
    # Prioritize columns with missing values
    missing_series = df.isna().sum()
    missing_pct = (missing_series / len(df)) * 100

    cols_with_missing = (
        missing_series[missing_series > 0].sort_values(ascending=False).index.tolist()
    )

    if len(cols_with_missing) > max_cols:
        display_cols = cols_with_missing[:max_cols]
    else:
        remaining = max_cols - len(cols_with_missing)
        cols_complete = missing_series[missing_series == 0].index.tolist()
        display_cols = cols_with_missing + cols_complete[:remaining]

    # Subset data
    df_plot = df[display_cols].copy()
    missing_pct_plot = missing_pct[display_cols]

    # --- 2. Row Subsampling (for Heatmap) ---
    original_rows = len(df_plot)
    if original_rows > max_rows:
        # Random sample for visualization speed & readability
        df_heatmap = df_plot.sample(n=max_rows, random_state=42).sort_index()
        y_axis_title = f"Row Index (Sampled {max_rows}/{original_rows})"
    else:
        df_heatmap = df_plot
        y_axis_title = "Row Index"

    # --- 3. Build Subplots ---
    fig = subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.2, 0.8],  # 20% Bar, 80% Heatmap
        specs=[[{"type": "bar"}], [{"type": "heatmap"}]],
    )

    # -- Top: Bar Chart --
    fig.add_trace(
        go.Bar(
            x=display_cols,
            y=missing_pct_plot.values,
            name="% Missing",
            marker_color="#E74856",  # Bootstrap danger red (or close)
            hovertemplate="%{x}: %{y:.1f}% Missing<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # -- Bottom: Heatmap --
    # Map: 0=Present, 1=Missing
    z = df_heatmap.isna().astype(int).transpose().values
    x = df_heatmap.index
    y = display_cols

    # Colors: 0 (Present) = Light Grey, 1 (Missing) = Red
    colorscale = [[0.0, "#f8f9fa"], [1.0, "#E74856"]]

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            showscale=False,
            hovertemplate="Row: %{x}<br>Var: %{y}<br>Status: %{z}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # --- 4. Layout ---
    fig.update_layout(
        title="<b>Missing Data Pattern</b>",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=40),
        height=min(900, max(500, len(display_cols) * 20 + 200)),
        showlegend=False,
    )

    # Bar chart axis
    fig.update_yaxes(title_text="% Missing", range=[0, 100], row=1, col=1)

    # Heatmap axis
    fig.update_xaxes(title_text=y_axis_title, row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)  # Top variable at top

    return fig

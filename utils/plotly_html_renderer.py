"""
📊 Plotly HTML Renderer Utility

Converts Plotly figures to HTML strings using CDN for better compatibility
in restricted environments (corporate firewalls, Private Spaces, etc.)

Usage:
    from utils.plotly_html_renderer import plotly_figure_to_html

    html_str = plotly_figure_to_html(
        fig=my_plotly_figure,
        div_id="my_unique_id",
        include_plotlyjs='cdn',
        responsive=True
    )
    return ui.HTML(html_str)
"""

from __future__ import annotations

import logging
import re
import uuid

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore

logger = logging.getLogger(__name__)


def plotly_figure_to_html(
    fig: "go.Figure" | None = None,
    div_id: str | None = None,
    include_plotlyjs: str | bool = "cdn",
    height: int | None = None,
    width: int | None = None,
    responsive: bool = True,
) -> str:
    """
    Render a Plotly Figure to an embeddable HTML string.

    Returns a sanitized HTML fragment suitable for embedding (e.g., in a UI). If the provided figure is None, Plotly is unavailable, the figure is invalid, or an error occurs during rendering, a styled placeholder HTML string is returned.

    Parameters:
        fig (go.Figure | None): Plotly Figure to render. If None, a placeholder is returned.
        div_id (str | None): Optional HTML id for the figure container. If provided it is sanitized to allow only letters, digits, underscores, and hyphens and to ensure it starts with a letter; if omitted a unique id is generated.
        include_plotlyjs (Union[str, bool]): How to include Plotly.js:
            - 'cdn' (default): Load Plotly.js from CDN.
            - True: Inline the full Plotly.js library.
            - False: Do not include Plotly.js (assumes it is already loaded).
        height (int | None): Fixed height in pixels to apply to the figure layout; if None the figure's existing height is used.
        width (int | None): Fixed width in pixels to apply to the figure layout; if None the figure's existing width is used.
        responsive (bool): When True (default), enable autosize/responsive layout behavior.

    Returns:
        str: HTML fragment containing the rendered Plotly figure, or a styled placeholder HTML string if rendering cannot be performed.
    """
    # Handle None or invalid figure
    if fig is None:
        logger.debug(
            "plotly_figure_to_html: Received None figure, returning placeholder"
        )
        return _create_placeholder_html("⏳ Waiting for data...")

    if go is None:
        logger.warning("plotly_figure_to_html: Plotly is not installed")
        return _create_placeholder_html("⚠️ Plotly is not installed")

    # Validate figure type
    if not isinstance(fig, go.Figure):
        logger.warning(
            "plotly_figure_to_html: Expected go.Figure, got %s", type(fig).__name__
        )
        return _create_placeholder_html("⚠️ Invalid figure type")

    # Generate or sanitize div_id
    if div_id is None:
        div_id = f"plotly-{uuid.uuid4().hex[:12]}"
    else:
        div_id = _sanitize_div_id(div_id)

    try:
        # Avoid mutating caller's figure
        fig = go.Figure(fig)

        # Update layout for responsiveness
        if responsive:
            fig.update_layout(autosize=True)

        # Apply fixed dimensions if specified
        if height is not None:
            fig.update_layout(height=height)
        if width is not None:
            fig.update_layout(width=width)

        # Generate HTML
        html_str = fig.to_html(
            full_html=False,
            include_plotlyjs=include_plotlyjs,
            div_id=div_id,
            config={
                "responsive": responsive,
                "displayModeBar": True,
                "displaylogo": False,
            },
        )

        logger.debug("plotly_figure_to_html: Generated HTML for div_id=%r", div_id)
        return html_str

    except Exception:
        logger.exception("plotly_figure_to_html: Error generating HTML")
        return _create_placeholder_html("Error rendering plot")


def _sanitize_div_id(div_id: str) -> str:
    """
    Produce a safe HTML id by removing characters not allowed in IDs and ensuring it starts with a letter.

    Parameters:
        div_id (str): Original div ID to sanitize.

    Returns:
        str: A sanitized string safe for use as an HTML element id. If the result would be empty, returns a generated id beginning with "plot-".
    """
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", str(div_id))

    # Ensure it starts with a letter (HTML ID requirement)
    if sanitized and not sanitized[0].isalpha():
        sanitized = "plot-" + sanitized

    # Fallback if empty
    if not sanitized:
        sanitized = f"plot-{uuid.uuid4().hex[:8]}"

    return sanitized


def _create_placeholder_html(message: str) -> str:
    """
    Create a styled placeholder HTML snippet displaying a message when a figure cannot be rendered.

    The provided message is HTML-escaped to mitigate cross-site scripting (XSS) risks and inserted into a centered, lightly styled div suitable for use where a Plotly figure would otherwise appear.

    Parameters:
        message (str): Text to display inside the placeholder; will be escaped before insertion.

    Returns:
        str: A HTML string containing the styled div with the escaped message.
    """
    # Escape the message to prevent XSS
    escaped_message = (
        str(message)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

    return f"""
    <div style="
        color: #999;
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        border: 1px dashed #dee2e6;
        font-size: 14px;
    ">
        {escaped_message}
    </div>
    """

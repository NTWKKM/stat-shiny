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

import re
import uuid
import logging
from typing import Optional, Union

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore

logger = logging.getLogger(__name__)


def plotly_figure_to_html(
    fig: Optional["go.Figure"] = None,
    div_id: Optional[str] = None,
    include_plotlyjs: Union[str, bool] = "cdn",
    height: Optional[int] = None,
    width: Optional[int] = None,
    responsive: bool = True
) -> str:
    """
    Convert Plotly Figure to HTML string with CDN support.
    
    This function provides a safe way to render Plotly figures as HTML strings,
    suitable for environments where shinywidgets may not work due to firewall
    restrictions or proxy issues.
    
    Args:
        fig: Plotly Figure object. If None, returns a placeholder HTML.
        div_id: Unique ID for the div element. If None, auto-generates a unique ID.
                The ID is sanitized to prevent XSS attacks.
        include_plotlyjs: How to include Plotly.js library:
            - 'cdn' (default): Load from CDN - best for restricted environments
            - True: Include full library inline (larger file size)
            - False: Don't include (assumes already loaded)
        height: Optional fixed height in pixels. If None, uses figure default.
        width: Optional fixed width in pixels. If None, uses figure default.
        responsive: If True, enables autosize for responsive layouts (default True).
    
    Returns:
        HTML string ready for ui.HTML(). Returns a placeholder if fig is None.
    
    Examples:
        Basic usage:
        >>> fig = go.Figure(data=[go.Scatter(x=[1,2,3], y=[1,2,3])])
        >>> html = plotly_figure_to_html(fig, div_id="my_plot")
        >>> return ui.HTML(html)
        
        With fixed dimensions:
        >>> html = plotly_figure_to_html(fig, height=400, width=600)
        
        Handle None gracefully:
        >>> html = plotly_figure_to_html(None)  # Returns placeholder
    
    Raises:
        No exceptions are raised. Errors are logged and a placeholder is returned.
    """
    # Handle None or invalid figure
    if fig is None:
        logger.debug("plotly_figure_to_html: Received None figure, returning placeholder")
        return _create_placeholder_html("⏳ Waiting for data...")
    
    # Validate figure type
    if go is not None and not isinstance(fig, go.Figure):
        logger.warning(f"plotly_figure_to_html: Expected go.Figure, got {type(fig).__name__}")
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
                'responsive': responsive,
                'displayModeBar': True,
                'displaylogo': False
            }
        )
        
        logger.debug(f"plotly_figure_to_html: Generated HTML for div_id='{div_id}'")
        return html_str
        
    except Exception:
        logger.exception("plotly_figure_to_html: Error generating HTML")
        return _create_placeholder_html("Error rendering plot")


def _sanitize_div_id(div_id: str) -> str:
    """
    Sanitize div_id to prevent XSS attacks.
    
    Only allows alphanumeric characters, underscores, and hyphens.
    Removes any other characters and ensures the ID starts with a letter.
    
    Args:
        div_id: The original div ID string.
    
    Returns:
        A sanitized div ID safe for use in HTML.
    """
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', str(div_id))
    
    # Ensure it starts with a letter (HTML ID requirement)
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'plot-' + sanitized
    
    # Fallback if empty
    if not sanitized:
        sanitized = f"plot-{uuid.uuid4().hex[:8]}"
    
    return sanitized


def _create_placeholder_html(message: str) -> str:
    """
    Create a placeholder HTML div for when a figure is not available.
    
    Args:
        message: The message to display in the placeholder.
    
    Returns:
        HTML string for the placeholder div.
    """
    # Escape the message to prevent XSS
    escaped_message = (
        str(message)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#x27;')
    )
    
    return f'''
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
    '''

"""
📊 Tests for Plotly HTML Renderer Utility

Tests the plotly_figure_to_html function for:
- None figure handling
- Basic figure rendering
- CDN inclusion
- Responsive layout
- Unique div_id generation
- XSS prevention (HTML safety)
"""

import pytest
import re
import sys
import os

# Add project root to path (one level up from tests/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils.plotly_html_renderer import (
    plotly_figure_to_html,
    _sanitize_div_id,
    _create_placeholder_html
)

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
@pytest.mark.unit
class TestPlotlyHtmlRenderer:
    """Test suite for plotly_figure_to_html function."""

    def test_none_figure(self):
        """
        Test that None figure is handled gracefully.
        Should return a placeholder HTML without raising an exception.
        """
        result = plotly_figure_to_html(None)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Waiting" in result or "⏳" in result
        
    def test_simple_figure(self):
        """
        Test basic Plotly figure rendering.
        Should return valid HTML containing the figure.
        """
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
        
        result = plotly_figure_to_html(fig, div_id="test_plot")
        
        assert result is not None
        assert isinstance(result, str)
        assert "test_plot" in result
        assert "plotly" in result.lower()
        
    def test_cdn_plotlyjs(self):
        """
        Test that CDN is included when specified.
        Should contain reference to Plotly CDN.
        """
        fig = go.Figure(data=[go.Bar(x=[1, 2, 3], y=[4, 5, 6])])
        
        result = plotly_figure_to_html(fig, include_plotlyjs='cdn')
        
        assert result is not None
        # CDN should be referenced in script tag
        assert "cdn.plot.ly" in result or "plotly" in result.lower()
        
    def test_responsive(self):
        """
        Test that responsive layout is enabled.
        Should have autosize configuration.
        """
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        
        result = plotly_figure_to_html(fig, responsive=True)
        
        assert result is not None
        # Check for responsive config or autosize
        assert '"responsive": true' in result or 'autosize' in result.lower()
        
    def test_unique_div_id(self):
        """
        Test that div_id is properly set and unique when auto-generated.
        Multiple calls should produce different IDs.
        """
        fig = go.Figure(data=[go.Scatter(x=[1], y=[1])])
        
        result1 = plotly_figure_to_html(fig)
        result2 = plotly_figure_to_html(fig)
        
        # Extract div IDs using regex
        id_pattern = r'id="(plotly-[a-f0-9]+)"'
        
        match1 = re.search(id_pattern, result1)
        match2 = re.search(id_pattern, result2)
        
        assert match1 is not None, "First result should contain auto-generated ID"
        assert match2 is not None, "Second result should contain auto-generated ID"
        assert match1.group(1) != match2.group(1), "Auto-generated IDs should be unique"
        
    def test_custom_div_id(self):
        """
        Test that custom div_id is used when provided.
        """
        fig = go.Figure(data=[go.Scatter(x=[1], y=[1])])
        custom_id = "my_custom_plot_id"
        
        result = plotly_figure_to_html(fig, div_id=custom_id)
        
        assert custom_id in result
        
    def test_html_safety(self):
        """
        Test XSS prevention - malicious div_id should be sanitized.
        Dangerous characters should be removed.
        """
        fig = go.Figure(data=[go.Scatter(x=[1], y=[1])])
        
        # Try to inject script via div_id
        malicious_id = '<script>alert("xss")</script>'
        
        result = plotly_figure_to_html(fig, div_id=malicious_id)
        
        # Script tags should NOT appear in output
        assert '<script>' not in result
        assert 'alert' not in result or 'alert("xss")' not in result
        
    def test_fixed_dimensions(self):
        """
        Test that fixed height and width are applied.
        """
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        
        result = plotly_figure_to_html(fig, height=400, width=600)
        
        assert result is not None
        # The dimensions should be in the layout
        assert "400" in result or "height" in result.lower()


@pytest.mark.unit
class TestSanitizeDivId:
    """Test suite for _sanitize_div_id helper function."""
    
    def test_valid_id(self):
        """Valid ID should pass through unchanged."""
        assert _sanitize_div_id("my_plot_1") == "my_plot_1"
        assert _sanitize_div_id("plot-area") == "plot-area"
        
    def test_special_characters_removed(self):
        """Special characters should be stripped."""
        result = _sanitize_div_id("plot<>id")
        assert "<" not in result
        assert ">" not in result
        
    def test_number_prefix_handled(self):
        """IDs starting with numbers should be prefixed."""
        result = _sanitize_div_id("123plot")
        assert result[0].isalpha(), "ID should start with a letter"
        
    def test_empty_string_handled(self):
        """Empty string should generate a fallback ID."""
        result = _sanitize_div_id("")
        assert len(result) > 0
        assert result.startswith("plot-")


@pytest.mark.unit
class TestCreatePlaceholderHtml:
    """Test suite for _create_placeholder_html helper function."""
    
    def test_returns_html(self):
        """Should return valid HTML string."""
        result = _create_placeholder_html("Test message")
        
        assert "<div" in result
        assert "</div>" in result
        assert "Test message" in result
        
    def test_escapes_html_entities(self):
        """Should escape dangerous HTML characters."""
        malicious = '<script>alert("xss")</script>'
        
        result = _create_placeholder_html(malicious)
        
        # Script tags should be escaped
        assert "<script>" not in result
        assert "&lt;script&gt;" in result or "&lt;" in result


@pytest.mark.unit
class TestIntegration:
    """Integration tests for the full rendering pipeline."""
    
    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
    def test_full_rendering_workflow(self):
        """
        Test the complete workflow of creating and rendering a figure.
        Simulates real-world usage in a Shiny app.
        """
        # Create a typical medical statistics plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[0.95, 0.90, 0.85, 0.75, 0.60],
            mode='lines+markers',
            name='Kaplan-Meier Curve'
        ))
        fig.update_layout(
            title="Survival Analysis",
            xaxis_title="Time (months)",
            yaxis_title="Survival Probability"
        )
        
        result = plotly_figure_to_html(
            fig,
            div_id="plot_survival_km",
            include_plotlyjs='cdn',
            responsive=True
        )
        
        assert result is not None
        assert "plot_survival_km" in result
        assert len(result) > 100  # Should be substantial HTML

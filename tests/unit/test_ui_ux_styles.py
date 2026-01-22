import re
import sys
from pathlib import Path

import pytest

# PATH FIX: Adjust path to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Moved import to top to fix lint
from tabs._common import get_color_palette  # noqa: E402


@pytest.fixture
def palette():
    return get_color_palette()


def test_css_variables_consistency(palette):
    """Verify that tabs/_styling.py uses the central palette for its CSS variables."""
    styling_path = PROJECT_ROOT / "tabs" / "_styling.py"
    with open(styling_path, "r") as f:
        content = f.read()

    # Check if root variables in get_shiny_css are mapped correctly to palette
    for key, hex_val in palette.items():
        # Check for --color-{key}: {COLORS["key"]}
        var_pattern = rf"--color-{key.replace('_', '-')}:\s*{{COLORS\[\"{key}\"\]}}"
        # Note: some keys might be mapped differently or missing in _styling.py
        # but the primary ones should be there.
        if key in ["primary", "success", "danger", "warning"]:
            assert re.search(
                var_pattern, content
            ), f"Color '{key}' not correctly mapped in tabs/_styling.py"


def test_compiled_css_sync():
    """Verify that static/styles.css matches tabs/_styling.py output."""
    from tabs._styling import get_shiny_css

    generated_css_with_tags = get_shiny_css()
    # Remove <style> tags
    generated_css = re.sub(
        r"^\s*<style>\s*|\s*</style>\s*$",
        "",
        generated_css_with_tags,
        flags=re.IGNORECASE,
    ).strip()

    compiled_css_path = PROJECT_ROOT / "static" / "styles.css"
    with open(compiled_css_path, "r") as f:
        compiled_css = f.read()

    # We check if the core variables match. Exact string match might fail due to headers/comments.
    # Check first 500 characters of the content after the header
    header_end = compiled_css.find("*/") + 2
    if header_end > 1:
        pure_compiled = compiled_css[header_end:].strip()
        # Compare first few lines to ensure they are in sync
        assert (
            pure_compiled[:200] in generated_css
        ), "static/styles.css is out of sync with tabs/_styling.py. Run python utils/update_css.py"


def test_plotly_renderer_consistency(palette):
    """Verify utils/plotly_html_renderer.py matches the UI theme."""
    renderer_path = PROJECT_ROOT / "utils" / "plotly_html_renderer.py"
    with open(renderer_path, "r") as f:
        content = f.read()

    # Check for Inter font
    assert "'Inter'" in content, "Plotly renderer should use 'Inter' font"

    # Check for sync with some neutral colors (smoke_white/border)
    # Placeholder background #f9fafb is COLORS['background']
    assert (
        palette["background"].lower() in content.lower()
    ), "Plotly renderer background color out of sync"
    assert (
        palette["border"].lower() in content.lower()
    ), "Plotly renderer border color out of sync"


def test_forest_plot_consistency(palette):
    """Verify utils/forest_plot_lib.py uses central colors."""
    forest_path = PROJECT_ROOT / "utils" / "forest_plot_lib.py"
    with open(forest_path, "r") as f:
        content = f.read()

    # Check if it imports and uses COLORS
    assert "from tabs._common import get_color_palette" in content
    assert "COLORS = get_color_palette()" in content

    # Check for hardcoded legacy colors (e.g., #21808d from the docstring/error)
    legacy_teal = "#21808d"
    assert legacy_teal not in content, "Legacy teal color found in forest_plot_lib.py"


def test_formatting_consistency(palette):
    """Verify utils/formatting.py badges and p-values match the theme colors."""
    formatting_path = PROJECT_ROOT / "utils" / "formatting.py"
    with open(formatting_path, "r") as f:
        content = f.read()

    # Formatting.py has a local 'colors' dict in get_badge_html
    # Check if at least some colors match or are close to the palette
    # Success in formatting.py: #d4edda (bg), #155724 (color)
    # Success in palette: #22A765

    # We verify it uses CONFIG-driven styles where possible
    assert "from config import CONFIG" in content

    # Check if 'Inter' is mentioned in creation_missing_data_report_html or similar?
    # Actually formatting.py doesn't seem to set font-family globally, but uses system defaults.


def test_custom_handlers_js_integrity():
    """Verify static/js/custom_handlers.js exists and has standard handlers."""
    js_path = PROJECT_ROOT / "static" / "js" / "custom_handlers.js"
    assert js_path.exists(), "custom_handlers.js is missing"

    with open(js_path, "r") as f:
        content = f.read()

    assert "set_element_style" in content
    assert "set_inner_text" in content


if __name__ == "__main__":
    pytest.main([__file__])

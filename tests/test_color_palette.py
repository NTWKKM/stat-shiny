"""Test suite for UI Styling System consistency. (Updated to match repository source)

Verifies:
1. Color variables in tabs/_common.py via get_color_palette()
2. CSS Generation logic in tabs/_styling.py (get_shiny_css)
3. Integration between Styling and Common variables
"""

import sys
import pytest
from pathlib import Path

# Add parent directory to path to allow importing from 'tabs'
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_styling_data():
    """Get color palette data from the actual source function."""
    try:
        # ใน tabs/_common.py มีฟังก์ชัน get_color_palette
        from tabs._common import get_color_palette

        palette = get_color_palette()
        return {"colors": palette, "fetcher": get_color_palette}
    except ImportError as e:
        pytest.skip(f"Could not import from tabs._common: {e}")


def test_styling_files_exist():
    """Verify that both styling system files exist."""
    base_path = Path(__file__).parent.parent / "tabs"
    assert (base_path / "_common.py").exists(), "tabs/_common.py is missing"
    assert (base_path / "_styling.py").exists(), "tabs/_styling.py is missing"


def test_essential_ui_colors():
    """Test that all brand and status colors are defined in the palette."""
    data = get_styling_data()
    assert data is not None, "Styling data not available (Import Error)"

    palette = data["colors"]

    # 1. Check Brand Colors (Navy Blue Theme)
    brand_colors = ["primary", "primary_light", "primary_dark", "smoke_white"]
    for color in brand_colors:
        assert color in palette, f"Missing Brand Color: {color}"
        assert palette[color].startswith("#"), f"Invalid HEX format for {color}"

    # 2. Check Status Colors
    status_colors = ["success", "danger", "warning", "info"]
    for color in status_colors:
        assert color in palette, f"Missing Status Color: {color}"

    # 3. Check Neutral/Text Colors
    neutral_colors = ["text", "text_secondary", "background", "surface", "border"]
    for color in neutral_colors:
        assert color in palette, f"Missing Neutral Color: {color}"


def test_color_format_validity():
    """Verify all colors are valid HEX codes."""
    data = get_styling_data()
    palette = data["colors"]

    import re

    # รองรับทั้ง #RGB และ #RRGGBB
    hex_regex = re.compile(r"^#([A-Fa-f0-9]{3}){1,2}$")

    for key, hex_val in palette.items():
        assert hex_regex.match(
            hex_val
        ), f"Color '{key}' has invalid HEX format: {hex_val}"


def test_styling_injector_integration():
    """Test if tabs/_styling.py can correctly generate CSS using colors."""
    try:
        # ใน tabs/_styling.py ใช้ฟังก์ชัน get_shiny_css
        from tabs._styling import get_shiny_css

        # Test that the function exists
        assert callable(get_shiny_css), "get_shiny_css is not a function"

        # Test CSS Generation
        css_content = get_shiny_css()
        assert isinstance(css_content, str), "get_shiny_css should return a string"
        assert "<style>" in css_content, "Output should contain style tags"

        # Verify it pulls the primary color from common.py
        data = get_styling_data()
        primary_hex = data["colors"]["primary"]
        assert (
            primary_hex in css_content
        ), "Generated CSS does not contain the primary color from _common.py"

    except ImportError as e:
        pytest.fail(f"Styling injection test failed due to import error: {e}")


def test_no_hardcoded_old_colors():
    """Check that old color keys are not being used."""
    data = get_styling_data()
    palette = data["colors"]

    # เช็คว่าไม่มี key เก่าที่เลิกใช้ไปแล้ว
    old_keys = ["text_primary", "bg_main"]
    for old_key in old_keys:
        assert (
            old_key not in palette
        ), f"Old key '{old_key}' found, please use the new naming convention"


if __name__ == "__main__":
    print("🎨 Running UI Styling System Tests (Production Ready)...\n")

    test_functions = [
        test_styling_files_exist,
        test_essential_ui_colors,
        test_color_format_validity,
        test_styling_injector_integration,
        test_no_hardcoded_old_colors,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {test_func.__name__}: Error: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"UI Test Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)

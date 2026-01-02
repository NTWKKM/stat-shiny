"""Test suite for UI Styling System consistency.

Verifies:
1. Color variables in tabs/_common.py
2. CSS Injection logic in tabs/_styling.py
3. Integration between Styling and Common variables
4. Essential UI elements (Primary, Secondary, Success, etc.)
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to allow importing from 'tabs'
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_styling_data():
    """Get color palette and styling data."""
    try:
        from tabs._common import DEFAULT_COLORS, get_color_palette
        return {
            "colors": DEFAULT_COLORS,
            "fetcher": get_color_palette
        }
    except ImportError as e:
        print(f"Warning: Could not import from tabs._common: {e}")
        return None

def test_styling_files_exist():
    """Verify that both styling system files exist."""
    base_path = Path(__file__).parent.parent / "tabs"
    assert (base_path / "_common.py").exists(), "tabs/_common.py is missing"
    assert (base_path / "_styling.py").exists(), "tabs/_styling.py is missing"

def test_essential_ui_colors():
    """Test that all brand and status colors are defined in DEFAULT_COLORS."""
    data = get_styling_data()
    assert data is not None, "Styling data not available"
    
    palette = data["colors"]
    
    # 1. Check Brand Colors (Must match UI Styling Guide)
    brand_colors = ['primary', 'primary_light', 'primary_dark', 'secondary']
    for color in brand_colors:
        assert color in palette, f"Missing Brand Color: {color}"
        assert palette[color].startswith('#'), f"Invalid format for {color}"

    # 2. Check Status Colors
    status_colors = ['success', 'danger', 'warning', 'info']
    for color in status_colors:
        assert color in palette, f"Missing Status Color: {color}"

    # 3. Check Neutral/Text Colors
    neutral_colors = ['text', 'text_secondary', 'background', 'surface', 'border']
    for color in neutral_colors:
        assert color in palette, f"Missing Neutral/Text Color: {color}"

def test_color_format_validity():
    """Verify all colors are valid 7-character HEX codes (#RRGGBB)."""
    data = get_styling_data()
    palette = data["colors"]
    
    import re
    hex_regex = re.compile(r'^#[0-9a-fA-F]{6}$')
    
    for key, hex_val in palette.items():
        assert hex_regex.match(hex_val), f"Color '{key}' has invalid HEX format: {hex_val}"

def test_styling_injector_integration():
    """Test if tabs/_styling.py can correctly import and use colors."""
    try:
        from tabs._styling import apply_custom_styling
        import streamlit as st
        
        # Test that the function exists and is callable
        assert callable(apply_custom_styling), "apply_custom_styling is not a function"
        
        # Verify it uses DEFAULT_COLORS internally (check source code briefly)
        import inspect
        source = inspect.getsource(apply_custom_styling)
        assert "DEFAULT_COLORS" in source or "get_color_palette" in source, \
            "apply_custom_styling might not be linked to the common color palette"
            
    except (ImportError, Exception) as e:
        assert False, f"Styling injection test failed: {str(e)}"

def test_no_hardcoded_old_colors():
    """Check that old color keys are not being used in key files."""
    data = get_styling_data()
    palette = data["colors"]
    
    # Example: 'text_primary' was an old key, now it should be 'text'
    assert 'text_primary' not in palette, "Old key 'text_primary' found, please use 'text'"

if __name__ == "__main__":
    print("🎨 Running UI Styling System Tests...\n")

    test_functions = [
        test_styling_files_exist,
        test_essential_ui_colors,
        test_color_format_validity,
        test_styling_injector_integration,
        test_no_hardcoded_old_colors
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

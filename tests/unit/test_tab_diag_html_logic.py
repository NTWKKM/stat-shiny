import pytest
from tabs.tab_diag import _create_status_elements


def test_create_status_elements_plain_text():
    """Test with simple text message."""
    msg = "Analysis completed successfully."
    elements = _create_status_elements(msg)

    # Expect single element wrapped in <p>
    assert len(elements) == 1
    assert elements[0]["type"] == "text"
    assert elements[0]["data"] == "Analysis completed successfully."


def test_create_status_elements_with_html_block():
    """Test with message containing the specific HTML block."""
    plain = "Note: Analysis completed."
    block = (
        "<div class='alert alert-light border mt-3'><h5>Likelihood Ratios</h5>...</div>"
    )
    msg = f"{plain}{block}"

    elements = _create_status_elements(msg)

    # Expect split into two elements
    assert len(elements) == 2

    # 1. Plain text wrapped in <p>
    assert elements[0]["type"] == "text"
    assert elements[0]["data"] == plain

    # 2. HTML block as raw HTML (no extra wrapper)
    assert elements[1]["type"] == "html"
    assert elements[1]["data"] == block


def test_create_status_elements_empty():
    """Test with empty message."""
    msg = ""
    elements = _create_status_elements(msg)
    # Should probably be empty or single empty p
    assert len(elements) == 1
    assert elements[0]["data"] == ""

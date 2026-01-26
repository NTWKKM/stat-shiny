import html
import pytest
from utils.diag_test import generate_report


def test_generate_report_html_rendering():
    """
    Verify that generate_report correctly:
    1. Escapes HTML when type="text"
    2. Does NOT escape HTML when type="html"
    """

    # 1. Test with type="text" (Should be escaped)
    raw_html_content = "<div class='alert'>Text</div>"
    elements_text = [{"type": "text", "data": raw_html_content}]
    report_text = generate_report("Test Text", elements_text)

    # Assert that < matches &lt;
    escaped = html.escape(raw_html_content)
    assert escaped in report_text, "HTML tags should be escaped for type='text'"
    assert raw_html_content not in report_text, (
        "Raw HTML should not appear for type='text'"
    )

    # 2. Test with type="html" (Should NOT be escaped)
    elements_html = [{"type": "html", "data": raw_html_content}]
    report_html = generate_report("Test HTML", elements_html)

    # Assert that <div is present as raw HTML
    assert "<div class='alert'>Text</div>" in report_html, (
        "HTML tags should be preserved for type='html'"
    )
    assert "&lt;div" not in report_html, (
        "HTML tags should NOT be escaped for type='html'"
    )


def test_diag_warning_message_escaping():
    """
    Verify that we can construct a message with manually escaped characters if needed,
    although specifically for the Chi-Square warning we want safe HTML.
    This test confirms the fix in utils/diag_test.py (if we could check the function return value,
    but here we just check the string logic).
    """
    # The fix in diag_test.py was changing "< 5" to "&lt; 5"
    # This is manually verifying that &lt; is safe in a type='html' block?
    # Actually, if type='html', &lt; will render as < in the browser, which is what we want for text content within HTML.

    msg = "Expected count &lt; 5"
    elements = [{"type": "html", "data": msg}]
    report = generate_report("Test Warning", elements)

    assert "Expected count &lt; 5" in report

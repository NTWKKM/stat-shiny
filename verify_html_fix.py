import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from utils.diag_test import generate_report


def verify_report_rendering():
    # Simulate the status text that caused issues
    status_text = "Note: <div class='alert alert-light border mt-3'> <h5>💡 Interpreting Likelihood Ratios (EBM Standards)</h5> ... </div>"

    # 1. Test with type="text" (Original problematic behavior)
    # This should be escaped in the output
    elements_text = [{"type": "text", "data": status_text}]
    html_text = generate_report("Test Report (Text)", elements_text)

    if "&lt;div" in html_text and "<div" not in html_text.replace("&lt;", "<"):
        # Note: replace check is to differentiate from the doctype/body tags
        # Actually simplest check:
        # If type is text, <div should become &lt;div
        print("[PASS] type='text' correctly escapes HTML tags")
    else:
        # It's okay if it doesn't pass this if generate_report was changed,
        # but we didn't change generate_report, we changed the caller.
        if "&lt;div" in html_text:
            print("[PASS] type='text' correctly escapes HTML tags")
        else:
            print("[FAIL] type='text' did NOT escape HTML tags (unexpected)")

    # 2. Test with type="html" (New fixed behavior)
    # This should NOT be escaped
    elements_html = [{"type": "html", "data": f"<p>{status_text}</p>"}]
    html_html = generate_report("Test Report (HTML)", elements_html)

    if "<div class='alert" in html_html:
        print("[PASS] type='html' correctly renders raw HTML tags")
    else:
        print(
            f"[FAIL] type='html' failed to render raw HTML tags. Output snippet: {html_html[:500]}..."
        )

    # 3. Verify the specific fix in calculate_chi2 (Mental check mainly, but we can verify generate_report handles it)
    # We changed the caller in tab_diag to use type='html'


if __name__ == "__main__":
    verify_report_rendering()

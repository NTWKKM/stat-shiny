# RabbitAI Report

In `@tests/unit/test_tab_diag_html_logic.py` around lines 5 - 44, The tests assume
HTML-wrapped results but the implementation returns escaped plain text entries
with type "text"; update the assertions in the three tests to match
_create_status_elements: check elements[*]["type"] == "text" for plain segments,
assert elements[0]["data"] equals the unwrapped plain string (e.g. "Analysis
completed successfully." and "Note: Analysis completed." respectively) rather
than "<p>...</p>", and for the empty message assert elements[0]["data"] == ""
instead of "<p></p>" so the tests reflect escaped/text output.

⚠️ Potential issue | 🟡 Minor

Align expectations with escaped text output for safety.

If _create_status_elements returns type: "text" for plain content, these assertions should reflect unwrapped text data and type: "text".

✏️ Proposed update

- assert elements[0]["type"] == "html"
- assert elements[0]["data"] == "<p>Analysis completed successfully.</p>"

+ assert elements[0]["type"] == "text"
- assert elements[0]["data"] == "Analysis completed successfully."
@@

- assert elements[0]["type"] == "html"
- assert elements[0]["data"] == f"<p>{plain}</p>"

+ assert elements[0]["type"] == "text"
- assert elements[0]["data"] == plain
@@

- assert elements[0]["data"] == "<p></p>"

+ assert elements[0]["data"] == ""

# RabbitAI Report

In `@tabs/tab_diag.py` around lines 680 - 683, status_text (aka msg from
utils/diag_test.calculate_chi2) may contain HTML blocks so wrapping it inside a
single "<p>" creates invalid markup; change the construction that currently
creates {"type":"html","data": f"<p>{status_text}</p>"} to split the plain
status line (escaped text) from any LR/HTML block: emit one element for the
escaped status string (use a non-HTML/text element or an HTML element with only
escaped text) and, if the msg contains the LR HTML block, append a separate
{"type":"html","data": html_block} element for that HTML content; locate the use
of status_text in tabs/tab_diag.py and adjust the code that builds the list of
message elements accordingly.

⚠️ Potential issue | 🟡 Minor

Avoid wrapping HTML-rich status_text inside <p>
msg can now include the LR explanation HTML block (see utils/diag_test.calculate_chi2), so wrapping it in <p> creates invalid markup and may break layout. Consider splitting the status line (as escaped text) from the LR HTML block and append the HTML as a separate "html" element instead of embedding both in one <p>

-----------------------------

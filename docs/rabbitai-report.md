# RabbitAI Report

In @.agent/workflows/format.md at line 16, The Markdown code block in the
workflow doc has the closing triple backticks on the same line as the command,
breaking the block; update the snippet so the command string ".venv\Scripts\ruff
format ." (the existing code content) remains on its own line and the
terminating ``` triple backticks are placed on the following new line to
properly close the code block.

⚠️ Potential issue | 🔴 Critical

Fix Markdown syntax error: missing newline before closing backticks.

The closing triple backticks must be on a new line to properly terminate the code block.

📝 Proposed fix

# Windows

-.venv\Scripts\ruff format .```
+.venv\Scripts\ruff format .
+```
------------------------------

In `@README.md` around lines 16 - 21, Replace the incorrect justification about
"modern f-string syntax" in the README with a statement that the codebase
requires Python 3.12+ because it uses PEP 695 features (e.g., type aliases/type
parameters like `type X = Literal[...]`); specifically update the descriptive
text at the two occurrences referenced (the paragraph around the Python
requirement and the explanatory note near line 246) to mention PEP 695 type
aliases/type parameters as the reason for requiring Python 3.12+, and ensure the
table entry still shows "Python ≥ 3.12".

⚠️ Potential issue | 🟠 Major

🧩 Analysis chain

🏁 Script executed:

find . -maxdepth 2 -type f \( -name "pyproject.toml" -o -name "setup.py" -o -name "setup.cfg" -o -name "requirements*.txt" \) | head -20
Repository: NTWKKM/stat-shiny

Length of output: 120

🏁 Script executed:

cat README.md | sed -n '240,250p'
Repository: NTWKKM/stat-shiny

Length of output: 378

🏁 Script executed:

rg -n --type=py "type\s+\w+\s*=" . | head -20
Repository: NTWKKM/stat-shiny

Length of output: 250

🏁 Script executed:

rg -n --type=py "def\s+\w+\[.*\]\s*\(" . | head -20
Repository: NTWKKM/stat-shiny

Length of output: 43

🏁 Script executed:

rg -n --type=py "class\s+\w+\[.*\]\s*:" . | head -20
Repository: NTWKKM/stat-shiny

Length of output: 43

🏁 Script executed:

rg -n "python" README.md -i
Repository: NTWKKM/stat-shiny

Length of output: 147

🏁 Script executed:

cat pyproject.toml | grep -A 5 -B 5 "python"
Repository: NTWKKM/stat-shiny

Length of output: 356

🏁 Script executed:

cat requirements.txt
Repository: NTWKKM/stat-shiny

Length of output: 2930

🏁 Script executed:

cat requirements-prod.txt
Repository: NTWKKM/stat-shiny

Length of output: 1000

Update Python 3.12+ justification in README.

The codebase uses PEP 695 type aliases (type X = Literal[...]), which require Python 3.12+. However, the README cites "modern f-string syntax" as the reason—f-strings have been available since Python 3.6 and are not a valid justification. Update lines 16 and 246 to cite PEP 695 type parameters/aliases instead
--------------------------------------------

In `@README.md` around lines 135 - 139, The README references
utils/data_quality.py but that module is missing from the repo; either add the
missing file (implementing the Data Health Report utilities referenced) or
update the README to point to the correct module/path or remove the mention.
Locate references to utils/data_quality.py in README.md and either (a) create
and commit the utils/data_quality.py file implementing the described functions
(missing-data reporting, non-standard numeric detection, categorical integrity
checks) or (b) change the README to reference the actual existing file or
example script that implements those features, ensuring the README’s bullet
points (Comprehensive Data Control, Data Health Report, Missing Data,
Non-standard Numeric, Categorical Integrity) map to real code symbols. Ensure
the README and repository are consistent before merging.

⚠️ Potential issue | 🟡 Minor

Docs reference a file that doesn’t appear in the tree.

utils/data_quality.py is mentioned, but the repository tree shown doesn’t list it. Please confirm the file exists or update the reference
--------------------------------------------

In `@tabs/tab_agreement.py` at line 25, Remove the dead assignment to COLORS from
get_color_palette() — the variable COLORS is never used in this module, so
delete the line "COLORS = get_color_palette()" and, if get_color_palette is only
imported for this unused assignment, also remove the unused import to avoid dead
code and unnecessary dependencies; ensure no other symbol in this file
references COLORS or get_color_palette before deleting.

🧹 Nitpick | 🔵 Trivial

Unused variable COLORS.

COLORS is assigned from get_color_palette() but never referenced in this module. Consider removing it to reduce unnecessary imports and dead code.

♻️ Suggested fix
-from tabs._common import (

- get_color_palette,
- select_variable_by_keyword,
-)
+from tabs._common import select_variable_by_keyword
 from utils import diag_test
...
-COLORS = get_color_palette()

-----------------------------------------
In `@tabs/tab_agreement.py` around lines 213 - 218, The bare except in the block
around input.radio_source() hides real errors; change it to catch the specific
expected exception (e.g., AttributeError or RuntimeError depending on how input
is initialized) and/or log the exception at debug level before falling back to
return df.get(); update the try/except that surrounds input.radio_source() so it
only handles the anticipated failure and retains the fallback to df.get() while
preserving visibility of unexpected errors from input.radio_source(),
referencing input.radio_source(), df_matched.get(), and df.get() in your
changes.

🧹 Nitpick | 🔵 Trivial

Broad exception handling may hide real errors.

The bare except Exception: pass silently swallows all errors when accessing input.radio_source(). While this provides a safe fallback, it could mask legitimate issues (e.g., input not yet initialized). Consider catching only the expected exception type or logging at debug level.

♻️ Suggested improvement
         if is_matched.get():
             try:
                 if input.radio_source() == "matched":
                     return df_matched.get()

-            except Exception:
-                pass

+            except (AttributeError, KeyError):
-                # Input not yet initialized, fall back to original
-                pass
         return df.get()

---------------------------------------
In `@tabs/tab_agreement.py` around lines 349 - 350, The error string from
calculate_kappa is being inserted directly into the HTML via
kappa_html.set(f"<div class='alert alert-danger'>{err}</div>") which can expose
user data to XSS; instead HTML-escape the err value before embedding (use a safe
escaping utility such as html.escape or MarkupSafe.escape) and then pass the
escaped string into kappa_html.set so the error message is rendered safely;
locate the branch that checks "if err:" and update the construction of the alert
HTML to use the escaped error value.

⚠️ Potential issue | 🟡 Minor

Error message from calculate_kappa is not HTML-escaped.

The err string may contain column names from user data. Embed it safely to prevent potential XSS, consistent with the exception handling below.

🐛 Proposed fix
             if err:

-                kappa_html.set(f"<div class='alert alert-danger'>{err}</div>")

+                kappa_html.set(f"<div class='alert alert-danger'>{_html.escape(str(err))}</div>")

---------------------------------------
In `@tabs/tab_agreement.py` around lines 418 - 421, The Bland-Altman error string
from stats_res is inserted into HTML without escaping; update the code around
ba_html.set(...) to HTML-escape stats_res['error'] (same approach used in the
Kappa error handling) before embedding it in the <div>, e.g., call the project’s
HTML-escape utility (or python's html.escape) on stats_res['error'] and pass the
escaped value to ba_html.set to prevent injection.

⚠️ Potential issue | 🟡 Minor

Error message from Bland-Altman result is not HTML-escaped.

Similar to the Kappa error handling, the error string should be escaped before embedding in HTML

🐛 Proposed fix
             if "error" in stats_res:
                 ba_html.set(

-                    f"<div class='alert alert-danger'>{stats_res['error']}</div>"

+                    f"<div class='alert alert-danger'>{_html.escape(str(stats_res['error']))}</div>"
                 )

---------------------------------------
In `@tabs/tab_agreement.py` around lines 494 - 495, The error message assigned to
the ICc display is not HTML-escaped; update the block that calls icc_html.set
(the one using the err variable returned from calculate_icc) to escape err
before embedding it into the HTML string (e.g., use Python's html.escape on err
or an equivalent HTML-escaping utility), guarding for None so you don't pass a
non-string, and then use the escaped value in the f-string to avoid injecting
raw HTML.

⚠️ Potential issue | 🟡 Minor

Error message from calculate_icc is not HTML-escaped.

For consistency and defense-in-depth, escape the error string.

🐛 Proposed fix
             if err:

-                icc_html.set(f"<div class='alert alert-danger'>{err}</div>")

+                icc_html.set(f"<div class='alert alert-danger'>{_html.escape(str(err))}</div>")

---------------------------------------
In `@tabs/tab_agreement.py` around lines 563 - 568, The filename lambda in the
render.download decorator can crash if input.icc_vars() returns None or an empty
sequence; update the btn_dl_icc_report filename generation to defensively read
vars = input.icc_vars() or [] (or use a safe getter), then take the first three
items safely (e.g., vars[:3] or use defaults when missing), join them, pass
through_safe_filename_part, and fallback to a fixed token like "untitled" when
the resulting name is empty so the render.download filename lambda never errors.

🧹 Nitpick | 🔵 Trivial

Defensive handling for filename generation.

If input.icc_vars() returns None or an empty sequence at download time, the slice operation will fail. Add a fallback for robustness.

♻️ Suggested improvement
     `@render.download`(

-        filename=lambda: f"icc_report_{_safe_filename_part('_'.join(input.icc_vars()[:3]))}.html"

+        filename=lambda: f"icc_report_{_safe_filename_part('_'.join((input.icc_vars() or [])[:3]) or 'analysis')}.html"
     )
     def btn_dl_icc_report():
         req(icc_html.get())
         yield icc_html.get()

--------------------------------------
In `@tabs/tab_agreement.py` around lines 589 - 599, Replace the inline
ui.div/ui.tags.small in the out_icc_validation render function with the shared
create_error_alert utility to match Kappa and Bland-Altman validations: in
out_icc_validation (which checks input.icc_vars()), when there are fewer than 2
columns return create_error_alert with the same message ("Please select at least
2 rater/method columns.") so styling and behavior are consistent across
validations; keep the conditional logic and return None otherwise.

🧹 Nitpick | 🔵 Trivial

Consider using create_error_alert for consistency.

The ICC validation uses an inline ui.div with text-warning styling, while Kappa and Bland-Altman validation use create_error_alert. Consider using a consistent approach across all validations for uniform UX.

♻️ Suggested improvement
     `@render.ui`
     def out_icc_validation():
         cols = input.icc_vars()
         if cols and len(cols) < 2:

-            return ui.div(
-                ui.tags.small(
-                    "Please select at least 2 rater/method columns.",
-                    class_="text-warning",
-                )
-            )

+            return create_error_alert(
-                "Please select at least 2 rater/method columns."
-            )
         return None

# RabbitAI Report

In @.agent/rules/architecture.md at line 25, Update the markdown image reference
line "![Navigation Workflow](./assets/navigation_sequence.png)" in
.agent/rules/architecture.md so it points to the actual image location by
replacing the target with "docs/assets/navigation_sequence.png"; ensure the alt
text and surrounding markup remain unchanged.

🧩 Analysis chain
🏁 Script executed:

fd "navigation_sequence.png" --type f
Repository: NTWKKM/stat-shiny

Length of output: 96

Fix the image reference path in the architecture documentation.

The document references ./assets/navigation_sequence.png, but the image is located at docs/assets/navigation_sequence.png. The relative path will not resolve correctly from .agent/rules/architecture.md. Update the reference to the correct path pointing to the actual image location
---------------------------------------------

In `@docs/DEEP-OPTIMIZATION-PLAN.md` around lines 496 - 525, The fenced code block
showing the CURRENT/TARGET plan lacks a language specifier; update the opening
backticks for that block (the triple-backtick block containing "CURRENT
(Monolithic):" and "TARGET (Modular):") to include a language identifier (e.g.,
text or txt) so syntax highlighting/accessibility works (replace ``` with

```text).

🧹 Nitpick | 🔵 Trivial

Add language specifier to fenced code block.

This code block lacks a language identifier, which affects syntax highlighting and accessibility.

♻️ Suggested fix
-```
+```text
 CURRENT (Monolithic):
   tabs/tab_core_regression.py (3,700 lines
------------------------------------------------
In `@docs/statiomed-detailed-roadmap.md` around lines 12 - 16, The TOC anchor
fragments (e.g., "Executive Summary", "Current System Architecture & Health",
"Verified Component Status", "Phase 2 - Deep Dive Implementation Plan", "Phase 3
- Advanced Features Roadmap") do not match the actual generated heading IDs
causing markdownlint errors; update each link to the correct slugified ID
(lowercase, spaces to hyphens, remove/translate punctuation/emoji) or add
explicit anchor tags/HTML anchors immediately before the corresponding headings
so the TOC links resolve correctly and lint warnings disappear.

⚠️ Potential issue | 🟡 Minor

Fix TOC anchor fragments so links resolve.

markdownlint flags the fragments; update the TOC to match the actual heading IDs (numbers/emoji/punctuation often alter anchors), or add explicit anchors in the headings
--------------------------------------------------
In `@docs/statiomed-detailed-roadmap.md` around lines 58 - 105, The fenced ASCII
diagram blocks (the triple-backtick sections showing "Shiny Web Layer" and the
module lists including app.py, tab_home.py, data_cleaning.py, etc.) lack
language identifiers; update each fenced block in
docs/statiomed-detailed-roadmap.md to include a language tag (e.g., ```text or
```ascii) immediately after the opening ``` so Markdown linting/renderers treat
them consistently, and apply the same change to the other similar fenced blocks
referenced in the review (the additional large ASCII diagrams elsewhere in the
file).

⚠️ Potential issue | 🟡 Minor

Add language identifiers to fenced blocks.

This improves Markdown linting and rendering consistency.

✅ Example fix (apply to each fenced block)
-```
+```text
Also applies to: 109-137, 337-363, 387-398

--------------------------------------------------
In `@tabs/tab_corr.py` around lines 70 - 74, The coefficient key used in
tabs/tab_corr.py is mismatched for Kendall: update out_corr_table() and
btn_dl_corr() (and the other similar lookup around lines 120-124) to first try
the new key "Coefficient (r/rho/tau)" returned by utils/correlation.py and, if
missing, fall back to the legacy key "Coefficient (r)"; implement this
conditional lookup when accessing the coefficient column in both display and
download code paths so a KeyError is avoided for Kendall and other methods.

⚠️ Potential issue | 🔴 Critical

🧩 Analysis chain
🏁 Script executed:

find . -name "correlation.py" -o -name "*corr*.py" | head -20
Repository: NTWKKM/stat-shiny

Length of output: 227

🏁 Script executed:

git ls-files | grep -E "(correlation|corr)" | head -20
Repository: NTWKKM/stat-shiny

Length of output: 217

🏁 Script executed:

rg -n "Coefficient" utils/correlation.py
Repository: NTWKKM/stat-shiny

Length of output: 110

🏁 Script executed:

rg -n "Coefficient" tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 338

🏁 Script executed:

sed -n '445,450p' utils/correlation.py
Repository: NTWKKM/stat-shiny

Length of output: 257

🏁 Script executed:

sed -n '68,75p' tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 439

🏁 Script executed:

sed -n '450,470p' tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 776

🏁 Script executed:

sed -n '520,530p' tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 296

🏁 Script executed:

rg -n "Coefficient \(r\)" tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 201

🏁 Script executed:

cat -n utils/correlation.py | sed -n '445,450p'
Repository: NTWKKM/stat-shiny

Length of output: 299

🏁 Script executed:

wc -l tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 81

🏁 Script executed:

sed -n '440,480p' utils/correlation.py
Repository: NTWKKM/stat-shiny

Length of output: 1482

🏁 Script executed:

sed -n '455,475p' tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 739

🏁 Script executed:

sed -n '520,545p' tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 822

🏁 Script executed:

sed -n '120,130p' tabs/tab_corr.py
Repository: NTWKKM/stat-shiny

Length of output: 524

🏁 Script executed:

rg -n "try|except" tabs/tab_corr.py | head -20
Repository: NTWKKM/stat-shiny

Length of output: 140

Fix coefficient key mismatch that breaks Kendall correlation

utils/correlation.py returns "Coefficient (r/rho/tau)" for all methods (line 447), but tabs/tab_corr.py hardcodes lookups for "Coefficient (r)" at lines 463 and 525. Selecting Kendall will raise a KeyError when displaying or downloading results. Implement a fallback that checks for the new key first, then falls back to the old one, and apply it consistently across both out_corr_table() and btn_dl_corr() functions.

🔧 Suggested fallback for coefficient key

@@ def out_corr_table():
-        display_data = {
-            "Metric": [
-                "Method",
-                "Correlation Coefficient (r)",
+        coef_key = (
+            "Coefficient (r/rho/tau)"
+            if "Coefficient (r/rho/tau)" in stats
+            else "Coefficient (r)"
+        )
+        coef_val = stats.get(coef_key)
+        display_data = {
+            "Metric": [
+                "Method",
+                "Correlation Coefficient",
@@
-                f"{stats['Coefficient (r)']:.4f}",
+                f"{coef_val:.4f}" if isinstance(coef_val, (int, float)) else str(coef_val),
@@ def btn_dl_corr():
-        for key in [
-            "Method",
-            "Coefficient (r)",
+        coef_key = (
+            "Coefficient (r/rho/tau)"
+            if "Coefficient (r/rho/tau)" in stats
+            else "Coefficient (r)"
+        )
+        for key in [
+            "Method",
+            coef_key,
---------------------------------------
In `@tests/benchmarks/r_scripts/test_firth.R` around lines 8 - 10, The test
currently auto-installs missing packages using install.packages with variables
required_packages and new_packages, which can fail in CI; instead remove the
runtime install step and either (a) document these packages in a project
DESCRIPTION (or similar CI setup) so CI installs them before tests run, or (b)
replace the install logic with a CI-friendly check that fails the test with a
clear message if packages in required_packages are missing (using
installed.packages() and new_packages) so CI/shell setup can install them;
reference the required_packages, new_packages and install.packages usage to
locate and update the code.

🧹 Nitpick | 🔵 Trivial

Consider CI environment implications for auto-installing packages.

Auto-installing packages (install.packages) may fail or behave unexpectedly in CI environments without a configured CRAN mirror or with restricted network access. Consider documenting the required packages in a separate DESCRIPTION file or adding a CI setup step
------------------------------------
In `@utils/correlation.py` around lines 445 - 457, The results dict key was
renamed to "Coefficient (r/rho/tau)" causing KeyError where the old key
"Coefficient (r)" is accessed; update all references in tabs/tab_corr.py (e.g.,
the formatted access f"{stats['Coefficient (r)']:.4f}" and the loop over keys
that includes "Coefficient (r)") to use the new key "Coefficient (r/rho/tau)";
alternatively, replace string literals with a single constant (e.g., COEFF_KEY)
used both in utils/correlation.py (where results is built) and in
tabs/tab_corr.py to avoid future mismatches.

⚠️ Potential issue | 🔴 Critical

🧩 Analysis chain
🏁 Script executed:

# Search for usages of the old key name "Coefficient (r)"
rg -n "Coefficient \(r\)" --type py
Repository: NTWKKM/stat-shiny

Length of output: 252

🏁 Script executed:

# Search for usages and imports of calculate_correlation function
rg -n "calculate_correlation" --type py -A 2
Repository: NTWKKM/stat-shiny

Length of output: 2024

🏁 Script executed:

# Check the actual function definition to understand its scope
rg -n "def calculate_correlation" --type py -A 10
Repository: NTWKKM/stat-shiny

Length of output: 695

🏁 Script executed:

# Look at the context around line 463 in tabs/tab_corr.py where the old key is accessed
cat -n tabs/tab_corr.py | sed -n '450,470p'
Repository: NTWKKM/stat-shiny

Length of output: 923

🏁 Script executed:

# Also check line 525
cat -n tabs/tab_corr.py | sed -n '515,535p'
Repository: NTWKKM/stat-shiny

Length of output: 796

🏁 Script executed:

# Check if test_corr_cleaning.py or other integration tests call the display code path
rg -n "display_data\|elements\|Statistical Results" tests/ --type py
Repository: NTWKKM/stat-shiny

Length of output: 43

🏁 Script executed:

# Check what line 364 in tabs/tab_corr.py does after calling calculate_correlation
cat -n tabs/tab_corr.py | sed -n '360,380p'
Repository: NTWKKM/stat-shiny

Length of output: 1003

Unresolved breaking change: production code will crash with KeyError.

The rename from "Coefficient (r)" to "Coefficient (r/rho/tau)" breaks production code in tabs/tab_corr.py at lines 463 and 525, where the old key name is accessed:

Line 463: f"{stats['Coefficient (r)']:.4f}"
Lines 525–532: Loop iterates over keys including "Coefficient (r)" and accesses stats[key]
These code paths are executed when users interact with the correlation analysis feature and will raise KeyError at runtime. Update all references to use the new key name "Coefficient (r/rho/tau)".

----------------------------------
In `@utils/diag_test.py` around lines 1180 - 1185, Add validation that input
scores are actual probabilities before clipping in the calibration plot: check
the array/column referred to as y_score (or the parameter from score_col) for
values outside [0,1], and if any exist emit a warning via the module logger (or
raise a clear ValueError) instead of silently clipping; keep the current
clipping to avoid errors but ensure the warning mentions that calibration_curve
(from sklearn.calibration) expects probabilities and that raw scores/logits
should be converted via a sigmoid or probability calibration first; also update
the function docstring to state that score_col must contain predicted
probabilities.

⚠️ Potential issue | 🟡 Minor

🧩 Analysis chain
🏁 Script executed:

#!/bin/bash
# Get the file and examine the analyze_roc function
cd /tmp && find . -name "diag_test.py" -o -name "utils" -type d 2>/dev/null | head -5
# If not found in /tmp, search from root
rg -l "diag_test.py" --type=py 2>/dev/null | head -3
Repository: NTWKKM/stat-shiny

Length of output: 43

🏁 Script executed:

#!/bin/bash
# Search for the analyze_roc function definition and its docstring
rg -A 50 "def analyze_roc" --type=py -n
Repository: NTWKKM/stat-shiny

Length of output: 3077

🏁 Script executed:

#!/bin/bash
# Check the context around lines 1180-1185 in diag_test.py
rg -B 20 -A 5 "calibration_curve" --type=py -n | head -100
Repository: NTWKKM/stat-shiny

Length of output: 1909

🏁 Script executed:

#!/bin/bash
# Get full docstring and more context
sed -n '1071,1150p' utils/diag_test.py
Repository: NTWKKM/stat-shiny

Length of output: 2801

🏁 Script executed:

#!/bin/bash
# Search for any validation or documentation about score ranges/types
rg "score" utils/diag_test.py -B2 -A2 -n | grep -E "(probability|calibr|range|clip|validate)" | head -30
Repository: NTWKKM/stat-shiny

Length of output: 263

🏁 Script executed:

#!/bin/bash
# Check if there are tests or usage examples that show what scores are expected
rg "analyze_roc" --type=py -B2 -A5 | head -100
Repository: NTWKKM/stat-shiny

Length of output: 6309

🏁 Script executed:

#!/bin/bash
# Look for any comments or documentation about what y_score represents
sed -n '1100,1200p' utils/diag_test.py | grep -E "(#|score|probability)"
Repository: NTWKKM/stat-shiny

Length of output: 1241

🏁 Script executed:

#!/bin/bash
# Look for how analyze_roc is called in the application
rg "analyze_roc" --type=py -B3 -A3 | grep -v test | head -80
Repository: NTWKKM/stat-shiny

Length of output: 1146

🏁 Script executed:

#!/bin/bash
# Check if there's any preprocessing or documentation about score ranges
rg "score.*range|probability|calibr" --type=py -i | grep -v test | head -30
Repository: NTWKKM/stat-shiny

Length of output: 2161

🏁 Script executed:

#!/bin/bash
# Check the sklearn documentation comments or any function parameters that hint at expected input
sed -n '1063,1076p' utils/diag_test.py && echo "---" && rg "score_col" utils/diag_test.py -B2 -A2 | head -40
Repository: NTWKKM/stat-shiny

Length of output: 1195

🏁 Script executed:

#!/bin/bash
# Look at the entire calibration section to see if clipping causes any actual issues
sed -n '1174,1210p' utils/diag_test.py
Repository: NTWKKM/stat-shiny

Length of output: 1330

🏁 Script executed:

#!/bin/bash
# Check if there's any validation that y_score is in [0,1] range
sed -n '1100,1160p' utils/diag_test.py | grep -E "(clip|range|0.*1|probability)"
Repository: NTWKKM/stat-shiny

Length of output: 43

🏁 Script executed:

#!/bin/bash
# Check what the UI says about score input
rg "Prediction.*Probability|Score.*Probability" tabs/tab_diag.py -B3 -A3
Repository: NTWKKM/stat-shiny

Length of output: 243

🏁 Script executed:

#!/bin/bash
# Look for any documentation or comments about expected score values in the codebase
rg "score.*probability|probability.*score" --type=py -i -B1 -A1 | grep -v test | head -40
Repository: NTWKKM/stat-shiny

Length of output: 587

Add validation or warning for non-probability score inputs to calibration plot.

The code clips y_score to [0, 1] only when computing the calibration curve, but applies no validation that the input scores actually represent probabilities. If score_col contains raw model outputs (logits, unbounded scores, etc.) rather than probabilities, the calibration plot will be misleading since clipping distorts the score distribution. The ROC and PR curves use raw scores (correct), but the asymmetric treatment creates confusion.

Given the UI explicitly labels this as "Prediction/Score (Probability):", add either:

A validation check that logs a warning if scores fall outside [0, 1], or
Documentation in the function docstring clarifying that scores must represent predicted probabilities.
---------------------------------------

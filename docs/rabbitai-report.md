# RabbitAI Report

In `@docs/DEEP-OPTIMIZATION-PLAN.md` around lines 38 - 41, Replace the bolded
section titles with proper Markdown headings (e.g., change "**Implementation
Plan:**" to an appropriate level heading like "### Implementation Plan" and
"**A. R Benchmark Generation (Test 1 of 3)**" to "### A. R Benchmark Generation
(Test 1 of 3)"); update any other similarly bolded section titles in the file
(the other occurrences of bolded headings noted in the review) to use consistent
heading syntax so MD036 is satisfied and document structure/navigation is
preserved.

🧹 Nitpick | 🔵 Trivial

Use proper headings instead of bold text for section titles.

These lines are used as headings but formatted with emphasis, which breaks structure and triggers MD036. Convert them to ### (or appropriate level) headings for readability and navigation.

♻️ Proposed fix

-**A. R Benchmark Generation (Test 1 of 3)**
+### A. R Benchmark Generation (Test 1 of 3)

-**B. Python Validation (Test 2 of 3)**
+### B. Python Validation (Test 2 of 3)

-**C. Integration Test (Test 3 of 3)**
+### C. Integration Test (Test 3 of 3)

--------------------------
In `@docs/DEEP-OPTIMIZATION-PLAN.md` around lines 653 - 681, Several fenced code
blocks under the milestone checklists (e.g., the blocks following "All Firth
regression tests PASSING", "Milestone 2: Core Module Stability", and "Milestone
3: Feature Parity with R") are missing language identifiers and trigger MD040;
update each triple-backtick fence to include the language identifier "text"
(i.e., replace ``` with ```text) so the checklist-style blocks are properly
marked, ensuring all similar checklist blocks in this file use ```text
consistently.

🧹 Nitpick | 🔵 Trivial

Add language identifiers to fenced blocks.

These fenced blocks lack a language, which triggers MD040. Use text for checklist-style blocks.

♻️ Proposed fix
-```
+```text
 ✅ All Firth regression tests PASSING
 ✅ Coefficients match R within ±0.0001
 ✅ P-values match within ±0.001
 ✅ Documentation updated

- +text
✅ Core regression module refactored & tested (85%+ coverage)
✅ All regression types documented
✅ Export to PDF/Word working
✅ Performance benchmarks established

-```
+```text
✅ All statistical outputs validated against R
✅ Diagnostic tests comprehensive
✅ Survival analysis complete with TVC
✅ Causal inference methods working
✅ >80% overall test coverage

- +text
✅ Advanced features fully implemented
✅ Performance: <2s median response time
✅ Security audit passed
✅ Comprehensive documentation
✅ Docker image optimized (<1.3GB)
✅ Production deployment checklist complete

-------------------------------------
tabs/tab_corr.py (1)
472-478: Consider consistent defensive access for stats dictionary.

Several keys like '95% CI Lower', '95% CI Upper', 'R-squared (R²)', and 'P-value' are accessed directly without .get(). While these are likely always present, applying consistent defensive access would prevent runtime errors if the upstream API changes.

♻️ Example defensive pattern
"Value": [
    stats.get("Method", "N/A"),
    f"{coef_val:.4f}",
    f"{stats.get('95% CI Lower', float('nan')):.4f}",
    f"{stats.get('95% CI Upper', float('nan')):.4f}",
    f"{stats.get('R-squared (R²)', float('nan')):.4f}",
    f"{stats.get('P-value', float('nan')):.4f}",
    str(stats.get("N", "N/A")),
    stats.get("Interpretation", "N/A"),
],

-----------------------------------
utils/diag_test.py (1)
904-919: Warn whenever weighted kappa falls back to string labels, not only on exceptions.

If numeric coercion yields NaNs (no exception), weighted kappa still uses string labels without warning, which can mislead.

🐛 Proposed fix

-            if y1_num.notna().all() and y2_num.notna().all():
-                kappa = cohen_kappa_score(y1_num, y2_num, weights=weights)
-            else:
-                kappa = cohen_kappa_score(
-                    y1.astype(str), y2.astype(str), weights=weights
-                )

-            if y1_num.notna().all() and y2_num.notna().all():

-                kappa = cohen_kappa_score(y1_num, y2_num, weights=weights)
-            else:
-                if weights:
-                    logger.warning(
-                        f"Weighted kappa requested but data is non-numeric; "
-                        f"weights='{weights}' may not be meaningful for categorical labels."
-                    )
-                kappa = cohen_kappa_score(
-                    y1.astype(str), y2.astype(str), weights=weights
-                )

------------------------------------
In `@tabs/tab_corr.py` around lines 451 - 458, The code currently sets coef_val =
stats.get(coef_key, 0.0) which causes missing coefficients to show as 0.0000;
change the default to a sentinel like None or the string "N/A" (e.g., coef_val =
stats.get(coef_key, None)), then wherever coef_val is formatted for display (the
formatting calls that assume numeric values), guard with a numeric check
(isinstance(coef_val, (int, float))) and only apply "{:.4f}".format for numeric
values, otherwise render "N/A"; apply the same change for the other coef_val
retrievals that use stats.get in this module.

⚠️ Potential issue | 🟡 Minor

Avoid displaying a missing coefficient as 0.0000.

Defaulting to 0.0 hides missing data and can mislead users. Prefer an explicit “N/A” and guard formatting for non-numeric values.

🔧 Suggested fix

-        coef_val = stats.get(coef_key, 0.0)

+        coef_val = stats.get(coef_key)
-        coef_display = (
-            f"{coef_val:.4f}"
-            if isinstance(coef_val, (int, float)) and not pd.isna(coef_val)
-            else "N/A"
-        )

@@

-                f"{coef_val:.4f}",

+                coef_display,

---------------------------------

In `@utils/diag_test.py` around lines 1114 - 1129, The current logic conflates an
empty/invalid score column and a missing pos_label_user; first validate y_score
after coercion (if y_score.empty, log and return a clear error about no numeric
scores in score_col), then align y_true_raw = data[truth_col].loc[y_score.index]
and verify it has exactly 2 unique classes (if not, log and return the
binary-outcome error using logger.error). If pos_label_user is None, infer a
default positive label from the two unique classes (e.g., choose one of the two
unique values deterministically, such as sorted[unique_values](1) or the second
element) and then set y_true = np.where(y_true_raw.astype(str) ==
chosen_pos_label, 1, 0); otherwise use pos_label_user. Ensure all returned error
messages are specific (separate messages for empty/invalid score column vs
non-binary truth column) and reference score_col, truth_col, y_score,
y_true_raw, and pos_label_user in your changes.

⚠️ Potential issue | 🟠 Major

pos_label_user=None now hard-fails despite being optional.

With the current check, any caller relying on the default (None) gets a misleading “binary outcome required” error. Separate the checks and either infer a default positive label or return a specific error. Also handle the case where the score column becomes empty after numeric coercion

🐛 Proposed fix

-        y_true_raw = data[truth_col]
-        y_score = pd.to_numeric(data[score_col], errors="coerce").dropna()
-        y_true_raw = y_true_raw.loc[y_score.index]
-
-        if y_true_raw.nunique() != 2 or pos_label_user is None:
-            logger.error("Binary outcome required")
-            return (
-                None,
-                "Error: Binary outcome required (must have exactly 2 unique classes).",
-                None,
-                None,
-            )

+        y_true_raw = data[truth_col]
-        y_score = pd.to_numeric(data[score_col], errors="coerce").dropna()
-        if y_score.empty:
-            logger.error("Prediction score is non-numeric or empty")
-            return (
-                None,
-                f"Error: '{score_col}' has no numeric values for ROC analysis.",
-                None,
-                None,
-            )
-        y_true_raw = y_true_raw.loc[y_score.index]
-
-        if y_true_raw.nunique() != 2:
-            logger.error("Binary outcome required")
-            return (
-                None,
-                "Error: Binary outcome required (must have exactly 2 unique classes).",
-                None,
-                None,
-            )
-        if pos_label_user is None:
-            # Infer a default positive label to preserve backward compatibility
-            pos_label_user = sorted(y_true_raw.astype(str).unique())[-1]
 
         y_true = np.where(y_true_raw.astype(str) == pos_label_user, 1, 0)

Also applies to: 1130-1145

--------------------------

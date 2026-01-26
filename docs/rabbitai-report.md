# RabbitAI Report

In `@utils/diag_test.py` around lines 1119 - 1338, The function currently accepts
a user-supplied pos_label_user without validating it and always formats the
Method field with "SE={se:.4f}" even when se is NaN; update the logic around
pos_label_user (where y_true_raw is converted and pos_label_user is inferred) to
first check if a provided pos_label_user exists in
y_true_raw.astype(str).unique() and if not return a clear error listing the
valid labels (use y_true_raw.unique() for the list) instead of the generic
missing-classes message; also change the Method formatting in stats_dict (the
code that sets "Method": f"{m_name} (SE={se:.4f})" if se else m_name) to include
the SE only when se is finite and >0 (use np.isfinite(se) and se>0) otherwise
just use m_name so you never display "SE=nan".

⚠️ Potential issue | 🟡 Minor

Clarify invalid pos_label_user and avoid “SE=nan” display.
Currently, a bad positive label yields a generic “need both classes” error, and SE=nan can appear in the Method string.

🐛 Suggested fix

-        if pos_label_user is None:

-        if pos_label_user is None:
             # Infer a default positive label to preserve backward compatibility
             # We take the second value in sorted list (usually correct for 0/1 or No/Yes)
             try:
                 unique_vals = sorted(y_true_raw.astype(str).unique())
                 pos_label_user = unique_vals[-1]
                 logger.info(f"Inferred positive label: {pos_label_user}")
             except Exception:
                 pos_label_user = str(y_true_raw.unique()[-1])

-        else:
-            valid_labels = set(y_true_raw.astype(str).unique())
-            if str(pos_label_user) not in valid_labels:
-                logger.error(
-                    f"Positive label '{pos_label_user}' not found in '{truth_col}'"
-                )
-                return (
-                    None,
-                    f"Error: Positive label '{pos_label_user}' not found in '{truth_col}'.",
-                    None,
-                    None,
-                )
 
         y_true = np.where(y_true_raw.astype(str) == pos_label_user, 1, 0)

@@

-        stats_dict = {

-        method_label = (

-            f"{m_name} (SE={se:.4f})"
-            if se is not None and np.isfinite(se)
-            else m_name
-        )
-        stats_dict = {
             "AUC": f"{auc_val:.4f}",
             "95% CI": auc_ci_str,
             "P-value": format_p_value(p_val_auc),

-            "Method": f"{m_name} (SE={se:.4f})" if se else m_name,

-            "Method": method_label,
             "Interpretation": f"{auc_badge}",
             "Best Threshold": f"{thresholds[best_idx]:.4f}",
             "Youden Index (J)": f"{youden_j:.4f}",
             "Sensitivity at Best": f"{tpr[best_idx]:.4f}",
             "Specificity at Best": f"{1 - fpr[best_idx]:.4f}",
             "Max F1-Score": f"{max_f1:.4f}",
             "calibration_plot": cal_fig,
         }

-----------------------
In `@utils/diag_test.py` around lines 731 - 824, table2_data is emitting raw "nan"
or "inf" strings for undefined metrics; update the value formatting to render
"-" for NaN/undefined (and "Inf" only where you intentionally want it) by using
explicit checks instead of plain f-strings. Concretely, in the table2_data
entries (and before constructing risk_df) replace direct f-string formatting for
variables like f1_score, dor, or_value, rr, rrr, nnt_abs, arr, rr_ci_display,
or_ci_display, etc., with conditional expressions that return "-" when
np.isnan(value) or not np.isfinite(value) (and preserve "Inf" only where nnt_abs
== np.inf is desired); ensure any CI display variables also fall back to "-"
when their numeric endpoints are NaN so the final DataFrame contains "-" for
undefined metrics.

⚠️ Potential issue | 🟡 Minor

Avoid “nan” strings in published metrics tables.
Several metrics format nan directly; prefer "-" for undefined values so the report reads cleanly.

🔧 Suggested fix

-                        "Value": f"{lr_plus:.4f}",

-                        "Value": f"{lr_plus:.4f}" if np.isfinite(lr_plus) else "-",

@@

-                        "Value": f"{lr_minus:.4f}",

-                        "Value": f"{lr_minus:.4f}" if np.isfinite(lr_minus) else "-",

@@

-                        "Value": f"{or_value:.4f}",

-                        "Value": f"{or_value:.4f}" if np.isfinite(or_value) else "-",

@@

-                        "Value": f"{rr:.4f}",

-                        "Value": f"{rr:.4f}" if np.isfinite(rr) else "-",

@@

-                        "Value": f"{dor:.4f}" if not np.isnan(dor) else "-",

-                        "Value": f"{dor:.4f}" if np.isfinite(dor) else "-",

----------------------------
In `@docs/DEEP-OPTIMIZATION-PLAN.md` around lines 629 - 646, Replace the
emphasized lines flagged by MD036 by converting the bolded totals into proper
Markdown headings: change "**Phase 3 Total: ~111 hours (~14 hours/week)**" to an
appropriate heading (e.g., "### Phase 3 Total: ~111 hours (~14 hours/week)") and
similarly change "**Grand Total: ~273 hours = 6.8 weeks full-time equivalent**"
to a suitable heading (e.g., "## Grand Total: ~273 hours = 6.8 weeks full-time
equivalent"), keeping heading levels consistent with the surrounding section
titles so the totals render as headings rather than emphasis.

⚠️ Potential issue | 🟡 Minor

Use headings instead of bold for section totals.
These lines read like headings; MD036 flags them as emphasis‑as‑heading.

✏️ Suggested fix
-**Phase 2B Total: ~162 hours (~20 hours/week)**
+### Phase 2B Total: ~162 hours (~20 hours/week)

-**Phase 3 Total: ~111 hours (~14 hours/week)**
+### Phase 3 Total: ~111 hours (~14 hours/week)

-**Grand Total: ~273 hours = 6.8 weeks full-time equivalent**
+### Grand Total: ~273 hours = 6.8 weeks full-time equivalent

-**END OF OPTIMIZATION PLAN**
+## END OF OPTIMIZATION PLAN
----------------

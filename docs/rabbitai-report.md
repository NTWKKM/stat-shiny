# RabbitAI Report

In `@config.py` around lines 57 - 67, _sync_missing_legacy currently always copies
nested "analysis.missing" into legacy keys, which overwrites explicit legacy
overrides; change it to preserve legacy values when present by: detect if
analysis contains legacy keys ("missing_strategy", "missing_threshold_pct") and
if so backfill the nested dict ("missing.strategy",
"missing.report_threshold_pct") when missing, otherwise copy nested to legacy
only if the legacy key is absent—update the logic in _sync_missing_legacy to
perform this bidirectional/backfill sync so explicit legacy overrides are not
clobbered.

In `@docs/UI_IMPROVEMENT_COMPLETE_10FILES.md` around lines 7 - 9, Update the
inconsistent timeline estimates in the document by choosing one unified duration
and applying it in both the header "Duration: 2-3 weeks" and the later aggregate
estimates (~15–20 hours / 2–3 days / 1–2 weeks) referenced around lines 714–720;
ensure the chosen estimate is reflected in the header line containing "Date:
January 21, 2026" and the summary totals so all mentions (header "Duration: 2-3
weeks" and the later "~15–20 hours (2–3 days) / 1–2 weeks") match exactly and
use the same units (hours/days/weeks) and wording.

In `@docs/UX_UI_audit_report.md` around lines 12 - 15, The document declares "ALL
ITEMS RESOLVED" while unchecked roadmap items (e.g., "empty states/skeleton
loaders") remain; update the UX_UI_audit_report.md to reconcile these by either
removing the all-resolved badge or marking specific checklist items as
completed, and ensure the generated design system reference (`_styling.py` →
`static/styles.css`) and checklist lines (around the "empty states/skeleton
loaders" entries) reflect the true status; update the Status line and
corresponding checklist entries so they are consistent and unambiguous.

In `@pytest.ini` around lines 8 - 14, The global ignores for statsmodels
ConvergenceWarning in pytest.ini (the entries matching "ignore:The MLE may be on
the boundary of the parameter
space:statsmodels.tools.sm_exceptions.ConvergenceWarning", "ignore:Maximum
Likelihood optimization failed to
converge:statsmodels.tools.sm_exceptions.ConvergenceWarning", and
"ignore:Retrying MixedLM optimization with
lbfgs:statsmodels.tools.sm_exceptions.ConvergenceWarning") should be removed or
narrowed; instead, delete these global filterwarnings lines and apply
pytest.mark.filterwarnings on the specific tests or test modules where
convergence instability is expected (or add local context managers in those
tests) so ConvergenceWarning is no longer suppressed project-wide while still
silencing it where intentional.

In `@requirements-prod.txt` around lines 9 - 34, Update the loose >= version pins
to bounded ranges for stability: replace entries like pandas>=3.0.0,
numpy>=2.0.0, starlette>=0.49.1 (and any other major libs: scikit-learn, scipy,
statsmodels, plotly, matplotlib, seaborn, etc.) with conservative ranges such as
>=X.Y.0,<X+1.0.0 (e.g., pandas>=3.0.0,<4.0.0) to prevent automatic upgrades to
the next major version; edit requirements-prod.txt to apply these upper bounds
consistently for each package named (pandas, numpy, starlette, scikit-learn,
scipy, statsmodels, plotly, matplotlib, seaborn, etc.).

In `@requirements.txt` around lines 13 - 33, The requirements pin currently allows
potentially incompatible major releases (pandas>=3.0.0, numpy>=2.0.0,
scipy>=1.15.0, scikit-learn>=1.6.0); update requirements.txt to prevent
unexpected breakage by adding conservative upper bounds (e.g., pandas<4,
numpy<3, scipy<2, scikit-learn<2) or switch to exact pins/compatible pins and
generate a lock file (poetry.lock/Pipfile.lock/requirements.lock) to lock
working combinations; annotate the pandas>=3.0.0 and numpy>=2.0.0 lines with a
short comment noting the need to test for ABI/compatibility and include
instructions to run integration tests before upgrading these majors.

In `@tabs/_styling.py` around lines 1450 - 1492, The file contains duplicated CSS
blocks for .loading-state, .spinner, `@keyframes` spin, .placeholder-state,
.skip-links and .skip-link:focus; remove or consolidate the duplicate
definitions so each selector/animation is defined only once (keep the canonical
styling and delete the redundant block shown in the diff or merge any differing
properties into the single canonical definition) to avoid later overrides and
reduce CSS size.

In `@tabs/tab_advanced_inference.py` around lines 1 - 32, The import block in
tabs.tab_advanced_inference.py is not sorted/formatted; run the automatic fixer
(e.g., ruff check --fix) or manually reorder and format imports so they follow
the project's import sorting rules (stdlib, third-party, local) and
line-wrapping conventions. Locate the top import section containing symbols like
matplotlib.pyplot as plt, numpy as np, pandas as pd, plotly.express as px,
seaborn as sns, statsmodels.api as sm, from shiny import
module/reactive/render/ui, and the local utility imports (get_color_palette,
select_variable_by_keyword, calculate_vif, calculate_heterogeneity,
analyze_mediation, calculate_cooks_distance, get_diagnostic_plot_data,
run_heteroscedasticity_test, run_reset_test, plotly_figure_to_html,
create_error_alert, create_input_group, create_loading_state,
create_placeholder_state, create_results_container,
create_missing_data_report_html) and reorder/format them to satisfy ruff/CI.

In `@config.py` around lines 57 - 67, _sync_missing_legacy currently always copies
nested "analysis.missing" into legacy keys, which overwrites explicit legacy
overrides; change it to preserve legacy values when present by: detect if
analysis contains legacy keys ("missing_strategy", "missing_threshold_pct") and
if so backfill the nested dict ("missing.strategy",
"missing.report_threshold_pct") when missing, otherwise copy nested to legacy
only if the legacy key is absent—update the logic in _sync_missing_legacy to
perform this bidirectional/backfill sync so explicit legacy overrides are not
clobbered.

tabs/_common.py (1)
24-50: Single source of truth for the color palette.

There is another get_color_palette() in utils/table_one.py Line 36-50 with different values. Consider consolidating to avoid palette drift (e.g., import from this module or move to a shared utilities module).

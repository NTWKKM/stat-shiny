# RabbitAI Report

In @.github/workflows/ui-styling.yml:

- Around line 196-207: The final-status check cannot detect compatibility
failures because the test-multi-version job masks failures with "|| true";
remove the "|| true" (or otherwise stop swallowing the exit code) in the
test-multi-version job so that needs.test-multi-version.result reflects real
failure, or if you intentionally allow the job to continue, export a job output
(e.g., multi_version_failed) from the test-multi-version steps and change the
Check final status condition to test that output instead of
needs.test-multi-version.result.

In `@docs/rabbitai-report.md`:

- Line 177: The markdown contains a dangling reference label "name" in the
expression around `irr_res["model"]["name"]` or `irr_res.get("model_type")`) or
parse the HTML — fix by either converting that markdown reference-style link to
an inline link (replace the [name] reference with a full URL in place) or add a
corresponding reference definition at the end of the document (e.g., add a
"name:" label with the target URL) so the reference resolves; locate occurrences
near the `irr_res["model"]["name"]` / `irr_res.get("model_type")` expression and
update them accordingly.
- Around line 12-17: The documentation mixes “pre-release” and “prerelease”;
standardize to a single variant (use "prerelease") and update the CI example so
the matrix entry uses "3.14-dev" (instead of "3.14") and the
actions/setup-python step references the matrix value (version:
matrix.python-version) and includes allow-prereleases: true to ensure the
prerelease Python is installed; look for the matrix.python-version declaration
and the actions/setup-python step to apply these changes.
- Line 1: Add a top-level H1 heading as the very first line of the Markdown file
to satisfy MD041 (e.g., "# RabbitAI Report") so the document no longer begins
with body text; place the H1 before the existing content and ensure it is a
plain Markdown H1 line at the top of the file.

In `@README.md`:

- Around line 235-237: Update the README line referencing the CI workflow so the
platform name is properly capitalized: change the prose that currently says
"github" or any lowercase form to "GitHub" in the sentence mentioning
ui-styling.yml (the workflow at .github/workflows/ui-styling.yml) — leave
filenames and paths like ui-styling.yml, .github/workflows/ui-styling.yml,
_common.py, and tests/unit/test_ui_ux_styles.py unchanged, only adjust the
human-readable platform name to "GitHub".

In `@static/js/custom_handlers.js`:

- Around line 54-59: The click handler for '#navbar_brand_home' currently
silently does nothing when homeLink (queried as '.navbar-nav
.nav-link[data-value="home"]') is null; update the handler to detect when
homeLink is missing and provide feedback—e.g., call console.warn with a clear
message including the selector and show a user-facing notification (toast or
alert) so developers/users know the navbar structure is unexpected; keep the
existing behavior of clicking homeLink when present.

In `@tabs/tab_data.py`:

- Around line 535-545: The scroll script in the reactive effect _jump_to_upload
uses a plain 'file_upload' ID which doesn't exist in the module DOM; update
_jump_to_upload to build the namespaced ID via session.ns('file_upload') and
inject that namespaced ID into the script string so document.getElementById(...)
targets the correct element in the "data" module context.

In `@tests/e2e/test_load_example_data.py`:

- Around line 47-49: The final visibility assertion for the mapping can be flaky
because it has no explicit timeout; after calling
page.locator("#data-sel_var_edit").select_option("Treatment_Group"), update the
expect on page.get_by_text("0=Standard Care") to include an explicit timeout
(e.g., timeout: 5000–10000 ms) so the test waits for the UI update before
failing; locate the expect call in tests/e2e/test_load_example_data.py and add
the timeout option to to_be_visible() accordingly.
- Around line 28-36: The current try/except around page.get_by_text("Loaded 1600
Clinical Records") swallows failures if both the toast and the fallback locator
"#data-ui_file_metadata" are missing; change the logic in
tests/e2e/test_load_example_data.py so that after attempting the primary expect
(page.get_by_text(...).to_be_visible) you catch the exception, then run the
fallback expect (page.locator("#data-ui_file_metadata").to_contain_text("1,600
rows")), and if that fallback also fails explicitly raise an AssertionError (or
re-raise the original exception) so the test fails rather than silently
proceeding.

In `@tests/e2e/test_smart_defaults.py`:

- Line 98: Replace the print("✅ Smart Variable Selection Verification Passed")
in the test with pytest-friendly logging or remove it: either import logging and
call a module logger (e.g. logger = logging.getLogger(__name__);
logger.info("Smart Variable Selection Verification Passed")) inside the test (or
at top-level), or use pytest's caplog fixture to record/assert messages instead
of printing; update the test function (e.g., test_smart_defaults or
test_smart_variable_selection) to use logger.info(...) or remove the debug
output entirely.
- Around line 17-20: Update the tab selection in the failing test to use the
correct tab label: change the get_by_role("tab", name="📁 Data") call to
get_by_role("tab", name="📁 Data Management") in the test (the call to
page.get_by_role is the unique symbol to edit) so the test clicks the actual "📁
Data Management" tab used in the app.

In `@tests/unit/test_color_palette.py`:

- Around line 50-55: Run Black on tests/unit/test_color_palette.py to apply the
required formatting changes; specifically reformat the assertion lines that
check (base_path / "_common.py").exists() and (base_path /
"_styling.py").exists() so they match Black's preferred
parenthesization/line-wrapping (you can run `black
tests/unit/test_color_palette.py`), ensuring the assertions that reference
base_path and the filenames "_common.py" and "_styling.py" are formatted cleanly
to unblock CI.

In `@tests/unit/test_heterogeneity.py`:

- Around line 30-34: Extend the single-study test to also assert that all other
keys returned by calculate_heterogeneity (e.g., "tau_squared", "p_value",
"degrees_of_freedom", and any other fields your implementation returns) are
present and have the expected values for a single-study input (typically zeros
or the neutral/statistical defaults your function uses); then add a new test
that calls calculate_heterogeneity with empty inputs (e.g., [], []) and asserts
the documented/expected behavior — either that it raises the appropriate
exception (ValueError/TypeError) or returns a defined sentinel (None or dict) —
matching how calculate_heterogeneity is implemented so the test enforces that
contract.

In `@tests/unit/test_linear_lib.py`:

- Around line 58-59: The fixture mock_dependencies currently uses autouse=True
which applies mocks to every test; change it so tests opt in by removing
autouse=True (or set autouse=False) and update only tests that require the mocks
(e.g., tests that call COLORS or format_ci_html such as test_plots_created) to
accept the fixture parameter mock_dependencies; ensure the fixture definition
remains named mock_dependencies so callers can request it explicitly and leave
tests that shouldn't be mocked (e.g., TestValidateOLSInputs,
TestPrepareDataForOLS) unchanged.
- Around line 58-84: The mock for format_ci_html in the mock_dependencies
fixture is brittle because mock_format_ci's lambda assumes numeric inputs and
will raise on None/non-numeric; update the mock_format_ci side_effect (for
utils.linear_lib.format_ci_html) to defensively handle None/non-numeric values
(e.g., check isinstance(lower, (int,float)) and isinstance(upper, (int,float))
and return a sensible fallback like "N/A" or formatted strings only when valid)
so tests mimic realistic function fallback behavior without raising
TypeError/ValueError.

In `@tests/unit/test_mediation.py`:

- Around line 145-160: The test test_analyze_mediation_perfect_collinearity is
non-deterministic and unsafe when inspecting result keys: set a fixed RNG seed
at the start (e.g., call np.random.seed(...) in that test) and replace direct
dict indexing of result["direct_effect"] / result["indirect_effect"] with safe
lookups (use result.get("direct_effect") / result.get("indirect_effect") or
check for keys with "direct_effect" in result) so the assertions won't raise
KeyError if the function returns an unexpected structure; reference
test_analyze_mediation_perfect_collinearity and the result variable in your
changes.

In `@tests/unit/test_ui_ux_styles.py`:

- Around line 1-14: Run Black on the failing test file to fix formatting errors:
reformat tests/unit/test_ui_ux_styles.py (the module containing PROJECT_ROOT and
the get_color_palette import) by running `black
tests/unit/test_ui_ux_styles.py`, verify imports and spacing around the
top-level PROJECT_ROOT assignment and the from tabs._common import
get_color_palette line comply with Black, then stage and commit the reformatted
file to resolve the pipeline failure.

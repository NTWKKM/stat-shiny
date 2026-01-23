In @.github/workflows/ui-styling.yml:

- Around line 43-44: The pytest commands currently append "|| true" which
swallows failures; remove the "|| true" suffix from the two pytest invocations
(the lines running "pytest tests/unit/test_color_palette.py -v
--junitxml=ui-styling-color-results.xml" and "pytest
tests/unit/test_ui_ux_styles.py -v --junitxml=ui-styling-ux-results.xml") and
instead set continue-on-error: true on that GitHub Actions step so the step
allows partial failures but preserves the actual exit code/artifacts for
downstream checks (leave the --junitxml flags intact so results are still
produced).
- Around line 57-60: Update the matrix entry and setup action to use the
prerelease Python specification: replace the matrix value "3.14" with "3.14-dev"
and in the actions/setup-python step, add allow-prereleases: true (and ensure
the version input there references the matrix variable, e.g., uses
matrix.python-version). This aligns the python-version matrix and the setup
action to properly install Python 3.14 pre-release.

In `@docs/ARCHITECTURE.md`:

- Line 7: Fix the grammatical error in the sentence "The application uses a
centralized styling system to ensures visual consistency across all modules." by
changing "ensures" to "ensure" so it reads "The application uses a centralized
styling system to ensure visual consistency across all modules."

In `@pytest.ini`:

- Around line 14-18: The pytest.ini contains redundant warning filters: the
existing "ignore:.*keyword-only.*::lifelines" entry already matches all
lifelines submodules, so remove the four specific submodule filters
("ignore:.*keyword-only.*::lifelines.statistics",
"ignore:.*keyword-only.*::lifelines.utils",
"ignore:.*keyword-only.*::lifelines.fitters.coxph_fitter",
"ignore:.*keyword-only.*::lifelines.fitters.cox_time_varying_fitter"); keep only
the parent-module filter ("ignore:.*keyword-only.*::lifelines") to eliminate
redundancy and simplify the file.

In `@static/js/custom_handlers.js`:

- Around line 42-49: The resize handler currently only calls sidebar.show() when
$(window).width() >= 768, so add the missing branch to hide the sidebar when
width < 768; inside the debounced function in the $(window).resize(...) handler
use the same resizeTimer/timeout logic but check if $(window).width() < 768 then
call sidebar.hide() (or the appropriate method on the sidebar object), ensuring
you reference the existing resizeTimer variable and the sidebar object so
behavior is symmetrical for shrinking and expanding.

In `@static/styles.css`:

- Around line 810-852: There are duplicate global CSS rules for .loading-state
(and related .spinner, .placeholder-state, .skip-link:focus) with conflicting
values; locate the second/wizard/stepper block that overrides the earlier
"enhanced" definitions and either consolidate the rules into one authoritative
definition or scope the wizard rules by renaming or namespacing them (e.g.,
.wizard-loading-state or .wizard .loading-state) so the enhanced layout
(display:flex, width:100%, height:100%) remains intact; update all usages to the
new class or remove the redundant block so only the intended styles apply.

In `@tabs/_styling.py`:

- Around line 1528-1537: The CSS in .empty-state h3 and .empty-state p uses
removed variables --text-heading and --text-body; update those to the project's
renamed typography tokens (replace --text-heading with the new heading token and
--text-body with the new body token used across the codebase, e.g.,
--font-size-heading / --font-size-body or the exact token names from the design
tokens file) so the rules in .empty-state h3 and .empty-state p reference valid
variables.

In `@tabs/tab_agreement.py`:

- Around line 589-593: The code reads input.radio_source() without ensuring the
UI input is present, so update the ICC handler where "data_label" is computed
(referencing is_matched.get() and input.radio_source()) to first safely obtain
the radio value: check that input exists and has a callable radio_source
attribute (e.g., via getattr(input, "radio_source", None) and callable) and only
call it when safe, otherwise use a sensible default (e.g., treat as not
"matched" or use "original"); then compute "data_label" from is_matched.get()
and the guarded radio value.

In `@tabs/tab_corr.py`:

- Around line 356-369: The code builds def_matrix and currently matches
desired_keywords against cols (all columns) and then calls
ui.update_selectize("matrix_vars", choices=cols, selected=def_matrix), which
allows non-numeric picks; change the matching and UI choices to use num_cols
instead of cols. Specifically, in the loop that builds def_matrix (referencing
desired_keywords, def_matrix, matches) search only within num_cols, ensure the
fallback uses num_cols[:5], and call ui.update_selectize("matrix_vars",
choices=num_cols, selected=def_matrix) so only numeric columns can be selected
for compute_correlation_matrix.
- Around line 323-337: The code computes numeric columns twice
(numeric_cols_list.set(data.select_dtypes(...)) and num_cols = [c for c in cols
if pd.api.types.is_numeric_dtype(data[c])]) which is redundant and may drift;
consolidate by deriving a single numeric column list (use the result of
select_dtypes once), assign that to numeric_cols_list and to num_cols, then pass
that unified list into select_variable_by_keyword and ui.update_select for
"cv1"/"cv2" so all places (numeric_cols_list, num_cols,
select_variable_by_keyword calls, and ui.update_select) use the same source of
truth.

In `@tabs/tab_data.py`:

- Around line 788-798: Remove the debug inline script that renders inside the UI
div: delete the ui.tags.script(...) block (the placeholder JS snippet) so the
return value only contains the ui.div with the ui.p message; locate the code
around the UI render that returns ui.div(...) (the block referencing
ui.tags.script and the helpBtn querySelector) and remove that script element to
avoid shipping inline debug artifacts in the UI.
- Around line 863-892: The empty-state "Upload File" button
(btn_jump_upload_trigger) is rendered but has no event handler; either remove
this button or add a reactive event handler mirroring the pattern used for
btn_load_example_trigger (use `@reactive.event`(input.btn_jump_upload_trigger))
and inside it call the existing upload-opening logic (e.g., invoke the same
function or reactive that shows the upload modal/component used elsewhere),
ensuring the handler name references btn_jump_upload_trigger so the UI button is
wired.

In `@tabs/tab_diag.py`:

- Around line 280-293: There is a duplicate declaration of the reactive variable
desc_processing (reactive.Value[bool]) — remove the redundant declaration so
desc_processing is declared and initialized exactly once; locate both
occurrences of desc_processing in the block that initializes reactive flags
(near roc_processing, chi_processing, dca_processing, desc_processing) and
delete the duplicate line while leaving the other reactive initializations
(roc_res, chi_res, desc_res, dca_res, roc_processing, chi_processing,
dca_processing) unchanged.

In `@tests/e2e/test_load_example_data.py`:

- Around line 33-35: Replace the fixed sleep after load_btn.click by waiting
explicitly for the UI change: remove page.wait_for_timeout(2000) and instead use
an explicit wait such as page.wait_for_selector(...) or
page.wait_for_response(...) that targets the expected result of load_btn.click;
for example, wait for the specific element/text/state that indicates the load
completed (use the selector or response URL relevant to your app) immediately
after calling load_btn.click to avoid flakiness.
- Around line 21-24: Remove the temporary debug evaluation that collects button
info and the subsequent print: specifically delete or guard the call to
page.evaluate("Array.from(document.querySelectorAll('button')).map(b => ({id:
b.id, text: b.innerText}))") that assigns all_btns and the print(f"DEBUG: All
buttons: {all_btns}"), or wrap them behind a test debug flag so they do not run
in normal CI; update any assertions to rely on the intended test selectors
instead of this debug output (look for variable all_btns and the page.evaluate
usage in the failing test).
- Line 4: Replace the hardcoded BASE_URL = "<http://localhost:8000>" in
tests/e2e/test_load_example_data.py with a parameterized value read from an
environment variable or pytest configuration (e.g., os.environ.get("BASE_URL")
or a pytest fixture/command-line option), and default to "<http://localhost:8000>"
if unset; update any references to BASE_URL in the file to use this new variable
so tests can run against CI/staging/custom ports without changing source.

In `@tests/e2e/test_smart_defaults.py`:

- Around line 1-4: Replace the hardcoded BASE_URL with a configurable value by
reading an environment variable (e.g., os.getenv("BASE_URL",
"<http://localhost:8000>")) so tests use the env-provided URL if present; update
the module-level BASE_URL symbol in tests/e2e/test_smart_defaults.py to derive
from the environment and ensure imports include os (or use pytest
config/fixture) so CI/local runs that bind to other ports can override the URL.

In `@tests/integration/test_advanced_features.py`:

- Around line 97-99: The test currently asserts that all propensity scores are
non-NaN using ps.notna().all(), which is brittle because calculate_ps can
produce NaNs for rows excluded during preprocessing; update the assertion to be
explicit about the assumption by either asserting only for synthetic fixture
rows or adding a clarifying message — e.g., keep the check on the ps Series
(ps.notna().all()) but append a message like "Expected all PS values (synthetic
data has no missing)" or alternatively assert non-NaNs over the subset used by
calculate_ps (reference variables: calculate_ps, ps, df, causal_data fixture) so
the test clearly documents the expectation.
- Around line 88-90: The assertion is fragile because irr_res is a dict and its
string won't reliably contain "NB"; update the test in test_advanced_features.py
to assert the model type from a deterministic field instead of relying on
str(irr_res). For example, read the model-type metadata from irr_res (e.g.,
irr_res["model"]["name"] or irr_res.get("model_type")) or parse the HTML
reliably (e.g., check for a specific element or label in html_rep that
explicitly states "Negative Binomial" or "NB"); replace the current assertion
that uses "NB" in str(irr_res) with a direct equality/contains check against
that metadata key (referencing html_rep and irr_res in the test).

In `@tests/integration/test_corr_cleaning.py`:

- Around line 31-34: The setup_method mutates global CONFIG keys
("analysis.missing.strategy", "analysis.missing.user_defined_values") without
restoring them; modify setup_method to first capture the original values (e.g.,
store them on self like self._orig_config = {...}) and add a teardown_method
that restores those originals via CONFIG.update for each key so other tests
aren’t polluted; reference setup_method, teardown_method, CONFIG, and the two
keys when implementing the save/restore logic.

In `@tests/integration/test_data_cleaning_pipeline.py`:

- Around line 1-11: Add pytest integration marker to this test module by
importing pytest and declaring pytestmark = pytest.mark.integration at top-level
(next to existing imports/logging). Update the file that defines tests in
tests/integration/test_data_cleaning_pipeline.py to include the pytest import
and the module-level pytestmark symbol so the test is categorized consistently
with other integration tests.

In `@tests/integration/test_diag_cleaning.py`:

- Around line 8-13: The CONFIG mutations at module scope (the three
CONFIG.update calls) should be moved into a test-scoped fixture (e.g.,
config_isolation or patch_config) that saves the previous values of
"analysis.missing.report_missing", "analysis.missing.strategy", and
"analysis.missing.user_defined_values", applies the updates before yielding to
tests, and restores the original values after tests finish; implement the
fixture using CONFIG.get or a shallow copy to capture prior state and ensure
teardown restores those keys so CONFIG is not mutated globally across the test
session.

In `@tests/integration/test_poisson_cleaning.py`:

- Around line 8-14: The test mutates the shared CONFIG (CONFIG.update calls) at
module level which leaks into other tests; wrap these changes in a pytest
fixture (e.g., a fixture named restore_config or temp_config) that captures the
previous values for keys "analysis.missing.report_missing",
"analysis.missing.strategy", and "analysis.missing.user_defined_values" (via
CONFIG.get), yields control to the test, and in the teardown restores them using
CONFIG.update (or deletes keys if they were absent); update the test to use this
fixture so CONFIG is restored after the test and no global state is leaked.
- Around line 31-35: Remove the debug print statement that logs the HTML report
in the test: delete the print("\n[DEBUG] HTML Report Content:\n", html_rep) line
in tests/integration/test_poisson_cleaning.py after analyze_poisson_outcome(...)
to avoid dumping large HTML into CI logs; keep the analyze_poisson_outcome call
and the html_rep variable intact so the test logic is unchanged.

In `@tests/unit/test_causal.py`:

- Around line 125-130: The test test_check_balance_missing_cols currently
expects a KeyError when a requested column is missing, but check_balance's
contract returns an empty DataFrame for invalid inputs; update the test to call
check_balance(df, "T", ["B"]) and assert the result is a pandas DataFrame and
result.empty is True (and optionally check columns are empty) instead of using
pytest.raises(KeyError), so the test aligns with check_balance's intended
behavior.
- Around line 104-122: The test test_calculate_ps_separation currently uses a
bare except which can mask real bugs; change the handler to only catch the
expected fitting/separation-related exceptions when calling calculate_ps (e.g.,
ValueError, RuntimeError, np.linalg.LinAlgError) and optionally statsmodels'
PerfectSeparationWarning/PerfectSeparationError if available, so only
perfect-separation/logit-fitting failures are swallowed and other errors
surface.

In `@tests/unit/test_color_palette.py`:

- Around line 28-38: The return None after calling pytest.skip in
get_styling_data is unreachable and should be removed; update the
get_styling_data function to call pytest.skip(...) in the ImportError except
block without the trailing "return None" so the code reads the exception
handling as just pytest.skip(...) and then proceed to the existing else block
that calls get_color_palette() and returns the palette dict.

In `@tests/unit/test_heterogeneity.py`:

- Line 8: The test defines a variable named "vars" which shadows Python's
built-in vars() function; rename the identifier (e.g., to "variances" or
"study_vars") wherever it appears in tests/unit/test_heterogeneity.py so
references (assignments and any downstream uses) are updated accordingly and no
built-in is shadowed—search for "vars" in that test and replace with the chosen
new name.

In `@tests/unit/test_linear_lib.py`:

- Around line 680-683: The test currently asserts len(result["history"]) >= 0
which is always true; update the assertion to verify meaningful backward
selection by asserting len(result["history"]) > 0 or by checking specific
expected entries in result["history"] (e.g., that it contains at least one step
or expected variable names), and keep the existing checks for "history" and
"final_model" (references: result, result["history"], result["final_model"]).

In `@tests/unit/test_mediation.py`:

- Around line 140-149: The test test_analyze_mediation_perfect_collinearity
currently only asserts result is not None; update it to assert explicit handling
of perfect collinearity by analyze_mediation: either expect a specific exception
(e.g., wrap the call in pytest.raises(ValueError) for collinearity) or assert a
documented diagnostic/warning (use pytest.warns(UserWarning or RuntimeWarning)
and then inspect the returned result for a collinearity indicator or NaN
coefficients). Reference analyze_mediation and ensure the test checks for one
clear, deterministic outcome (exception OR warning + result containing a
'collinearity' flag or NaN mediator/treatment coefficients).
- Around line 125-137: The test test_analyze_mediation_constant_variable
currently only asserts result is not None; update it to assert the expected
behavior of analyze_mediation when treatment is constant: call
analyze_mediation(df, outcome="Y", treatment="X", mediator="M") and then assert
either that the returned dict contains an explicit error flag/message (e.g.,
result.get("error") or result.get("status") indicates failure) or that numeric
effect fields (e.g., result.get("indirect_effect"), result.get("direct_effect"),
or similar keys used by analyze_mediation) are NaN/None; also consider asserting
that a warnings list (e.g., result.get("warnings")) mentions constant treatment.
Use the actual keys returned by analyze_mediation to make precise assertions
instead of only checking non-None.

In `@tests/unit/test_ui_ux_styles.py`:

- Around line 1-10: Run ruff format on the test file to fix styling issues;
specifically reformat imports and spacing in tests/unit/test_ui_ux_styles.py
(affecting the import block and the PROJECT_ROOT path insertion using Path and
sys.path) so the file conforms to Ruff rules—e.g., reorder/compact imports,
ensure proper blank lines, and remove any unnecessary whitespace—then commit the
formatted file.
- Around line 93-94: Replace the brittle string assertions with AST-based
checks: parse the tested file content into an ast.Module, walk Import and
ImportFrom nodes to assert that a symbol named "get_color_palette" is imported
from the module "tabs._common" (handle both ImportFrom and aliased imports), and
then parse Assign nodes to confirm there is an assignment to the name "COLORS"
where the value is a Call to the Name "get_color_palette"; update the test
assertions to use these AST checks instead of exact string matches for "from
tabs._common import get_color_palette" and "COLORS = get_color_palette()".
- Around line 62-64: The current assertion compares pure_compiled[:200] to
generated_css which is brittle; update the test in
tests/unit/test_ui_ux_styles.py to normalize both pure_compiled and
generated_css before comparing (e.g., strip CSS comments, collapse/normalize
whitespace and newlines) and then assert on the normalized strings (or compare
tokens/AST) instead of raw character slices; reference the variables
pure_compiled and generated_css in the modified assertion so the test is robust
to formatting/comment changes.

In `@utils/collinearity_lib.py`:

- Around line 110-112: Update condition_index to return missing-data information
like calculate_vif: change the signature of condition_index (function name:
condition_index) to return a tuple (pd.DataFrame, dict) instead of just
pd.DataFrame, collect the same kind of missing-info dict you use in
calculate_vif (or reuse the helper that produces it) and return (result_df,
missing_info). Then update any internal returns in condition_index to return the
tuple and adjust all callers to unpack the tuple or accept the extra value so
API behavior is consistent with calculate_vif.
- Around line 140-141: The column normalization can divide by zero when a column
in X_clean is constant; modify the normalization step that computes X_norm =
X_clean / np.sqrt((X_clean**2).sum(axis=0)) to guard against zero norms by
computing the column norms, replacing any zeros with a safe value (e.g., 1.0 or
a small epsilon) before dividing, and use those nonzero norms to produce X_norm
so no inf/nan values are introduced (locate the normalization around the X_norm
/ X_clean computation in collinearity_lib.py).

In `@utils/data_cleaning.py`:

- Around line 953-961: The prepare_data_for_analysis function declares
return_info but never uses it; implement conditional return behavior: keep the
return_info parameter, update the signature/type hints to reflect conditional
return (e.g., -> tuple[pd.DataFrame, dict[str, Any]] | pd.DataFrame or use
typing.Union), and change the final return to return (df_clean, info) when
return_info is True and just df_clean when False; also update the function
docstring to document return types and ensure any callers handle both return
forms (or update callers to always expect the tuple if you prefer removing the
parameter instead).
- Around line 1019-1025: The current handling of the handle_missing parameter in
the block that assigns df_clean/rows_excluded silently ignores unsupported
strategies; update the logic in the if/else that checks handle_missing (the
branch handling "complete_case"/"complete-case") to raise a clear ValueError (or
custom exception) when handle_missing is not one of the supported values instead
of silently passing through; reference the variables handle_missing, df_subset,
df_clean, and rows_excluded so the exception is raised before returning, with a
message listing the accepted strategies.

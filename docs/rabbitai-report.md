# RabbitAI Report

In @.github/workflows/ui-styling.yml:

- Around line 40-46: The "Run UI Styling Tests" workflow step currently uses
continue-on-error: true which masks pytest failures; remove the
continue-on-error: true line from that step so pytest exit codes fail the job
(the step with name "Run UI Styling Tests" that runs the two pytest commands),
relying on the artifact upload step's if: always() to still publish results.

In `@static/js/custom_handlers.js`:

- Around line 25-70: The handlers in initMobileMenu are re-registered on every
shiny:connected event causing duplicate handlers; modify initMobileMenu to be
idempotent by adding a guard (e.g., a module/global flag like
mobileMenuInitialized) and return immediately if already initialized, or remove
existing handlers before binding (use $(document).off for '#mobile_menu_btn' and
$(window).off('resize') for the resize handler) and then bind; ensure you set
the flag (mobileMenuInitialized = true) or rely on Shiny.initializedPromise for
one-time init so symbols to update are initMobileMenu, the '#mobile_menu_btn'
click handler, the window resize handler (resizeTimer), and the shiny:connected
hookup.

In `@tests/unit/test_heterogeneity.py`:

- Around line 42-56: The test test_heterogeneity_empty_input must assert the
concrete sentinel returned by calculate_heterogeneity for empty inputs instead
of allowing None/NaN/exceptions; update the test_heterogeneity_empty_input to
remove the try/except and replace the loose assertions with a single explicit
equality/assertion against the concrete sentinel dict (fetch the actual sentinel
from the implementation or module constant used by calculate_heterogeneity) —
reference calculate_heterogeneity and the test_heterogeneity_empty_input
function when making the change.

In `@tests/unit/test_ui_ux_styles.py`:

- Around line 40-48: Update the var_pattern so it accepts either single or
double quotes around the COLORS key: change the current var_pattern (defined in
the loop over palette) to use a character class for quotes (e.g. ['"]) when
matching COLORS[...] and properly escape the literal braces and brackets;
reference the var_pattern variable and the COLORS lookup in the test to locate
where to change the regex.

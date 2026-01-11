
```
================================================================================
PROJECT: stat-shiny Migration from shinywidgets to HTML/Plotly CDN Rendering
STATUS: Ready for Implementation
REPOSITORY: GitHub - NTWKKM/stat-shiny (branch: patch)
================================================================================

I need your help migrating a Shiny Python project from shinywidgets to HTML/Plotly CDN rendering.

CONTEXT:
- Current: Using @render_widget decorators with shinywidgets library
- Problem: Blocked by firewalls/proxies in restricted environments
- Solution: Convert all Plotly visualizations to HTML strings using CDN
- Goal: Make app work in Private Spaces and corporate networks

TARGET DELIVERABLES:
1. Utility function for Plotly → HTML conversion
2. Test suite for the utility
3. All 5 tabs migrated (13-15 @render_widget functions)
4. Updated documentation
5. Clean requirements.txt

================================================================================
PHASE 1: CREATE INFRASTRUCTURE (2-3 hours estimated work)
================================================================================

TASK 1.1: Create utils/plotly_html_renderer.py

Create a production-ready utility function with these requirements:

```python
def plotly_figure_to_html(
    fig: Optional[go.Figure] = None,
    div_id: Optional[str] = None,
    include_plotlyjs: str = 'cdn',
    height: Optional[int] = None,
    width: Optional[int] = None,
    responsive: bool = True
) -> str:
    """
    Convert Plotly Figure to HTML string with CDN support.
    
    Args:
        fig: Plotly Figure object (None returns safe empty HTML)
        div_id: Unique ID for div (auto-generates if None)
        include_plotlyjs: 'cdn' (default), True (inline), or False (none)
        height/width: Optional fixed dimensions
        responsive: Enable autosize (default True)
    
    Returns:
        HTML string ready for ui.HTML()
    
    Handles:
        - None figures gracefully
        - Sanitizes div_id for security (no XSS)
        - Auto-generates unique div_id if not provided
        - Error handling with logging
        - Invalid figure objects
    """
```

Include:

- Docstring with usage examples
- Error handling with try/except
- Logging for debug info
- div_id sanitization (remove special chars)
- Input validation

TASK 1.2: Create tests/test_plotly_html_rendering.py

Write 6+ pytest test cases:

1. test_none_figure() - Handle None gracefully
2. test_simple_figure() - Basic Plotly rendering
3. test_cdn_plotlyjs() - Verify CDN included
4. test_responsive() - Verify autosize
5. test_unique_div_id() - Check div_id uniqueness
6. test_html_safety() - Check XSS prevention

All tests must:

- Pass with pytest
- Use clear assertions
- Have docstrings

VERIFICATION (after Phase 1):

```bash
pytest tests/test_plotly_html_rendering.py -v  # All pass ✓
python -c "from utils.plotly_html_renderer import plotly_figure_to_html; print('✓ Works')"
```

================================================================================
PHASE 2: MIGRATE ALL TABS (3-4 hours estimated work)
================================================================================

MIGRATION PATTERN (Apply to ALL @render_widget functions):

BEFORE (Current):

```python
from shinywidgets import output_widget, render_widget

@render_widget
def my_plot():
    res = results.get()
    return res['fig'] if res else None
```

AFTER (New):

```python
from utils.plotly_html_renderer import plotly_figure_to_html
from shiny import render, ui

@render.ui
def my_plot():
    res = results.get()
    if res is None:
        return ui.div(
            ui.markdown("⏳ *Waiting for results...*"),
            style="color: #999; text-align: center; padding: 20px;"
        )
    
    html_str = plotly_figure_to_html(
        res['fig'],
        div_id="plot_descriptive_name",  # ← UNIQUE per function
        include_plotlyjs='cdn',
        responsive=True
    )
    return ui.HTML(html_str)
```

KEY RULES:

1. Change decorator: @render_widget → @render.ui
2. Wrap figure: plotly_figure_to_html(fig, ...)
3. Return type: ui.HTML(html_str) not just string
4. div_id format: "plot_<tab>_<analysis>" (must be unique per function)
5. Error handling: Always check if res is None
6. CDN: Always use include_plotlyjs='cdn' (not True)

APPLY TO THESE 5 FILES:

FILE 1: tabs/tab_survival.py (4 functions)

- out_curves_plot() → div_id: "plot_curves_km_na"
- out_landmark_plot() → div_id: "plot_landmark_analysis"
- out_cox_forest() → div_id: "plot_cox_forest"
- out_sg_forest() → div_id: "plot_subgroup_forest"

FILE 2: tabs/tab_logit.py (3-4 functions)

- Find all @render_widget
- Apply pattern
- div_ids: "plot_logit_*"

FILE 3: tabs/tab_corr.py (2-3 functions)

- Find all @render_widget
- Apply pattern
- div_ids: "plot_corr_*"

FILE 4: tabs/tab_baseline_matching.py (2-3 functions)

- Find all @render_widget
- Apply pattern
- div_ids: "plot_balance_*"

FILE 5: tabs/tab_diag.py (1-2 functions)

- Find all @render_widget
- Apply pattern
- div_ids: "plot_diag_*"

FOR EACH FILE:

1. Remove: from shinywidgets import output_widget, render_widget
2. Add: from utils.plotly_html_renderer import plotly_figure_to_html
3. For each @render_widget function:
    - Change decorator to @render.ui
    - Wrap figure with plotly_figure_to_html()
    - Return ui.HTML(html_str)
    - Use UNIQUE div_id per function

VERIFICATION (after Phase 2):

```bash
grep -r "@render_widget" tabs/  # Should find 0
grep -r "shinywidgets" . --include="*.py"  # Should find 0
python -m py_compile tabs/*.py  # All compile ✓
python -m shiny run app.py  # App starts ✓
```

================================================================================
PHASE 3: CLEANUP \& DOCUMENTATION (1-2 hours estimated work)
================================================================================

TASK 3.1: Update requirements.txt

- Find lines: shinywidgets>=0.1.0 and shimmy>=0.2.0
- Either DELETE them or COMMENT them out (add \#)

TASK 3.2: Update README.md

- Add new section "✅ Deployment Features" that explains:
    * No external JavaScript runtime dependencies
    * Works behind corporate proxies/firewalls
    * Compatible with restricted platforms (Hugging Face Private Spaces)
    * Uses Plotly CDN for interactive visualization
    * Not affected by library version updates

TASK 3.3: Create MIGRATION_NOTES.md in folder /md/
Include sections:

1. **Why:** Explain firewall/proxy compatibility need
2. **What:** Explain shinywidgets → HTML rendering change
3. **How:** Explain Plotly fig → HTML string → CDN approach
4. **Impact:** Emphasize zero breaking changes
5. **Testing:** What to verify
6. **Troubleshooting:** Common issues and solutions

================================================================================
QUALITY REQUIREMENTS (MUST MEET ALL)
================================================================================

✓ All code is production-ready (no debug code)
✓ All tests pass: pytest tests/test_plotly_html_rendering.py -v
✓ All files compile: python -m py_compile tabs/*.py utils/plotly_html_renderer.py
✓ NO @render_widget decorators remain anywhere
✓ NO shinywidgets imports remain (except comments)
✓ ALL Plotly figures wrapped with plotly_figure_to_html()
✓ ALL div_ids are UNIQUE and DESCRIPTIVE
✓ ALL div_ids follow naming: "plot_<tab>_<analysis>"
✓ ALL functions return ui.HTML() not bare strings
✓ ALL None figures handled gracefully
✓ ALL error handling is robust
✓ CDN always uses include_plotlyjs='cdn' not True

================================================================================
EXECUTION INSTRUCTIONS
================================================================================

After each phase pls : 

- testing of all plots
- Responsive design verification (desktop, tablet, mobile)


STEP 1: Do Phase 1 FIRST

- Create utils/plotly_html_renderer.py
- Create tests/test_plotly_html_rendering.py
- Show all code
- Explain your implementation

STEP 2: After Phase 1 works, do Phase 2

- Migrate all 5 tab files
- Show complete migrated code for each file
- Explain any edge cases

STEP 3: After Phase 2 works, do Phase 3

- Update requirements.txt
- Update README.md
- Create MIGRATION_NOTES.md


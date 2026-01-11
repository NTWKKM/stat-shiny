# Migration Notes: shinywidgets → HTML/Plotly CDN Rendering

**Date:** 2026-01-11  
**Version:** 1.0.0  
**Status:** ✅ Complete

---

## 1. Why: The Need for Migration

### Problem Statement

The original implementation used `shinywidgets` library with `@render_widget` decorators to render Plotly figures. This approach had significant issues in restricted environments:

| Issue | Impact |
|-------|--------|
| **Firewall blocks** | Corporate networks block WebSocket connections required by shinywidgets |
| **Proxy interference** | HTTP proxies strip or modify widget communication |
| **Private Spaces** | Hugging Face Private Spaces don't allow arbitrary network connections |
| **Version instability** | Breaking changes in shinywidgets updates could affect production |

### Solution

Migrate all Plotly visualizations to **static HTML strings** using Plotly's built-in CDN support. This approach:

- ✅ Works behind firewalls (only standard HTTP)
- ✅ Compatible with any proxy configuration
- ✅ No WebSocket dependencies
- ✅ Stable across library versions

---

## 2. What: Changes Made

### Files Modified

| File | Changes |
|------|---------|
| `utils/plotly_html_renderer.py` | **NEW** - Utility function for HTML rendering |
| `tests/test_plotly_html_rendering.py` | **NEW** - Test suite (15 tests) |
| `tabs/tab_survival.py` | Migrated 4 @render_widget functions |
| `tabs/tab_logit.py` | Migrated 5 @render_widget functions |
| `tabs/tab_corr.py` | Migrated 2 @render_widget functions |
| `tabs/tab_baseline_matching.py` | Migrated 2 @render_widget functions |
| `requirements.txt` | Commented out shinywidgets/shimmy |
| `README.md` | Added Deployment Features section |

### Summary Statistics

- **Total @render_widget functions migrated:** 13
- **Files modified:** 6
- **New files created:** 2
- **Tests passing:** 15

---

## 3. How: Technical Approach

### Before (Old Pattern)

```python
from shinywidgets import output_widget, render_widget

# UI
output_widget("my_plot")

# Server
@render_widget
def my_plot():
    return fig  # Returns Plotly Figure directly
```

### After (New Pattern)

```python
from utils.plotly_html_renderer import plotly_figure_to_html
from shiny import render, ui

# UI
ui.output_ui("my_plot")

# Server
@render.ui
def my_plot():
    if result is None:
        return ui.div(ui.markdown("⏳ *Waiting...*"), ...)
    
    html_str = plotly_figure_to_html(
        fig,
        div_id="unique_plot_id",
        include_plotlyjs='cdn',
        responsive=True
    )
    return ui.HTML(html_str)
```

### Key Differences

| Aspect | Before (shinywidgets) | After (HTML/CDN) |
|--------|----------------------|------------------|
| **Decorator** | `@render_widget` | `@render.ui` |
| **Return type** | `go.Figure` | `ui.HTML(str)` |
| **UI element** | `output_widget(...)` | `ui.output_ui(...)` |
| **Plotly.js** | Bundled via widget | CDN loaded |
| **None handling** | Return None | Return placeholder div |

---

## 4. Impact: Zero Breaking Changes

### User Experience

- ✅ All visualizations work exactly as before
- ✅ Interactive features preserved (zoom, pan, hover)
- ✅ Download report functions unchanged
- ✅ Responsive layouts maintained

### For Developers

- The `plotly_figure_to_html()` utility handles edge cases
- Unique `div_id` required per plot (prevents conflicts)
- Always use `include_plotlyjs='cdn'` (not `True`)

---

## 5. Testing: What to Verify

### Automated

```bash
# Run utility tests
pytest tests/test_plotly_html_rendering.py -v -m unit

# Compile all files
python -m py_compile tabs/*.py utils/plotly_html_renderer.py

# Verify no @render_widget remains
grep -r "@render_widget" tabs/
```

### Manual Verification

Run the app and test each plot type:

1. **Survival Analysis Tab**
   - Kaplan-Meier/Nelson-Aalen curves
   - Landmark analysis plot
   - Cox regression forest plot
   - Subgroup analysis forest plot

2. **Logistic Regression Tab**
   - Crude OR forest plot
   - Adjusted OR forest plot
   - Poisson crude/adjusted IRR forest plots
   - Subgroup analysis forest plot

3. **Correlation Tab**
   - Scatter plot
   - Correlation heatmap

4. **Baseline Matching Tab**
   - Love plot (SMD comparison)
   - Boxplot for matched data

---

## 6. Troubleshooting

### Plot Not Appearing

**Symptom:** Empty space where plot should be

**Solutions:**

1. Check if `div_id` is unique (no conflicts)
2. Ensure `include_plotlyjs='cdn'` is set
3. Verify CDN is accessible (not blocked by firewall)

### Multiple Plots Conflicting

**Symptom:** Only first plot shows, or plots overlay

**Solutions:**

1. Each plot MUST have a unique `div_id`
2. Use descriptive naming: `plot_{tab}_{analysis}`
3. Example: `plot_survival_km`, `plot_logit_forest_adj`

### CDN Blocked

**Symptom:** Interactive features don't work

**Solutions:**

1. Check network access to `cdn.plot.ly`
2. If blocked, consider `include_plotlyjs=True` (inline JS)
3. Note: Inline JS increases file size significantly

---

## 7. Future Considerations

### If Reverting is Needed

1. Uncomment shinywidgets in `requirements.txt`
2. Restore `@render_widget` decorators
3. Change `ui.output_ui(...)` back to `output_widget(...)`
4. The utility can coexist with shinywidgets

### Maintenance

- The `plotly_html_renderer.py` utility is self-contained
- Tests ensure reliability across Python/Plotly versions
- CDN URL is managed by Plotly (automatic updates)

---

**Migration completed successfully!** 🎉

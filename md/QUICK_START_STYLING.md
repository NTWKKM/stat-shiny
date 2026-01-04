# 🚀 Quick Start Guide - Styling Implementation

**TL;DR Version** - Get up and running with the new styling in 5 minutes.

---

## 🔐 One-Time Setup

Nothing to setup! The styling is **automatically included** in your app.

Just verify this line exists in `app.py`:

```python
from tabs._styling import get_shiny_css

# In app_ui:
header=ui.tags.head(
    ui.HTML(get_shiny_css())
)
```

✅ **Done!** All components now have professional styling.

---

## 💱 Most Used Styling Classes

### Buttons

```python
# Primary (Blue)
ui.input_action_button("btn", "Click Me", class_="btn-primary")

# Secondary (Light Blue)
ui.input_action_button("btn", "Cancel", class_="btn-secondary")

# Success (Green)
ui.input_action_button("btn", "Save", class_="btn-success")

# Danger (Red)
ui.input_action_button("btn", "Delete", class_="btn-danger")

# Warning (Orange)
ui.input_action_button("btn", "Alert", class_="btn-warning")
```

### Cards

```python
ui.card(
    ui.card_header("📊 Results"),
    ui.card_body("Your content here")
)
```

### Spacing

```python
# Margin bottom
ui.div("Content", class_="mb-4")  # 16px margin-bottom

# Margin top
ui.div("Content", class_="mt-3")  # 12px margin-top

# Padding
ui.div("Content", class_="p-4")   # 16px padding
```

### Text Colors

```python
ui.HTML('<p class="text-primary">Primary</p>')     # Navy
ui.HTML('<p class="text-success">Success</p>')     # Green
ui.HTML('<p class="text-danger">Danger</p>')       # Red
ui.HTML('<p class="text-warning">Warning</p>')     # Orange
ui.HTML('<p class="text-secondary">Secondary</p>') # Gray
```

### Status Badges

```python
from tabs._styling import style_status_badge

ui.HTML(style_status_badge('success', '✅ Matched'))
ui.HTML(style_status_badge('danger', '❌ Unmatched'))
ui.HTML(style_status_badge('warning', '⚠️ Check'))
```

### Alerts

```python
from tabs._styling import style_alert

ui.HTML(style_alert('success', 'Successfully saved', 'Success'))
ui.HTML(style_alert('danger', 'An error occurred', 'Error'))
ui.HTML(style_alert('warning', 'Please review', 'Warning'))
```

---

## 📋 Color Reference

```
Navy Blue:     #1E3A5F  ← Use for primary actions
Dark Navy:     #0F2440  ← Use for headers
Light Navy:    #E8EEF7  ← Use for backgrounds
Smoke White:   #F8F9FA  ← Use for navbar

Green:         #22A765  ← Success/positive
Red:           #E74856  ← Danger/error
Orange:        #FFB900  ← Warning/caution
Gray-blue:     #5A7B8E  ← Info/secondary
```

---

## 📊 Common Patterns

### Simple Data Card

```python
ui.card(
    ui.card_header("📄 Patient Data"),
    ui.card_body(
        ui.input_text("name", "Patient Name"),
        ui.input_numeric("age", "Age", value=60),
        ui.input_action_button("save", "Save", class_="btn-primary")
    )
)
```

### Two-Column Layout

```python
ui.layout_columns(
    ui.card(
        ui.card_header("📄 Input"),
        ui.card_body("Controls here")
    ),
    ui.card(
        ui.card_header("📊 Results"),
        ui.card_body(ui.output_data_frame("results"))
    ),
    col_widths=(5, 7)  # 5:7 ratio
)
```

### Status Display

```python
ui.layout_columns(
    ui.div(
        class_="stat-box",
        ui.HTML("""
            <div class="stat-box-label">Total</div>
            <div class="stat-box-value">1,500</div>
            <div class="stat-box-subtext">Patients</div>
        """)
    ),
    ui.div(
        class_="stat-box",
        ui.HTML("""
            <div class="stat-box-label">Matched</div>
            <div class="stat-box-value">1,342</div>
            <div class="stat-box-subtext">89.5%</div>
        """)
    ),
    col_widths=(6, 6)
)
```

### Form with Buttons

```python
ui.card(
    ui.card_header("📋 Input Form"),
    ui.card_body(
        ui.input_text("var1", "Variable 1"),
        ui.input_select("var2", "Variable 2", choices=["A", "B"]),
        ui.br(),
        ui.layout_columns(
            ui.input_action_button("submit", "Submit", class_="btn-primary"),
            ui.input_action_button("reset", "Reset", class_="btn-secondary"),
            col_widths=(6, 6)
        )
    )
)
```

---

## 🛠 Customization

### Change Primary Color

**File:** `tabs/_common.py`

```python
def get_color_palette():
    return {
        'primary': '#1E3A5F',        # ← Change this
        'primary_dark': '#0F2440',   # ← And this
        'primary_light': '#E8EEF7',  # ← And this
        # ... rest stays same
    }
```

**That's it!** All styling updates automatically.

### Change Default Spacing

**File:** `tabs/_styling.py`

Find this section:

```css
:root {
    --spacing-lg: 16px;  /* ← Change default spacing */
    --spacing-xl: 24px;  /* ← Change section spacing */
}
```

### Add Custom Color

**File:** `tabs/_common.py`

```python
def get_color_palette():
    return {
        # ... existing colors ...
        'custom_blue': '#0066CC',    # ← Add new color
    }
```

Then in `tabs/_styling.py` create CSS rule:

```css
.custom-bg-blue {
    background-color: var(--custom-blue);
}
```

Use it:

```python
ui.div("Text", class_="custom-bg-blue")
```

---

## 🐛 Common Issues

### **Q: Buttons don't have colors**

**A:** Make sure `get_shiny_css()` is in app header:

```python
header=ui.tags.head(
    ui.HTML(get_shiny_css())  # ← This must exist
)
```

### **Q: Forms look default**

**A:** Check that form inputs have correct Shiny classes:

```python
# Must use Shiny's input functions
ui.input_text("name", "Name")          # ✅ Correct
ui.input_select("type", "Type", [])    # ✅ Correct
ui.input_text_area("notes", "Notes")  # ✅ Correct
```

### **Q: Colors don't look right**

**A:** Check the color palette in `tabs/_common.py`. Make sure colors are valid hex codes:

```python
'primary': '#1E3A5F'  # ✅ Valid hex
'primary': '1E3A5F'   # ❌ Missing #
'primary': 'navy'     # ❌ Not supported
```

### **Q: Layout broken on mobile**

**A:** Add responsive breakpoints to layout:

```python
ui.layout_columns(
    ui.card("Column 1"),
    ui.card("Column 2"),
    col_widths=(6, 6),  # Desktop: 50/50
    # Mobile: automatically stacks 100%
)
```

---

## 📊 Spacing Cheat Sheet

```
Margin Bottom:     mb-1 (4px), mb-2 (8px), mb-3 (12px), mb-4 (16px), mb-5 (24px)
Margin Top:        mt-1 (4px), mt-2 (8px), mt-3 (12px), mt-4 (16px), mt-5 (24px)
Padding:           p-2 (8px), p-3 (12px), p-4 (16px), p-5 (24px)
```

---

## 🚀 Performance Tips

1. **Minimize custom CSS** - Use existing classes
2. **Cache styling** - `get_shiny_css()` is computed once
3. **Use CSS variables** - Faster than changing hex codes
4. **Avoid inline styles** - Use classes instead

---

## 🌟 Quick Links

- **Full Guide:** See `UI_ENHANCEMENT_GUIDE.md`
- **Code Examples:** See `STYLING_EXAMPLES.md`
- **Technical Details:** See `CHANGELOG_UI_ENHANCEMENT.md`
- **Color Palette:** See `tabs/_common.py`
- **CSS Styles:** See `tabs/_styling.py`

---

## 💸 Button Types at a Glance

| Type | Class | Color | Use Case |
|------|-------|-------|----------|
| Primary | `btn-primary` | Navy Blue | Main actions |
| Secondary | `btn-secondary` | Light Blue | Cancel, Back |
| Success | `btn-success` | Green | Save, Confirm |
| Danger | `btn-danger` | Red | Delete |
| Warning | `btn-warning` | Orange | Alert |

---

## 🎈 Pro Tips

1. **Always include icon emojis** in button/card headers for visual interest:
   ```python
   ui.input_action_button("run", "🔍 Run Analysis", class_="btn-primary")
   ```

2. **Use status badges** for quick status feedback:
   ```python
   ui.HTML(style_status_badge('success', '✅ Ready to analyze'))
   ```

3. **Combine cards in layouts** for better organization:
   ```python
   ui.layout_columns(card1, card2, col_widths=(5, 7))
   ```

4. **Use proper heading hierarchy** for accessibility:
   ```python
   ui.h2("Section")      # Main section
   ui.h3("Subsection")   # Under section
   ```

5. **Test on mobile** - Use browser DevTools mobile view (F12 → Toggle device toolbar)

---

## ✅ Ready to Go!

You now have everything you need to style your app professionally.

**Next:** Review specific sections in `UI_ENHANCEMENT_GUIDE.md` for detailed information.

---

**Last Updated:** December 31, 2025  
**Styling Version:** 1.0

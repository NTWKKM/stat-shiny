# 🖺 Styling Examples & Code Snippets

**Quick Reference Guide** for implementing professional UI styling in the Statistical Analysis Tool.

---

## 💱 Table of Contents

1. [Button Examples](#buttons)
2. [Card Layouts](#cards)
3. [Form Examples](#forms)
4. [Layout Patterns](#layouts)
5. [Status Displays](#status)
6. [Common Patterns](#patterns)

---

## 🔘 Buttons <a name="buttons"></a>

### Basic Buttons

```python
from shiny import ui

# Primary button (blue, filled)
ui.input_action_button("btn_submit", "📋 Submit", class_="btn-primary")

# Secondary button (light blue, outlined)
ui.input_action_button("btn_cancel", "↩️ Cancel", class_="btn-secondary")

# Success button (green)
ui.input_action_button("btn_confirm", "✅ Confirm", class_="btn-success")

# Danger button (red)
ui.input_action_button("btn_delete", "🗑 Delete", class_="btn-danger")

# Warning button (orange)
ui.input_action_button("btn_caution", "⚠️ Warning", class_="btn-warning")

# Outline button (navy border, transparent)
ui.input_action_button("btn_outline", "Outline", class_="btn-outline-primary")
```

### Button Groups

```python
ui.layout_columns(
    ui.input_action_button("btn_prev", "Previous", class_="btn-secondary"),
    ui.input_action_button("btn_next", "Next", class_="btn-primary"),
    col_widths=(6, 6)
)
```

### Button with Icon

```python
ui.input_action_button(
    "btn_download",
    "💾 Download Results",
    class_="btn-primary"
)
```

---

## 🃅 Cards <a name="cards"></a>

### Simple Card

```python
ui.card(
    ui.card_header("📊 Results"),
    ui.card_body(
        ui.HTML("<p>Your analysis results appear here</p>")
    )
)
```

### Card with Multiple Sections

```python
ui.card(
    ui.card_header("📋 Summary Statistics"),
    ui.card_body(
        ui.h5("Baseline Characteristics"),
        ui.output_data_frame("table_baseline"),
        ui.br(),
        ui.h5("Additional Metrics"),
        ui.output_ui("ui_metrics")
    ),
    ui.card_footer(
        ui.HTML('<span class="text-secondary">Data updated: 2025-12-31</span>')
    )
)
```

### Stat Cards (Side by Side)

```python
ui.layout_columns(
    ui.div(
        class_="stat-box",
        ui.HTML("""
            <div class="stat-box-label">Total Patients</div>
            <div class="stat-box-value">1,500</div>
            <div class="stat-box-subtext">Baseline cohort</div>
        """)
    ),
    ui.div(
        class_="stat-box",
        ui.HTML("""
            <div class="stat-box-label">Matched Pairs</div>
            <div class="stat-box-value">1,342</div>
            <div class="stat-box-subtext">89.5% retained</div>
        """)
    ),
    ui.div(
        class_="stat-box",
        ui.HTML("""
            <div class="stat-box-label">Balance Achieved</div>
            <div class="stat-box-value">Yes</div>
            <div class="stat-box-subtext">All SMD < 0.1</div>
        """)
    ),
    col_widths=(4, 4, 4)
)
```

### Collapsible Card

```python
ui.card(
    ui.card_header(
        ui.span("Advanced Options"),
        ui.input_action_button("btn_toggle", "Collapse", class_="btn-outline-primary")
    ),
    ui.card_body(
        ui.input_select("var1", "Variable 1", choices=["A", "B", "C"]),
        ui.input_select("var2", "Variable 2", choices=["X", "Y", "Z"])
    )
)
```

---

## 📝 Forms <a name="forms"></a>

### Basic Form

```python
ui.card(
    ui.card_header("📋 Data Input"),
    ui.card_body(
        # Text input
        ui.input_text(
            "txt_patient_id",
            "Patient ID",
            placeholder="Enter patient ID"
        ),
        
        # Number input
        ui.input_numeric(
            "num_age",
            "Age (years)",
            value=60,
            min=18,
            max=100
        ),
        
        # Dropdown
        ui.input_select(
            "sel_gender",
            "Gender",
            choices={
                "M": "Male",
                "F": "Female",
                "O": "Other"
            }
        ),
        
        # Checkbox
        ui.input_checkbox(
            "chk_diabetes",
            "Has Diabetes?"
        ),
        
        # Radio buttons
        ui.input_radio_buttons(
            "radio_status",
            "Status",
            choices={
                "active": "Active",
                "inactive": "Inactive",
                "excluded": "Excluded"
            },
            selected="active",
            inline=True
        ),
        
        # Text area
        ui.input_text_area(
            "txt_notes",
            "Clinical Notes",
            placeholder="Enter any additional notes...",
            rows=4
        ),
        
        ui.br(),
        
        # Form buttons
        ui.layout_columns(
            ui.input_action_button("btn_save", "💾 Save", class_="btn-primary"),
            ui.input_action_button("btn_clear", "🗑 Clear", class_="btn-secondary"),
            col_widths=(6, 6)
        )
    )
)
```

### Inline Form

```python
ui.card(
    ui.card_header("🔍 Quick Filter"),
    ui.card_body(
        ui.layout_columns(
            ui.input_text(
                "txt_search",
                "Search",
                placeholder="Enter search term"
            ),
            ui.input_action_button("btn_search", "🔍 Search", class_="btn-primary"),
            col_widths=(8, 4),
            gap="1rem"
        )
    )
)
```

### Required Fields Indicator

```python
ui.card(
    ui.card_body(
        ui.HTML('<label class="form-label required">Email Address</label>'),
        ui.input_text("email", None, placeholder="user@example.com"),
        
        ui.HTML('<label class="form-label">Notes (Optional)</label>'),
        ui.input_text_area("notes", None, placeholder="Enter notes...", rows=3)
    )
)
```

---

## 📐 Layouts <a name="layouts"></a>

### Two-Column Layout

```python
ui.layout_columns(
    ui.card(
        ui.card_header("📄 Input Parameters"),
        ui.card_body(
            # Left column content
            ui.input_select("var_outcome", "Outcome Variable", choices=["Y1", "Y2"]),
            ui.input_select("var_treatment", "Treatment Variable", choices=["T1", "T2"])
        )
    ),
    ui.card(
        ui.card_header("📊 Results"),
        ui.card_body(
            # Right column content
            ui.output_data_frame("tbl_results")
        )
    ),
    col_widths=(5, 7)  # 5/12 + 7/12
)
```

### Three-Column Layout

```python
ui.layout_columns(
    ui.card(
        ui.card_header("🎯 Column 1"),
        ui.card_body("Content 1")
    ),
    ui.card(
        ui.card_header("🎯 Column 2"),
        ui.card_body("Content 2")
    ),
    ui.card(
        ui.card_header("🎯 Column 3"),
        ui.card_body("Content 3")
    ),
    col_widths=(4, 4, 4)  # Equal width
)
```

### Sidebar Layout

```python
ui.layout_sidebar(
    ui.sidebar(
        ui.h5("🔍 Filters"),
        ui.input_select("filter_category", "Category", choices=["A", "B", "C"]),
        ui.input_slider("filter_range", "Range", 0, 100, 50),
        ui.input_action_button("btn_apply", "✓ Apply", class_="btn-primary"),
        width=250,
        bg="#f8f9fa"
    ),
    # Main content
    ui.card(
        ui.card_header("📊 Main Content"),
        ui.card_body(
            "Your main content appears here"
        )
    )
)
```

---

## 🔔 Status Displays <a name="status"></a>

### Status Badges

```python
from tabs._styling import style_status_badge

# Success status
ui.HTML(style_status_badge('success', '✅ Data Matched (n=1,342)'))

# Warning status
ui.HTML(style_status_badge('warning', '⚠️ Imbalance Detected'))

# Danger status
ui.HTML(style_status_badge('danger', '❌ Error in Matching'))

# Info status
ui.HTML(style_status_badge('info', 'ℹ️ Processing...'))
```

### Alert Messages

```python
from tabs._styling import style_alert

# Success alert
ui.HTML(style_alert(
    'success',
    'Your analysis has been completed successfully',
    'Success'
))

# Error alert
ui.HTML(style_alert(
    'danger',
    'Please check your input data and try again',
    'Error'
))

# Warning alert
ui.HTML(style_alert(
    'warning',
    'Some variables have missing values. Rows will be excluded.',
    'Warning'
))

# Info alert
ui.HTML(style_alert(
    'info',
    'Propensity score matching is in progress...',
    'Information'
))
```

### Info Panels

```python
# Basic info panel
ui.HTML("""
<div class="info-panel">
    <strong>ℹ️ Note</strong><br>
    This analysis uses the default matching algorithm.
</div>
""")

# Success panel
ui.HTML("""
<div class="info-panel success">
    <strong>✅ Matching Complete</strong><br>
    Successfully matched 1,342 pairs with excellent balance (all SMD < 0.1)
</div>
""")

# Error panel
ui.HTML("""
<div class="info-panel danger">
    <strong>❌ Error</strong><br>
    Unable to create matched pairs. Please check your selection criteria.
</div>
""")
```

---

## 🖾 Common Patterns <a name="patterns"></a>

### Analysis Workflow

```python
ui.navset_tab(
    # Step 1: Data Input
    ui.nav_panel(
        "📄 1. Data",
        ui.card(
            ui.card_header("💾 Upload Data"),
            ui.card_body(
                ui.input_file("file_upload", "Choose CSV file"),
                ui.input_action_button("btn_load", "✓ Load", class_="btn-primary")
            )
        )
    ),
    
    # Step 2: Variable Setup
    ui.nav_panel(
        "🔍 2. Variables",
        ui.card(
            ui.card_header("🔍 Variable Configuration"),
            ui.card_body(
                ui.input_select("var_outcome", "Outcome", choices=[]),
                ui.input_select("var_treatment", "Treatment", choices=[])
            )
        )
    ),
    
    # Step 3: Matching
    ui.nav_panel(
        "🗐 3. Matching",
        ui.card(
            ui.card_header("🗐 Propensity Score Matching"),
            ui.card_body(
                ui.input_action_button("btn_match", "🔍 Run Matching", class_="btn-primary")
            )
        )
    ),
    
    # Step 4: Results
    ui.nav_panel(
        "📊 4. Results",
        ui.card(
            ui.card_header("📊 Matching Results"),
            ui.card_body(
                ui.output_data_frame("tbl_matched_summary")
            )
        )
    )
)
```

### Data Quality Checklist

```python
ui.card(
    ui.card_header("✅ Data Quality Checks"),
    ui.card_body(
        ui.HTML("""
        <div class="mb-3">
            <strong>✅ Duplicates Check</strong> - No duplicates found
        </div>
        <div class="mb-3">
            <strong>❌ Missing Values</strong> - 15 missing values in variable X
        </div>
        <div class="mb-3">
            <strong>⚠️ Type Check</strong> - Age should be numeric (currently: text)
        </div>
        <div>
            <strong>✅ Ranges Check</strong> - All values within expected ranges
        </div>
        """)
    )
)
```

### Results Summary Box

```python
ui.layout_columns(
    ui.card(
        ui.card_header("📄 Matching Summary"),
        ui.card_body(
            ui.HTML("""
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 10px;"><strong>Original Sample:</strong></td>
                    <td style="padding: 10px; text-align: right;"><strong>1,500</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 10px;">Matched Pairs:</td>
                    <td style="padding: 10px; text-align: right;"><strong>1,342</strong></td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 10px;">Retention Rate:</td>
                    <td style="padding: 10px; text-align: right;"><strong>89.5%</strong></td>
                </tr>
                <tr>
                    <td style="padding: 10px;">Balance Status:</td>
                    <td style="padding: 10px; text-align: right;">
                        <span class="badge badge-success">Excellent</span>
                    </td>
                </tr>
            </table>
            """)
        )
    ),
    col_widths=(12)
)
```

---

## 🛠 Tips & Tricks

### Using Color Classes

```python
# Text colors
ui.HTML('<p class="text-primary">Primary color text</p>')
ui.HTML('<p class="text-success">Success message</p>')
ui.HTML('<p class="text-danger">Error message</p>')

# Background colors
ui.HTML('<div class="bg-primary-light p-3">Light background</div>')

# Combined
ui.HTML('<div class="bg-primary-light text-primary p-3">Info box</div>')
```

### Spacing with Classes

```python
# Margin bottom (creates spacing below)
ui.card(
    ui.card_body("Content"),
    class_="mb-4"  # margin-bottom: 16px
)

# Padding (internal spacing)
ui.div(
    "Content with padding",
    class_="p-4"  # padding: 16px
)

# Margin top
ui.div(
    "Content with top margin",
    class_="mt-5"  # margin-top: 24px
)
```

### Responsive Columns

```python
ui.layout_columns(
    ui.card("Column 1"),
    ui.card("Column 2"),
    col_widths=(6, 6),  # Desktop: 50/50
    # Mobile: automatically stacks to 100% width
)
```

---

## 🐛 Accessibility Best Practices

### Always Include Labels

```python
# ✅ Good
ui.input_text("name", "Full Name", placeholder="Enter name")

# ❌ Avoid
ui.input_text("name", "", placeholder="Full Name")
```

### Use Semantic HTML

```python
# ✅ Good - proper heading hierarchy
ui.h2("Main Section")
ui.h3("Subsection")

# ❌ Avoid - inconsistent sizing
ui.HTML('<div style="font-size: 24px;">Main Section</div>')
```

### Ensure Color Contrast

All colors meet WCAG AA standards. If customizing:
- Text on background: minimum 4.5:1 contrast ratio
- Large text: minimum 3:1 contrast ratio

---

## 📊 Visual Example

See the live application at:
- **Development:** `https://huggingface.co/spaces/[your-space]`
- **Main Branch:** Production version
- **Fix Branch:** Development with latest styling

---

**Last Updated:** December 31, 2025  
**Styling Version:** 1.0

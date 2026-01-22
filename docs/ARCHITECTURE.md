# 🏗️ System Architecture & AI Context

This document provides a deep technical overview of the **Medical Statistical Tool** for developers and AI agents (Antigravity/Bots).

## 🎨 UI Styling System

The application uses a centralized styling system to ensures visual consistency across all modules.

### Core Components

| File | Role | Description |
| :--- | :--- | :--- |
| `tabs/_common.py` | **Source of Truth** | Defines the central `get_color_palette()` and common UI constants. |
| `tabs/_styling.py` | **CSS Generator** | Main injector that generates CSS using the palette from `_common.py`. |
| `static/styles.css` | **Compiled CSS** | Auto-generated output from `_styling.py`. Used for performance and deployment. |
| `utils/update_css.py` | **Sync Utility** | Script to update `static/styles.css` whenever `_styling.py` changes. |
| `static/js/custom_handlers.js` | **JS Hooks** | Shiny custom message handlers for dynamic client-side styling. |

### Visual Consistency (Plotly)

- `utils/plotly_html_renderer.py`: Standardizes Plotly figure rendering with "Inter" font and theme-matching placeholders.
- `utils/forest_plot_lib.py`: Interactive forest plots synced with the central color palette.

### Formatting

- `utils/formatting.py`: Handles P-value styling, badge generation, and statistical report HTML structure (syncing with `config.py`).

---

## 🔄 Data Processing & Statistical Pipeline

The data flow is standardized to ensure consistent handling of missing values and variable metadata.

### 1. Ingestion (`tab_data.py`)

- Users upload files (CSV/Excel) or load example data.
- **Initial Quality Check**: Immediate identification of missingness and data types.
- **Configuration**: Individual variable type casting and missing value strategy selection.

### 2. Central Preparation (`utils/data_cleaning.py`)

- Every statistical module calls `prepare_data_for_analysis()` before execution.
- **Tasks**: Handles missing value exclusion, type enforcement, and logging of analyzed indices.
- **Reporting**: Generates a `missing_data_info` dictionary.

### 3. Reporting (`utils/formatting.py`)

- When a module finishes analysis, it generates an HTML report.
- The `missing_data_info` is passed to `create_missing_data_report_html()`.
- **Outcome**: A standardized "Missing Data Summary" is automatically included at the start of every statistical output.

---

## 🧪 Testing & Quality Assurance

- `.github/workflows/ui-styling.yml`: Automated CI that runs:
  - `tests/unit/test_color_palette.py`: Basic palette checks.
  - `tests/unit/test_ui_ux_styles.py`: Comprehensive cross-file consistency checks (Python/CSS/JS/Plotly).

---

> [!NOTE]
> **AI AGENT GUIDANCE**: When analyzing this repository, always verify that new UI components use `get_color_palette()` and that data-heavy modules integrate with `prepare_data_for_analysis()`.

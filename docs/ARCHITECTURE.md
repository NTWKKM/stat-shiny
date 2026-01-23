# 🏗️ System Architecture & AI Context

This document provides a deep technical overview of the **Medical Statistical Tool** for developers and AI agents (Antigravity/Bots).

---

## 📱 App Shell & Navigation

The application uses a fluid, multi-menu navbar designed for responsiveness and accessibility.

### Fluid UI & Layout

- **Container**: Built using `ui.page_fluid` for full-width flexibility across devices.
- **Navigation**: Utilizes nested `ui.page_navbar` and `ui.nav_menu` to organize 10+ statistical modules into logical groups (General Stats, Inference, etc.).
- **Accessibility**: Includes skip-links, ARIA landmarks, and standardized focus states.

### ⚡ Lazy Loading Mechanism

Modules are loaded **on-demand** to minimize initial bundle size and improve startup performance.

1. **Browser** triggers a navigation event.
2. **App Shell** checks if the module (UI/Server) is already initialized.
3. If not, it performs an **asynchronous import** of the tab module.
4. Static assets (`styles.css`, `custom_handlers.js`) are served once and shared across all modules.

![Navigation Workflow](./assets/navigation_sequence.png)

---

## 🎨 UI Styling System

The application uses a centralized styling system to ensure visual consistency across all modules.

### Core Components & Sync Workflow

| File | Role | Description |
| :--- | :--- | :--- |
| `tabs/_common.py` | **Source of Truth** | Defines the central `get_color_palette()` and common UI constants. |
| `tabs/_styling.py` | **CSS Generator** | Main injector that generates CSS using the palette from `_common.py`. |
| `static/styles.css` | **Compiled CSS** | Auto-generated output. **DO NOT EDIT DIRECTLY.** |
| `utils/update_css.py` | **Sync Utility** | Script to update `static/styles.css` whenever `_styling.py` changes. |

> [!TIP]
**To update styles**: Edit `tabs/_styling.py` then run:

- Unix/Linux/macOS: `.venv/bin/python utils/update_css.py`
- Windows: `.venv\Scripts\python.exe utils/update_css.py`

### Visual Consistency (Plotly & HTML)

- `utils/plotly_html_renderer.py`: Standardizes Plotly figure rendering (Inter font, theme-sync).
- `utils/formatting.py`: Handles P-value styling, logic-driven badges, and statistical report HTML structure (syncing with `config.py`).

---

## 📊 Analysis Modules

The application covers a wide range of medical statistical needs, organized into functional clusters:

| Category | Modules | Key Features |
| :--- | :--- | :--- |
| **Standard** | `tab_corr`, `tab_diag`, `tab_agreement` | Correlation Matrix, ROC/AUC, Kappa, ICC, Bland-Altman. |
| **Inference** | `tab_core_regression`, `tab_advanced_inference` | Linear/Logistic/Cox Regressions, Subgroup analysis. |
| **Causal** | `tab_causal_inference`, `tab_baseline_matching` | EconML Integration, Propensity Score Matching (PSM). |
| **Specialized** | `tab_survival`, `tab_advanced_stats`, `tab_sample_size` | Kaplan-Meier, Time-Varying Cox, G-Computation, Power Analysis. |

---

## 🔄 Data Processing & Statistical Pipeline

The data flow is standardized to ensure consistent handling of missing values and variable metadata.

### 1. Ingestion & Quality Check (`tab_data.py` & `utils/data_quality.py`)

- Users upload files (CSV/Excel) or load example data.
- **Immediate Data Health Report**: Uses `check_data_quality()` to perform deep validation:
  - **Numeric Validation**: Detects non-standard values like `"<5"`, `"10%"`, or symbols (`<`, `>`, `,`, `%`, `$`, `€`, `£`) that often appear in medical data but break standard numeric parsing.
  - **Categorical Validation**: Identifies numbers accidentally placed in categorical columns and flags rare categories (threshold < 5) which might lead to unstable statistical estimates.
  - **Row-level Reporting**: Provides exact row indices and unique offending values for fast debugging.
- **Configuration**: Individual variable type casting and missing value strategy selection based on the health report.

### 2. Central Preparation (`utils/data_cleaning.py`)

- **`prepare_data_for_analysis()`**: Every statistical module calls this before execution.
- **Tasks**: Handles missing value exclusion (Listwise/Pairwise), type enforcement, and logging analyzed indices.
- **Metadata**: Generates a `missing_data_info` dictionary detailing what was excluded and why.

### 3. Reporting (`utils/formatting.py`)

- When a module finishes analysis, it generates an HTML report.
- The `missing_data_info` is passed to `create_missing_data_report_html()`.
- **Outcome**: A standardized "Missing Data Summary" is automatically included at the start of every statistical output.

---

## 🧪 Testing & Quality Assurance

### Continuous Integration (CI)

- `.github/workflows/quality_check.yml`: Runs the full suite of Unit, Integration, and E2E (Playwright) tests.
- `.github/workflows/ui-styling.yml`: Verifies CSS/JS/Python styling consistency.
- `.github/workflows/docker-publish.yml`: Automates production image builds.

### Test Suites

- **Unit**: Core utilities, statistics math, color palette.
- **Integration**: PSM pipelines, data cleaning flow.
- **E2E**: Multi-tab navigation, smart variable defaults, loading states.

---

---

## 🤖 AI Bot Technical Guidelines

This repository is optimized for autonomous AI agents (Antigravity, Cursor, Copilot, etc.). To ensure reliable code generation, all bots **MUST** adhere to the standard architecture regardless of the environment (Local, Docker, Hugging Face).

### 📝 Scannable Technical Summary (for LLMs)

```yaml
project:
  name: Medical Statistical Tool
  stack: [Python 3.12+, Shiny for Python, Pandas, Statsmodels, Plotly]
  architecture: Modular / Asynchronous Lazy Loading
  source_of_truth:
    styling: tabs/_common.py (get_color_palette)
    compilation: tabs/_styling.py
    data_pipeline: utils/data_cleaning.py (prepare_data_for_analysis)
    formatting: utils/formatting.py
  constraints:
    - Never edit static/styles.css directly (regenerate via utils/update_css.py)
    - Python 3.12+ (PEP 695 type aliases required)
    - Architecture-First: Research tabs/_common.py and utils/ before new logic.
```

### AI Context & Automation

- **Memory Preservation**: Project-specific logic and rules are in `.agent/` and `.cursorrules`.
- **Workflow Execution**: Use `.agent/workflows/*.md` for automated formatting, linting, and testing.
- **Bot Persona**: Be proactive, architectural-first, and always verify cross-file consistency.

### 🧠 Deep Implementation Details (Bot Confidence)

To adjust or improve statistical modules with confidence, follow these deep patterns:

#### 1. Statistical Pipeline & Data Health

- **Gatekeeper**: `utils/data_cleaning.py -> prepare_data_for_analysis()`.  
  - ALWAYS call this in any analysis module. It handles `complete-case` logic and returns `missing_data_info`.
  - **Threshold**: Numeric conversion defaults to a **30% threshold** (if >30% cells are numeric, the column is treated as numeric and problematic cells are coerced to `NaN`).
- **P-Value Standard**: Use `utils/formatting.py -> format_p_value()`. It respects NEJM standards and `CONFIG` boundaries (e.g., `< 0.001`).

#### 2. Standardized Reporting Schema

Analysis modules should return or generate an **item-based report** list, rendered via `utils.formatting.generate_standard_report`. This ensures visual consistency and automatic inclusion of missing data reports.

```python
from utils.formatting import generate_standard_report

elements = [
    {"type": "text", "data": "Analysis description..."},
    {"type": "table", "header": "Results Table", "data": df_results},
    {"type": "plot", "header": "ROC Curve", "data": fig_plotly},
    {"type": "interpretation", "data": "High confidence result..."},
]
# Unified generator (handles CSS, Missing Data, and premium themes)
report_html = generate_standard_report("Analysis Title", elements, missing_data_info=missing_info)
```

#### 3. UI/UX Component Library (`utils/ui_helpers.py`)

- **Main Container**: `create_results_container(title, *content)`
- **States**:
  - `create_loading_state(msg)`: Use during calculations.
  - `create_placeholder_state(msg, icon)`: Use for initial landing.
  - `create_error_alert(msg)`: Use for validation or runtime errors.
- **Consistency**: Use `create_badge_html` (from `formatting.py` or `diag_test.py`) for semantic labels (success, warning, etc.).

#### 4. Reactive State Pattern (Shiny)

Modules must follow a consistent naming convention for reactive values to ensure predictable UI behavior:

- `*_processing = reactive.Value(False)`: Toggle before/after heavy calculations to show `create_loading_state()`.
- `*_html = reactive.Value("")`: Store the final rendered report HTML.
- `*_fig = reactive.Value(go.Figure())`: Store Plotly figures for independent rendering.
- `*_df = reactive.Calc`: Use for derived data that multiple outputs depend on.

> [!IMPORTANT]
> **AI BOT CRITICAL CHECKLIST**:
>
> 1. Verify UI components use `get_color_palette()`.
> 2. Ensure all data-heavy modules call `prepare_data_for_analysis()`.
> 3. Add tests to the corresponding suite in `tests/`.
> 4. Use `_html.escape()` for any dynamic HTML strings to prevent XSS.

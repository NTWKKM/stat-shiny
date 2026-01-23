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
> **To update styles**: Edit `tabs/_styling.py` then run `.venv/bin/python utils/update_css.py`.

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

### 1. Ingestion (`tab_data.py`)

- Users upload files (CSV/Excel) or load example data.
- **Initial Quality Check**: Immediate identification of missingness and data types.
- **Configuration**: Individual variable type casting and missing value strategy selection.

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

> [!NOTE]
> **AI AGENT GUIDANCE**: When adding new features:
>
> 1. Verify UI components use `get_color_palette()`.
> 2. Ensure all data-heavy modules call `prepare_data_for_analysis()`.
> 3. Add tests to the corresponding suite in `tests/`.

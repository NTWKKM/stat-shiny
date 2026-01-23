# 🤖 AI Context: Universal Entry Point

This file is a machine-readable entry point for all AI bots (Antigravity, Cursor, GitHub Copilot, etc.) interacting with this repository.

## 🏗️ Architecture First

To maintain system integrity, bots **MUST** read and adhere to the architectural standards defined in:
👉 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## ⚡ Core Rules & Environment

- **Environment**: Strict Python 3.12+ requirement. Use `.venv` for all commands.
- **Styling**: Never edit `static/styles.css`. Modify `tabs/_styling.py` or `tabs/_common.py`.
- **Data Pipeline**: All analysis modules must use `utils/data_cleaning.py`.

## 🧠 Deep Implementation Details (Bot Confidence)

- **Data Pipeline**: Analysis MUST call `utils/data_cleaning.py -> prepare_data_for_analysis()` to handle `complete-case` logic.
- **Reporting**: Follow the element-based schema using `utils.formatting.generate_standard_report()`.
- **UI States**: Standardize on `create_loading_state`, `create_placeholder_state`, and `create_error_alert` from `utils/ui_helpers.py`.
- **P-Values**: Always use `utils/formatting.py -> format_p_value()` to respect global `CONFIG`.

## 🛠️ Automated Workflows

Common tasks are automated via markdown workflows in:
📂 **[.agent/workflows/](.agent/workflows/)**

### Quick Start for Bots

1. Run `/ai-readfirst` to verify the environment.
2. Read `docs/ARCHITECTURE.md` before making UI or Data changes.
3. Follow Rule 0 in `.cursorrules`.

---
*This repository is optimized for AI-Human pair programming. Stay architecture-first.*

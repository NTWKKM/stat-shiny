# RabbitAI Report

In `@utils/logic.py` around lines 1125 - 1160, The AUC interpretation thresholds
used when building diag_rows (variable auc_interp derived from mv_stats["auc"])
need an inline comment documenting the source of those cutoffs; update the block
that sets auc_interp (around the AUC/Discrimination logic) to add a brief
comment citing the reference or guideline that justifies ">0.8 Excellent, >0.7
Acceptable, else Poor" (e.g., Hosmer & Lemeshow or another authoritative paper),
and if appropriate include a short note linking to the citation or paper title
so future readers know where the thresholds came from.

🧹 Nitpick | 🔵 Trivial

AUC interpretation thresholds are reasonable but consider documenting the source.

The interpretation thresholds (>0.8 Excellent, >0.7 Acceptable, else Poor) follow common conventions but vary across literature. Consider adding a comment citing the source (e.g., Hosmer-Lemeshow 2000, or similar)
------------------------------------------------------------

In `@tabs/tab_corr.py` around lines 159 - 161, The markdown uses GitHub-flavored
alert syntax (the literal string "> [!WARNING]") which Shiny's ui.markdown()
won't render as a styled alert; locate the ui.markdown(...) call(s) that include
the "> [!WARNING]" block in tab_corr.py (and the similar occurrences in
tab_data.py) and replace them with HTML alert markup (e.g., a <div class="alert
alert-warning">...</div>) or use Shiny's alert component API so the warning is
rendered properly; ensure you update each occurrence of the exact string ">
[!WARNING]" and keep the same message text ("Correlation does not imply
causation...") inside the new HTML or Shiny alert wrapper.

⚠️ Potential issue | 🟡 Minor

🧩 Analysis chain
🏁 Script executed:

# !/bin/bash

# Check if there are other usages of GFM alert syntax in the codebase

rg -n '\[!WARNING\]|\[!NOTE\]|\[!TIP\]|\[!CAUTION\]|\[!IMPORTANT\]' --type=py
Repository: NTWKKM/stat-shiny

Length of output: 243

🏁 Script executed:

# !/bin/bash

# Also check the file context to understand how the markdown is being used in Shiny

head -n 180 tabs/tab_corr.py | tail -n 30
Repository: NTWKKM/stat-shiny

Length of output: 1855

🏁 Script executed:

# !/bin/bash

# Find imports to understand if this is a Shiny app and how markdown is used

grep -n "import\|from.*import" tabs/tab_corr.py | head -20
Repository: NTWKKM/stat-shiny

Length of output: 470

🌐 Web query:

Shiny Python ui.markdown GFM alert syntax support [!WARNING] rendering

💡 Result:

Shiny for Python’s ui.markdown() renders Markdown server-side using a customized markdown_id.main.MarkdownIt (from the markdown-it ecosystem) and generally supports “GitHub-flavored markdown”, but GitHub’s alert blocks syntax (> [!WARNING], etc.) typically won’t render as a styled alert unless an alert plugin is enabled—otherwise it will just look like a normal blockquote with text. [1]

Workaround: supply your own renderer (with a GFM-alerts plugin)
Use render_func= to plug in a MarkdownIt instance that enables a GFM-alerts plugin (Python ecosystem example shown below). Shiny explicitly supports providing a custom renderer via render_func. [1]

from shiny import ui
from markdown_it import MarkdownIt
from mdformat_gfm_alerts.mdit_plugins import gfm_alerts_plugin

md = MarkdownIt("gfm").use(gfm_alerts_plugin)

ui.markdown(
    """> [!WARNING]
> This is a warning.""",
    render_func=md.render,
)
The mdformat-gfm-alerts plugin is designed to turn > [!WARNING] into HTML with alert classes (e.g., markdown-alert markdown-alert-warning). [2]

Styling
Alert plugins usually emit specific CSS classes; you’ll likely want to add CSS for those classes (either your own, or the plugin’s provided styles in its ecosystem). GFM-alert plugins document the expected syntax and behavior. [3]

Sources: Shiny ui.markdown API (custom render_func) [1]; mdformat-gfm-alerts plugin example output [2]; GFM alert plugin syntax overview [3].

[1] (shiny.posit.co)
[2] (pypi.org)
[3] (mdit-plugins.github.io)

Citations:

1: <https://shiny.posit.co/py/api/core/ui.markdown.html?utm_source=openai>
2: <https://pypi.org/project/mdformat-gfm-alerts/?utm_source=openai>
3: <https://mdit-plugins.github.io/alert.html?utm_source=openai>
GFM alert syntax won't render as styled alerts in Shiny.

The > [!WARNING] syntax is GitHub-flavored markdown, but Shiny's ui.markdown() doesn't support it by default—it will render as a plain blockquote with literal [!WARNING] text. This pattern appears in at least three files (tab_corr.py:159, tab_data.py:84, tab_data.py:1055).

Use HTML markup, Shiny alert components, or a custom markdown renderer with the mdformat-gfm-alerts plugin instead
-------------------------------------

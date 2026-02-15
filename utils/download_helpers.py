"""
Centralized download helpers for HTML report generation.

Provides utilities to ensure download handlers yield valid HTML,
handle errors gracefully, and show user notifications.
"""

import html as _html
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _notify(message: str, *, type: str = "message", duration: int = 3) -> None:
    """Show a Shiny notification, silently skipping if no session is active."""
    try:
        from shiny import ui

        ui.notification_show(message, type=type, duration=duration)
    except Exception:
        # Outside a Shiny session (e.g. tests, CLI) – just log instead
        logger.info("Notification (%s): %s", type, message)


def _build_error_html(title: str, message: str) -> str:
    """Build a minimal standalone HTML error page."""
    safe_title = _html.escape(str(title))
    safe_message = _html.escape(str(message))
    return (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        "display: flex; justify-content: center; align-items: center; min-height: 80vh; "
        "background: #f8f9fa; color: #333; }"
        ".error-box { text-align: center; max-width: 480px; background: white; "
        "padding: 40px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }"
        ".error-icon { font-size: 3em; margin-bottom: 12px; }"
        "h1 { color: #d63384; font-size: 1.5em; margin-bottom: 8px; }"
        "p { color: #666; line-height: 1.6; }"
        "</style>"
        "</head><body>"
        "<div class='error-box'>"
        "<div class='error-icon'>⚠️</div>"
        f"<h1>{safe_title}</h1>"
        f"<p>{safe_message}</p>"
        f"<p style='font-size:0.8em;color:#999;margin-top:20px;'>"
        f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        "</div></body></html>"
    )


def safe_download_html(
    content: str | None,
    *,
    label: str = "Report",
) -> str:
    """
    Validate HTML content for download, returning an error page if empty.

    Use this inside ``@render.download`` handlers to guarantee the user
    always receives a valid HTML file and sees a notification about the
    outcome.

    Args:
        content: The HTML string to download. ``None`` or empty strings
            are treated as "no results available".
        label: Human-readable report name used in notifications and the
            fallback error page (e.g. ``"ROC Report"``).

    Returns:
        A valid HTML string – either the original *content* or a styled
        error page explaining that results are unavailable.

    Example::

        @render.download(filename="roc_report.html")
        def btn_dl_roc_report():
            yield safe_download_html(roc_html.get(), label="ROC Report")
    """
    if not content or not content.strip():
        logger.warning("Download attempted with no content for: %s", label)
        _notify(
            f"⚠️ {label}: No results available. Please run the analysis first.",
            type="warning",
            duration=5,
        )
        return _build_error_html(
            f"{label} – No Results",
            "No analysis results are available for download. "
            "Please run the analysis first, then try downloading again.",
        )

    # Content is valid – notify success
    _notify(
        f"✅ {label} downloaded successfully.",
        type="message",
        duration=3,
    )
    return content


def safe_report_generation(
    generate_fn,
    *,
    label: str = "Report",
) -> str:
    """
    Safely execute a report-generation callable, catching exceptions.

    Wraps *generate_fn* in a try/except so that any error during inline
    report building (e.g. Plotly rendering, DataFrame formatting) is
    caught and converted to a downloadable error page instead of
    silently yielding nothing.

    Args:
        generate_fn: A zero-argument callable that returns an HTML string.
        label: Human-readable name for notifications.

    Returns:
        The generated HTML string, or an error-page HTML on failure.

    Example::

        @render.download(filename="cox_report.html")
        def btn_dl_cox():
            def _build():
                res = cox_result.get()
                if not res:
                    return None
                elements = [...]
                return survival_lib.generate_report_survival(...)

            yield safe_report_generation(_build, label="Cox Report")
    """
    try:
        result = generate_fn()
        return safe_download_html(result, label=label)
    except Exception:
        logger.exception("Report generation failed for: %s", label)
        _notify(
            f"❌ {label}: Report generation failed. See logs for details.",
            type="error",
            duration=8,
        )
        return _build_error_html(
            f"{label} – Generation Failed",
            "An error occurred while generating the report. "
            "Please check the analysis results and try again.",
        )

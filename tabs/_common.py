from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shiny.ui import TagChild


def wrap_with_container(content: "TagChild") -> "TagChild":
    """
    Wraps UI content with the .app-container CSS class.

    Args:
        content: The UI content (TagChild) to wrap.

    Returns:
        A shiny.ui.div element with class='app-container'.
    """
    from shiny import ui  # Import inside function to avoid circular imports

    return ui.div(content, class_="app-container")


def get_color_palette() -> dict[str, str]:
    """
    Returns a unified color palette dictionary for all modules.
    Ensures consistency across the application.

    Returns:
        dict[str, str]: A dictionary mapping color names to hex codes.
    """
    return {
        # Primary colors - Navy Blue theme
        "primary": "#1E3A5F",
        "primary_dark": "#0F2440",
        "primary_light": "#E8EEF7",
        # Neutral colors - Light theme
        "smoke_white": "#F8F9FA",
        "text": "#1F2328",
        "text_secondary": "#6B7280",
        "border": "#E5E7EB",
        "background": "#F9FAFB",
        "surface": "#FFFFFF",
        # Status/Semantic colors
        "success": "#22A765",
        "danger": "#E74856",
        "warning": "#FFB900",
        "info": "#5A7B8E",
        "neutral": "#D1D5DB",
    }


def select_variable_by_keyword(
    columns: list[str], keywords: list[str], default_to_first: bool = True
) -> str | None:
    """
    Intelligently attempts to select a default variable from a list of columns
    based on a list of keywords.

    Args:
        columns: List of available column names.
        keywords: List of keywords to search for (case-insensitive).
        default_to_first: If True, returns the first column if no keyword match is found.

    Returns:
        The matched column name, or the first column (if default_to_first is True),
        or None if no match/no columns.
    """
    if not columns:
        return None

    # Try to find a match by keyword priority
    for k in keywords:
        k_lower = k.lower()
        for col in columns:
            if k_lower in col.lower():
                return col

    # Default fallback
    if default_to_first:
        return columns[0]

    return None

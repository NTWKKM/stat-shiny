from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

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
    slate_50 = "#F8FAFC"  # Slate 50
    return {
        # Primary colors - Slate theme
        "primary": "#0F172A",  # Slate 900
        "primary_dark": "#020617",  # Slate 950
        "primary_light": slate_50,
        "secondary": "#64748B",  # Slate 500
        # Neutral colors - Light theme
        "smoke_white": slate_50,
        "text": "#0F172A",  # Slate 900
        "text_secondary": "#64748B",  # Slate 500
        "border": "#E2E8F0",  # Slate 200
        "background": "#FAFAFA",  # Very clean off-white
        "surface": "#FFFFFF",
        # Status/Semantic colors - Soft/desaturated pastels
        "success": "#059669",  # Emerald 600
        "danger": "#DC2626",  # Red 600
        "warning": "#D97706",  # Amber 600
        "info": "#475569",  # Slate 600
        "neutral": "#CBD5E1",  # Slate 300
        "stale": "#D97706",  # Amber 600 - Stale inputs
        "stale_bg": "#FFFBEB",  # Amber 50
        "stale_border": "#FDE68A",  # Amber 200
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

    # Tier 1: Exact match (case-insensitive) across keyword priority
    for k in keywords:
        k_lower = k.lower().strip()
        for col in columns:
            if k_lower == col.lower().strip():
                return col

    # Tier 2: Word / Underscore / Token Boundary match
    # Enforces token boundaries to prevent false positive substring matches (e.g., 'n_statin' matching 'mean_statin')
    for k in keywords:
        k_lower = k.lower().strip()
        pattern = rf"(^|[^a-zA-Z0-9]){re.escape(k_lower)}([^a-zA-Z0-9]|$)"
        for col in columns:
            if re.search(pattern, col.lower()):
                return col

    # Default fallback
    if default_to_first:
        return columns[0]

    return None


class VariableRoles:
    """
    Standardized dataclass holding inferred or user-assigned variable roles across the application.
    Acts as a single source of truth for downstream analysis modules.
    """

    def __init__(
        self,
        outcome: str | None = None,
        exposure: str | None = None,
        covariates: list[str] | None = None,
        time: str | None = None,
        event: str | None = None,
        strata: str | None = None,
    ):
        self.outcome = outcome
        self.exposure = exposure
        self.covariates = covariates or []
        self.time = time
        self.event = event
        self.strata = strata

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "exposure": self.exposure,
            "covariates": self.covariates,
            "time": self.time,
            "event": self.event,
            "strata": self.strata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VariableRoles":
        if not d:
            return cls()
        return cls(
            outcome=d.get("outcome"),
            exposure=d.get("exposure"),
            covariates=d.get("covariates", []),
            time=d.get("time"),
            event=d.get("event"),
            strata=d.get("strata"),
        )


def infer_variable_roles(columns: list[str]) -> VariableRoles:
    """
    Intelligently infers variable roles based on biostatistical naming conventions.

    Args:
        columns: List of available dataset column names.

    Returns:
        VariableRoles instance with inferred column assignments.
    """
    if not columns:
        return VariableRoles()

    # Heuristic Keyword Dictionaries
    TIME_KEYWORDS = [
        "time",
        "duration",
        "followup",
        "survival_time",
        "days",
        "months",
        "tt_event",
        "tte",
        "surv_time",
    ]
    EVENT_KEYWORDS = [
        "status",
        "event",
        "death",
        "died",
        "recurrence",
        "censored",
        "mortality",
        "endpoint",
        "dead",
    ]
    EXPOSURE_KEYWORDS = [
        "treatment",
        "treat",
        "group",
        "arm",
        "exposure",
        "rx",
        "intervention",
        "drug",
        "therapy",
    ]
    OUTCOME_KEYWORDS = [
        "outcome",
        "target",
        "y",
        "disease",
        "response",
        "case",
        "diagnosis",
        "result",
    ]
    STRATA_KEYWORDS = [
        "subgroup",
        "strata",
        "stratification",
        "cohort",
        "center",
        "site",
        "cluster",
    ]

    assigned_cols: set[str] = set()

    def _select(keywords: list[str]) -> str | None:
        available = [c for c in columns if c not in assigned_cols]
        chosen = select_variable_by_keyword(available, keywords, default_to_first=False)
        if chosen is not None:
            assigned_cols.add(chosen)
        return chosen

    time_col = _select(TIME_KEYWORDS)
    event_col = _select(EVENT_KEYWORDS)
    exposure_col = _select(EXPOSURE_KEYWORDS)
    outcome_col = _select(OUTCOME_KEYWORDS)
    strata_col = _select(STRATA_KEYWORDS)

    covariates = [c for c in columns if c not in assigned_cols]

    return VariableRoles(
        outcome=outcome_col,
        exposure=exposure_col,
        covariates=covariates,
        time=time_col,
        event=event_col,
        strata=strata_col,
    )

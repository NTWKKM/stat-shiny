"""
🚦 Result State Machine Module
------------------------------
Manages lifecycle states for statistical calculation modules:
- EMPTY: No data or variables selected.
- CONFIGURING: Variables are selected, awaiting run.
- COMPUTING: Model fitting / calculation in progress.
- FRESH: Calculated output is strictly in sync with current inputs.
- STALE: User modified input parameters after running the model.

Provides input fingerprinting to deterministically detect stale results.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from shiny import ui


class ResultState(str, Enum):
    EMPTY = "empty"
    CONFIGURING = "configuring"
    COMPUTING = "computing"
    FRESH = "fresh"
    STALE = "stale"


def _canonicalize_item(val: Any) -> Any:
    if isinstance(val, dict):
        return {
            str(k): _canonicalize_item(v)
            for k, v in sorted(val.items(), key=lambda x: str(x[0]))
        }
    elif isinstance(val, (list, tuple, set)):
        items = [_canonicalize_item(item) for item in val]
        return sorted(
            items,
            key=lambda x: json.dumps(x, sort_keys=True)
            if isinstance(x, (dict, list))
            else str(x),
        )
    elif val is None or isinstance(val, (int, float, bool, str)):
        return val
    else:
        return str(val)


def compute_input_fingerprint(params: dict[str, Any]) -> str:
    """
    Computes a deterministic MD5 hash string of input parameters.

    Args:
        params: Dictionary of input parameter names and values.

    Returns:
        Hexadecimal hash string.
    """
    normalized = _canonicalize_item(params)
    raw_str = json.dumps(normalized, sort_keys=True)
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()


def get_result_state(
    has_run: bool,
    is_computing: bool,
    current_fingerprint: str | None,
    last_run_fingerprint: str | None,
    has_required_inputs: bool = True,
) -> ResultState:
    """
    Determines the current ResultState based on calculation flags and parameter fingerprints.

    Args:
        has_run: True if calculation has executed at least once.
        is_computing: True if calculation is currently running.
        current_fingerprint: Hash of the active UI inputs.
        last_run_fingerprint: Hash of the inputs used in the latest successful run.
        has_required_inputs: True if mandatory variables (e.g. outcome/exposure) are set.

    Returns:
        ResultState enum member.
    """
    if is_computing:
        return ResultState.COMPUTING
    if not has_required_inputs:
        return ResultState.EMPTY
    if not has_run:
        return ResultState.CONFIGURING
    if (
        current_fingerprint
        and last_run_fingerprint
        and current_fingerprint != last_run_fingerprint
    ):
        return ResultState.STALE
    return ResultState.FRESH


def render_state_badge(state: ResultState) -> ui.Tag:
    """
    Renders an accessible, styled status pill badge representing the ResultState.

    Args:
        state: The ResultState enum member.

    Returns:
        A Shiny UI span element with appropriate styling and accessibility attributes.
    """
    if state == ResultState.FRESH:
        return ui.span(
            ui.HTML("✓ Synchronized"),
            class_="result-state-badge state-fresh",
            title="Analysis results match current inputs",
            role="status",
            aria_label="Results are synchronized with current inputs",
        )
    elif state == ResultState.STALE:
        return ui.span(
            ui.HTML("⚠️ Inputs Changed"),
            class_="result-state-badge state-stale",
            title="Inputs have been modified since last run. Click 'Run Analysis' to update.",
            role="alert",
            aria_label="Inputs have changed since last calculation. Please re-run analysis.",
        )
    elif state == ResultState.COMPUTING:
        return ui.span(
            ui.HTML("⏳ Computing..."),
            class_="result-state-badge state-computing",
            role="status",
            aria_live="polite",
        )
    elif state == ResultState.CONFIGURING:
        return ui.span(
            ui.HTML("⚙️ Ready to Run"),
            class_="badge bg-light text-secondary border",
            role="status",
        )
    else:
        return ui.span(
            ui.HTML("⚪ No Analysis"),
            class_="badge bg-light text-muted border",
            role="status",
        )


def render_stale_warning_banner(action_button_id: str = "btn_run") -> ui.Tag:
    """
    Renders a prominent amber warning callout when results are stale.

    Args:
        action_button_id: The ID of the button to click for re-computation.

    Returns:
        A Shiny UI div element.
    """
    return ui.div(
        ui.div(
            ui.strong("⚠️ Inputs Modified: "),
            "The model settings or selected variables have changed since the last calculation. Displayed results may not reflect current inputs.",
            class_="mb-1",
        ),
        ui.div(
            "Please click ",
            ui.strong("Run Analysis"),
            " to refresh the statistics and confidence intervals.",
            class_="text-muted-sm",
        ),
        class_="stale-warning-banner",
        role="alert",
    )

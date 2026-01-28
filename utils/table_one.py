"""
📈 Table One Generator Module (Wrapper for Advanced Generator)
Backwards compatible wrapper around `utils.table_one_advanced`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from logger import get_logger

# New Advanced Implementation
from utils.table_one_advanced import (
    TableOneGenerator,
    VariableClassifier,
)

logger = get_logger(__name__)

# --- Legacy Functions (Kept for compatibility if imported directly) ---


def check_normality(series: pd.Series) -> bool:
    """Wrapper for VariableClassifier logic"""
    return VariableClassifier.classify(series) == "continuous_normal"


def format_p(p: float | str) -> str:
    """Wrapper for p-value formatting"""
    if pd.isna(p) or isinstance(p, str):
        return "-"
    if float(p) < 0.001:
        return "<0.001"
    return f"{float(p):.3f}"


# Note: These legacy calculation functions are kept just in case external scripts import them.
# Internally, generate_table will uses the new Engine.


def compute_or_ci(a, b, c, d) -> str:
    # See original implementation if needed, but for now we provide a functional stub or copy logic
    # Copying original logic to be safe for direct imports
    try:
        if min(a, b, c, d) == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
        or_val = (a * d) / (b * c)
        if or_val == 0 or np.isinf(or_val):
            return "-"
        ln_or = np.log(or_val)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        return f"{or_val:.2f} ({np.exp(ln_or - 1.96 * se):.2f}-{np.exp(ln_or + 1.96 * se):.2f})"
    except Exception:
        return "-"


# Re-expose other functions if critical...
# For now, we assume most consumers only use generate_table


# --- MAIN ENTRY POINT ---


def generate_table(
    df: pd.DataFrame,
    selected_vars: list[str],
    group_col: str | None,
    var_meta: dict[str, Any] | None,
    or_style: str = "all_levels",
) -> str:
    """
    Generate baseline characteristics table using the new Advanced TableOneGenerator.

    Args:
        df: Input DataFrame
        selected_vars: List of variables to include
        group_col: Column to stratify by (optional)
        var_meta: Metadata dictionary (labels, etc.)
        or_style: 'all_levels' or 'simple' (Passed to generator if implemented, currently only standard supported)

    Returns:
        HTML string of the table.
    """
    try:
        # Instantiate the new Generator
        generator = TableOneGenerator(df, var_meta=var_meta)

        # Determine stratify_by
        stratify = group_col if (group_col and group_col != "None") else None

        # Generate HTML
        # Note: The new generate method handles classification, stats, and HTML rendering internally.
        html_output = generator.generate(selected_vars, stratify_by=stratify)

        return html_output

    except Exception as e:
        logger.error(f"Error in Table One generation (Advanced Wrapper): {e}")
        # Fallback or re-raise?
        # Re-raise to match previous behavior so UI handles the error message
        raise e

from __future__ import annotations

import pandas as pd

from typing import Any, Sequence


def _is_numeric_column(
    series: pd.Series, total_rows: int
) -> tuple[bool, pd.Series, pd.Series, pd.Series, int]:
    """
    Determine whether a pandas Series should be treated as a numeric column and identify non-standard or unparseable numeric values.
    
    Parameters:
        series (pd.Series): Column values to analyze.
        total_rows (int): Total number of rows in the DataFrame (used to compute the numeric ratio).
    
    Returns:
        tuple:
            is_numeric_col (bool): `True` if more than 50% of entries parse as numbers after coercion, `False` otherwise.
            numeric_strict (pd.Series): Result of a strict numeric conversion where non-numeric entries are `NaN`.
            numeric_coerced (pd.Series): Numeric conversion after cleaning common non-numeric symbols (e.g., percent signs, currency, comparison symbols); may parse values that strict conversion did not.
            is_strict_nan (pd.Series): Boolean mask marking positions where strict conversion is `NaN` but the original entry was non-empty (i.e., non-standard numeric-like values).
            strict_nan_count (int): Number of `True` values in `is_strict_nan`.
    """
    # Try strict conversion
    numeric_strict = pd.to_numeric(series, errors="coerce")

    # Try coerced conversion (handling common symbols like <, >, %)
    if pd.api.types.is_object_dtype(series):
        clean_s = (
            series.astype(str)
            .str.replace(r"[<|>|,|%|$|€|£]", "", regex=True)
            .str.strip()
        )
        numeric_coerced = pd.to_numeric(clean_s, errors="coerce")
    else:
        numeric_coerced = numeric_strict

    # Valid if at least 50% can be numeric
    valid_numeric_count = numeric_coerced.notna().sum()
    is_numeric_col = (
        (valid_numeric_count / total_rows) > 0.5 if total_rows > 0 else False
    )

    # Non-standard: it's NaN in strict but has a value in coerced (like "<5")
    # OR it has a value in original but NaN in coerced (like "abc")
    is_strict_nan = (
        numeric_strict.isna() & series.notna() & (series.astype(str).str.strip() != "")
    )
    strict_nan_count = is_strict_nan.sum()

    return (
        is_numeric_col,
        numeric_strict,
        numeric_coerced,
        is_strict_nan,
        strict_nan_count,
    )


def _format_row_list(rows: Sequence[Any], max_show: int = 400) -> str:
    """
    Format a sequence of row indices or values into a concise, comma-separated string for reporting.
    
    Parameters:
        rows (Sequence[Any]): Sequence of values (e.g., row indices or category values); each element will be converted to a string.
        max_show (int): Maximum number of items to include before truncating the list and appending a summary of remaining items.
    
    Returns:
        str: A comma-separated string of the items. If `rows` is empty returns an empty string. If the sequence length exceeds `max_show`, the returned string contains the first `max_show` items followed by ", ... (+N more)" where N is the number of omitted items.
    """
    # FIX: Use len(rows) == 0 instead of 'if not rows' to handle numpy arrays safely
    if len(rows) == 0:
        return ""

    count = len(rows)
    if count <= max_show:
        return ",".join(map(str, rows))
    else:
        shown = ",".join(map(str, rows[:max_show]))
        remaining = count - max_show
        return f"{shown}, ... (+{remaining} more)"


def check_data_quality(df: pd.DataFrame) -> list[str]:
    """
    Generate human-readable data-quality warnings for each column in the provided DataFrame.
    
    Checks performed for each column:
    - Missing values: reports count and row indices where present.
    - Numeric columns: detects non-standard numeric representations (e.g., "<5", "10%") and reports their row indices and example values.
    - Categorical columns: detects values that look numeric and reports their row indices and example values.
    - Rare categories: when the column is not dominated by unique values (unique ratio < 0.8), reports categories that occur fewer than 5 times.
    
    Parameters:
        df (pd.DataFrame): DataFrame to analyze.
    
    Returns:
        list[str]: A list of formatted warning messages, one per column with detected issues.
    """
    warnings = []
    total_rows = len(df)

    if total_rows == 0:
        return warnings

    for col in df.columns:
        col_issues = []
        series = df[col]

        # 0. Missing Data Check
        is_na = series.isna()
        missing_count = is_na.sum()
        if missing_count > 0:
            error_rows = df.index[is_na].tolist()
            # Show detailed list (up to 10000 - practically all) to serve as full log
            row_str = _format_row_list(error_rows, max_show=10000)
            col_issues.append(f"Missing {missing_count} values at rows `{row_str}`.")

        # Type detection helper
        (
            is_numeric_col,
            numeric_strict,
            numeric_coerced,
            is_strict_nan,
            strict_nan_count,
        ) = _is_numeric_column(series, total_rows)

        # CASE 1: Numeric Column
        if is_numeric_col:
            if strict_nan_count > 0:
                error_rows = df.index[is_strict_nan].tolist()
                bad_values = series.loc[is_strict_nan].unique()

                row_str = _format_row_list(error_rows, max_show=10000)
                val_str = _format_row_list(
                    bad_values, max_show=20
                )  # Keep values concise

                col_issues.append(
                    f"Found {strict_nan_count} non-standard values at rows `{row_str}` (Values: `{val_str}`)."
                )

        # CASE 2: Categorical Column
        else:
            # Check for numbers in categorical text
            original_vals = series.astype(str).str.strip()
            # It is a number if numeric_strict is not NaN and original was not empty
            is_numeric_in_text = (numeric_strict.notna()) & (original_vals != "")
            numeric_in_text_count = is_numeric_in_text.sum()

            if numeric_in_text_count > 0:
                error_rows = df.index[is_numeric_in_text].tolist()
                bad_values = series.loc[is_numeric_in_text].unique()

                row_str = _format_row_list(error_rows, max_show=10000)
                val_str = _format_row_list(bad_values, max_show=20)

                col_issues.append(
                    f"Found {numeric_in_text_count} numeric values inside categorical column at rows `{row_str}` (Values: `{val_str}`)."
                )

            # Rare categories check
            unique_ratio = series.nunique() / total_rows
            # Only check if it's truly categorical (not unique IDs)
            if unique_ratio < 0.8:
                val_counts = series.value_counts()
                rare_threshold = 5
                rare_mask = val_counts < rare_threshold
                rare_vals = val_counts[rare_mask].index.tolist()

                if rare_vals:
                    val_str = _format_row_list(rare_vals, max_show=20)
                    col_issues.append(
                        f"Found rare categories (<{rare_threshold} times): `{val_str}`."
                    )

        if col_issues:
            full_msg = " ".join(col_issues)
            warnings.append(f"**Column '{col}':** {full_msg}")

    return warnings
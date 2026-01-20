"""
🧪 Propensity Score Matching (PSM) Library (Shiny Compatible)

Propensity score calculation and matching without Streamlit dependencies.
OPTIMIZED for Python 3.12 with strict type hints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

from config import CONFIG
from logger import get_logger
from utils.data_cleaning import apply_missing_values_to_df, get_missing_summary_df

logger = get_logger(__name__)


def calculate_propensity_score(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list[str],
    var_meta: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Calculate propensity scores using logistic regression.
    """
    try:
        # --- MISSING DATA HANDLING ---
        # PSM in this library currently uses simple mean imputation
        df_subset = df[covariate_cols].copy()

        missing_data_info = {}
        if var_meta:
            missing_cfg = CONFIG.get("analysis.missing", {}) or {}
            missing_codes = missing_cfg.get("user_defined_values", [])
            missing_summary = get_missing_summary_df(df_subset, var_meta, missing_codes)
            df_processed = apply_missing_values_to_df(
                df_subset, var_meta, missing_codes
            )

            # 3. Create report info (Strategy: Mean Imputation)
            # We don't drop rows here, we impute.
            # Calculate how many rows WOULD have been dropped if we did complete-case
            n_rows = len(df)
            n_complete = len(df_processed.dropna())
            n_imputed = n_rows - n_complete

            missing_data_info = {
                "strategy": "Mean Imputation (Automatic)",
                "rows_analyzed": n_rows,
                "rows_excluded": 0,  # We keep all rows via imputation
                "note": f"⚠️ {n_imputed} rows containing missing values were imputed with the mean.",
                "summary_before": missing_summary.to_dict("records"),
            }
            X = df_processed.fillna(df_processed.mean())
        else:
            # Fallback path (existing logic but slightly cleaner)
            X = df[covariate_cols].fillna(df[covariate_cols].mean())

        y = df[treatment_col].astype(int)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        ps = model.predict_proba(X)[:, 1]
        logger.info(
            f"Propensity scores calculated: mean={ps.mean():.3f}, std={ps.std():.3f}"
        )
        return pd.Series(ps, index=df.index), missing_data_info
    except Exception as e:
        logger.error(f"PSM calculation error: {e}")
        raise


def perform_matching(
    df: pd.DataFrame, treatment_col: str, ps_col: str, caliper: float = 0.1
) -> pd.DataFrame:
    """
    Perform 1:1 nearest neighbor matching on propensity scores.
    """
    try:
        treated = df[df[treatment_col] == 1].copy()
        control = df[df[treatment_col] == 0].copy()

        matched_pairs: list[int] = []
        used_controls: set[Any] = set()

        # Simple nearest neighbor matching
        for idx, treated_row in treated.iterrows():
            ps_treated = treated_row[ps_col]

            distances = (control[ps_col] - ps_treated).abs()
            # Only consider controls not yet used
            available_control_distances = distances[
                ~distances.index.isin(used_controls)
            ]

            if not available_control_distances.empty:
                min_dist_idx = available_control_distances.idxmin()
                min_dist = available_control_distances[min_dist_idx]

                if min_dist <= caliper:
                    matched_pairs.append(idx)
                    matched_pairs.append(min_dist_idx)
                    used_controls.add(min_dist_idx)

        matched_df = df.loc[matched_pairs].copy()
        logger.info(f"Matched {len(matched_pairs) // 2} pairs (caliper={caliper})")
        return matched_df
    except Exception as e:
        logger.error(f"Matching error: {e}")
        raise


def compute_smd(
    df: pd.DataFrame, treatment_col: str, covariate_cols: list[str]
) -> pd.DataFrame:
    """
    Calculate Standardized Mean Difference (SMD) for balance checking.
    """
    smd_data = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    for col in covariate_cols:
        # Basic check for numeric or boolean encoded as numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        m1 = treated[col].mean()
        m0 = control[col].mean()
        v1 = treated[col].var()
        v0 = control[col].var()

        # Standardized Mean Difference formula
        pooled_sd = np.sqrt((v1 + v0) / 2)
        if pooled_sd < 1e-10:
            smd = 0
        else:
            smd = abs(m1 - m0) / pooled_sd

        smd_data.append({"Variable": col, "SMD": smd})

    return pd.DataFrame(smd_data)


def plot_love_plot(smd_pre: pd.DataFrame, smd_post: pd.DataFrame) -> go.Figure:
    """
    Create a Love Plot (Dot Plot for SMD) using Plotly.
    """
    # Merge data for plotting
    df_plot = smd_pre.merge(
        smd_post, on="Variable", suffixes=("_Unmatched", "_Matched")
    )

    fig = go.Figure()

    # Add reference lines
    fig.add_vline(
        x=0.1, line_dash="dash", line_color="gray", annotation_text="0.1 (Target)"
    )
    fig.add_vline(x=0, line_color="black")

    # Add Unmatched points (Red Circles)
    fig.add_trace(
        go.Scatter(
            x=df_plot["SMD_Unmatched"],
            y=df_plot["Variable"],
            mode="markers",
            name="Unmatched",
            marker={"color": "red", "size": 10, "symbol": "circle"},
        )
    )

    # Add Matched points (Green Diamonds)
    fig.add_trace(
        go.Scatter(
            x=df_plot["SMD_Matched"],
            y=df_plot["Variable"],
            mode="markers",
            name="Matched",
            marker={"color": "green", "size": 12, "symbol": "diamond"},
        )
    )

    fig.update_layout(
        title="Love Plot: Balance Assessment (SMD)",
        xaxis_title="Standardized Mean Difference (SMD)",
        yaxis_title="Variables",
        template="plotly_white",
        height=max(400, len(df_plot) * 30),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return fig


def generate_psm_report(title: str, elements: list[dict[str, Any]]) -> str:
    """
    Generate a simple HTML report for PSM results.
    """
    html = f"<html><head><title>{title}</title>"
    html += "<style>body {font-family: sans-serif; padding: 20px;} table {border-collapse: collapse; width: 100%; margin: 10px 0;} th, td {border: 1px solid #ddd; padding: 8px; text-align: left;} th {background-color: #f2f2f2;}</style>"
    html += f"</head><body><h1>{title}</h1>"

    for el in elements:
        if el["type"] == "text":
            html += f"<h3>{el['data']}</h3>"
        elif el["type"] == "table":
            if isinstance(el["data"], pd.DataFrame):
                html += el["data"].to_html()
            else:
                html += str(el["data"])
        elif el["type"] == "plot":
            # Convert Plotly figure to HTML div
            if hasattr(el["data"], "to_html"):
                html += el["data"].to_html(full_html=False, include_plotlyjs="cdn")

    html += "</body></html>"
    return html

from typing import Any, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable, Independence
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.regression.mixed_linear_model import MixedLM


def run_gee(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    subject_col: str,
    covariates: List[str] = [],
    cov_struct: str = "exchangeable",
    family_str: str = "gaussian",
) -> Any:
    """
    Run Generalized Estimating Equations (GEE) model.

    Args:
        df: DataFrame containing the data
        outcome_col: Name of the outcome variable
        treatment_col: Name of the treatment/group variable
        time_col: Name of the time variable
        subject_col: Name of the subject ID variable
        covariates: List of additional covariate names
        cov_struct: Correlation structure ('exchangeable', 'independence', 'ar1')
        family_str: Distribution family ('gaussian', 'binomial', 'poisson', 'gamma')

    Returns:
        Fitted GEE model results
    """
    # Construct formula
    formula = (
        f"{outcome_col} ~ {treatment_col} + {time_col} + {treatment_col}:{time_col}"
    )
    if covariates:
        formula += " + " + " + ".join(covariates)

    # Set family
    family_map = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(),
    }
    family = family_map.get(family_str.lower(), sm.families.Gaussian())

    # Set correlation structure
    cov_struct_map = {
        "exchangeable": Exchangeable(),
        "independence": Independence(),
        "ar1": Autoregressive(),
    }
    covariance = cov_struct_map.get(cov_struct.lower(), Exchangeable())

    try:
        model = GEE.from_formula(
            formula, groups=subject_col, data=df, family=family, cov_struct=covariance
        )
        results = model.fit()
        return results
    except Exception as e:
        return f"Error running GEE: {str(e)}"


def run_lmm(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_col: str,
    subject_col: str,
    covariates: List[str] = [],
    random_slope: bool = False,
) -> Any:
    """
    Run Linear Mixed Model (LMM).

    Args:
        df: DataFrame containing the data
        outcome_col: Name of the outcome variable
        treatment_col: Name of the treatment/group variable
        time_col: Name of the time variable
        subject_col: Name of the subject ID variable
        covariates: List of additional covariate names
        random_slope: If True, includes random slope for time variable

    Returns:
        Fitted MixedLM model results
    """
    # Construct formula
    formula = (
        f"{outcome_col} ~ {treatment_col} + {time_col} + {treatment_col}:{time_col}"
    )
    if covariates:
        formula += " + " + " + ".join(covariates)

    try:
        if random_slope:
            # Random intercept and slope for time
            # re_formula="~time_col"
            # Note: statsmodels requires the variable name in re_formula
            # If time_col is "Time", re_formula="~Time"
            re_formula = f"~{time_col}"
            model = MixedLM.from_formula(
                formula, groups=subject_col, re_formula=re_formula, data=df
            )
        else:
            # Random intercept only
            model = MixedLM.from_formula(formula, groups=subject_col, data=df)

        results = model.fit()
        return results
    except Exception as e:
        return f"Error running LMM: {str(e)}"


def create_trajectory_plot(
    df: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    group_col: str,
    subject_col: Optional[str] = None,
) -> go.Figure:
    """
    Create a trajectory plot showing mean trends and optional individual spaghetti lines.
    """
    # Calculate means and SE per group at each time point
    summary = (
        df.groupby([group_col, time_col])[outcome_col]
        .agg(["mean", "sem"])
        .reset_index()
    )
    summary["ci_upper"] = summary["mean"] + 1.96 * summary["sem"]
    summary["ci_lower"] = summary["mean"] - 1.96 * summary["sem"]

    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip("#")
        lv = len(hex_color)
        rgb = tuple(int(hex_color[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"

    fig = go.Figure()

    # Get unique groups
    groups = df[group_col].unique()
    colors = px.colors.qualitative.Plotly

    for i, group in enumerate(groups):
        group_data = summary[summary[group_col] == group]
        # Ensure we have enough colors, cycle if needed
        color_hex = colors[i % len(colors)]

        # Add Mean Line
        fig.add_trace(
            go.Scatter(
                x=group_data[time_col],
                y=group_data["mean"],
                mode="lines+markers",
                name=f"{group} (Mean)",
                line=dict(color=color_hex, width=3),
                legendgroup=str(group),
            )
        )

        # Add CI Band
        fill_color = hex_to_rgba(color_hex, 0.2)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([group_data[time_col], group_data[time_col][::-1]]),
                y=pd.concat([group_data["ci_upper"], group_data["ci_lower"][::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=str(group),
            )
        )

    # Add individual lines (Spaghetti) if subject_col is provided
    # This can get very heavy if many subjects.
    # For now, let's just stick to the Mean Trajectories with CI as requested.

    fig.update_layout(
        title=f"Mean Trajectory of {outcome_col} by {group_col}",
        xaxis_title=time_col,
        yaxis_title=outcome_col,
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def extract_model_results(results: Any, model_type: str) -> pd.DataFrame:
    """
    Extract coeffs, CI, p-values from GEE/LMM results into a DataFrame.
    """
    if isinstance(results, str):  # Error message
        return pd.DataFrame()

    try:
        # Both GEE and MixedLM results have .params, .bse, .pvalues, .conf_int()
        df_res = pd.DataFrame(
            {
                "Beta": results.params,
                "SE": results.bse,
                "P-value": results.pvalues,
                "CI Lower": results.conf_int()[0],
                "CI Upper": results.conf_int()[1],
            }
        )

        # Rounding
        df_res = df_res.round(4)
        df_res["Variable"] = df_res.index
        df_res = df_res[["Variable", "Beta", "SE", "CI Lower", "CI Upper", "P-value"]]
        df_res.reset_index(drop=True, inplace=True)

        return df_res
    except Exception:
        return pd.DataFrame()

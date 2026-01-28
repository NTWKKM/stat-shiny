from __future__ import annotations

import numpy as np
import pandas as pd
import pingouin as pg
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


class AgreementAnalysis:
    """
    Comprehensive inter-rater/inter-method agreement analysis.
    Includes: Cohen's Kappa, Fleiss' Kappa, ICC, and Advanced Bland-Altman.
    """

    @staticmethod
    def cohens_kappa(
        df: pd.DataFrame,
        rater1: str,
        rater2: str,
        weights: str | None = None,
        ci: float = 0.95,
    ) -> tuple[pd.DataFrame, str | None, pd.DataFrame, dict]:
        """
        Calculate Cohen's Kappa with Confidence Intervals.
        Returns: (Stats DataFrame, Error/None, Confusion Matrix, Missing Info)
        """
        try:
            # Data Cleaning
            clean_df = df[[rater1, rater2]].dropna()
            missing_count = len(df) - len(clean_df)
            missing_info = (
                {f"Missing rows ({rater1}, {rater2})": missing_count}
                if missing_count > 0
                else {}
            )

            if len(clean_df) == 0:
                return (
                    pd.DataFrame(),
                    "No valid data pairs found.",
                    pd.DataFrame(),
                    missing_info,
                )

            y1, y2 = clean_df[rater1], clean_df[rater2]

            # Kappa Calculation
            kappa = cohen_kappa_score(y1, y2, weights=weights)

            # Standard Error Calculation (Approximate)
            n = len(clean_df)

            # Use analytical variance if possible (not standard in sklearn)
            # We will use a basic CI based on asymptotic standard error for H0=0 for significance,
            # but for the CI around the estimate, we assume normality.
            z_score = stats.norm.ppf(1 - (1 - ci) / 2)

            # Better SE approximation for CI construction
            # Using simple asymptotic standard error for binary/nominal case
            p_o = np.sum(y1 == y2) / n
            pe = 0  # Calculate expected chance agreement manually to get correct SE
            labels = np.unique(np.concatenate([y1, y2]))
            for k in labels:
                p1 = np.sum(y1 == k) / n
                p2 = np.sum(y2 == k) / n
                pe += p1 * p2

            se_kappa = (
                np.sqrt((p_o * (1 - p_o)) / (n * (1 - pe) ** 2)) if (1 - pe) != 0 else 0
            )

            # For weighted kappa, SE is more complex; falling back to simple approximation if needed
            if weights:
                # Placeholder for complex weighted SE - often approximated
                # Using a very simplified heuristic if se_kappa is 0 or invalid
                if se_kappa == 0:
                    se_kappa = np.sqrt((kappa * (1 - kappa)) / n)

            lower = kappa - z_score * se_kappa
            upper = kappa + z_score * se_kappa

            # Results
            stats_data = {
                "Metric": [
                    "Cohen's Kappa",
                    "Sample Size (N)",
                    "Standard Error",
                    f"{int(ci * 100)}% CI Lower",
                    f"{int(ci * 100)}% CI Upper",
                ],
                "Value": [kappa, n, se_kappa, lower, upper],
            }

            # Confusion Matrix
            cm = confusion_matrix(y1, y2, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index=[f"{rater1}: {lbl}" for lbl in labels],
                columns=[f"{rater2}: {lbl}" for lbl in labels],
            )

            return pd.DataFrame(stats_data), None, cm_df, missing_info

        except Exception as e:
            return pd.DataFrame(), str(e), pd.DataFrame(), {}

    @staticmethod
    def fleiss_kappa(
        df: pd.DataFrame, raters: list[str]
    ) -> tuple[pd.DataFrame, str | None, dict]:
        """
        Calculate Fleiss' Kappa for multiple raters.
        """
        try:
            # Data Cleaning
            clean_df = df[raters].dropna()
            missing_count = len(df) - len(clean_df)
            missing_info = (
                {"Missing rows (Multiple Raters)": missing_count}
                if missing_count > 0
                else {}
            )

            if len(clean_df) == 0:
                return pd.DataFrame(), "No valid data found.", missing_info

            # Convert to Subject x Category counts format
            # aggregate_raters expects (Subject, Rater) but we have columns as raters.
            # We need to ensure we are passing the correct format to aggregate_raters.
            # statsmodels aggregate_raters takes a 2d array (n_subjects, n_raters) containing category labels.

            agg_data, categories = aggregate_raters(clean_df.values)

            kappa = fleiss_kappa(agg_data)

            stats_data = {
                "Metric": ["Fleiss' Kappa", "Subjects (N)", "Raters (k)", "Categories"],
                "Value": [kappa, len(clean_df), len(raters), len(categories)],
            }

            return pd.DataFrame(stats_data), None, missing_info

        except Exception as e:
            return pd.DataFrame(), str(e), {}

    @staticmethod
    def bland_altman_advanced(
        df: pd.DataFrame,
        method1: str,
        method2: str,
        ci: float = 0.95,
        show_ci_bands: bool = True,
    ) -> tuple[dict, go.Figure, dict]:
        """
        Generate Bland-Altman statistics and Plotly figure with optional CI bands.
        """
        try:
            # Data Cleaning
            clean_df = df[[method1, method2]].dropna()
            missing_count = len(df) - len(clean_df)
            missing_info = {"Missing rows": missing_count} if missing_count > 0 else {}

            if len(clean_df) < 2:
                return {"error": "Insufficient data (N < 2)"}, go.Figure(), missing_info

            m1 = clean_df[method1].values
            m2 = clean_df[method2].values

            diffs = m1 - m2
            means = (m1 + m2) / 2
            n = len(diffs)

            mean_diff = np.mean(diffs)
            sd_diff = np.std(diffs, ddof=1)
            # SE of the mean difference
            se_mean_diff = sd_diff / np.sqrt(n)

            # Limits of Agreement (1.96 * SD)
            # For 95% LoA, we use +/- 1.96 * SD
            loa_range = 1.96 * sd_diff
            upper_loa = mean_diff + loa_range
            lower_loa = mean_diff - loa_range

            # Confidence Intervals (Carkeet, 2015 / Bland & Altman 1999)
            # t-score for CI construction
            t_crit = stats.t.ppf(1 - (1 - ci) / 2, df=n - 1)

            # CI for Mean Difference (Bias)
            ci_md_low = mean_diff - t_crit * se_mean_diff
            ci_md_high = mean_diff + t_crit * se_mean_diff

            # Standard Error for Limits of Agreement
            # Approx SE for LoA limits = sqrt(3 * SD^2 / n)
            se_loa = np.sqrt(3 * sd_diff**2 / n)

            ci_upper_loa_low = upper_loa - t_crit * se_loa
            ci_upper_loa_high = upper_loa + t_crit * se_loa
            ci_lower_loa_low = lower_loa - t_crit * se_loa
            ci_lower_loa_high = lower_loa + t_crit * se_loa

            # --- Plotting ---
            fig = go.Figure()

            # Scatter points
            fig.add_trace(
                go.Scatter(
                    x=means,
                    y=diffs,
                    mode="markers",
                    marker=dict(color="rgba(0,0,0,0.5)", size=6),
                    name="Difference",
                    hovertemplate="Mean: %{x:.2f}<br>Diff: %{y:.2f}<extra></extra>",
                )
            )

            # Mean Difference Line
            fig.add_hline(
                y=mean_diff,
                line_dash="solid",
                line_color="blue",
                annotation_text=f"Mean Diff: {mean_diff:.2f}",
                annotation_position="top right",
            )

            # LoA Lines
            fig.add_hline(
                y=upper_loa,
                line_dash="dash",
                line_color="red",
                annotation_text=f"+1.96 SD: {upper_loa:.2f}",
                annotation_position="top right",
            )
            fig.add_hline(
                y=lower_loa,
                line_dash="dash",
                line_color="red",
                annotation_text=f"-1.96 SD: {lower_loa:.2f}",
                annotation_position="bottom right",
            )

            # CI Bands (Optional)
            if show_ci_bands:
                # CI for Mean Diff
                fig.add_hrect(
                    y0=ci_md_low,
                    y1=ci_md_high,
                    line_width=0,
                    fillcolor="blue",
                    opacity=0.15,
                    annotation_text="CI Mean",
                    annotation_position="left",
                )

                # CI for Upper LoA
                fig.add_hrect(
                    y0=ci_upper_loa_low,
                    y1=ci_upper_loa_high,
                    line_width=0,
                    fillcolor="red",
                    opacity=0.1,
                    annotation_text="CI Upper LoA",
                    annotation_position="left",
                )

                # CI for Lower LoA
                fig.add_hrect(
                    y0=ci_lower_loa_low,
                    y1=ci_lower_loa_high,
                    line_width=0,
                    fillcolor="red",
                    opacity=0.1,
                    annotation_text="CI Lower LoA",
                    annotation_position="left",
                )

            fig.update_layout(
                title=f"Bland-Altman Plot: {method1} vs {method2}",
                xaxis_title=f"Mean of {method1} & {method2}",
                yaxis_title=f"Difference ({method1} - {method2})",
                template="plotly_white",
                height=500,
                hovermode="closest",
            )

            stats_res = {
                "n": n,
                "mean_diff": mean_diff,
                "ci_mean_diff": [ci_md_low, ci_md_high],
                "sd_diff": sd_diff,
                "upper_loa": upper_loa,
                "lower_loa": lower_loa,
                "ci_upper_loa": [ci_upper_loa_low, ci_upper_loa_high],
                "ci_lower_loa": [ci_lower_loa_low, ci_lower_loa_high],
            }

            return stats_res, fig, missing_info

        except Exception as e:
            return {"error": str(e)}, go.Figure(), {}

    @staticmethod
    def icc(
        df: pd.DataFrame,
        cols: list[str],
        icc_type: str = "ICC2k",  # Default to Two-way random, average measures
    ) -> tuple[pd.DataFrame, str | None, dict, dict]:
        """
        Calculate Intraclass Correlation Coefficient using Pingouin.
        Input DF: Wide format (Subjects as rows, Raters as columns).
        """
        try:
            # Data Cleaning
            clean_df = df[cols].dropna()
            missing_count = len(df) - len(clean_df)
            missing_info = {"Missing rows": missing_count} if missing_count > 0 else {}

            if len(clean_df) == 0:
                return pd.DataFrame(), "No valid data found.", {}, missing_info

            # Pingouin expects Long format: [Subject, Rater, Rating]
            # Add Subject ID
            clean_df = clean_df.copy()
            clean_df["Subject_ID"] = range(len(clean_df))

            # Melt to Long format
            df_long = clean_df.melt(
                id_vars=["Subject_ID"],
                value_vars=cols,
                var_name="Rater",
                value_name="Rating",
            )

            # Calculate ICC
            icc_res = pg.intraclass_corr(
                data=df_long, targets="Subject_ID", raters="Rater", ratings="Rating"
            )

            # Add interpretation helper
            def interpret_icc(v):
                if v < 0.40:
                    return "Poor"
                if v < 0.60:
                    return "Fair"
                if v < 0.75:
                    return "Good"
                return "Excellent"

            icc_res["Strength"] = icc_res["ICC"].apply(interpret_icc)

            return icc_res, None, {}, missing_info

        except Exception as e:
            return pd.DataFrame(), str(e), {}, {}

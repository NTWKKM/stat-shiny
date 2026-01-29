"""
📈 Advanced Table One Generator (OOP Architecture)

This module implements an object-oriented approach to Table 1 generation,
separating classification, statistical calculation, and formatting logic.
Refactored from `utils/table_one.py` for better maintainability and testing.
"""

from __future__ import annotations

import html as _html
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import CONFIG
from logger import get_logger
from tabs._common import get_color_palette
from utils.data_cleaning import (
    clean_numeric_vector,
    prepare_data_for_analysis,
)
from utils.formatting import create_missing_data_report_html

logger = get_logger(__name__)


# --- 1. Data Structures ---
@dataclass
class VariableAnalysis:
    """
    Holds the full analysis results for a single variable row in Table 1.
    """

    name: str  # Column name
    label: str  # Display label
    var_type: Literal[
        "continuous_normal",
        "continuous_non_normal",
        "categorical",
        "ordinal",
        "unknown",
    ]
    stats_overall: str  # e.g., "50.2 ± 10.1" or "120 (45.3%)"
    stats_groups: Dict[str, str] = field(default_factory=dict)  # GroupID -> Stat string
    p_value: Optional[float] = None
    test_name: Optional[str] = None
    warning: Optional[str] = None
    extra_stats: Dict[str, Any] = field(default_factory=dict)  # e.g., OR, SMD
    categorical_counts: Optional[pd.Series] = None  # Store raw counts for OR/SMD calcs


# --- 2. Classifier Logic ---
class VariableClassifier:
    """
    Intelligent variable type inference and distribution checking.
    """

    @staticmethod
    def classify(series: pd.Series, max_cat_unique: int = 10) -> str:
        """
        Classify a variable into a specific type for Table 1 reporting.
        """
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return "unknown"

        is_numeric = pd.api.types.is_numeric_dtype(clean_series)
        unique_count = clean_series.nunique()

        # Heuristic for Categorical
        if (
            not is_numeric
            or pd.api.types.is_object_dtype(clean_series)
            or isinstance(clean_series.dtype, pd.CategoricalDtype)
            or unique_count <= max_cat_unique
        ):
            return "categorical"

        # Continuous Distribution Check
        return VariableClassifier._check_normality_advanced(clean_series)

    @staticmethod
    def _check_normality_advanced(series: pd.Series) -> str:
        """
        Robust normality check.
        """
        n = len(series)
        if n < 3:
            return "continuous_non_normal"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if n < 5000:
                    stat, p = stats.shapiro(series)
                else:
                    stat, p = stats.jarque_bera(series)

            # P > 0.05 indicates normality
            is_normal_test = p > 0.05

            # Descriptive Checks
            skew = stats.skew(series)
            kurt = stats.kurtosis(series)
            is_normal_desc = abs(skew) < 1.0 and abs(kurt) < 2.0

            if is_normal_test or (n > 50 and is_normal_desc):
                return "continuous_normal"
            else:
                return "continuous_non_normal"

        except Exception as e:
            logger.warning(f"Normality check failed: {e}")
            return "continuous_non_normal"


# --- 3. Statistical Engine ---
class StatisticalEngine:
    """
    Encapsulates all statistical calculations.
    """

    @staticmethod
    def get_stats_continuous(series: pd.Series, normal: bool = True) -> str:
        """Mean ± SD or Median [IQR]"""
        clean = clean_numeric_vector(series).dropna()
        if len(clean) == 0:
            return "-"

        if normal:
            return f"{clean.mean():.1f} ± {clean.std():.1f}"
        else:
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            return f"{clean.median():.1f} [{q1:.1f}, {q3:.1f}]"

    @staticmethod
    def get_stats_categorical(
        series: pd.Series, total_n: int = None
    ) -> Tuple[str, pd.Series]:
        """n (%)"""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return "-", pd.Series([], dtype=object)

        counts = clean_series.value_counts().sort_index()
        denominator = total_n if total_n is not None else len(clean_series)

        if denominator == 0:
            return "-", counts

        res_parts = []
        for cat, count in counts.items():
            pct = (count / denominator) * 100
            res_parts.append(f"{_html.escape(str(cat))}: {int(count)} ({pct:.1f}%)")

        return "<br>".join(res_parts), counts

    @staticmethod
    def calculate_p_continuous(
        groups: List[pd.Series], normal: bool = True
    ) -> Tuple[Optional[float], str]:
        """T-test/ANOVA or MWU/Kruskal"""
        clean_groups = [clean_numeric_vector(g).dropna() for g in groups]
        clean_groups = [g for g in clean_groups if len(g) > 0]

        if len(clean_groups) < 2:
            return None, "-"

        try:
            if normal:
                if len(clean_groups) == 2:
                    _, p = stats.ttest_ind(
                        clean_groups[0], clean_groups[1], nan_policy="omit"
                    )
                    return p, "t-test"
                else:
                    _, p = stats.f_oneway(*clean_groups)
                    return p, "ANOVA"
            else:
                if len(clean_groups) == 2:
                    _, p = stats.mannwhitneyu(
                        clean_groups[0], clean_groups[1], alternative="two-sided"
                    )
                    return p, "Mann-Whitney U"
                else:
                    _, p = stats.kruskal(*clean_groups)
                    return p, "Kruskal-Wallis"
        except Exception:
            return None, "Error"

    @staticmethod
    def calculate_p_categorical(
        df: pd.DataFrame, col: str, group_col: str
    ) -> Tuple[Optional[float], str]:
        """Chi-square or Fisher's Exact"""
        try:
            tab = pd.crosstab(df[col], df[group_col])
            if tab.size == 0:
                return None, "-"

            _, p, _, expected = stats.chi2_contingency(tab)
            min_expected = expected.min()

            # Use Fisher if small samples (2x2 only mostly)
            if tab.shape == (2, 2) and min_expected < 5:
                try:
                    _, p_fisher = stats.fisher_exact(tab)
                    return p_fisher, "Fisher's Exact"
                except Exception:
                    pass

            test_name = "Chi-square" if min_expected >= 5 else "Chi-square (Low N)"
            return p, test_name
        except Exception:
            return None, "Error"

    @staticmethod
    def calculate_smd(
        df: pd.DataFrame,
        col: str,
        group_col: str,
        g1_val: Any,
        g2_val: Any,
        is_cat: bool,
    ) -> str:
        """Calcluate Standardized Mean Difference"""
        try:
            # Reusing robust logic from original table_one for reliability
            mask1 = df[group_col].astype(str) == str(g1_val)
            mask2 = df[group_col].astype(str) == str(g2_val)

            if is_cat:
                s1 = df.loc[mask1, col].dropna()
                s2 = df.loc[mask2, col].dropna()

                # Get union of categories
                cats = set(s1.unique()) | set(s2.unique())
                if not cats:
                    return "-"

                smd_vals = []
                for cat in sorted(list(cats)):
                    p1 = (s1.astype(str) == str(cat)).mean()
                    p2 = (s2.astype(str) == str(cat)).mean()
                    pooled_sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
                    smd = abs(p1 - p2) / pooled_sd if pooled_sd > 0 else 0.0

                    val_str = f"{smd:.3f}"
                    if smd >= 0.1:
                        val_str = f"<b>{val_str}</b>"
                    smd_vals.append(val_str)
                return "<br>".join(smd_vals)
            else:
                v1 = clean_numeric_vector(df.loc[mask1, col]).dropna()
                v2 = clean_numeric_vector(df.loc[mask2, col]).dropna()
                if len(v1) == 0 or len(v2) == 0:
                    return "-"

                pooled_sd = np.sqrt((v1.std() ** 2 + v2.std() ** 2) / 2)
                smd = abs(v1.mean() - v2.mean()) / pooled_sd if pooled_sd > 0 else 0.0

                val_str = f"{smd:.3f}"
                if smd >= 0.1:
                    val_str = f"<b>{val_str}</b>"
                return val_str
        except Exception:
            return "-"

    @staticmethod
    def calculate_or(
        df: pd.DataFrame, col: str, group_col: str, group_1_val: Any, is_cat: bool
    ) -> str:
        """Wrapper for OR calculation"""
        # Logic similar to original but integrated here
        # For brevity in this artifact, relying on statsmodels Logit for continuous
        # and 2x2 logic for binary.
        # (Full robust implementation would go here)
        return "-"  # Placeholder for full implementation if not using original functions directly


# --- 4. Formatter ---
class TableOneFormatter:
    """
    Handles HTML rendering matching Shiny style.
    """

    COLORS = get_color_palette()

    @staticmethod
    def format_p(p: Optional[float]) -> str:
        if p is None or np.isnan(p):
            return "-"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    @classmethod
    def render_html(
        cls,
        results: List[VariableAnalysis],
        groups: List[Dict],
        group_masks: Dict,
        total_n: int,
        missing_info: Dict,
    ) -> str:
        # Helper helpers
        def _p_cell(p, test):
            if p is None:
                return "<td>-</td><td>-</td>"
            p_str = cls.format_p(p)
            is_sig = p is not None and p < 0.05
            p_cls = "p-significant" if is_sig else "p-not-significant"
            star = "*" if is_sig else ""
            return (
                f"<td class='numeric-cell'><span class='{p_cls}'>{p_str}{star}</span></td>"
                f"<td class='numeric-cell'>{test}</td>"
            )

        # Basic CSS (same as original to ensure UI consistency)
        css = f"""
        <style>
            .numeric-cell {{ font-family: 'Courier New', monospace; font-size: 12px; }}
            .p-significant {{ color: {cls.COLORS["danger"]}; font-weight: 600; }}
            .p-not-significant {{ color: {cls.COLORS["success"]}; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
            th {{ background-color: {cls.COLORS["primary_dark"]}; color: white; padding: 10px; text-align: center; }}
            td {{ padding: 8px; border: 1px solid {cls.COLORS["border"]}; }}
            tr:nth-child(even) {{ background-color: {cls.COLORS["primary_light"]}; }}
        </style>
        """

        # Table Header
        header_cols = (
            f"<th>Characteristic</th><th class='numeric-cell'>Total (N={total_n})</th>"
        )
        for g in groups:
            n_g = group_masks[g["val"]].sum()
            header_cols += f"<th class='numeric-cell'>{_html.escape(str(g['label']))} (n={n_g})</th>"

        if len(groups) == 2:
            header_cols += "<th>SMD</th><th>P-value</th><th>Test</th>"
        elif len(groups) > 0:
            header_cols += "<th>P-value</th><th>Test</th>"

        rows_html = ""
        for res in results:
            row = f"<tr><td><strong>{_html.escape(res.label)}</strong>"
            if res.var_type == "continuous_non_normal":
                row += "<br><small>(Median [IQR])</small>"
            elif res.var_type == "continuous_normal":
                row += "<br><small>(Mean ± SD)</small>"
            row += "</td>"

            # Total
            row += f"<td class='numeric-cell'>{res.stats_overall}</td>"

            # Groups
            for g in groups:
                val = res.stats_groups.get(g["val"], "-")
                row += f"<td class='numeric-cell'>{val}</td>"

            # Stats (SMD/P)
            if len(groups) == 2:
                smd = res.extra_stats.get("smd", "-")
                row += f"<td class='numeric-cell'>{smd}</td>"
                row += _p_cell(res.p_value, res.test_name)
            elif len(groups) > 0:
                row += _p_cell(res.p_value, res.test_name)

            row += "</tr>"
            rows_html += row

        html = f"""
        {css}
        <div class="table-wrapper">
            <table>
                <thead><tr>{header_cols}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """
        # Append missing data info
        html += create_missing_data_report_html(missing_info, {})
        return html


# --- 5. Main Generator ---
class TableOneGenerator:
    """
    Orchestrator class.
    """

    def __init__(self, df: pd.DataFrame, var_meta: Dict = None):
        self.raw_df = df
        self.var_meta = var_meta or {}
        self.classifier = VariableClassifier()
        self.stats_engine = StatisticalEngine()

    def generate(self, selected_vars: List[str], stratify_by: str = None) -> str:
        # 1. Clean Data (Reuse robust logic)
        missing_cfg = CONFIG.get("analysis.missing", {})
        df_clean, missing_info = prepare_data_for_analysis(
            self.raw_df,
            numeric_cols=[
                c
                for c in selected_vars
                if c in self.raw_df.columns
                and pd.api.types.is_numeric_dtype(self.raw_df[c])
            ],
            required_cols=list(
                set(list(selected_vars) + ([stratify_by] if stratify_by else []))
            ),
            var_meta=self.var_meta,
            handle_missing=missing_cfg.get("strategy", "complete-case"),
        )

        # 2. Setup Groups
        groups = []
        group_masks = {}
        if stratify_by and stratify_by != "None":
            uniques = df_clean[stratify_by].dropna().unique()
            # Sort naturally
            uniques = sorted(
                uniques,
                key=lambda x: (0, float(x))
                if str(x).replace(".", "").isdigit()
                else (1, str(x)),
            )
            for u in uniques:
                label = str(u)  # Could map here using var_meta
                groups.append({"val": u, "label": label})
                group_masks[u] = df_clean[stratify_by] == u

        # 3. Analyze Variables
        results = []
        for var in selected_vars:
            if var == stratify_by:
                continue

            # Classify
            vtype = self.classifier.classify(df_clean[var])
            res = VariableAnalysis(
                name=var,
                label=self.var_meta.get(var, {}).get("label", var),
                var_type=vtype,
                stats_overall="-",
            )

            # Stats Logic
            is_cat = vtype == "categorical"

            # Overall Stats
            if is_cat:
                res.stats_overall, _ = self.stats_engine.get_stats_categorical(
                    df_clean[var]
                )
            else:
                is_normal = vtype == "continuous_normal"
                res.stats_overall = self.stats_engine.get_stats_continuous(
                    df_clean[var], is_normal
                )

            # Group Stats
            if stratify_by:
                group_series_list = []
                for g in groups:
                    sub_s = df_clean.loc[group_masks[g["val"]], var]
                    group_series_list.append(sub_s)

                    if is_cat:
                        val, _ = self.stats_engine.get_stats_categorical(
                            sub_s, total_n=len(sub_s)
                        )
                        res.stats_groups[g["val"]] = val
                    else:
                        is_normal = vtype == "continuous_normal"
                        res.stats_groups[g["val"]] = (
                            self.stats_engine.get_stats_continuous(sub_s, is_normal)
                        )

                # Hypothesis Testing
                if is_cat:
                    res.p_value, res.test_name = (
                        self.stats_engine.calculate_p_categorical(
                            df_clean, var, stratify_by
                        )
                    )
                else:
                    is_normal = vtype == "continuous_normal"
                    res.p_value, res.test_name = (
                        self.stats_engine.calculate_p_continuous(
                            group_series_list, is_normal
                        )
                    )

                # SMD (if 2 groups)
                if len(groups) == 2:
                    smd = self.stats_engine.calculate_smd(
                        df_clean,
                        var,
                        stratify_by,
                        groups[0]["val"],
                        groups[1]["val"],
                        is_cat,
                    )
                    res.extra_stats["smd"] = smd

            results.append(res)

        # 4. Render
        return TableOneFormatter.render_html(
            results, groups, group_masks, len(df_clean), missing_info
        )

"""
📄 Publication Table Renderer Module (NEJM / JAMA / APA 7)
----------------------------------------------------------
Standardizes statistical regression and estimate outputs into publication-grade tables
adhering strictly to international medical journal formatting guidelines.

Features:
- Dual-style rendering: NEJM/JAMA vs APA 7.
- Strict precision matching between point estimates and confidence bounds.
- Tabular numeric alignment with en-dashes for ranges.
- Zero significance asterisk bias (confidence intervals carry statistical inference).
- Word-compatible rich HTML clipboard output.
- Auto-generated reproducible Methods & Analysis paragraph.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Estimate:
    term: str
    label: str
    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    scale: str = "OR"  # "OR", "HR", "RR", "Beta", "IRR", "MD"
    reference: bool = False
    ref_label: str = "Reference"
    std_error: Optional[float] = None
    n: Optional[int] = None
    events: Optional[int] = None


@dataclass
class ModelMeta:
    estimator: str
    n_total: int
    n_events: Optional[int] = None
    outcome_name: Optional[str] = None
    exposure_name: Optional[str] = None
    adjusted_for: List[str] = field(default_factory=list)
    software_version: str = (
        "Medical Stat Platform 2026 (Python statsmodels/lifelines/firthmodels)"
    )
    seed: Optional[int] = None
    notes: Optional[str] = None


@dataclass
class EstimateTable:
    title: str
    rows: List[Estimate]
    meta: Optional[ModelMeta] = None
    confidence_level: float = 0.95


def format_journal_p_value(p: float, style: str = "NEJM") -> str:
    """
    Formats p-values according to specific journal conventions.

    NEJM/JAMA: P=0.03, P<0.001 (capital P, leading zero)
    APA 7: p = .03, p < .001 (italic/lower p, no leading zero)
    """
    if p is None or (isinstance(p, float) and (p != p)):  # NaN check
        return "—"

    if style.upper() in ("NEJM", "JAMA"):
        if p < 0.001:
            return "P<0.001"
        elif p < 0.01:
            return f"P={p:.3f}"
        elif p >= 0.99:
            return "P>0.99"
        else:
            return f"P={p:.2f}"
    else:  # APA 7
        if p < 0.001:
            return "p < .001"
        elif p < 0.01:
            p_str = f"{p:.3f}".lstrip("0")
            return f"p = {p_str}"
        elif p >= 0.99:
            return "p > .99"
        else:
            p_str = f"{p:.2f}".lstrip("0")
            return f"p = {p_str}"


def format_journal_estimate_ci(
    est: float,
    ci_lo: float,
    ci_hi: float,
    scale: str = "OR",
    style: str = "NEJM",
    decimals: int = 2,
    confidence_level: float = 0.95,
) -> str:
    """
    Formats point estimate and CI according to journal style.

    NEJM/JAMA: 1.24 (0.98–1.57) using en-dash
    APA 7: OR = 1.24, 95% CI [0.98, 1.57] (reflecting configured confidence level)
    """
    est_str = f"{est:.{decimals}f}"
    lo_str = f"{ci_lo:.{decimals}f}"
    hi_str = f"{ci_hi:.{decimals}f}"
    ci_pct = int(confidence_level * 100)

    if style.upper() in ("NEJM", "JAMA"):
        return f"{est_str} ({lo_str}–{hi_str})"
    else:
        return f"{html.escape(scale)} = {est_str}, {ci_pct}% CI [{lo_str}, {hi_str}]"


def render_publication_html(
    table: EstimateTable,
    style: str = "NEJM",
    include_meta_paragraph: bool = True,
    for_clipboard: bool = False,
) -> str:
    """
    Renders an EstimateTable into rich, publication-ready HTML.

    Args:
        table: The EstimateTable instance holding estimates and metadata.
        style: "NEJM", "JAMA", or "APA7".
        include_meta_paragraph: Whether to append the auto-generated Methods text.
        for_clipboard: If True, injects full inline styling for direct MS Word pasting.

    Returns:
        HTML string.
    """
    style_upper = style.upper()
    if style_upper not in ("NEJM", "JAMA", "APA7"):
        raise ValueError(
            f"Unsupported publication style '{style}'. Supported styles are 'NEJM', 'JAMA', 'APA7'."
        )
    normalized_style = style_upper.lower()

    ci_pct = int(table.confidence_level * 100)
    scale_label = table.rows[0].scale if table.rows else "Estimate"

    if style_upper in ("NEJM", "JAMA"):
        header_est = f"{html.escape(scale_label)} ({ci_pct}% CI)"
        header_p = "P Value"
    else:
        header_est = f"{html.escape(scale_label)} [{ci_pct}% CI]"
        header_p = "p"

    # Inline styles for Word pasting
    tbl_style = (
        "width: 100%; border-collapse: collapse; font-family: 'Times New Roman', Times, serif; font-size: 11pt; margin: 16px 0;"
        if for_clipboard
        else ""
    )
    th_top = "border-top: 2pt solid #000; border-bottom: 1pt solid #000; padding: 6px 10px; text-align: left; font-weight: bold;"
    th_num = "border-top: 2pt solid #000; border-bottom: 1pt solid #000; padding: 6px 10px; text-align: right; font-weight: bold;"
    td_text = "padding: 5px 10px; text-align: left; border-bottom: 0.5pt solid #E2E8F0;"
    td_num = "padding: 5px 10px; text-align: right; font-variant-numeric: tabular-nums; border-bottom: 0.5pt solid #E2E8F0;"
    td_foot = "padding: 8px 10px; font-size: 9.5pt; color: #475569; border-top: 1pt solid #000; border-bottom: 2pt solid #000;"

    html_parts = []
    html_parts.append('<div class="publication-table-container">')
    html_parts.append(
        f'<table class="table-publication table-{normalized_style}" style="{tbl_style}">'
    )
    html_parts.append(
        f"  <caption><strong>{html.escape(table.title)}</strong></caption>"
    )
    html_parts.append("  <thead>")
    html_parts.append("    <tr>")
    html_parts.append(f'      <th style="{th_top}">Variable</th>')
    html_parts.append(f'      <th style="{th_num}">{header_est}</th>')
    html_parts.append(f'      <th style="{th_num}">{header_p}</th>')
    html_parts.append("    </tr>")
    html_parts.append("  </thead>")
    html_parts.append("  <tbody>")

    for row in table.rows:
        escaped_label = html.escape(row.label or row.term)
        if row.reference:
            null_val = (
                "0.00"
                if (row.scale or "").upper() in ("BETA", "MD", "MEAN DIFF", "DIFF")
                else "1.00"
            )
            est_text = (
                f"{null_val} ({html.escape(row.ref_label)})"
                if style_upper in ("NEJM", "JAMA")
                else f"{null_val} [{html.escape(row.ref_label)}]"
            )
            p_text = "—"
        else:
            est_text = format_journal_estimate_ci(
                row.estimate,
                row.ci_lower,
                row.ci_upper,
                scale=row.scale,
                style=style_upper,
                confidence_level=table.confidence_level,
            )
            p_text = format_journal_p_value(row.p_value, style=style_upper)

        html_parts.append("    <tr>")
        html_parts.append(f'      <td style="{td_text}">{escaped_label}</td>')
        html_parts.append(f'      <td style="{td_num}">{est_text}</td>')
        html_parts.append(f'      <td style="{td_num}">{p_text}</td>')
        html_parts.append("    </tr>")

    html_parts.append("  </tbody>")

    # Footnote
    footnotes = [
        f"CI denotes confidence interval; {scale_label}, {get_scale_full_name(scale_label)}."
    ]
    if table.meta and table.meta.adjusted_for:
        adj_str = ", ".join(table.meta.adjusted_for)
        footnotes.append(f"Model adjusted for: {adj_str}.")
    if table.meta and table.meta.n_total:
        n_str = f"N = {table.meta.n_total:,}"
        if table.meta.n_events is not None:
            n_str += f" (events = {table.meta.n_events:,})"
        footnotes.append(n_str)

    footnote_text = " ".join(footnotes)
    html_parts.append("  <tfoot>")
    html_parts.append("    <tr>")
    html_parts.append(
        f'      <td colspan="3" style="{td_foot}">{html.escape(footnote_text)}</td>'
    )
    html_parts.append("    </tr>")
    html_parts.append("  </tfoot>")
    html_parts.append("</table>")

    # Methods text
    if include_meta_paragraph and table.meta:
        methods_p = generate_methods_paragraph(
            table.meta,
            scale=scale_label,
            confidence_level=table.confidence_level,
        )
        html_parts.append(
            '<div class="methods-paragraph-box" style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;padding:12px 16px;margin-top:12px;font-size:12px;color:#334155;line-height:1.6;">'
        )
        html_parts.append(
            "  <strong>📝 Reproducible Methods Paragraph (for Manuscript):</strong>"
        )
        html_parts.append(
            f'  <p class="mb-0 mt-1" style="font-family:serif;font-size:13px;color:#0F172A;">{html.escape(methods_p)}</p>'
        )
        html_parts.append("</div>")

    html_parts.append("</div>")
    return "\n".join(html_parts)


def get_scale_full_name(scale: str) -> str:
    mapping = {
        "OR": "odds ratio",
        "HR": "hazard ratio",
        "RR": "risk ratio",
        "IRR": "incidence rate ratio",
        "Beta": "regression coefficient",
        "MD": "mean difference",
    }
    return mapping.get(scale, "effect estimate")


def generate_methods_paragraph(
    meta: ModelMeta,
    scale: str = "OR",
    confidence_level: float = 0.95,
) -> str:
    """
    Generates a formal, reproducible methodology paragraph ready for academic papers.
    """
    parts = []
    parts.append(f"Statistical analyses were performed using {meta.estimator}.")
    if meta.n_total:
        events_clause = (
            f" with {meta.n_events} events observed"
            if meta.n_events is not None
            else ""
        )
        parts.append(
            f"The total analyzed cohort comprised N = {meta.n_total:,} patients{events_clause}."
        )

    if meta.adjusted_for:
        cov_str = ", ".join(meta.adjusted_for)
        parts.append(
            f"Multivariable adjustment included the following covariates: {cov_str}."
        )

    ci_pct_str = f"{confidence_level * 100:g}%"
    parts.append(
        f"Effect estimates are presented as {get_scale_full_name(scale)}s alongside two-sided {ci_pct_str} confidence intervals."
    )
    parts.append(
        f"All computational procedures were executed in {meta.software_version}."
    )
    if meta.seed is not None:
        parts.append(
            f"Random seed was fixed at {meta.seed} for deterministic reproducibility."
        )

    return " ".join(parts)

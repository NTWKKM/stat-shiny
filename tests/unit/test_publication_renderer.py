import pytest
from utils.publication_renderer import (
    Estimate,
    EstimateTable,
    ModelMeta,
    format_journal_estimate_ci,
    format_journal_p_value,
    generate_methods_paragraph,
    render_publication_html,
)


def test_format_journal_p_value_nejm():
    assert format_journal_p_value(0.0001, style="NEJM") == "P<0.001"
    assert format_journal_p_value(0.034, style="NEJM") == "P=0.03"
    assert format_journal_p_value(0.005, style="NEJM") == "P=0.005"
    assert format_journal_p_value(0.995, style="NEJM") == "P>0.99"


def test_format_journal_p_value_apa():
    assert format_journal_p_value(0.0001, style="APA7") == "p < .001"
    assert format_journal_p_value(0.034, style="APA7") == "p = .03"
    assert format_journal_p_value(0.005, style="APA7") == "p = .005"
    assert format_journal_p_value(0.995, style="APA7") == "p > .99"


def test_format_journal_estimate_ci():
    nejm = format_journal_estimate_ci(1.24, 0.98, 1.57, scale="OR", style="NEJM")
    assert nejm == "1.24 (0.98–1.57)"

    apa = format_journal_estimate_ci(1.24, 0.98, 1.57, scale="OR", style="APA7")
    assert apa == "OR = 1.24, 95% CI [0.98, 1.57]"


def test_generate_methods_paragraph():
    meta = ModelMeta(
        estimator="Multivariable Logistic Regression (Firth Penalized)",
        n_total=500,
        n_events=120,
        adjusted_for=["Age", "Sex", "Baseline eGFR"],
        seed=42,
    )
    p = generate_methods_paragraph(meta, scale="OR")
    assert "Multivariable Logistic Regression" in p
    assert "N = 500" in p
    assert "120 events" in p
    assert "Age, Sex, Baseline eGFR" in p
    assert "seed was fixed at 42" in p


def test_render_publication_html():
    rows = [
        Estimate(
            term="treat",
            label="Statin Therapy",
            estimate=1.45,
            ci_lower=1.12,
            ci_upper=1.88,
            p_value=0.004,
            scale="OR",
        ),
        Estimate(
            term="placebo",
            label="Placebo",
            estimate=1.0,
            ci_lower=1.0,
            ci_upper=1.0,
            p_value=1.0,
            scale="OR",
            reference=True,
        ),
    ]
    meta = ModelMeta(
        estimator="Cox Proportional Hazards",
        n_total=1000,
        n_events=250,
        adjusted_for=["Age", "Hypertension"],
    )
    table = EstimateTable(
        title="Table 2. Multivariable Hazard Ratios", rows=rows, meta=meta
    )

    html_nejm = render_publication_html(table, style="NEJM")
    assert "Table 2. Multivariable Hazard Ratios" in html_nejm
    assert "Statin Therapy" in html_nejm
    assert "1.45 (1.12–1.88)" in html_nejm
    assert "1.00 (Reference)" in html_nejm
    assert "P=0.004" in html_nejm
    assert "Reproducible Methods Paragraph" in html_nejm

    html_apa = render_publication_html(table, style="APA7")
    assert "OR = 1.45, 95% CI [1.12, 1.88]" in html_apa
    assert "p = .004" in html_apa

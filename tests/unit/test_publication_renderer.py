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


def test_publication_renderer_custom_confidence_level():
    rows = [
        Estimate(
            term="treat",
            label="Treatment Effect",
            estimate=0.85,
            ci_lower=0.75,
            ci_upper=0.96,
            p_value=0.012,
            scale="HR",
        )
    ]
    table = EstimateTable(
        title="Table 3. 90% CI Estimates",
        rows=rows,
        confidence_level=0.90,
    )
    html_apa = render_publication_html(table, style="APA7")
    assert "HR [90% CI]" in html_apa
    assert "HR = 0.85, 90% CI [0.75, 0.96]" in html_apa


def test_publication_renderer_additive_reference_value():
    rows = [
        Estimate(
            term="group_ref",
            label="Control Group",
            estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            scale="Beta",
            reference=True,
            ref_label="Ref",
        ),
        Estimate(
            term="group_treat",
            label="Active Group",
            estimate=2.5,
            ci_lower=1.2,
            ci_upper=3.8,
            p_value=0.001,
            scale="Beta",
        ),
    ]
    table = EstimateTable(title="Table 4. Linear Regression", rows=rows)
    html_nejm = render_publication_html(table, style="NEJM")
    assert "0.00 (Ref)" in html_nejm

    html_apa = render_publication_html(table, style="APA7")
    assert "0.00 [Ref]" in html_apa


def test_publication_renderer_invalid_style():
    table = EstimateTable(title="Test", rows=[])
    with pytest.raises(ValueError, match="Unsupported publication style"):
        render_publication_html(table, style="INVALID_STYLE")


def test_generate_methods_paragraph_custom_confidence_level():
    meta = ModelMeta(
        estimator="Multivariable Logistic Regression",
        n_total=200,
        n_events=40,
    )
    p = generate_methods_paragraph(meta, scale="OR", confidence_level=0.90)
    assert "two-sided 90% confidence intervals" in p

    rows = [
        Estimate(
            term="drug",
            label="Study Drug",
            estimate=1.5,
            ci_lower=1.1,
            ci_upper=2.1,
            p_value=0.01,
            scale="OR",
        )
    ]
    table = EstimateTable(
        title="Table 90% CI",
        rows=rows,
        meta=meta,
        confidence_level=0.90,
    )
    html_out = render_publication_html(table, style="NEJM", include_meta_paragraph=True)
    assert "two-sided 90% confidence intervals" in html_out


def test_publication_renderer_html_escaping():
    malicious_scale = "<script>alert('xss')</script>"
    rows = [
        Estimate(
            term="x",
            label="Safe Label",
            estimate=1.2,
            ci_lower=0.9,
            ci_upper=1.6,
            p_value=0.2,
            scale=malicious_scale,
        )
    ]
    table = EstimateTable(title="Escaped Table", rows=rows)
    html_nejm = render_publication_html(table, style="NEJM")
    assert "<script>" not in html_nejm
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_nejm

    html_apa = render_publication_html(table, style="APA7")
    assert "<script>" not in html_apa
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_apa
    assert (
        "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt; = 1.20, 95% CI [0.90, 1.60]"
        in html_apa
    )

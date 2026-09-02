import pytest

from utils.publication_renderer import (
    Estimate,
    EstimateTable,
    ModelMeta,
    canonicalize_scale,
    format_confidence_level,
    format_journal_estimate_ci,
    format_journal_p_value,
    generate_methods_paragraph,
    get_scale_full_name,
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


def test_format_confidence_level():
    assert format_confidence_level(0.95) == "95%"
    assert format_confidence_level(0.90) == "90%"
    assert format_confidence_level(0.975) == "97.5%"
    assert format_confidence_level(0.99) == "99%"
    assert format_confidence_level(0.999) == "99.9%"


def test_format_confidence_level_validation_boundaries():
    # Boundary values 0 and 1
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(0)
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(0.0)
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(1)
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(1.0)

    # Values outside the range (0, 1)
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(-0.05)
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(1.5)
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(95)

    # Non-finite and invalid values
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(float("nan"))
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(float("inf"))
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(float("-inf"))
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(None)
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level(True)
    with pytest.raises(
        ValueError,
        match="confidence_level must be a finite number strictly between 0 and 1",
    ):
        format_confidence_level("0.95")


def test_render_publication_html_inconsistent_scales_rejected():
    rows = [
        Estimate(
            term="var1",
            label="Exposure A",
            estimate=1.5,
            ci_lower=1.1,
            ci_upper=2.0,
            p_value=0.01,
            scale="OR",
        ),
        Estimate(
            term="var2",
            label="Exposure B",
            estimate=0.8,
            ci_lower=0.6,
            ci_upper=0.95,
            p_value=0.02,
            scale="HR",
        ),
    ]
    table = EstimateTable(title="Incompatible Scales Table", rows=rows)
    with pytest.raises(
        ValueError, match="EstimateTable rows have incompatible effect scales"
    ):
        render_publication_html(table, style="NEJM")
    with pytest.raises(
        ValueError, match="EstimateTable rows have incompatible effect scales"
    ):
        render_publication_html(table, style="APA7")


def test_render_publication_html_case_insensitive_scales():
    rows = [
        Estimate(
            term="var1",
            label="Exposure A",
            estimate=1.5,
            ci_lower=1.1,
            ci_upper=2.0,
            p_value=0.01,
            scale="OR",
        ),
        Estimate(
            term="var2",
            label="Exposure B",
            estimate=1.8,
            ci_lower=1.2,
            ci_upper=2.7,
            p_value=0.005,
            scale="or",
        ),
    ]
    table = EstimateTable(title="Case Insensitive Scales Table", rows=rows)
    html_out = render_publication_html(table, style="NEJM")
    assert "OR (95% CI)" in html_out


def test_render_publication_html_lowercase_hr_scale():
    rows = [
        Estimate(
            term="statin",
            label="Statin Treatment",
            estimate=0.72,
            ci_lower=0.55,
            ci_upper=0.94,
            p_value=0.015,
            scale="hr",
        ),
        Estimate(
            term="control",
            label="Standard Care",
            estimate=1.0,
            ci_lower=1.0,
            ci_upper=1.0,
            p_value=1.0,
            scale="HR",
            reference=True,
        ),
    ]
    meta = ModelMeta(
        estimator="Cox Proportional Hazards Model",
        n_total=450,
        n_events=85,
    )
    table = EstimateTable(title="Survival Table (HR)", rows=rows, meta=meta)

    # 1. Check NEJM style
    html_nejm = render_publication_html(
        table, style="NEJM", include_meta_paragraph=True
    )
    assert "HR (95% CI)" in html_nejm
    assert "CI denotes confidence interval; HR, hazard ratio." in html_nejm
    assert "hazard ratios" in html_nejm

    # 2. Check APA7 style
    html_apa = render_publication_html(table, style="APA7", include_meta_paragraph=True)
    assert "HR [95% CI]" in html_apa
    assert "HR = 0.72, 95% CI [0.55, 0.94]" in html_apa
    assert "CI denotes confidence interval; HR, hazard ratio." in html_apa
    assert "hazard ratios" in html_apa

    # 3. Direct helper checks
    assert get_scale_full_name("hr") == "hazard ratio"
    assert get_scale_full_name("HR") == "hazard ratio"
    apa_ci = format_journal_estimate_ci(0.72, 0.55, 0.94, scale="hr", style="APA7")
    assert apa_ci == "HR = 0.72, 95% CI [0.55, 0.94]"


def test_render_publication_html_mixed_additive_scale_aliases():
    rows = [
        Estimate(
            term="placebo",
            label="Placebo Control",
            estimate=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            scale="MD",
            reference=True,
        ),
        Estimate(
            term="drug_low",
            label="Low-Dose Intervention",
            estimate=-2.3,
            ci_lower=-3.8,
            ci_upper=-0.8,
            p_value=0.003,
            scale="Mean Diff",
        ),
        Estimate(
            term="drug_high",
            label="High-Dose Intervention",
            estimate=-4.1,
            ci_lower=-5.9,
            ci_upper=-2.3,
            p_value=0.0001,
            scale="DIFF",
        ),
        Estimate(
            term="drug_combo",
            label="Combination Therapy",
            estimate=-5.5,
            ci_lower=-7.2,
            ci_upper=-3.8,
            p_value=0.00005,
            scale="mean difference",
        ),
    ]
    meta = ModelMeta(
        estimator="Linear Mixed-Effects Model",
        n_total=320,
    )
    table = EstimateTable(
        title="Table 4. Primary Outcome Differences", rows=rows, meta=meta
    )

    # 1. NEJM check
    html_nejm = render_publication_html(
        table, style="NEJM", include_meta_paragraph=True
    )
    assert "MD (95% CI)" in html_nejm
    assert "0.00 (Reference)" in html_nejm
    assert "-2.30 (-3.80–-0.80)" in html_nejm
    assert "CI denotes confidence interval; MD, mean difference." in html_nejm
    assert "mean differences" in html_nejm

    # 2. APA 7 check
    html_apa = render_publication_html(table, style="APA7", include_meta_paragraph=True)
    assert "MD [95% CI]" in html_apa
    assert "0.00 [Reference]" in html_apa
    assert "MD = -2.30, 95% CI [-3.80, -0.80]" in html_apa
    assert "MD = -4.10, 95% CI [-5.90, -2.30]" in html_apa
    assert "CI denotes confidence interval; MD, mean difference." in html_apa
    assert "mean differences" in html_apa

    # 3. Direct canonicalize_scale checks
    assert canonicalize_scale("MD") == "MD"
    assert canonicalize_scale("Mean Diff") == "MD"
    assert canonicalize_scale("DIFF") == "MD"
    assert canonicalize_scale("mean difference") == "MD"
    assert canonicalize_scale("Beta") == "Beta"
    assert canonicalize_scale("COEF") == "Beta"
    assert canonicalize_scale("regression coefficient") == "Beta"


def test_publication_renderer_fractional_confidence_level_975():
    # 1. format_journal_estimate_ci APA7 check
    apa_ci = format_journal_estimate_ci(
        1.24, 0.98, 1.57, scale="OR", style="APA7", confidence_level=0.975
    )
    assert apa_ci == "OR = 1.24, 97.5% CI [0.98, 1.57]"

    # 2. generate_methods_paragraph check
    meta = ModelMeta(
        estimator="Multivariable Logistic Regression",
        n_total=500,
        n_events=120,
    )
    methods = generate_methods_paragraph(meta, scale="OR", confidence_level=0.975)
    assert "two-sided 97.5% confidence intervals" in methods

    # 3. render_publication_html table headers and content check
    rows = [
        Estimate(
            term="drug",
            label="Study Drug",
            estimate=1.45,
            ci_lower=1.12,
            ci_upper=1.88,
            p_value=0.004,
            scale="OR",
        )
    ]
    table = EstimateTable(
        title="Table with 97.5% CI",
        rows=rows,
        meta=meta,
        confidence_level=0.975,
    )

    html_nejm = render_publication_html(
        table, style="NEJM", include_meta_paragraph=True
    )
    assert "OR (97.5% CI)" in html_nejm
    assert "two-sided 97.5% confidence intervals" in html_nejm

    html_apa = render_publication_html(table, style="APA7", include_meta_paragraph=True)
    assert "OR [97.5% CI]" in html_apa
    assert "OR = 1.45, 97.5% CI [1.12, 1.88]" in html_apa
    assert "two-sided 97.5% confidence intervals" in html_apa

"""
Unit tests for medstat missing data handling and reporting integrity verification.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure medstat package is on sys.path
scripts_path = str(
    Path(__file__).resolve().parents[2] / ".agents/skills/shinystat/scripts"
)
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from medstat.data.missing import (  # noqa: E402
    MissingStrategyRequiredError,
    prepare_data_for_analysis,
)
from medstat.reporting.integrity import verify_report_integrity  # noqa: E402
from medstat.reporting.ir import ReportDocument  # noqa: E402


def test_prepare_data_outcome_protection_complete_case():
    """Verify that complete-case is permitted when primary outcome has missing values."""
    df = pd.DataFrame(
        {
            "outcome": [1.0, 0.0, np.nan, 1.0, 0.0],
            "treatment": [1.0, 0.0, 1.0, 0.0, 1.0],
            "mediator": [2.1, 1.8, 3.4, 2.0, 1.5],
        }
    )

    # complete-case should succeed and drop the row with missing outcome
    df_clean, info = prepare_data_for_analysis(
        df,
        required_cols=["outcome", "treatment", "mediator"],
        handle_missing="complete-case",
        missing_justification="Pre-planned complete-case sensitivity analysis",
        outcome_cols=["outcome"],
    )

    assert len(df_clean) == 4
    assert df_clean["outcome"].isna().sum() == 0


def test_prepare_data_outcome_protection_imputation_prohibited():
    """Verify that imputation strategies (mice, knn, indicator) raise for missing outcomes."""
    df = pd.DataFrame(
        {
            "outcome": [1.0, 0.0, np.nan, 1.0, 0.0],
            "treatment": [1.0, 0.0, 1.0, 0.0, 1.0],
            "mediator": [2.1, 1.8, 3.4, 2.0, 1.5],
        }
    )

    for strat in ("mice", "knn", "indicator"):
        with pytest.raises(MissingStrategyRequiredError) as exc_info:
            prepare_data_for_analysis(
                df,
                required_cols=["outcome", "treatment", "mediator"],
                handle_missing=strat,
                missing_justification="Test missingness rationale",
                outcome_cols=["outcome"],
            )
        assert "Primary outcome column(s) ['outcome'] contain missing values" in str(
            exc_info.value
        )
        assert "outcomes must not be imputed" in str(exc_info.value)


def test_verify_report_integrity_structural_vs_untraced():
    """
    Verify that numbers in IGNORED_NUMBERS (0.05, 2024, etc.) are exempt ONLY when in structural context,
    and are flagged as untraced when reported as unsupported effect estimates or counts.
    """
    # Document A: Structural context for 0.05 (significance cutoff) and 2024 (calendar year)
    doc_structural = ReportDocument(
        title="Test Structural Study",
        results_dict={"mean_age": 45.0},
    )
    doc_structural.add_heading("Methods and Statistical Analysis", level=1)
    doc_structural.add_paragraph(
        "Between 2020 and 2024, we enrolled a cohort. Statistical significance was set at alpha = 0.05. "
        "The mean age was 45.0 years. Cohort retention and inclusion flow were documented."
    )

    rep_structural = verify_report_integrity(doc_structural)
    # 2020, 2024, 0.05 should be recognized as structural constants, 45.0 is in results_dict
    assert 2020.0 not in rep_structural.untraced_numbers
    assert 2024.0 not in rep_structural.untraced_numbers
    assert 0.05 not in rep_structural.untraced_numbers
    assert len(rep_structural.untraced_numbers) == 0

    # Document B: Non-structural narrative numbers matching IGNORED_NUMBERS values
    # e.g., OR = 0.05 and sample size 2024 patients, not present in results_dict
    doc_untraced = ReportDocument(
        title="Test Untraced Study",
        results_dict={"other_metric": 12.3},
    )
    doc_untraced.add_heading("Methods and Statistical Analysis", level=1)
    doc_untraced.add_paragraph(
        "A total of 2024 patients were analyzed. The treatment effect yielded an odds ratio of 0.05. "
        "Cohort retention flow was recorded."
    )

    rep_untraced = verify_report_integrity(doc_untraced)
    # 2024.0 (sample count) and 0.05 (unsupported effect estimate) must NOT be exempt!
    assert 2024.0 in rep_untraced.untraced_numbers
    assert 0.05 in rep_untraced.untraced_numbers


def test_renderers_graceful_import_error(monkeypatch):
    """Verify that render_docx and render_pptx raise clear ImportError when packages are missing."""
    import medstat.reporting.renderers.docx as docx_mod
    import medstat.reporting.renderers.pptx as pptx_mod

    doc = ReportDocument(title="Test Export")

    monkeypatch.setattr(docx_mod, "DOCX_AVAILABLE", False)
    with pytest.raises(ImportError) as exc_docx:
        docx_mod.render_docx(doc, "dummy.docx")
    assert "python-docx is required" in str(exc_docx.value)

    monkeypatch.setattr(pptx_mod, "PPTX_AVAILABLE", False)
    with pytest.raises(ImportError) as exc_pptx:
        pptx_mod.render_pptx(doc, "dummy.pptx")
    assert "python-pptx is required" in str(exc_pptx.value)

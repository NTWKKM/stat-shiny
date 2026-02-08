"""
Unit tests for MI reporting logic.
"""

from utils.logic import generate_mi_pooled_report


def test_generate_mi_pooled_report_columns():
    """Test that MI pooled report contains Crude OR and aOR columns."""

    # Mock data
    pooled_or = {
        "age": {
            "or": 1.2,
            "ci_low": 1.1,
            "ci_high": 1.3,
            "p_value": 0.001,
            "fmi": 0.1,
            "label": "Age",
        }
    }

    pooled_aor = {
        "age": {
            "aor": 1.15,
            "ci_low": 1.05,
            "ci_high": 1.25,
            "p_value": 0.002,
            "fmi": 0.12,
            "label": "Age",
        }
    }

    html_out = generate_mi_pooled_report(
        n_imputations=5, pooled_or=pooled_or, pooled_aor=pooled_aor
    )

    # Check for headers
    assert "<th>Crude OR (95% CI)</th>" in html_out
    assert "<th>aOR (95% CI)</th>" in html_out

    # Check for values
    assert "1.20" in html_out  # Crude OR
    assert "1.15" in html_out  # Adjusted OR


def test_generate_mi_pooled_report_missing_crude():
    """Test MI pooled report when Crude OR is missing for some reason."""

    pooled_aor = {
        "age": {
            "aor": 1.15,
            "ci_low": 1.05,
            "ci_high": 1.25,
            "p_value": 0.002,
            "fmi": 0.12,
            "label": "Age",
        }
    }

    html_out = generate_mi_pooled_report(
        n_imputations=5, pooled_or=None, pooled_aor=pooled_aor
    )

    assert "<th>Crude OR (95% CI)</th>" in html_out
    assert "<td>-</td>" in html_out  # Placeholder for missing crude

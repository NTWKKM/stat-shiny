from config import CONFIG
from utils.formatting import MissingDataStatement, PublicationFormatter


def test_publication_formatter():
    # Test Data
    coef = 3.14159
    se = 0.1
    ci_lower = 2.812
    ci_upper = 3.471
    p_val = 0.0031

    # NEJM
    nejm = PublicationFormatter.format_nejm(coef, se, ci_lower, ci_upper, p_val)
    print(f"NEJM: {nejm}")
    assert "3.14 (95% CI, 2.81 to 3.47); P=0.0031" in nejm or "P=0.003" in nejm

    # JAMA
    jama = PublicationFormatter.format_jama(coef, se, ci_lower, ci_upper, p_val)
    print(f"JAMA: {jama}")
    assert "3.14 (95% CI, 2.81-3.47); P=0.003" in jama or "P=.003" in jama

    # Lancet
    lancet = PublicationFormatter.format_lancet(coef, se, ci_lower, ci_upper, p_val)
    print(f"Lancet: {lancet}")
    assert "3.14 (95% CI 2.81–3.47), p=0.003" in lancet

    # BMJ
    bmj = PublicationFormatter.format_bmj(coef, se, ci_lower, ci_upper, p_val)
    print(f"BMJ: {bmj}")
    assert "3.14 (95% confidence interval 2.81 to 3.47); P=0.003" in bmj

    # Generic Dispatch
    CONFIG.update("analysis.publication_style", "lancet")
    generic = PublicationFormatter.format(coef, se, ci_lower, ci_upper, p_val)
    assert generic == lancet


def test_missing_data_statement():
    missing_info = {
        "strategy": "complete-case",
        "rows_analyzed": 90,
        "rows_excluded": 10,
        "summary_before": [
            {"Variable": "Age", "N_Missing": 5, "Pct_Missing": "5.0%"},
            {"Variable": "BMI", "N_Missing": 5, "Pct_Missing": "5.0%"},
        ],
    }

    text = MissingDataStatement.generate(missing_info)
    print(f"Missing Statement: {text}")

    assert "Of 100 observations, 10 (10.0%) were excluded" in text
    assert "Age (n=5, 5.0%)" in text
    assert "complete-case analysis" in text


if __name__ == "__main__":
    test_publication_formatter()
    test_missing_data_statement()

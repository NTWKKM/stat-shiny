import numpy as np
import pandas as pd
import pytest

from utils.data_quality import DataQualityReport


class TestDataQualityReport:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": ["a", "b", "c", "d", "e"],
                "C": [1.1, 2.2, np.nan, 4.4, 5.5],
                "D": [1, 1, 2, 2, 3],  # Duplicates potential
            }
        )

    def test_completeness_score(self, sample_data):
        report = DataQualityReport(sample_data)
        # Total cells = 20. 1 missing. 19/20 = 0.95 -> 95.0
        assert report.completeness_score() == 95.0

    def test_consistency_score_clean(self, sample_data):
        report = DataQualityReport(sample_data)
        # Assuming sample_data is relatively clean for types (excluding nan which is completeness)
        # A is int, B is str, C is float (with nan), D is int.
        # Strict nan count should be 0 for A, B, D.
        # For C: NaN is standard NaN, not "strict nan" (which implies non-standard string like "<5")
        # So consistency should be 100.
        assert report.consistency_score() == 100.0

    def test_consistency_score_dirty(self):
        df = pd.DataFrame({"A": ["1", "2", "<5", "4", "10%"]})
        # 5 rows. 2 dirty values (<5, 10%).
        # _is_numeric_column logic:
        # numeric_strict: 1, 2, NaN, 4, NaN.
        # numeric_coerced: 1, 2, 5, 4, 10.
        # is_strict_nan: False, False, True, False, True.
        # strict_nan_count = 2.
        # Score = 100 * (1 - 2/5) = 100 * 0.6 = 60.0
        report = DataQualityReport(df)
        assert report.consistency_score() == 60.0

    def test_uniqueness_score(self):
        df = pd.DataFrame({"A": [1, 1, 2, 3, 3], "B": ["a", "a", "b", "c", "c"]})
        # 5 rows. Row 1 (dup of 0) and Row 4 (dup of 3).
        # duplicated() returns [False, True, False, False, True] -> sum=2.
        # Score = 100 * (1 - 2/5) = 60.0
        report = DataQualityReport(df)
        assert report.uniqueness_score() == 60.0

    def test_validity_score(self, sample_data):
        report = DataQualityReport(sample_data)
        assert report.validity_score() == 100.0

    def test_generate_report(self, sample_data):
        report = DataQualityReport(sample_data)
        result = report.generate_report()

        assert "overall_score" in result
        assert "dimension_scores" in result
        assert "issues" in result
        assert "recommendations" in result

        scores = result["dimension_scores"]
        assert scores["completeness"] == 95.0
        assert scores["validity"] == 100.0

    def test_recommendations(self):
        df = pd.DataFrame({"A": [np.nan] * 10})  # 0% completeness
        report = DataQualityReport(df)
        result = report.generate_report()
        recs = result["recommendations"]
        assert any("significant missing values" in r for r in recs)

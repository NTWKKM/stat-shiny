import logging
import unittest

import pandas as pd

from utils.data_cleaning import prepare_data_for_analysis
from utils.formatting import create_missing_data_report_html
from utils.logic import analyze_outcome

# Configure logging to avoid noise during tests
logging.basicConfig(level=logging.ERROR)


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.df_clean = pd.DataFrame(
            {
                "outcome": [0, 1, 0, 1, 0],
                "age": [20, 30, 40, 50, 60],
                "score": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        self.df_dirty = pd.DataFrame(
            {
                "outcome": [0, 1, None, 1, 0, 1],
                "age": [20, 30, 40, None, 60, -99],  # -99 is a custom missing code
                "score": [1.1, 2.2, 3.3, 4.4, "nan", 6.6],
            }
        )

    def test_prepare_data_simple(self):
        """Test basic cleaning without custom codes."""
        df_out, info = prepare_data_for_analysis(
            self.df_dirty,
            required_cols=["outcome", "age", "score"],
            numeric_cols=["age", "score"],
        )
        # Should drop rows with None or "nan"
        # Row 2 (index 2) has outcome=None
        # Row 3 (index 3) has age=None
        # Row 4 (index 4) has score="nan"
        # Row 5 (index 5) has age=-99 (valid numeric here unless coded)

        # Expected: Rows 0, 1, 5 are kept.
        self.assertEqual(len(df_out), 3)
        self.assertEqual(info["rows_excluded"], 3)

    def test_prepare_data_custom_codes(self):
        """Test cleaning with custom missing codes."""
        missing_codes = {"age": [-99]}
        df_out, info = prepare_data_for_analysis(
            self.df_dirty,
            required_cols=["outcome", "age", "score"],
            numeric_cols=["age", "score"],
            missing_codes=missing_codes,
        )
        # Row 5 has age=-99 which should be treated as missing
        # So independent rows dropped: 2, 3, 4, 5
        # Expected: Rows 0, 1 kept.
        self.assertEqual(len(df_out), 2)
        self.assertEqual(info["rows_excluded"], 4)

    def test_html_report_generation(self):
        """Test HTML report generation."""
        info = {
            "strategy": "complete-case",
            "rows_analyzed": 100,
            "rows_excluded": 20,
            "summary_before": [
                {"Variable": "age", "N_Missing": 10, "Pct_Missing": "8.3%"},
                {"Variable": "score", "N_Missing": 20, "Pct_Missing": "16.7%"},
            ],
        }
        html = create_missing_data_report_html(info, {})
        self.assertIn("Missing Data Summary", html)
        self.assertIn("Strategy:", html)
        self.assertIn("<b>20</b>", html)
        self.assertIn("16.7%", html)

    def test_logic_integration(self):
        """Test integration with logic.analyze_outcome."""
        # Using dirty data that allows for a valid model after cleaning
        # Need enough data points.
        df = pd.DataFrame(
            {
                "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "x2": [1, 1, 1, 1, 1, None, 1, 1, 1, 1],  # One missing
            }
        )

        html_table, or_res, _, _ = analyze_outcome("y", df)

        # Check if missing data report is in the HTML
        self.assertIn("Missing Data Summary", html_table)
        self.assertIn("Excluded:", html_table)


if __name__ == "__main__":
    unittest.main()

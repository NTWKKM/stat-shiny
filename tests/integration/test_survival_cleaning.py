
import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from unittest.mock import MagicMock, patch
from utils.survival_lib import fit_km_logrank, fit_nelson_aalen, fit_cox_ph, check_cph_assumptions

class TestSurvivalPipeline(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with missing values
        self.df = pd.DataFrame({
            "time": [10, 20, np.nan, 40, 50, 10, 20],
            "event": [1, 0, 1, 1, 0, 1, 0],
            "group": ["A", "A", "B", "B", "A", "B", "A"],
            "age": [50, 60, 70, 80, np.nan, 55, 65],
            "marker": [1.1, 2.2, 3.3, 4.4, 5.5, np.nan, 7.7]
        })
        self.var_meta = {
            "time": {"label": "Survival Time"},
            "event": {"label": "Status"},
            "group": {"label": "Treatment Group"},
            "age": {"label": "Age (Years)"},
            "marker": {"label": "Biomarker"}
        }

    def test_fit_km_logrank_pipeline(self):
        """Test KM Logrank with unified pipeline integration."""
        fig, stats, missing_info = fit_km_logrank(
            self.df, "time", "event", "group", var_meta=self.var_meta
        )
        
        # Check structure
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertIsInstance(missing_info, dict)
        
        # Check missing info content
        self.assertEqual(missing_info["rows_original"], 7)
        self.assertEqual(missing_info["rows_analyzed"], 6) # One row with missing time
        self.assertEqual(missing_info["rows_excluded"], 1)
        self.assertEqual(len(missing_info["summary_before"]), 1) # Only 'time' is missing in checked cols
        
    def test_fit_nelson_aalen_pipeline(self):
        """Test Nelson-Aalen with unified pipeline integration."""
        fig, stats, missing_info = fit_nelson_aalen(
            self.df, "time", "event", group_col="group", var_meta=self.var_meta
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(missing_info, dict)
        self.assertEqual(missing_info["rows_analyzed"], 6)
        
    def test_fit_cox_ph_pipeline(self):
        """Test Cox PH with unified pipeline integration."""
        model, res_df, clean_data, error, stats, missing_info = fit_cox_ph(
            self.df, "time", "event", ["age", "marker"], var_meta=self.var_meta
        )
        
        if error:
            self.fail(f"Cox PH fit failed: {error}")
            
        self.assertIsInstance(missing_info, dict)
        # Should exclude row with missing time (idx 2), row with missing age (idx 4), row with missing marker (idx 5)
        # Original: 7
        # Missing time: 1 (idx 2)
        # Missing age: 1 (idx 4)
        # Missing marker: 1 (idx 5)
        # Total excluded: 3 (indices 2, 4, 5)
        # Analyzed: 4
        self.assertEqual(missing_info["rows_analyzed"], 4)
        self.assertEqual(missing_info["rows_excluded"], 3)
        self.assertEqual(len(missing_info["summary_before"]), 3) # time, age, marker all have missing

if __name__ == "__main__":
    unittest.main()

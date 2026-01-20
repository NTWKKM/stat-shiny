"""
🔗 Integration Tests for Logic Module Pipeline
File: tests/integration/test_logic_pipeline.py

Tests the complete flow of functions from logic.py AND visualization:
1. Data loading and validation
2. Numeric cleaning
3. Model execution (run_binary_logit)
4. Integration with Forest Plot (forest_plot_lib)
"""

import numpy as np
import pandas as pd
import pytest

from utils.forest_plot_lib import create_forest_plot
from utils.logic import (
    analyze_outcome,
    clean_numeric_value,
    run_binary_logit,
    validate_logit_data,
)

# Mark as integration test
pytestmark = pytest.mark.integration


class TestLogicPipeline:
    """
    Integration tests for Logic module (Logistic Regression flow)
    """

    @pytest.fixture
    def sample_medical_data(self):
        """
        🏥 Create realistic medical dataset for testing
        """
        np.random.seed(42)
        n_patients = 150

        # Realistic age distribution
        age = np.random.normal(55, 15, n_patients)
        age = np.clip(age, 18, 90)

        # Blood pressure (SBP in mmHg)
        sbp = np.random.normal(130, 20, n_patients)
        sbp = np.clip(sbp, 80, 200)

        # Cholesterol (mg/dL)
        cholesterol = np.random.normal(200, 50, n_patients)
        cholesterol = np.clip(cholesterol, 100, 400)

        # Binary outcome: disease status (0=no disease, 1=disease)
        prob = 1 / (1 + np.exp(-(0.05 * age + 0.01 * cholesterol - 10)))
        outcome = np.random.binomial(1, prob)
        # Ensure at least one of each class for model stability
        if len(np.unique(outcome)) < 2:
            outcome[0] = 0
            outcome[1] = 1

        # Create DataFrame
        df = pd.DataFrame({"age": age, "sbp": sbp, "cholesterol": cholesterol})

        return df, pd.Series(outcome, name="disease")

    def test_complete_logic_flow(self, sample_medical_data):
        """🔄 Test complete logistic regression pipeline from logic.py"""
        df, y = sample_medical_data

        # Step 1: Validate input data
        valid, msg = validate_logit_data(y, df)
        assert valid is True, f"Data validation failed: {msg}"

        # Step 2: Run logistic regression
        params, _conf, _pvals, status, metrics = run_binary_logit(y, df)

        # Step 3: Verify output
        assert status == "OK", f"Logit regression failed: {status}"
        assert params is not None
        assert len(params) == len(df.columns) + 1  # Including 'const'

        # Verify metrics
        assert isinstance(metrics, dict)
        assert "mcfadden" in metrics
        assert "p_value" in metrics

    def test_logic_to_forest_plot_integration(self, sample_medical_data):
        """
        🌲 Test Integration: Logic Results -> Forest Plot
        Verifies that logistic regression output can be visualized as a Forest Plot
        """
        df, y = sample_medical_data

        # 1. Run Logistic Regression via analyze_outcome (which returns structured results)
        # analyze_outcome combines run_binary_logit and formatting
        full_df = df.copy()
        full_df["disease"] = y

        # Use analyze_outcome to get Odds Ratios (OR) and Confidence Intervals
        _html_table, or_results, aor_results, _int_results = analyze_outcome(
            "disease", full_df
        )

        # Verify analyze_outcome returned results
        assert or_results is not None or aor_results is not None

        # Rerun raw logit to get numerical values for plotting
        params, conf, pvals, _status, _metrics = run_binary_logit(y, df)

        # Construct DataFrame for Forest Plot
        plot_df = pd.DataFrame(
            {
                "Subgroup": params.index,
                "Level": [""] * len(params),  # No levels for continuous
                "Est": np.exp(params.values),  # Odds Ratio
                "Lower": np.exp(conf[0].values),
                "Upper": np.exp(conf[1].values),
                "P-value": pvals.values,
            }
        )

        # 3. Create Forest Plot
        fig = create_forest_plot(
            plot_df,
            "Est",
            "Lower",
            "Upper",
            "Subgroup",
            title="Logistic Regression Results (Forest Plot)",
            xlabel="Odds Ratio (95% CI)",
        )

        assert fig is not None
        assert hasattr(fig, "layout")
        assert "Logistic Regression" in fig.layout.title.text
        # Verify data points are plotted
        assert len(fig.data) > 0

    def test_data_flow_with_numeric_cleaning(self, sample_medical_data):
        """🧹 Test data flow with numeric value cleaning"""
        df, y = sample_medical_data

        # Simulate real-world dirty data
        df_formatted = df.copy()
        df_formatted["age"] = df_formatted["age"].apply(lambda x: f"{x:.0f}")
        df_formatted["cholesterol"] = df_formatted["cholesterol"].apply(
            lambda x: f"{x:,.1f}"
        )

        # Clean the formatted values using logic.clean_numeric_value
        df_clean = df_formatted.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_numeric_value)

        # Run pipeline with cleaned data
        valid, _msg = validate_logit_data(y, df_clean)
        assert valid is True

        _params, _conf, _pvals, status, _metrics = run_binary_logit(y, df_clean)
        assert status == "OK"

    def test_pipeline_with_multiple_covariates(self, sample_medical_data):
        """📊 Test pipeline with multiple covariates"""
        df, y = sample_medical_data

        params, _conf, pvals, status, metrics = run_binary_logit(y, df)

        assert status == "OK"
        assert len(params) == 4  # age, sbp, cholesterol + const
        assert len(pvals) == 4
        assert metrics["mcfadden"] > 0

    def test_pipeline_robustness_with_subset(self, sample_medical_data):
        """🎯 Test pipeline robustness with variable subsets"""
        df, y = sample_medical_data

        subsets = [
            ["age"],  # Single variable
            ["age", "sbp"],  # Two variables
            ["age", "sbp", "cholesterol"],  # All variables
        ]

        for subset_vars in subsets:
            df_subset = df[subset_vars]

            valid, _msg = validate_logit_data(y, df_subset)
            assert valid is True

            params, _conf, _pvals, status, _metrics = run_binary_logit(y, df_subset)
            assert status == "OK"
            assert len(params) == len(subset_vars) + 1  # Including 'const'

    def test_outcome_variable_consistency(self, sample_medical_data):
        """✅ Test outcome variable consistency"""
        df, y = sample_medical_data

        unique_values = y.unique()
        assert len(unique_values) <= 2

        _params, _conf, _pvals, status, _metrics = run_binary_logit(y, df)
        # run_binary_logit doesn't return n_obs in metrics anymore
        assert status == "OK"

    def test_pipeline_with_missing_values(self):
        """⚠️ Test pipeline handles missing values"""
        df = pd.DataFrame(
            {"age": [25, 30, np.nan, 45, 50], "sbp": [120, 130, 135, np.nan, 150]}
        )
        y = pd.Series([0, 1, 0, 1, 0])

        valid, _msg = validate_logit_data(y, df)
        # Just ensure it doesn't crash and returns valid boolean
        assert isinstance(valid, bool)

    def test_pipeline_with_constant_outcome(self):
        """🚫 Test pipeline rejects constant outcome"""
        df = pd.DataFrame(
            {"age": [25, 30, 35, 40, 45], "sbp": [120, 130, 135, 140, 150]}
        )
        y = pd.Series([1, 1, 1, 1, 1])  # No variation

        valid, msg = validate_logit_data(y, df)
        assert valid is False or "constant" in msg.lower() or "variance" in msg.lower()

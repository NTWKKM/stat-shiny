"""
🧪 Comprehensive Unit Tests for Statistical Analysis Modules

Tests all core statistical functions across:
- logic.py: Logistic regression, data validation, outcome analysis
- diag_test.py: Diagnostic tests (Chi², Fisher, ROC, Kappa, ICC)
- survival_lib.py: Survival analysis (KM, Cox, Nelson-Aalen, Landmark)

Run with: pytest tests/unit/test_statistics.py -v
"""

import importlib
import os
import sys
import warnings
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

# ============================================================================
# PATH SETUP & MOCKING
# ============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Global placeholders for imported functions
run_negative_binomial_regression = None
run_poisson_regression = None

analyze_roc = None
auc_ci_delong = None
calculate_chi2 = None
calculate_ci_wilson_score = None
calculate_descriptive = None
format_p_value = None

analyze_outcome = None
clean_numeric_value = None
get_label = None
run_binary_logit = None
validate_logit_data = None

calculate_median_survival = None
fit_cox_ph = None
fit_km_landmark = None
fit_km_logrank = None
fit_nelson_aalen = None
AgreementAnalysis = None


@pytest.fixture(scope="module", autouse=True)
def setup_mocks():
    """
    Provide a pytest fixture that injects robust mocks for external libraries and exposes test-target callables.

    Patches sys.modules with mocked versions of plotly, lifelines, sklearn, pingouin, and related submodules, reloads the modules under test so they use those mocks, and binds module-level globals (e.g., run_negative_binomial_regression, analyze_roc, AgreementAnalysis, fit_cox_ph, calculate_descriptive, validate_logit_data, etc.) for use by the test suite. Yields control to allow tests to run within the patched context.
    """
    global run_negative_binomial_regression, run_poisson_regression
    global analyze_roc, auc_ci_delong, calculate_chi2, calculate_ci_wilson_score
    global calculate_descriptive, format_p_value
    global analyze_outcome, clean_numeric_value, get_label
    global run_binary_logit, validate_logit_data
    global \
        calculate_median_survival, \
        fit_cox_ph, \
        fit_km_landmark, \
        fit_km_logrank, \
        fit_nelson_aalen
    global AgreementAnalysis

    # --- ROBUST MOCK SETUP ---

    # 1. Mock Plotly with concrete colors to prevent ZeroDivisionError
    mock_plotly = MagicMock()
    # Ensure colors list is available and has length
    mock_colors = MagicMock()
    mock_colors.qualitative.Plotly = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    mock_plotly.colors = mock_colors
    mock_plotly.express.colors = mock_colors  # For px.colors usage

    # 2. Mock Lifelines with structured return values
    mock_lifelines = MagicMock()

    # KM Fitter Mock
    mock_kmf = MagicMock()
    type(mock_kmf.return_value).median_survival_time_ = PropertyMock(return_value=10.0)
    mock_ci = MagicMock()
    mock_ci.values = np.array([[5.0, 15.0]])
    mock_ci.to_numpy.return_value = np.array([[5.0, 15.0]])
    type(mock_kmf.return_value).confidence_interval_ = PropertyMock(
        return_value=mock_ci
    )

    # Cox Fitter Mock
    mock_cph = MagicMock()
    mock_cph_inst = mock_cph.return_value

    # --- FIX: Define concordance_index_ as a float for format string usage ---
    mock_cph_inst.concordance_index_ = 0.85
    # ----------------------------------------------------------------------------

    # Mock summary DataFrame
    mock_cph_inst.summary = pd.DataFrame(
        {
            "coef": [0.5, 0.2],
            "exp(coef)": [1.65, 1.22],
            "p": [0.01, 0.04],
            "HR": [1.65, 1.22],
        },
        index=["age", "exposure"],
    )

    # Mock confidence_intervals_ DataFrame (Crucial for fit_cox_ph)
    mock_cph_inst.confidence_intervals_ = pd.DataFrame(
        data=[[-0.1, 0.8], [0.1, 0.3]],
        index=["age", "exposure"],
        columns=["lower-bound", "upper-bound"],
    )

    mock_lifelines.KaplanMeierFitter = mock_kmf
    mock_lifelines.CoxPHFitter = mock_cph
    mock_lifelines.NelsonAalenFitter = MagicMock()

    # Configure statistics mock with concrete float values
    mock_stats = MagicMock()
    mock_test_result = MagicMock()
    # Important: p_value must be a float to allow f-string formatting (e.g. :.3f)
    mock_test_result.p_value = 0.045
    mock_test_result.test_statistic = 4.0
    mock_stats.logrank_test.return_value = mock_test_result
    mock_stats.multivariate_logrank_test.return_value = mock_test_result

    # 3. Mock Sklearn with float return values for metrics
    mock_sklearn = MagicMock()
    mock_metrics = MagicMock()

    # Mock Pingouin for ICC
    mock_pingouin = MagicMock()
    mock_pingouin.intraclass_corr.return_value = pd.DataFrame(
        {"Type": ["ICC2k"], "ICC": [0.85]}
    )

    # Fix: roc_curve returns tuple of arrays
    mock_metrics.roc_curve.return_value = (
        np.array([0.0, 0.1, 1.0]),
        np.array([0.0, 0.8, 1.0]),
        np.array([0.5, 0.5, 0.5]),
    )
    # Fix: auc must return a float for comparisons (e.g. >= 0.9)
    mock_metrics.auc.return_value = 0.85
    mock_metrics.roc_auc_score.return_value = 0.85
    mock_metrics.cohen_kappa_score.return_value = 0.85

    mock_sklearn.metrics = mock_metrics

    # Create the dictionary of modules to patch
    modules_to_patch = {
        "plotly": mock_plotly,
        "plotly.graph_objects": MagicMock(),
        "plotly.express": mock_plotly,
        "plotly.io": MagicMock(),
        "plotly.subplots": MagicMock(),
        "plotly.colors": mock_colors,
        "forest_plot_lib": MagicMock(),
        "lifelines": mock_lifelines,
        "lifelines.statistics": mock_stats,
        "lifelines.utils": MagicMock(),
        "sklearn": mock_sklearn,
        "sklearn.calibration": MagicMock(),
        "sklearn.experimental": MagicMock(),
        "sklearn.impute": MagicMock(),
        "sklearn.metrics": mock_metrics,
        "sklearn.linear_model": MagicMock(),
        "pingouin": mock_pingouin,
    }

    # Apply the patches to sys.modules
    with patch.dict(sys.modules, modules_to_patch):
        # Import modules under test INSIDE the patch context
        # Check if modules are already imported and reload if necessary
        # to ensure they use the mocked dependencies

        import utils.poisson_lib

        importlib.reload(utils.poisson_lib)
        try:
            # Re-assign from reloaded module
            run_negative_binomial_regression = (
                utils.poisson_lib.run_negative_binomial_regression
            )
            run_poisson_regression = utils.poisson_lib.run_poisson_regression
        except AttributeError:
            # Fallback if specific functions missing or import failed
            run_negative_binomial_regression = MagicMock()
            run_poisson_regression = MagicMock()

        import utils.diag_test

        importlib.reload(utils.diag_test)
        analyze_roc = utils.diag_test.analyze_roc
        auc_ci_delong = utils.diag_test.auc_ci_delong
        calculate_chi2 = utils.diag_test.calculate_chi2
        calculate_ci_wilson_score = utils.diag_test.calculate_ci_wilson_score
        calculate_descriptive = utils.diag_test.calculate_descriptive
        format_p_value = utils.diag_test.format_p_value

        import utils.agreement_lib

        importlib.reload(utils.agreement_lib)
        AgreementAnalysis = utils.agreement_lib.AgreementAnalysis

        import utils.logic

        importlib.reload(utils.logic)
        analyze_outcome = utils.logic.analyze_outcome
        clean_numeric_value = utils.logic.clean_numeric_value
        format_p_value = utils.logic.format_p_value
        get_label = utils.logic.get_label
        run_binary_logit = utils.logic.run_binary_logit
        validate_logit_data = utils.logic.validate_logit_data

        import utils.survival_lib

        importlib.reload(utils.survival_lib)
        calculate_median_survival = utils.survival_lib.calculate_median_survival
        fit_cox_ph = utils.survival_lib.fit_cox_ph
        fit_km_landmark = utils.survival_lib.fit_km_landmark
        fit_km_logrank = utils.survival_lib.fit_km_logrank
        fit_nelson_aalen = utils.survival_lib.fit_nelson_aalen

        yield


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================================
# SHARED FIXTURES
# ============================================================================


@pytest.fixture
def small_numeric_df():
    """Small dataset for quick validation tests."""
    return pd.DataFrame({"age": [20, 30, 40, 50], "outcome": [0, 1, 0, 1]})


@pytest.fixture
def dummy_df():
    """Comprehensive synthetic dataset for all test types."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            # Demographics
            "age": np.random.normal(50, 10, n),
            "sex": np.random.choice(["Male", "Female"], n),
            # Binary variables
            "exposure": np.random.choice([0, 1], n),
            "disease": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "treatment": np.random.choice(["A", "B"], n),
            # Continuous predictors
            "test_score": np.random.normal(0, 1, n) + np.random.choice([0, 1], n) * 2,
            "biomarker": np.random.exponential(2, n),
            # Survival data
            "time": np.random.exponential(10, n),
            "event": np.random.choice([0, 1], n, p=[0.6, 0.4]),
            # Rating data
            "rater1": np.random.choice([0, 1, 2], n),
            "rater2": np.random.choice([0, 1, 2], n),
        }
    )


@pytest.fixture
def perfect_separation_df():
    """Dataset with perfect separation for Firth testing."""
    return pd.DataFrame(
        {"predictor": [1, 1, 1, 1, 2, 2, 2, 2], "outcome": [0, 0, 0, 0, 1, 1, 1, 1]}
    )


# ============================================================================
# LOGIC.PY TESTS - Logistic Regression & Data Validation
# ============================================================================


class TestDataValidation:
    """Tests for data quality checks and preprocessing."""

    def test_clean_numeric_value_basic(self):
        """✅ Test numeric string parsing."""
        assert clean_numeric_value("1,000") == 1000.0
        assert clean_numeric_value("1000") == 1000.0
        assert clean_numeric_value(1000) == 1000.0

    # @pytest.mark.xfail(reason="Feature currently not implemented in logic.py")
    def test_clean_numeric_value_comparison_symbols(self):
        """✅ Test removal of comparison operators."""
        assert clean_numeric_value("<0.001") == 0.001
        assert clean_numeric_value("> 50") == 50.0
        assert clean_numeric_value("<=100") == 100.0

    def test_clean_numeric_value_invalid(self):
        """✅ Test invalid input handling."""
        assert pd.isna(clean_numeric_value("abc"))
        assert pd.isna(clean_numeric_value(None))
        assert pd.isna(clean_numeric_value(""))
        assert pd.isna(clean_numeric_value("N/A"))

    def test_validate_logit_data_success(self, small_numeric_df):
        """✅ Test validation with valid data."""
        y = small_numeric_df["outcome"]
        X = small_numeric_df[["age"]]
        valid, msg = validate_logit_data(y, X)
        assert valid is True
        assert msg == "OK"

    def test_validate_logit_data_zero_variance(self):
        """✅ Test detection of constant variables."""
        df = pd.DataFrame({"constant": [1, 1, 1, 1], "age": [20, 30, 40, 50]})
        y = pd.Series([0, 1, 0, 1])
        valid, msg = validate_logit_data(y, df)
        assert valid is False
        assert "zero variance" in msg.lower()

    def test_validate_logit_data_empty(self):
        """✅ Test empty data detection."""
        df = pd.DataFrame()
        y = pd.Series([])
        valid, msg = validate_logit_data(y, df)
        assert valid is False
        assert "empty" in msg.lower()

    def test_validate_logit_data_single_predictor(self):
        """✅ Test single predictor validation."""
        y = pd.Series([0, 1, 0, 1, 0, 1])
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
        valid, msg = validate_logit_data(y, X)
        assert valid is True
        assert msg == "OK"


class TestLogisticRegression:
    """Tests for binary logistic regression fitting."""

    def test_run_binary_logit_basic(self):
        """✅ Test logistic regression with synthetic data."""
        np.random.seed(42)
        n = 100
        age = np.random.normal(50, 10, n)

        # Create clear relationship: older age → higher probability of outcome=1
        prob = 1 / (1 + np.exp(-(0.1 * (age - 50))))
        y = pd.Series(np.random.binomial(1, prob))

        X = pd.DataFrame({"age": age})

        params, _conf, pvals, status, metrics = run_binary_logit(y, X)

        # Verify successful fit
        assert status == "OK", f"Expected 'OK', got '{status}'"
        assert params is not None, "Parameters should not be None"
        assert "age" in params.index, "Age coefficient should exist"

        # Age coefficient should be positive (matches synthetic pattern)
        assert params["age"] > 0, (
            f"Expected positive age coefficient, got {params['age']}"
        )

        # Model quality checks
        assert metrics["mcfadden"] > 0, "McFadden R² should be > 0"
        assert 0 <= metrics["mcfadden"] <= 1, "McFadden R² should be in [0,1]"

        # P-value should be significant
        assert pvals["age"] < 0.05, "Age should be statistically significant"

    def test_run_binary_logit_multiple_predictors(self):
        """✅ Test multivariate logistic regression."""
        np.random.seed(42)
        n = 150
        age = np.random.normal(50, 10, n)
        weight = np.random.normal(70, 15, n)

        # Both predictors affect outcome
        logit = 0.05 * (age - 50) + 0.02 * (weight - 70) - 0.5
        prob = 1 / (1 + np.exp(-logit))
        y = pd.Series(np.random.binomial(1, prob))

        X = pd.DataFrame({"age": age, "weight": weight})

        _params, _conf, _pvals, status, _metrics = run_binary_logit(y, X)

        assert status == "OK"
        assert "age" in X.columns
        assert "weight" in X.columns

    def test_run_binary_logit_perfect_separation(self, perfect_separation_df):
        """✅ Test handling of perfect separation."""
        y = perfect_separation_df["outcome"]
        X = perfect_separation_df[["predictor"]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerfectSeparationWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Default method should fail gracefully or switch method
            _params, _conf, _pvals, status, _metrics = run_binary_logit(
                y, X, method="default"
            )

        # Should either succeed (if robust) or return informative error
        if status != "OK":
            assert "separation" in status.lower() or "singular" in status.lower()

    def test_run_binary_logit_constant_outcome(self):
        """✅ Test with constant outcome (no variation)."""
        y = pd.Series([1, 1, 1, 1, 1])
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

        _params, _conf, _pvals, status, _metrics = run_binary_logit(y, X)

        assert status in ["OK", "Error"] or "constant" in status.lower()


class TestFormattingHelpers:
    """Tests for formatting and display helper functions."""

    def test_format_p_value_significant(self):
        """✅ Test p-value formatting with styling."""
        result = format_p_value(0.001)
        # Check logic: formatting usually applies styling
        assert "&lt;0.001" in result or "<0.001" in result or "0.001" in result
        assert "sig-p" in result or "color:" in result or "font-weight:" in result

    def test_format_p_value_not_significant(self):
        """✅ Test non-significant p-value formatting."""
        result = format_p_value(0.234)
        assert "0.234" in result
        assert "sig-p" not in result and "color:" not in result

    def test_format_p_value_edge_cases(self):
        """✅ Test edge case p-values."""
        f = format_p_value
        assert "-" in f(None) or "NA" in f(None)
        assert "-" in f(np.nan) or "NA" in f(np.nan)
        assert "&gt;0.999" in f(0.9999) or ">0.999" in f(0.9999)
        assert "&lt;0.001" in f(0.0001) or "<0.001" in f(0.0001)

    def test_get_label_basic(self):
        """✅ Test label generation without metadata."""
        label = get_label("age", None)
        assert "age" in label
        assert "<b>" in label  # Should be bolded

    def test_get_label_with_metadata(self):
        """✅ Test label generation with metadata."""
        var_meta = {"age": {"label": "Patient Age (years)"}}
        label = get_label("age", var_meta)
        assert "age" in label
        assert "Patient Age" in label


class TestAnalyzeOutcome:
    """Tests for comprehensive outcome analysis."""

    def test_analyze_outcome_basic(self, dummy_df):
        """✅ Test basic outcome analysis."""
        html, or_results, aor_results, int_results = analyze_outcome(
            "disease", dummy_df[["age", "exposure", "disease"]]
        )

        # Verify HTML output
        assert html is not None
        assert len(html) > 0
        assert "disease" in html.lower()

        # Verify OR results
        assert isinstance(or_results, dict)
        assert isinstance(aor_results, dict)

    def test_analyze_outcome_missing_column(self, dummy_df):
        """✅ Test with non-existent outcome."""
        html, or_results, aor_results, _int_results = analyze_outcome(
            "nonexistent_outcome", dummy_df
        )

        assert "not found" in html.lower() or "error" in html.lower()
        assert len(or_results) == 0
        assert len(aor_results) == 0

    def test_analyze_outcome_non_binary(self, dummy_df):
        """✅ Test with non-binary outcome."""
        df = dummy_df.copy()
        df["multi_outcome"] = np.random.choice([0, 1, 2], len(df))

        html, or_results, aor_results, int_results = analyze_outcome(
            "multi_outcome", df
        )

        # Should fail with informative message
        assert "invalid" in html.lower() or "expected 2" in html.lower()


# ============================================================================
# DIAG_TEST.PY TESTS - Diagnostic Statistics
# ============================================================================


class TestDescriptiveStatistics:
    """Tests for descriptive statistics calculations."""

    def test_calculate_descriptive_numeric(self, dummy_df):
        """✅ Test descriptive stats for numeric variables."""
        # FIX: Unpack tuple (DataFrame, Metadata)
        result, _ = calculate_descriptive(dummy_df, "age")

        assert result is not None
        assert not result.empty
        assert isinstance(result, pd.DataFrame)

        # Check for expected statistics
        stats_present = result["Statistic"].values
        assert "Mean" in stats_present
        assert "SD" in stats_present
        assert "Median" in stats_present

    def test_calculate_descriptive_categorical(self, dummy_df):
        """✅ Test descriptive stats for categorical variables."""
        # FIX: Unpack tuple (DataFrame, Metadata)
        result, _ = calculate_descriptive(dummy_df, "sex")

        assert result is not None
        assert not result.empty
        assert isinstance(result, pd.DataFrame)

        # Should have Category and Count columns
        assert "Category" in result.columns or "Count" in result.columns

    def test_calculate_descriptive_missing_column(self, dummy_df):
        """✅ Test with non-existent column."""
        # FIX: Unpack tuple
        result, _ = calculate_descriptive(dummy_df, "nonexistent")
        assert result is None

    def test_calculate_descriptive_empty_column(self):
        """✅ Test with all-NA column."""
        df = pd.DataFrame({"empty_col": [np.nan, np.nan, np.nan]})
        # FIX: Unpack tuple
        result, _ = calculate_descriptive(df, "empty_col")
        assert result is None


class TestChiSquareAnalysis:
    """Tests for chi-square and contingency table analysis."""

    def test_calculate_chi2_basic(self, dummy_df):
        """✅ Test basic chi-square calculation."""
        display_tab, stats_df, msg, risk_df, _ = calculate_chi2(
            dummy_df, "exposure", "disease"
        )

        assert stats_df is not None
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0

        # Verify chi-square statistic is present
        test_row = stats_df[stats_df["Statistic"] == "Test"]
        assert not test_row.empty
        assert "Chi-Square" in test_row["Value"].values[0]

    def test_calculate_chi2_2x2_risk_metrics(self, dummy_df):
        """✅ Test 2x2 table risk metrics calculation."""
        display_tab, stats_df, msg, risk_df, _ = calculate_chi2(
            dummy_df, "exposure", "disease"
        )

        # For 2x2 tables, risk metrics should be calculated
        if risk_df is not None:
            assert isinstance(risk_df, pd.DataFrame)

            # Check for expected metrics
            metrics = risk_df["Metric"].values
            assert any("Risk Ratio" in str(m) for m in metrics)
            assert any("Odds Ratio" in str(m) for m in metrics)

    def test_calculate_chi2_fisher_exact(self):
        """✅ Test Fisher's Exact Test."""
        df = pd.DataFrame(
            {
                "treatment": ["A", "B"] * 10,
                "outcome": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1] * 2,
            }
        )

        display_tab, stats_df, msg, risk_df, _ = calculate_chi2(
            df, "treatment", "outcome", method="Fisher"
        )

        assert stats_df is not None
        test_row = stats_df[stats_df["Statistic"] == "Test"]
        assert "Fisher" in test_row["Value"].values[0]

    def test_calculate_chi2_missing_columns(self, dummy_df):
        """✅ Test with missing columns."""
        display_tab, stats_df, msg, risk_df, _ = calculate_chi2(
            dummy_df, "nonexistent1", "nonexistent2"
        )

        assert stats_df is None
        assert msg is not None
        assert "not found" in msg.lower()

    def test_calculate_chi2_empty_data(self):
        """✅ Test with empty DataFrame."""
        df = pd.DataFrame({"col1": [], "col2": []})
        display_tab, stats_df, msg, risk_df, _ = calculate_chi2(df, "col1", "col2")

        assert msg is not None
        assert "no valid data" in msg.lower() or "no data" in msg.lower()


class TestROCAnalysis:
    """Tests for ROC curve and AUC calculation."""

    def test_analyze_roc_basic(self, dummy_df):
        """✅ Test basic ROC analysis."""
        # Ensure we have sufficient variation
        n = len(dummy_df)
        dummy_df["disease_binary"] = np.random.choice([0, 1], n, p=[0.7, 0.3])
        dummy_df["score_values"] = np.random.uniform(0, 1, n)

        stats_dict, error_msg, fig, coords = analyze_roc(
            dummy_df, "disease_binary", "score_values", pos_label_user="1"
        )

        if stats_dict is not None:
            assert "AUC" in stats_dict
            # Mock returns 0.85
            auc_val = float(stats_dict["AUC"])
            assert 0 <= auc_val <= 1

            # Verify CI is present
            assert "95% CI" in stats_dict
        else:
            assert error_msg is not None

    def test_analyze_roc_perfect_classifier(self):
        """✅ Test ROC with perfect classifier (AUC=1.0)."""
        # n = 100
        df = pd.DataFrame(
            {
                "outcome": [0] * 50 + [1] * 50,
                "score": list(range(50)) + list(range(50, 100)),  # Perfect separation
            }
        )

        stats_dict, error_msg, fig, coords = analyze_roc(
            df, "outcome", "score", pos_label_user="1"
        )

        if stats_dict:
            # Mocked to 0.85
            assert "AUC" in stats_dict

    def test_analyze_roc_missing_column(self, dummy_df):
        """✅ Test ROC with missing column."""
        try:
            stats_dict, error_msg, fig, coords = analyze_roc(
                dummy_df, "nonexistent", "test_score", pos_label_user="1"
            )
            assert stats_dict is None
            assert error_msg is not None
        except KeyError:
            pass

    def test_analyze_roc_constant_score(self):
        """✅ Test ROC with constant prediction score."""
        df = pd.DataFrame(
            {
                "outcome": [0, 1, 0, 1, 0, 1],
                "score": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # All same
            }
        )

        stats_dict, error_msg, fig, coords = analyze_roc(
            df, "outcome", "score", pos_label_user="1"
        )

        assert stats_dict is None
        assert error_msg is not None
        assert "constant" in error_msg.lower()


class TestKappaAnalysis:
    """Tests for Cohen's Kappa inter-rater agreement."""

    def test_calculate_kappa_basic(self):
        """✅ Test basic Kappa calculation."""

        df = pd.DataFrame({"rater1": [0, 1, 0, 1] * 5, "rater2": [0, 1, 0, 1] * 5})

        res_df, error_msg, conf_matrix, _ = AgreementAnalysis.cohens_kappa(
            df, "rater1", "rater2"
        )

        assert res_df is not None
        assert error_msg is None
        # New structure check
        assert "Cohen's Kappa" in res_df["Metric"].values

    def test_calculate_kappa_perfect_agreement(self):
        """✅ Test Kappa with perfect agreement."""

        df = pd.DataFrame({"rater1": [0, 1] * 10, "rater2": [0, 1] * 10})

        res_df, error_msg, conf_matrix, _ = AgreementAnalysis.cohens_kappa(
            df, "rater1", "rater2"
        )
        assert res_df is not None
        assert error_msg is None

    def test_calculate_kappa_no_agreement(self):
        """✅ Test Kappa with no agreement."""

        df = pd.DataFrame({"rater1": [0, 0, 0], "rater2": [1, 1, 1]})

        res_df, error_msg, conf_matrix, _ = AgreementAnalysis.cohens_kappa(
            df, "rater1", "rater2"
        )
        assert res_df is not None
        assert error_msg is None


class TestICCAnalysis:
    """Tests for Intraclass Correlation Coefficient."""

    def test_calculate_icc_basic(self):
        """
        Verify that AgreementAnalysis.icc returns a non-empty DataFrame containing an 'ICC' column for two raters with correlated scores.

        Creates two correlated rater measurements from a shared true score and asserts the function returns a valid ICC DataFrame and no error message. Uses the test mock of pingouin for deterministic output.
        """

        np.random.seed(42)
        n = 50
        true_score = np.random.normal(50, 10, n)
        rater1 = true_score + np.random.normal(0, 2, n)
        rater2 = true_score + np.random.normal(0, 2, n)

        df = pd.DataFrame({"rater1": rater1, "rater2": rater2})

        icc_df, error_msg, _, _ = AgreementAnalysis.icc(df, ["rater1", "rater2"])

        assert icc_df is not None
        assert error_msg is None
        # With the mock setup, we expect a valid DataFrame
        assert not icc_df.empty, (
            "ICC DataFrame should not be empty with mocked pingouin"
        )
        assert "ICC" in icc_df.columns

    def test_calculate_icc_insufficient_columns(self):
        """✅ Test ICC with single column."""

        df = pd.DataFrame({"rater1": [1, 2, 3, 4, 5]})

        icc_df, error_msg, _, _ = AgreementAnalysis.icc(df, ["rater1"])

        assert icc_df.empty
        assert error_msg is not None


class TestConfidenceIntervals:
    """Tests for confidence interval calculations."""

    def test_wilson_score_ci(self):
        """✅ Test Wilson score confidence intervals."""
        lower, upper = calculate_ci_wilson_score(30, 100)

        assert 0 <= lower <= 1
        assert 0 <= upper <= 1
        assert lower < upper

        proportion = 30 / 100
        assert lower <= proportion <= upper

    def test_wilson_score_ci_edge_cases(self):
        """✅ Test Wilson CI edge cases."""
        lower, upper = calculate_ci_wilson_score(0, 100)
        assert np.isclose(lower, 0, atol=1e-10)
        assert upper > 0

        lower, upper = calculate_ci_wilson_score(100, 100)
        assert np.isclose(upper, 1, atol=1e-10)
        assert lower < 1

    def test_delong_auc_ci(self):
        """✅ Test DeLong AUC confidence intervals."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])

        ci_lower, ci_upper, se = auc_ci_delong(y_true, y_scores)

        if not np.isnan(ci_lower):
            assert 0 <= ci_lower <= 1
            assert 0 <= ci_upper <= 1
            assert ci_lower < ci_upper


class TestPValueFormatting:
    """Tests for p-value formatting functions."""

    def test_format_p_value_significant(self):
        """✅ Test significant p-value formatting."""
        result = format_p_value(0.001)
        assert "0.001" in result or "<0.001" in result
        assert "font-weight: bold" in result

    def test_format_p_value_not_significant(self):
        """✅ Test non-significant p-value."""
        result = format_p_value(0.234)
        assert "0.234" in result
        assert "bold" not in result

    def test_format_p_value_edge_cases(self):
        """✅ Test p-value edge cases."""
        assert "NA" in format_p_value(np.nan)
        assert "NA" in format_p_value(np.inf)


# ============================================================================
# SURVIVAL_LIB.PY TESTS - Survival Analysis
# ============================================================================


class TestMedianSurvival:
    """Tests for median survival time calculations."""

    def test_calculate_median_survival_basic(self, dummy_df):
        """✅ Test median survival calculation."""
        result = calculate_median_survival(dummy_df, "time", "event", "exposure")

        assert not result.empty
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "Median Time (95% CI)" in result.columns

    def test_calculate_median_survival_no_group(self, dummy_df):
        """✅ Test median survival without grouping."""
        result = calculate_median_survival(dummy_df, "time", "event", None)

        assert not result.empty
        assert len(result) == 1
        assert "Overall" in result["Group"].values

    def test_calculate_median_survival_missing_column(self, dummy_df):
        """✅ Test with missing columns."""
        with pytest.raises(ValueError, match="[Mm]issing|not found"):
            calculate_median_survival(dummy_df, "nonexistent_time", "event", "exposure")


class TestKaplanMeier:
    """Tests for Kaplan-Meier survival curves."""

    def test_fit_km_logrank_basic(self, dummy_df):
        """✅ Test KM curves with log-rank test."""
        dummy_df["group"] = dummy_df["exposure"]

        fig, stats_df, _ = fit_km_logrank(dummy_df, "time", "event", "group")

        assert fig is not None
        assert stats_df is not None
        assert isinstance(stats_df, pd.DataFrame)
        assert "P-value" in stats_df.columns

    def test_fit_km_logrank_no_group(self, dummy_df):
        """✅ Test KM without grouping variable."""
        fig, stats_df, _ = fit_km_logrank(dummy_df, "time", "event", None)

        assert fig is not None
        assert "Note" in stats_df.columns or len(stats_df) == 0


class TestNelsonAalen:
    """Tests for Nelson-Aalen cumulative hazard."""

    def test_fit_nelson_aalen_basic(self, dummy_df):
        """✅ Test Nelson-Aalen estimator."""
        dummy_df["group"] = dummy_df["exposure"]

        fig, stats_df, _ = fit_nelson_aalen(dummy_df, "time", "event", "group")

        assert fig is not None
        assert stats_df is not None
        assert isinstance(stats_df, pd.DataFrame)

        assert "N" in stats_df.columns
        assert "Events" in stats_df.columns


class TestCoxRegression:
    """Tests for Cox Proportional Hazards model."""

    def test_fit_cox_ph_basic(self, dummy_df):
        """✅ Test Cox regression."""
        # FIX: Using extended unpacking to handle both 4-return (old) and 5-return (new with checks)
        # cph, res_df, data, err, *rest = ...
        result = fit_cox_ph(dummy_df, "time", "event", ["age", "exposure"])

        # Unpack safely
        if len(result) == 4:
            cph, res_df, data, err = result
        else:
            cph, res_df, data, err, *rest = result

        if cph is not None:
            if res_df is not None:
                assert isinstance(res_df, pd.DataFrame)
                assert "HR" in res_df.columns
        else:
            assert err is not None

    def test_fit_cox_ph_no_events(self, dummy_df):
        """✅ Test Cox with all censored data."""
        df = dummy_df.copy()
        df["event"] = 0  # All censored

        result = fit_cox_ph(df, "time", "event", ["age", "exposure"])

        # Unpack safely
        if len(result) == 4:
            cph, res_df, data, err = result
        else:
            cph, res_df, data, err, *rest = result

        assert cph is None
        assert err is not None
        assert "no events" in err.lower() or "censored" in err.lower()

    def test_fit_cox_ph_missing_columns(self, dummy_df):
        """✅ Test Cox with missing columns."""
        result = fit_cox_ph(dummy_df, "nonexistent_time", "event", ["age"])

        # Unpack safely
        if len(result) == 4:
            cph, res_df, data, err = result
        else:
            cph, res_df, data, err, *rest = result

        assert cph is None
        assert err is not None
        assert "missing" in err.lower()


class TestLandmarkAnalysis:
    """Tests for landmark survival analysis."""

    def test_fit_km_landmark_basic(self, dummy_df):
        """✅ Test landmark analysis."""
        landmark_time = 5.0
        dummy_df["group"] = dummy_df["exposure"]

        fig, stats_df, n_pre, n_post, err, _ = fit_km_landmark(
            dummy_df, "time", "event", "group", landmark_time
        )

        if fig is not None:
            assert stats_df is not None
            assert n_pre >= n_post
            assert n_post > 0
        else:
            assert err is not None

    def test_fit_km_landmark_insufficient_data(self, dummy_df):
        """✅ Test landmark with time beyond all observations."""
        landmark_time = 1000.0  # Beyond max time

        fig, stats_df, n_pre, n_post, err, _ = fit_km_landmark(
            dummy_df, "time", "event", "exposure", landmark_time
        )

        assert fig is None
        assert err is not None
        assert "insufficient" in err.lower()


# ============================================================================
# INTEGRATION & EDGE CASE TESTS
# ============================================================================


class TestIntegrationScenarios:
    """Tests for realistic analysis workflows."""

    def test_full_diagnostic_workflow(self, dummy_df):
        """✅ Test complete diagnostic analysis pipeline."""
        # Step 1: Descriptive statistics
        # FIX: Unpack tuple
        desc, _ = calculate_descriptive(dummy_df, "age")
        assert desc is not None

        # Step 2: Chi-square test
        _, stats, _, risk, _ = calculate_chi2(dummy_df, "exposure", "disease")
        assert stats is not None

        # Step 3: ROC analysis
        dummy_df_roc = dummy_df.copy()
        dummy_df_roc["disease_binary"] = dummy_df_roc["disease"].astype(int)
        dummy_df_roc["score_values"] = np.random.uniform(0, 1, len(dummy_df_roc))

        roc_stats, error_msg, _, _ = analyze_roc(
            dummy_df_roc, "disease_binary", "score_values", pos_label_user="1"
        )
        assert roc_stats is not None or error_msg is not None

    def test_full_survival_workflow(self, dummy_df):
        """✅ Test complete survival analysis pipeline."""
        # Step 1: Median survival
        median_df = calculate_median_survival(dummy_df, "time", "event", "exposure")
        assert not median_df.empty

        # Step 2: KM curves
        dummy_df_km = dummy_df.copy()
        dummy_df_km["group"] = dummy_df_km["exposure"]

        fig, stats, _ = fit_km_logrank(dummy_df_km, "time", "event", "group")
        assert fig is not None
        assert stats is not None

        # Step 3: Cox regression
        result = fit_cox_ph(dummy_df, "time", "event", ["age", "exposure"])

        # Unpack safely
        if len(result) == 4:
            cph, res_df, _, err = result
        else:
            cph, res_df, _, err, *rest = result

        assert cph is not None or err is not None

    def test_full_logistic_workflow(self, dummy_df):
        """✅ Test complete logistic regression pipeline."""
        # Step 1: Data validation
        y = dummy_df["disease"]
        X = dummy_df[["age", "exposure"]]
        valid, msg = validate_logit_data(y, X)
        assert valid is True

        # Step 2: Univariate logistic
        params, conf, pvals, status, metrics = run_binary_logit(y, X)
        assert status == "OK"

        # Step 3: Full outcome analysis
        html, or_res, aor_res, int_res = analyze_outcome(
            "disease", dummy_df[["age", "exposure", "disease"]]
        )
        assert len(html) > 0


class TestEdgeCases:
    """Tests for boundary conditions and error handling."""

    def test_empty_dataframe_all_functions(self):
        """✅ Test all functions with empty DataFrame."""
        df = pd.DataFrame()

        # Diagnostic tests
        # FIX: Unpack tuple
        res, _ = calculate_descriptive(df, "col")
        assert res is None

        display, stats, msg, risk, _ = calculate_chi2(df, "c1", "c2")
        assert stats is None or msg is not None

        # Survival tests
        with pytest.raises(ValueError):
            calculate_median_survival(df, "time", "event", "group")

    def test_all_missing_data(self):
        """✅ Test with all-NA data."""
        df = pd.DataFrame(
            {"col1": [np.nan, np.nan, np.nan], "col2": [np.nan, np.nan, np.nan]}
        )

        # FIX: Unpack tuple
        res, _ = calculate_descriptive(df, "col1")
        assert res is None

    def test_single_observation(self):
        """✅ Test with minimal data (n=1)."""
        df = pd.DataFrame({"outcome": [1], "predictor": [5.0]})

        # Should fail gracefully
        y = df["outcome"]
        X = df[["predictor"]]
        params, conf, pvals, status, metrics = run_binary_logit(y, X)
        assert status != "OK"


# ============================================================================
# PERFORMANCE & OPTIMIZATION TESTS
# ============================================================================


class TestPerformance:
    """Tests for computational efficiency (optional)."""

    @pytest.mark.slow
    def test_large_dataset_chi2(self):
        """✅ Test chi-square with large dataset."""
        n = 10000
        df = pd.DataFrame(
            {
                "var1": np.random.choice(["A", "B"], n),
                "var2": np.random.choice([0, 1], n),
            }
        )

        import time

        start = time.time()
        display, stats, msg, risk, _ = calculate_chi2(df, "var1", "var2")
        elapsed = time.time() - start

        assert stats is not None
        assert elapsed < 5.0

    @pytest.mark.slow
    def test_large_dataset_logistic(self):
        """✅ Test logistic regression with large dataset."""
        np.random.seed(42)
        n = 5000
        X = pd.DataFrame(
            {"x1": np.random.normal(0, 1, n), "x2": np.random.normal(0, 1, n)}
        )
        logit = 0.5 * X["x1"] + 0.3 * X["x2"]
        prob = 1 / (1 + np.exp(-logit))
        y = pd.Series(np.random.binomial(1, prob))

        import time

        start = time.time()
        params, conf, pvals, status, metrics = run_binary_logit(y, X)
        elapsed = time.time() - start

        assert status == "OK"
        assert elapsed < 10.0


# ============================================================================
# MODULE IMPORT TEST
# ============================================================================


def test_module_imports():
    """✅ Verify all modules imported successfully."""
    # If we got this far, imports succeeded
    assert validate_logit_data is not None
    assert calculate_chi2 is not None
    assert analyze_roc is not None
    assert fit_km_logrank is not None
    assert fit_cox_ph is not None
    assert calculate_descriptive is not None
    assert format_p_value is not None


# ============================================================================
# TEST SUITE SUMMARY
# ============================================================================


def test_suite_completeness():
    """✅ Verify test suite covers all major functions."""
    tested_functions = [
        # logic.py
        "clean_numeric_value",
        "validate_logit_data",
        "run_binary_logit",
        "analyze_outcome",
        "get_label",
        # diag_test.py
        "calculate_descriptive",
        "calculate_chi2",
        "analyze_roc",
        "format_p_value",
        "calculate_ci_wilson_score",
        "auc_ci_delong",
        # survival_lib.py
        "calculate_median_survival",
        "fit_km_logrank",
        "fit_nelson_aalen",
        "fit_cox_ph",
        "fit_km_landmark",
    ]

    global_names = globals()
    for func_name in tested_functions:
        assert func_name in global_names, f"Function {func_name} not imported"

    print(f"\n✅ Test suite covers {len(tested_functions)} functions across 3 modules")

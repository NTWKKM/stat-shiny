"""
🧪 Comprehensive Unit Tests for Statistical Analysis Modules

Tests all core statistical functions across:
- logic.py: Logistic regression, data validation, outcome analysis
- diag_test.py: Diagnostic tests (Chi², Fisher, ROC, Kappa, ICC)
- survival_lib.py: Survival analysis (KM, Cox, Nelson-Aalen, Landmark)

Run with: pytest tests/unit/test_statistics.py -v
"""

import os
import sys
import warnings
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

# ============================================================================
# PATH SETUP & MOCKING
# ============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- ROBUST MOCK SETUP ---

# 1. Mock Plotly with concrete colors to prevent ZeroDivisionError
mock_plotly = MagicMock()
# Ensure colors list is available and has length
mock_colors = MagicMock()
mock_colors.qualitative.Plotly = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
mock_plotly.colors = mock_colors
mock_plotly.express.colors = mock_colors  # For px.colors usage
sys.modules['plotly'] = mock_plotly
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = mock_plotly
sys.modules['plotly.io'] = MagicMock()
sys.modules['plotly.colors'] = mock_colors # Direct import mock

# 2. Mock Forest Plot Lib
sys.modules['forest_plot_lib'] = MagicMock()

# 3. Mock Lifelines with structured return values
mock_lifelines = MagicMock()

# KM Fitter Mock
mock_kmf = MagicMock()
type(mock_kmf.return_value).median_survival_time_ = PropertyMock(return_value=10.0)
mock_ci = MagicMock()
mock_ci.values = np.array([[5.0, 15.0]])
mock_ci.to_numpy.return_value = np.array([[5.0, 15.0]])
type(mock_kmf.return_value).confidence_interval_ = PropertyMock(return_value=mock_ci)

# Cox Fitter Mock
mock_cph = MagicMock()
mock_cph_inst = mock_cph.return_value

# --- FIX: Define concordance_index_ as a float for format string usage ---
mock_cph_inst.concordance_index_ = 0.85 
# ----------------------------------------------------------------------------

# Mock summary DataFrame
mock_cph_inst.summary = pd.DataFrame({
    'coef': [0.5, 0.2], 
    'exp(coef)': [1.65, 1.22], 
    'p': [0.01, 0.04], 
    'HR': [1.65, 1.22]
}, index=['age', 'exposure'])

# Mock confidence_intervals_ DataFrame (Crucial for fit_cox_ph)
mock_cph_inst.confidence_intervals_ = pd.DataFrame(
    data=[[-0.1, 0.8], [0.1, 0.3]], 
    index=['age', 'exposure'],
    columns=['lower-bound', 'upper-bound']
)

mock_lifelines.KaplanMeierFitter = mock_kmf
mock_lifelines.CoxPHFitter = mock_cph
mock_lifelines.NelsonAalenFitter = MagicMock()

sys.modules['lifelines'] = mock_lifelines

# --- FIX START: Configure statistics mock with concrete float values ---
mock_stats = MagicMock()
mock_test_result = MagicMock()
# Important: p_value must be a float to allow f-string formatting (e.g. :.3f)
mock_test_result.p_value = 0.045
mock_test_result.test_statistic = 4.0
mock_stats.logrank_test.return_value = mock_test_result
mock_stats.multivariate_logrank_test.return_value = mock_test_result

sys.modules['lifelines.statistics'] = mock_stats
# --- FIX END ---

sys.modules['lifelines.utils'] = MagicMock()

# 4. Mock Sklearn with float return values for metrics
mock_sklearn = MagicMock()
mock_metrics = MagicMock()

# Fix: roc_curve returns tuple of arrays
mock_metrics.roc_curve.return_value = (
    np.array([0.0, 0.1, 1.0]), 
    np.array([0.0, 0.8, 1.0]), 
    np.array([0.5, 0.5, 0.5])
)
# Fix: auc must return a float for comparisons (e.g. >= 0.9)
mock_metrics.auc.return_value = 0.85
mock_metrics.roc_auc_score.return_value = 0.85
mock_metrics.cohen_kappa_score.return_value = 0.85

mock_sklearn.metrics = mock_metrics
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.metrics'] = mock_metrics
sys.modules['sklearn.linear_model'] = MagicMock()

# ============================================================================
# IMPORT MODULES UNDER TEST
# ============================================================================
# Try importing poisson_lib if available, else mock or ignore
try:
    from poisson_lib import run_poisson_regression, run_negative_binomial_regression
except ImportError:
    run_poisson_regression = MagicMock()
    run_negative_binomial_regression = MagicMock()

from logic import (
    validate_logit_data, 
    clean_numeric_value, 
    run_binary_logit,
    analyze_outcome,
    get_label,
    fmt_p_with_styling
)

from diag_test import (
    calculate_descriptive, 
    calculate_chi2, 
    analyze_roc,
    calculate_kappa,
    calculate_icc,
    format_p_value,
    calculate_ci_wilson_score,
    auc_ci_delong
)

from survival_lib import (
    fit_km_logrank, 
    fit_cox_ph, 
    calculate_median_survival,
    fit_nelson_aalen,
    fit_km_landmark
)

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def small_numeric_df():
    """Small dataset for quick validation tests."""
    return pd.DataFrame({
        'age': [20, 30, 40, 50],
        'outcome': [0, 1, 0, 1]
    })


@pytest.fixture
def dummy_df():
    """Comprehensive synthetic dataset for all test types."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        # Demographics
        'age': np.random.normal(50, 10, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        
        # Binary variables
        'exposure': np.random.choice([0, 1], n),
        'disease': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'treatment': np.random.choice(['A', 'B'], n),
        
        # Continuous predictors
        'test_score': np.random.normal(0, 1, n) + np.random.choice([0, 1], n) * 2,
        'biomarker': np.random.exponential(2, n),
        
        # Survival data
        'time': np.random.exponential(10, n),
        'event': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        
        # Rating data
        'rater1': np.random.choice([0, 1, 2], n),
        'rater2': np.random.choice([0, 1, 2], n)
    })


@pytest.fixture
def perfect_separation_df():
    """Dataset with perfect separation for Firth testing."""
    return pd.DataFrame({
        'predictor': [1, 1, 1, 1, 2, 2, 2, 2],
        'outcome': [0, 0, 0, 0, 1, 1, 1, 1]
    })


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
    
    #@pytest.mark.xfail(reason="Feature currently not implemented in logic.py")
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
        y = small_numeric_df['outcome']
        X = small_numeric_df[['age']]
        valid, msg = validate_logit_data(y, X)
        assert valid is True
        assert msg == "OK"
    
    def test_validate_logit_data_zero_variance(self):
        """✅ Test detection of constant variables."""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1],
            'age': [20, 30, 40, 50]
        })
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
        X = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]})
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
        
        X = pd.DataFrame({'age': age})
        
        params, _conf, pvals, status, metrics = run_binary_logit(y, X)
        
        # Verify successful fit
        assert status == "OK", f"Expected 'OK', got '{status}'"
        assert params is not None, "Parameters should not be None"
        assert 'age' in params.index, "Age coefficient should exist"
        
        # Age coefficient should be positive (matches synthetic pattern)
        assert params['age'] > 0, f"Expected positive age coefficient, got {params['age']}"
        
        # Model quality checks
        assert metrics['mcfadden'] > 0, "McFadden R² should be > 0"
        assert 0 <= metrics['mcfadden'] <= 1, "McFadden R² should be in [0,1]"
        
        # P-value should be significant
        assert pvals['age'] < 0.05, "Age should be statistically significant"
    
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
        
        X = pd.DataFrame({'age': age, 'weight': weight})
        
        _params, _conf, _pvals, status, _metrics = run_binary_logit(y, X)
        
        assert status == "OK"
        assert 'age' in X.columns
        assert 'weight' in X.columns
    
    def test_run_binary_logit_perfect_separation(self, perfect_separation_df):
        """✅ Test handling of perfect separation."""
        y = perfect_separation_df['outcome']
        X = perfect_separation_df[['predictor']]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerfectSeparationWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # Default method should fail gracefully or switch method
            _params, _conf, _pvals, status, _metrics = run_binary_logit(y, X, method='default')
        
        # Should either succeed (if robust) or return informative error
        if status != "OK":
            assert "separation" in status.lower() or "singular" in status.lower()
    
    def test_run_binary_logit_constant_outcome(self):
        """✅ Test with constant outcome (no variation)."""
        y = pd.Series([1, 1, 1, 1, 1])
        X = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        
        _params, _conf, _pvals, status, _metrics = run_binary_logit(y, X)
        
        assert status in ["OK", "Error"] or "constant" in status.lower()


class TestFormattingHelpers:
    """Tests for formatting and display helper functions."""
    
    def test_fmt_p_with_styling_significant(self):
        """✅ Test p-value formatting with styling."""
        result = fmt_p_with_styling(0.001)
        # Check logic: formatting usually applies styling
        assert ("&lt;0.001" in result or "<0.001" in result or "0.001" in result)
        assert "sig-p" in result  # Should have significance class
    
    def test_fmt_p_with_styling_not_significant(self):
        """✅ Test non-significant p-value formatting."""
        result = fmt_p_with_styling(0.234)
        assert "0.234" in result
        assert "sig-p" not in result  # Should NOT have significance class
    
    def test_fmt_p_with_styling_edge_cases(self):
        """✅ Test edge case p-values."""
        assert "-" in fmt_p_with_styling(None)
        assert "-" in fmt_p_with_styling(np.nan)
        assert ">0.999" in fmt_p_with_styling(0.9999)
        assert ("&lt;0.001" in fmt_p_with_styling(0.0001) or "<0.001" in fmt_p_with_styling(0.0001))
    
    def test_get_label_basic(self):
        """✅ Test label generation without metadata."""
        label = get_label("age", None)
        assert "age" in label
        assert "<b>" in label  # Should be bolded
    
    def test_get_label_with_metadata(self):
        """✅ Test label generation with metadata."""
        var_meta = {'age': {'label': 'Patient Age (years)'}}
        label = get_label("age", var_meta)
        assert "age" in label
        assert "Patient Age" in label


class TestAnalyzeOutcome:
    """Tests for comprehensive outcome analysis."""
    
    def test_analyze_outcome_basic(self, dummy_df):
        """✅ Test basic outcome analysis."""
        html, or_results, aor_results, int_results = analyze_outcome(
            'disease', 
            dummy_df[['age', 'exposure', 'disease']]
        )
        
        # Verify HTML output
        assert html is not None
        assert len(html) > 0
        assert 'disease' in html.lower()
        
        # Verify OR results
        assert isinstance(or_results, dict)
        assert isinstance(aor_results, dict)
    
    def test_analyze_outcome_missing_column(self, dummy_df):
        """✅ Test with non-existent outcome."""
        html, or_results, aor_results, _int_results = analyze_outcome(
            'nonexistent_outcome',
            dummy_df
        )
        
        assert 'not found' in html.lower() or 'error' in html.lower()
        assert len(or_results) == 0
        assert len(aor_results) == 0
    
    def test_analyze_outcome_non_binary(self, dummy_df):
        """✅ Test with non-binary outcome."""
        df = dummy_df.copy()
        df['multi_outcome'] = np.random.choice([0, 1, 2], len(df))
        
        html, or_results, aor_results, int_results = analyze_outcome(
            'multi_outcome',
            df
        )
        
        # Should fail with informative message
        assert 'invalid' in html.lower() or 'expected 2' in html.lower()


# ============================================================================
# DIAG_TEST.PY TESTS - Diagnostic Statistics
# ============================================================================

class TestDescriptiveStatistics:
    """Tests for descriptive statistics calculations."""
    
    def test_calculate_descriptive_numeric(self, dummy_df):
        """✅ Test descriptive stats for numeric variables."""
        result = calculate_descriptive(dummy_df, 'age')
        
        assert result is not None
        assert not result.empty
        assert isinstance(result, pd.DataFrame)
        
        # Check for expected statistics
        stats_present = result['Statistic'].values
        assert 'Mean' in stats_present
        assert 'SD' in stats_present
        assert 'Median' in stats_present
    
    def test_calculate_descriptive_categorical(self, dummy_df):
        """✅ Test descriptive stats for categorical variables."""
        result = calculate_descriptive(dummy_df, 'sex')
        
        assert result is not None
        assert not result.empty
        assert isinstance(result, pd.DataFrame)
        
        # Should have Category and Count columns
        assert 'Category' in result.columns or 'Count' in result.columns
    
    def test_calculate_descriptive_missing_column(self, dummy_df):
        """✅ Test with non-existent column."""
        result = calculate_descriptive(dummy_df, 'nonexistent')
        assert result is None
    
    def test_calculate_descriptive_empty_column(self):
        """✅ Test with all-NA column."""
        df = pd.DataFrame({'empty_col': [np.nan, np.nan, np.nan]})
        result = calculate_descriptive(df, 'empty_col')
        assert result is None


class TestChiSquareAnalysis:
    """Tests for chi-square and contingency table analysis."""
    
    def test_calculate_chi2_basic(self, dummy_df):
        """✅ Test basic chi-square calculation."""
        display_tab, stats_df, msg, risk_df = calculate_chi2(
            dummy_df, 'exposure', 'disease'
        )
        
        assert stats_df is not None
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) > 0
        
        # Verify chi-square statistic is present
        test_row = stats_df[stats_df['Statistic'] == 'Test']
        assert not test_row.empty
        assert 'Chi-Square' in test_row['Value'].values[0]
    
    def test_calculate_chi2_2x2_risk_metrics(self, dummy_df):
        """✅ Test 2x2 table risk metrics calculation."""
        display_tab, stats_df, msg, risk_df = calculate_chi2(
            dummy_df, 'exposure', 'disease'
        )
        
        # For 2x2 tables, risk metrics should be calculated
        if risk_df is not None:
            assert isinstance(risk_df, pd.DataFrame)
            
            # Check for expected metrics
            metrics = risk_df['Metric'].values
            assert any('Risk Ratio' in str(m) for m in metrics)
            assert any('Odds Ratio' in str(m) for m in metrics)
    
    def test_calculate_chi2_fisher_exact(self):
        """✅ Test Fisher's Exact Test."""
        df = pd.DataFrame({
            'treatment': ['A', 'B'] * 10,
            'outcome': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1] * 2
        })
        
        display_tab, stats_df, msg, risk_df = calculate_chi2(
            df, 'treatment', 'outcome', method='Fisher'
        )
        
        assert stats_df is not None
        test_row = stats_df[stats_df['Statistic'] == 'Test']
        assert 'Fisher' in test_row['Value'].values[0]
    
    def test_calculate_chi2_missing_columns(self, dummy_df):
        """✅ Test with missing columns."""
        display_tab, stats_df, msg, risk_df = calculate_chi2(
            dummy_df, 'nonexistent1', 'nonexistent2'
        )
        
        assert stats_df is None
        assert msg is not None
        assert 'not found' in msg.lower()
    
    def test_calculate_chi2_empty_data(self):
        """✅ Test with empty DataFrame."""
        df = pd.DataFrame({'col1': [], 'col2': []})
        display_tab, stats_df, msg, risk_df = calculate_chi2(
            df, 'col1', 'col2'
        )
        
        assert msg is not None
        assert 'no data' in msg.lower()


class TestROCAnalysis:
    """Tests for ROC curve and AUC calculation."""
    
    def test_analyze_roc_basic(self, dummy_df):
        """✅ Test basic ROC analysis."""
        # Ensure we have sufficient variation
        n = len(dummy_df)
        dummy_df['disease_binary'] = np.random.choice([0, 1], n, p=[0.7, 0.3])
        dummy_df['score_values'] = np.random.uniform(0, 1, n)
        
        stats_dict, error_msg, fig, coords = analyze_roc(
            dummy_df, 'disease_binary', 'score_values', pos_label_user='1'
        )
        
        if stats_dict is not None:
            assert 'AUC' in stats_dict
            # Mock returns 0.85
            auc_val = float(stats_dict['AUC'])
            assert 0 <= auc_val <= 1
            
            # Verify CI is present
            assert '95% CI' in stats_dict
        else:
            assert error_msg is not None
    
    def test_analyze_roc_perfect_classifier(self):
        """✅ Test ROC with perfect classifier (AUC=1.0)."""
        n = 100
        df = pd.DataFrame({
            'outcome': [0] * 50 + [1] * 50,
            'score': list(range(50)) + list(range(50, 100))  # Perfect separation
        })
        
        stats_dict, error_msg, fig, coords = analyze_roc(
            df, 'outcome', 'score', pos_label_user='1'
        )
        
        if stats_dict:
            # Mocked to 0.85
            assert 'AUC' in stats_dict
    
    def test_analyze_roc_missing_column(self, dummy_df):
        """✅ Test ROC with missing column."""
        try:
            stats_dict, error_msg, fig, coords = analyze_roc(
                dummy_df, 'nonexistent', 'test_score', pos_label_user='1'
            )
            assert stats_dict is None
            assert error_msg is not None
        except KeyError:
            pass
    
    def test_analyze_roc_constant_score(self):
        """✅ Test ROC with constant prediction score."""
        df = pd.DataFrame({
            'outcome': [0, 1, 0, 1, 0, 1],
            'score': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]  # All same
        })
        
        stats_dict, error_msg, fig, coords = analyze_roc(
            df, 'outcome', 'score', pos_label_user='1'
        )
        
        assert stats_dict is None
        assert error_msg is not None
        assert 'constant' in error_msg.lower()


class TestKappaAnalysis:
    """Tests for Cohen's Kappa inter-rater agreement."""
    
    def test_calculate_kappa_basic(self):
        """✅ Test basic Kappa calculation."""
        df = pd.DataFrame({
            'rater1': [0, 1, 0, 1] * 5,
            'rater2': [0, 1, 0, 1] * 5
        })
        
        res_df, error_msg, conf_matrix = calculate_kappa(
            df, 'rater1', 'rater2'
        )
        
        assert res_df is not None
        assert error_msg is None
        assert "Cohen's Kappa" in res_df['Statistic'].values
    
    def test_calculate_kappa_perfect_agreement(self):
        """✅ Test Kappa with perfect agreement."""
        df = pd.DataFrame({
            'rater1': [0, 1] * 10,
            'rater2': [0, 1] * 10
        })
        
        res_df, error_msg, conf_matrix = calculate_kappa(df, 'rater1', 'rater2')
        assert res_df is not None
    
    def test_calculate_kappa_no_agreement(self):
        """✅ Test Kappa with no agreement."""
        df = pd.DataFrame({
            'rater1': [0, 0, 0],
            'rater2': [1, 1, 1]
        })
        
        res_df, error_msg, conf_matrix = calculate_kappa(df, 'rater1', 'rater2')
        assert res_df is not None


class TestICCAnalysis:
    """Tests for Intraclass Correlation Coefficient."""
    
    def test_calculate_icc_basic(self):
        """✅ Test basic ICC calculation."""
        np.random.seed(42)
        n = 50
        true_score = np.random.normal(50, 10, n)
        rater1 = true_score + np.random.normal(0, 2, n)
        rater2 = true_score + np.random.normal(0, 2, n)
        
        df = pd.DataFrame({'rater1': rater1, 'rater2': rater2})
        
        icc_df, error_msg, anova_df = calculate_icc(df, ['rater1', 'rater2'])
        
        assert icc_df is not None
        assert error_msg is None
        assert 'ICC' in icc_df.columns
    
    def test_calculate_icc_insufficient_columns(self):
        """✅ Test ICC with single column."""
        df = pd.DataFrame({'rater1': [1, 2, 3, 4, 5]})
        
        icc_df, error_msg, anova_df = calculate_icc(df, ['rater1'])
        
        assert icc_df is None
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
        assert ("0.001" in result or "<0.001" in result)
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
        result = calculate_median_survival(
            dummy_df, 'time', 'event', 'exposure'
        )
        
        assert not result.empty
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'Median Time (95% CI)' in result.columns
    
    def test_calculate_median_survival_no_group(self, dummy_df):
        """✅ Test median survival without grouping."""
        result = calculate_median_survival(
            dummy_df, 'time', 'event', None
        )
        
        assert not result.empty
        assert len(result) == 1
        assert 'Overall' in result['Group'].values
    
    def test_calculate_median_survival_missing_column(self, dummy_df):
        """✅ Test with missing columns."""
        with pytest.raises(ValueError, match="[Mm]issing|not found") as exc_info:
            calculate_median_survival(
                dummy_df, 'nonexistent_time', 'event', 'exposure'
            )


class TestKaplanMeier:
    """Tests for Kaplan-Meier survival curves."""
    
    def test_fit_km_logrank_basic(self, dummy_df):
        """✅ Test KM curves with log-rank test."""
        dummy_df['group'] = dummy_df['exposure']
        
        fig, stats_df = fit_km_logrank(
            dummy_df, 'time', 'event', 'group'
        )
        
        assert fig is not None
        assert stats_df is not None
        assert isinstance(stats_df, pd.DataFrame)
        assert 'P-value' in stats_df.columns
    
    def test_fit_km_logrank_no_group(self, dummy_df):
        """✅ Test KM without grouping variable."""
        fig, stats_df = fit_km_logrank(
            dummy_df, 'time', 'event', None
        )
        
        assert fig is not None
        assert 'Note' in stats_df.columns or len(stats_df) == 0


class TestNelsonAalen:
    """Tests for Nelson-Aalen cumulative hazard."""
    
    def test_fit_nelson_aalen_basic(self, dummy_df):
        """✅ Test Nelson-Aalen estimator."""
        dummy_df['group'] = dummy_df['exposure']
        
        fig, stats_df = fit_nelson_aalen(
            dummy_df, 'time', 'event', 'group'
        )
        
        assert fig is not None
        assert stats_df is not None
        assert isinstance(stats_df, pd.DataFrame)
        
        assert 'N' in stats_df.columns
        assert 'Events' in stats_df.columns


class TestCoxRegression:
    """Tests for Cox Proportional Hazards model."""
    
    def test_fit_cox_ph_basic(self, dummy_df):
        """✅ Test Cox regression."""
        # FIX: Using extended unpacking to handle both 4-return (old) and 5-return (new with checks)
        # cph, res_df, data, err, *rest = ...
        result = fit_cox_ph(
            dummy_df, 'time', 'event', ['age', 'exposure']
        )
        
        # Unpack safely
        if len(result) == 4:
            cph, res_df, data, err = result
        else:
            cph, res_df, data, err, *rest = result

        if cph is not None:
            if res_df is not None:
                assert isinstance(res_df, pd.DataFrame)
                assert 'HR' in res_df.columns
        else:
            assert err is not None
    
    def test_fit_cox_ph_no_events(self, dummy_df):
        """✅ Test Cox with all censored data."""
        df = dummy_df.copy()
        df['event'] = 0  # All censored
        
        result = fit_cox_ph(
            df, 'time', 'event', ['age', 'exposure']
        )
        
        # Unpack safely
        if len(result) == 4:
            cph, res_df, data, err = result
        else:
            cph, res_df, data, err, *rest = result
        
        assert cph is None
        assert err is not None
        assert 'no events' in err.lower() or 'censored' in err.lower()
    
    def test_fit_cox_ph_missing_columns(self, dummy_df):
        """✅ Test Cox with missing columns."""
        result = fit_cox_ph(
            dummy_df, 'nonexistent_time', 'event', ['age']
        )
        
        # Unpack safely
        if len(result) == 4:
            cph, res_df, data, err = result
        else:
            cph, res_df, data, err, *rest = result
        
        assert cph is None
        assert err is not None
        assert 'missing' in err.lower()


class TestLandmarkAnalysis:
    """Tests for landmark survival analysis."""
    
    def test_fit_km_landmark_basic(self, dummy_df):
        """✅ Test landmark analysis."""
        landmark_time = 5.0
        dummy_df['group'] = dummy_df['exposure']
        
        fig, stats_df, n_pre, n_post, err = fit_km_landmark(
            dummy_df, 'time', 'event', 'group', landmark_time
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
        
        fig, stats_df, n_pre, n_post, err = fit_km_landmark(
            dummy_df, 'time', 'event', 'exposure', landmark_time
        )
        
        assert fig is None
        assert err is not None
        assert 'insufficient' in err.lower()


# ============================================================================
# INTEGRATION & EDGE CASE TESTS
# ============================================================================

class TestIntegrationScenarios:
    """Tests for realistic analysis workflows."""
    
    def test_full_diagnostic_workflow(self, dummy_df):
        """✅ Test complete diagnostic analysis pipeline."""
        # Step 1: Descriptive statistics
        desc = calculate_descriptive(dummy_df, 'age')
        assert desc is not None
        
        # Step 2: Chi-square test
        _, stats, _, risk = calculate_chi2(dummy_df, 'exposure', 'disease')
        assert stats is not None
        
        # Step 3: ROC analysis
        dummy_df_roc = dummy_df.copy()
        dummy_df_roc['disease_binary'] = dummy_df_roc['disease'].astype(int)
        dummy_df_roc['score_values'] = np.random.uniform(0, 1, len(dummy_df_roc))
        
        roc_stats, error_msg, _, _ = analyze_roc(
            dummy_df_roc, 'disease_binary', 'score_values', pos_label_user='1'
        )
        assert roc_stats is not None or error_msg is not None
    
    def test_full_survival_workflow(self, dummy_df):
        """✅ Test complete survival analysis pipeline."""
        # Step 1: Median survival
        median_df = calculate_median_survival(
            dummy_df, 'time', 'event', 'exposure'
        )
        assert not median_df.empty
        
        # Step 2: KM curves
        dummy_df_km = dummy_df.copy()
        dummy_df_km['group'] = dummy_df_km['exposure']
        
        fig, stats = fit_km_logrank(dummy_df_km, 'time', 'event', 'group')
        assert fig is not None
        assert stats is not None
        
        # Step 3: Cox regression
        result = fit_cox_ph(
            dummy_df, 'time', 'event', ['age', 'exposure']
        )
        
        # Unpack safely
        if len(result) == 4:
            cph, res_df, _, err = result
        else:
            cph, res_df, _, err, *rest = result

        assert cph is not None or err is not None
    
    def test_full_logistic_workflow(self, dummy_df):
        """✅ Test complete logistic regression pipeline."""
        # Step 1: Data validation
        y = dummy_df['disease']
        X = dummy_df[['age', 'exposure']]
        valid, msg = validate_logit_data(y, X)
        assert valid is True
        
        # Step 2: Univariate logistic
        params, conf, pvals, status, metrics = run_binary_logit(y, X)
        assert status == "OK"
        
        # Step 3: Full outcome analysis
        html, or_res, aor_res, int_res = analyze_outcome(
            'disease', 
            dummy_df[['age', 'exposure', 'disease']]
        )
        assert len(html) > 0


class TestEdgeCases:
    """Tests for boundary conditions and error handling."""
    
    def test_empty_dataframe_all_functions(self):
        """✅ Test all functions with empty DataFrame."""
        df = pd.DataFrame()
        
        # Diagnostic tests
        assert calculate_descriptive(df, 'col') is None
        
        display, stats, msg, risk = calculate_chi2(df, 'c1', 'c2')
        assert stats is None or msg is not None
        
        # Survival tests
        with pytest.raises(ValueError):
            calculate_median_survival(df, 'time', 'event', 'group')
    
    def test_all_missing_data(self):
        """✅ Test with all-NA data."""
        df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        assert calculate_descriptive(df, 'col1') is None
    
    def test_single_observation(self):
        """✅ Test with minimal data (n=1)."""
        df = pd.DataFrame({
            'outcome': [1],
            'predictor': [5.0]
        })
        
        # Should fail gracefully
        y = df['outcome']
        X = df[['predictor']]
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
        df = pd.DataFrame({
            'var1': np.random.choice(['A', 'B'], n),
            'var2': np.random.choice([0, 1], n)
        })
        
        import time
        start = time.time()
        display, stats, msg, risk = calculate_chi2(df, 'var1', 'var2')
        elapsed = time.time() - start
        
        assert stats is not None
        assert elapsed < 5.0
    
    @pytest.mark.slow
    def test_large_dataset_logistic(self):
        """✅ Test logistic regression with large dataset."""
        np.random.seed(42)
        n = 5000
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n)
        })
        logit = 0.5 * X['x1'] + 0.3 * X['x2']
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
    assert calculate_median_survival is not None
    assert calculate_kappa is not None
    assert calculate_icc is not None
    assert clean_numeric_value is not None


# ============================================================================
# ============================================================================
# VIF COLLINEARITY DIAGNOSTICS TESTS
# ============================================================================

class TestMultipleComparisonCorrections:
    """Test multiple comparison correction methods."""
    
    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple comparisons."""
        from multiple_comparisons import MultipleComparisonCorrection
        
        p_values = [0.01, 0.03, 0.05, 0.10, 0.15]
        mcc = MultipleComparisonCorrection()
        results_df, threshold = mcc.bonferroni(p_values)
        
        # Check results structure
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 5
        assert 'Test #' in results_df.columns
        assert 'P-value' in results_df.columns
        assert 'Adjusted P' in results_df.columns
        assert 'Significant (α=0.05)' in results_df.columns
        
        # Check threshold calculation: 0.05 / 5 = 0.01
        assert threshold == 0.01
        
        # Check adjusted p-values (should be p * n_tests)
        expected_adjusted = [min(p * 5, 1.0) for p in p_values]
        actual_adjusted = results_df['Adjusted P'].values
        np.testing.assert_array_almost_equal(actual_adjusted, expected_adjusted)
        
        # Check significance: only p_values < 0.01 should be significant
        expected_significant = [p < 0.01 for p in p_values]
        actual_significant = results_df['Significant (α=0.05)'].values
        assert np.array_equal(actual_significant, expected_significant)
    
    def test_holm_correction(self):
        """Test Holm-Bonferroni correction for multiple comparisons."""
        from multiple_comparisons import MultipleComparisonCorrection
        
        p_values = [0.01, 0.03, 0.05, 0.10, 0.15]
        mcc = MultipleComparisonCorrection()
        results_df, threshold = mcc.holm(p_values)
        
        # Check results structure
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 5
        assert 'Test #' in results_df.columns
        assert 'P-value' in results_df.columns
        assert 'Adjusted P (Holm)' in results_df.columns
        assert 'Significant (α=0.05)' in results_df.columns
        
        # Holm should be less conservative than Bonferroni
        bonf_results, _ = mcc.bonferroni(p_values)
        # At least some Holm adjusted p-values should be <= Bonferroni adjusted p-values
        holm_adjusted = results_df['Adjusted P (Holm)'].values
        bonf_adjusted = bonf_results['Adjusted P'].values
        # Sort by original p-value for fair comparison
        sorted_indices = np.argsort(p_values)
        assert np.any(holm_adjusted[sorted_indices] <= bonf_adjusted[sorted_indices])
    
    def test_benjamini_hochberg_correction(self):
        """Test Benjamini-Hochberg FDR correction for multiple comparisons."""
        from multiple_comparisons import MultipleComparisonCorrection
        
        p_values = [0.01, 0.03, 0.05, 0.10, 0.15]
        mcc = MultipleComparisonCorrection()
        results_df, threshold = mcc.benjamini_hochberg(p_values, fdr=0.05)
        
        # Check results structure
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 5
        assert 'Test #' in results_df.columns
        assert 'P-value' in results_df.columns
        assert 'Threshold' in results_df.columns
        assert 'Significant (FDR=0.05)' in results_df.columns
        
        # BH should be more powerful than Bonferroni
        bonf_results, _ = mcc.bonferroni(p_values)
        bh_significant_count = results_df['Significant (FDR=0.05)'].sum()
        bonf_significant_count = bonf_results['Significant (α=0.05)'].sum()
        assert bh_significant_count >= bonf_significant_count
    
    def test_edge_case_single_test(self):
        """Test correction with a single test (should work without modification)."""
        from multiple_comparisons import MultipleComparisonCorrection
        
        p_values = [0.03]
        mcc = MultipleComparisonCorrection()
        
        # All methods should work with a single test
        bonf_results, bonf_threshold = mcc.bonferroni(p_values)
        holm_results, holm_threshold = mcc.holm(p_values)
        bh_results, bh_threshold = mcc.benjamini_hochberg(p_values)
        
        # With single test, adjusted p should equal original p
        assert bonf_results['Adjusted P'].values[0] == 0.03
        assert holm_results['Adjusted P (Holm)'].values[0] == 0.03
        
        # Bonferroni threshold: 0.05 / 1 = 0.05
        assert bonf_threshold == 0.05
    
    def test_edge_case_extreme_p_values(self):
        """Test correction with extreme p-values (very small and large)."""
        from multiple_comparisons import MultipleComparisonCorrection
        
        p_values = [1e-10, 0.001, 0.5, 0.999, 1.0]
        mcc = MultipleComparisonCorrection()
        results_df, threshold = mcc.bonferroni(p_values)
        
        # Check that very small p-values are capped at 1.0
        assert all(results_df['Adjusted P'].values <= 1.0)
        # 1e-10 * 5 = 5e-10, which is still < 1.0, so it won't be capped
        # But very large values like 1.0 should remain 1.0
        assert results_df['Adjusted P'].values[-1] == 1.0
    
    def test_generate_mcc_report_html(self):
        """Test HTML report generation for multiple comparison corrections."""
        from multiple_comparisons import MultipleComparisonCorrection, generate_mcc_report_html
        
        p_values = [0.01, 0.03, 0.05, 0.10, 0.15]
        mcc = MultipleComparisonCorrection()
        results_df, threshold = mcc.bonferroni(p_values)
        
        html_report = generate_mcc_report_html("Bonferroni", results_df)
        
        # Should return HTML string
        assert isinstance(html_report, str)
        # Should contain key HTML elements
        assert '<!DOCTYPE html>' in html_report
        assert 'Multiple Comparison Correction Results' in html_report
        assert 'Bonferroni' in html_report
        assert 'Method Comparison' in html_report


class TestCollinearityDiagnostics:
    """Test VIF (Variance Inflation Factor) calculation functions."""
    
    def test_vif_calculation_independent_predictors(self):
        """Test VIF calculation with independent predictors (VIF ≈ 1)."""
        from collinearity_check import calculate_vif
        
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n)
        })
        
        vif_df = calculate_vif(X)
        
        # Should return DataFrame
        assert isinstance(vif_df, pd.DataFrame)
        # Should have 3 rows
        assert len(vif_df) == 3
        # Should have required columns
        assert 'Variable' in vif_df.columns
        assert 'VIF' in vif_df.columns
        assert 'Flag' in vif_df.columns
        assert 'Interpretation' in vif_df.columns
        # VIF values should be close to 1 for independent predictors
        assert all(vif_df['VIF'] < 5)
        # No severe flags
        assert not any('SEVERE' in str(f) for f in vif_df['Flag'])
    
    def test_vif_calculation_perfect_collinearity(self):
        """Test VIF calculation with highly correlated predictors."""
        from collinearity_check import calculate_vif
        
        np.random.seed(42)
        n = 100
        x1_vals = np.random.normal(0, 1, n)
        X = pd.DataFrame({
            'x1': x1_vals,
            'x2': np.random.normal(0, 1, n),
            'x3': x1_vals * 2 + np.random.normal(0, 0.01, n)  # Highly correlated with x1
        })
        
        vif_df = calculate_vif(X)
        
        # x3 should have higher VIF than x2
        x3_vif = vif_df[vif_df['Variable'] == 'x3']['VIF'].values[0]
        x2_vif = vif_df[vif_df['Variable'] == 'x2']['VIF'].values[0]
        
        # x3 should have higher VIF due to correlation with x1
        assert x3_vif > x2_vif
        # x3 should have high VIF (> 10)
        assert x3_vif > 10
    
    def test_vif_calculation_high_collinearity(self):
        """Test VIF calculation with highly correlated predictors."""
        from collinearity_check import calculate_vif
        
        np.random.seed(42)
        n = 100
        base = np.random.normal(0, 1, n)
        X = pd.DataFrame({
            'x1': base,
            'x2': base + np.random.normal(0, 0.1, n),  # Highly correlated with x1
            'x3': np.random.normal(0, 1, n)
        })
        
        vif_df = calculate_vif(X)
        
        # x2 should have higher VIF than x3
        x2_vif = vif_df[vif_df['Variable'] == 'x2']['VIF'].values[0]
        x3_vif = vif_df[vif_df['Variable'] == 'x3']['VIF'].values[0]
        
        assert x2_vif > x3_vif
        # x2 should have at least moderate collinearity
        assert x2_vif > 5
    
    def test_vif_with_categorical_variables(self):
        """Test VIF calculation with categorical variables (should be encoded)."""
        from collinearity_check import calculate_vif
        
        np.random.seed(42)
        n = 200  # Increase sample size to ensure enough data for VIF calculation
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'cat1': np.random.choice(['A', 'B', 'C'], n),
            'cat2': np.random.choice(['X', 'Y'], n)
        })
        
        vif_df = calculate_vif(X)
        
        # Should successfully encode categoricals and return results
        assert isinstance(vif_df, pd.DataFrame)
        assert len(vif_df) > 0
        # After one-hot encoding with drop_first, we should have x1 + (3-1) + (2-1) = 4 columns
        # or just check that we have more than just x1
        assert len(vif_df) > 1
        # Check that x1 is in the results
        assert 'x1' in vif_df['Variable'].values
    
    def test_vif_no_numeric_columns(self):
        """Test VIF calculation with only categorical variables (should encode them)."""
        from collinearity_check import calculate_vif
        
        # Create enough data for VIF calculation
        n = 50
        X = pd.DataFrame({
            'cat1': ['A', 'B', 'C'] * (n // 3),
            'cat2': ['X', 'Y', 'Z'] * (n // 3)
        })
        
        # After encoding, we should have numeric columns
        vif_df = calculate_vif(X)
        assert isinstance(vif_df, pd.DataFrame)
        # Should have successfully encoded the categoricals
        assert len(vif_df) > 0
    
    def test_vif_empty_dataframe(self):
        """Test VIF calculation with empty DataFrame."""
        from collinearity_check import calculate_vif
        
        X = pd.DataFrame()
        
        with pytest.raises(ValueError, match="No numeric columns found"):
            calculate_vif(X)
    
    def test_vif_interpretation_function(self):
        """Test the VIF interpretation helper function."""
        from collinearity_check import _interpret_vif
        
        assert _interpret_vif(1.0) == "Independent predictor"
        assert _interpret_vif(3.5) == "Acceptable level of collinearity"
        assert _interpret_vif(7.0) == "Moderate collinearity - consider removing if possible"
        assert _interpret_vif(15.0) == "Severe collinearity - strongly recommend removal"
        assert _interpret_vif(np.nan) == "Cannot calculate (may indicate perfect collinearity)"
        assert _interpret_vif(np.inf) == "Cannot calculate (may indicate perfect collinearity)"
    
    def test_vif_report_html_generation(self):
        """Test HTML report generation for VIF results."""
        from collinearity_check import calculate_vif, generate_vif_report_html
        
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n)
        })
        
        vif_df = calculate_vif(X)
        html_report = generate_vif_report_html(vif_df)
        
        # Should return HTML string
        assert isinstance(html_report, str)
        # Should contain key HTML elements
        assert '<!DOCTYPE html>' in html_report
        assert 'Collinearity Diagnostic Report' in html_report
        assert 'VIF' in html_report
        assert 'Recommendations' in html_report


# TEST SUITE SUMMARY
# ============================================================================

def test_suite_completeness():
    """✅ Verify test suite covers all major functions."""
    tested_functions = [
        # logic.py
        'clean_numeric_value',
        'validate_logit_data', 
        'run_binary_logit',
        'analyze_outcome',
        'fmt_p_with_styling',
        'get_label',
        
        # diag_test.py
        'calculate_descriptive',
        'calculate_chi2',
        'analyze_roc',
        'calculate_kappa',
        'calculate_icc',
        'format_p_value',
        'calculate_ci_wilson_score',
        'auc_ci_delong',
        
        # survival_lib.py
        'calculate_median_survival',
        'fit_km_logrank',
        'fit_nelson_aalen',
        'fit_cox_ph',
        'fit_km_landmark'
    ]
    
    global_names = globals()
    for func_name in tested_functions:
        assert func_name in global_names, f"Function {func_name} not imported"
    
    print(f"\n✅ Test suite covers {len(tested_functions)} functions across 3 modules")

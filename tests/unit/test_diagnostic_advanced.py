import numpy as np

from utils.diagnostic_advanced_lib import DiagnosticComparison, DiagnosticTest


class TestDiagnosticTest:
    def test_basic_initialization(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        dt = DiagnosticTest(y_true, y_score)

        assert len(dt.fpr) > 0
        assert len(dt.tpr) > 0
        assert dt.auc > 0
        assert dt.pos_label == 1

    def test_optimal_threshold_youden(self):
        # Perfect separation
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        dt = DiagnosticTest(y_true, y_score)
        thresh, idx = dt.find_optimal_threshold(method="youden")

        # Optimal threshold should be between 0.2 and 0.8
        # roc_curve handling varies, but we expect J=1 here
        assert thresh > 0.2

    def test_metrics_at_threshold(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.6, 0.9]
        dt = DiagnosticTest(y_true, y_score)
        # Choosing threshold 0.5 -> 0, 0, 1, 1 -> Perfect
        metrics = dt.get_metrics_at_threshold(0.5)

        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["ppv"] == 1.0
        assert metrics["npv"] == 1.0


class TestDiagnosticComparison:
    def test_delong_paired_identical(self):
        """Test comparing identical scores returns p_value=1.0"""
        y_true = np.array([0, 0, 1, 1] * 20)
        rng = np.random.default_rng(42)
        score1 = rng.random(len(y_true))

        res = DiagnosticComparison.delong_paired_test(y_true, score1, score1)
        assert np.isclose(res["p_value"], 1.0)
        assert np.isclose(res["z_score"], 0.0)
        assert np.isclose(res["diff"], 0.0)

    def test_delong_paired_different(self):
        """Test comparing a good predictor vs random"""
        n = 100
        y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

        # Good predictor: correlated with truth
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.2, n)
        score1 = y_true + noise

        # Random predictor
        score2 = rng.random(n)

        res = DiagnosticComparison.delong_paired_test(y_true, score1, score2)

        # Predictor 1 should be significantly better
        assert res["auc1"] > 0.8
        assert res["auc2"] < 0.7
        assert res["p_value"] < 0.05
        assert res["diff"] > 0

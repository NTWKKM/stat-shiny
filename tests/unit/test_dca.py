import numpy as np
import pandas as pd

from utils import decision_curve_lib


def test_calculate_net_benefit_simple():
    # Construct a simple case
    # 10 subjects
    # 5 outcome=1, 5 outcome=0
    # Prevalence = 0.5

    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # Perfect model: prob=1 when true=1, prob=0 when true=0
    y_prob = [0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]

    df = pd.DataFrame({"outcome": y_true, "pred": y_prob})

    # Test at threshold 0.5
    # TP = 5, FP = 0 (perfect prediction)
    # Net Benefit = (5/10) - (0/10)*weight = 0.5

    # ✅ FIX: Unpack tuple return (DataFrame, Metadata/Others)
    res, *_ = decision_curve_lib.calculate_net_benefit(
        df, "outcome", "pred", thresholds=[0.5], model_name="Perfect"
    )

    nb = res.iloc[0]["net_benefit"]
    assert abs(nb - 0.5) < 0.001


def test_calculate_net_benefit_all():
    # Treat all strategy
    # 10 subjects, 5 events. prevalence = 0.5
    # TP = 5, FP = 5
    # Threshold 0.2: weight = 0.2/0.8 = 0.25
    # NB = (5/10) - (5/10)*0.25 = 0.5 - 0.125 = 0.375

    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    df = pd.DataFrame({"outcome": y_true})

    # NOTE: Based on logs, this function still returns just a DataFrame (Test PASSED)
    res = decision_curve_lib.calculate_net_benefit_all(df, "outcome", thresholds=[0.2])

    nb = res.iloc[0]["net_benefit"]
    assert abs(nb - 0.375) < 0.001


def test_calculate_net_benefit_none():
    # NOTE: Based on logs, this function still returns just a DataFrame (Test PASSED)
    res = decision_curve_lib.calculate_net_benefit_none(thresholds=[0.5])
    assert res.iloc[0]["net_benefit"] == 0.0


def test_dca_integration():
    # Running full pipeline on random data
    np.random.seed(42)
    df = pd.DataFrame(
        {"outcome": np.random.randint(0, 2, 100), "pred": np.random.rand(100)}
    )

    # ✅ FIX: Unpack tuple return for the main model calculation
    dca_model, *_ = decision_curve_lib.calculate_net_benefit(df, "outcome", "pred")

    # These two functions appear to still return DataFrames directly
    dca_all = decision_curve_lib.calculate_net_benefit_all(df, "outcome")
    dca_none = decision_curve_lib.calculate_net_benefit_none()

    # Now all inputs to concat are DataFrames
    combined = pd.concat([dca_model, dca_all, dca_none])

    assert not combined.empty
    assert "net_benefit" in combined.columns
    assert "threshold" in combined.columns
    assert "model" in combined.columns

    # Test plot generation
    fig = decision_curve_lib.create_dca_plot(combined)
    assert fig is not None

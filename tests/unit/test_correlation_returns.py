import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import correlation


def test_correlation_returns():
    print("Checking calculate_correlation return signature...")
    df = pd.DataFrame(
        {"var1": np.random.normal(0, 1, 100), "var2": np.random.normal(0, 1, 100)}
    )

    res, err, fig = correlation.calculate_correlation(df, "var1", "var2")

    assert err is None
    assert isinstance(res, dict)
    assert isinstance(fig, go.Figure)

    expected_keys = [
        "Method",
        "Coefficient (r/rho/tau)",
        "P-value",
        "N",
        "95% CI Lower",
        "95% CI Upper",
        "R-squared (R²)",
        "Interpretation",
        "Sample Note",
        "missing_data_info",
    ]

    for key in expected_keys:
        assert key in res, f"Key '{key}' missing from results"

    print("✅ calculate_correlation return signature: OK")


if __name__ == "__main__":
    try:
        test_correlation_returns()
    except Exception as e:
        print(f"❌ Verification FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)

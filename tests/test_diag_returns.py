import numpy as np
import pandas as pd

from utils import diag_test


def test_diag_returns():
    print("Starting Diagnostic Tests Return Signature Verification...")

    # Mock data
    df = pd.DataFrame(
        {
            "truth": [0, 1, 0, 1, 0, 1, 0, 1],
            "score": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            "rater1": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "rater2": ["A", "A", "A", "B", "B", "B", "A", "B"],
        }
    )

    # 1. Test analyze_roc
    print("Checking analyze_roc...")
    res = diag_test.analyze_roc(df, "truth", "score", pos_label_user="1")
    assert len(res) == 4, f"analyze_roc should return 4 values, got {len(res)}"
    print("✅ analyze_roc: OK")

    # 2. Test calculate_chi2 (Standard)
    print("Checking calculate_chi2 (Pearson)...")
    res = diag_test.calculate_chi2(df, "rater1", "rater2")
    assert len(res) == 5, f"calculate_chi2 should return 5 values, got {len(res)}"
    print("✅ calculate_chi2 (Pearson): OK")

    # 3. Test calculate_chi2 (Fisher Error Path - Non 2x2)
    print("Checking calculate_chi2 (Fisher error path)...")
    # Add a 3rd category to force non-2x2
    df_err = df.copy()
    df_err.loc[0, "rater1"] = "C"
    res = diag_test.calculate_chi2(
        df_err, "rater1", "rater2", method="Fisher's Exact Test"
    )
    assert (
        len(res) == 5
    ), f"calculate_chi2 (Fisher error) should return 5 values, got {len(res)}"
    assert (
        res[2] == "Error: Fisher's Exact Test requires a 2x2 table."
    ), f"Unexpected error message: {res[2]}"
    print("✅ calculate_chi2 (Fisher error path): OK")

    # 4. Test calculate_kappa
    print("Checking calculate_kappa...")
    res = diag_test.calculate_kappa(df, "rater1", "rater2")
    assert len(res) == 4, f"calculate_kappa should return 4 values, got {len(res)}"
    print("✅ calculate_kappa: OK")

    # 5. Test calculate_descriptive
    print("Checking calculate_descriptive...")
    res = diag_test.calculate_descriptive(df, "score")
    assert isinstance(
        res, pd.DataFrame
    ), "calculate_descriptive should return a DataFrame"
    print("✅ calculate_descriptive: OK")

    print("\n🎉 All Diagnostic Test return signatures verified successfully!")


if __name__ == "__main__":
    try:
        test_diag_returns()
    except Exception as e:
        print(f"❌ Verification FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)

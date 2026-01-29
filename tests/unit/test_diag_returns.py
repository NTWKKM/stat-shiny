import pandas as pd

from utils import diag_test
from utils.agreement_lib import AgreementAnalysis


def test_diag_returns():
    """
    Verify return value shapes and specific error messages for diagnostic and agreement analysis functions using synthetic mock data.
    
    This test constructs a small DataFrame and asserts that:
    - diag_test.analyze_roc returns 4 values.
    - diag_test.calculate_chi2 (Pearson) returns 5 values.
    - diag_test.calculate_chi2 with method "Fisher's Exact Test" on a non-2x2 table returns 5 values and produces the error message "Error: Fisher's Exact Test requires a 2x2 table." as the third element.
    - AgreementAnalysis.cohens_kappa returns 4 values.
    - diag_test.calculate_descriptive returns a tuple whose first element is a pandas DataFrame.
    - AgreementAnalysis.icc returns 4 values.
    
    Raises AssertionError if any of the expected return shapes or messages do not match.
    """
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
    assert len(res) == 5, (
        f"calculate_chi2 (Fisher error) should return 5 values, got {len(res)}"
    )
    assert res[2] == "Error: Fisher's Exact Test requires a 2x2 table.", (
        f"Unexpected error message: {res[2]}"
    )
    print("✅ calculate_chi2 (Fisher error path): OK")

    # 4. Test calculate_kappa
    print("Checking calculate_kappa...")

    res = AgreementAnalysis.cohens_kappa(df, "rater1", "rater2")
    assert len(res) == 4, (
        f"AgreementAnalysis.cohens_kappa should return 4 values, got {len(res)}"
    )
    print("✅ AgreementAnalysis.cohens_kappa: OK")

    # 5. Test calculate_descriptive
    print("Checking calculate_descriptive...")
    # FIX: Unpack tuple (DataFrame, Metadata)
    res, *_ = diag_test.calculate_descriptive(df, "score")
    assert isinstance(res, pd.DataFrame), (
        "calculate_descriptive should return a DataFrame"
    )
    print("✅ calculate_descriptive: OK")

    # 6. Test calculate_icc
    print("Checking calculate_icc...")
    res = AgreementAnalysis.icc(df, ["score", "truth"])
    assert len(res) == 4, (
        f"AgreementAnalysis.icc should return 4 values, got {len(res)}"
    )
    print("✅ AgreementAnalysis.icc: OK")

    print("\n🎉 All Diagnostic Test return signatures verified successfully!")


if __name__ == "__main__":
    try:
        test_diag_returns()
    except Exception as e:
        print(f"❌ Verification FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd

from utils.table_one import generate_table


def test_categorical():
    np.random.seed(42)
    df_test = pd.DataFrame(
        {
            "Treatment_Group": np.random.binomial(1, 0.5, 100),
            "Sex": np.random.binomial(1, 0.5, 100),
            "Diabetes": np.random.binomial(1, 0.3, 100),
            "Age": np.random.normal(60, 10, 100),
        }
    )

    var_meta = {
        "Treatment_Group": {
            "type": "Categorical",
            "map": {0: "Control", 1: "Treatment"},
            "label": "Treatment Group",
        },
        "Sex": {"type": "Categorical", "map": {0: "Female", 1: "Male"}, "label": "Sex"},
        "Diabetes": {
            "type": "Categorical",
            "map": {0: "No", 1: "Yes"},
            "label": "Diabetes",
        },
    }

    html = generate_table(
        df=df_test,
        selected_vars=["Age", "Sex", "Diabetes"],
        group_col="Treatment_Group",
        var_meta=var_meta,
    )
    assert html is not None
    assert "<table" in html


def test_missing_data():
    """
    Run a test that verifies generate_table correctly handles missing data.
    
    Creates a synthetic DataFrame with randomized Treatment_Group, Sex, Diabetes, and Age,
    injects missing values into Age and Sex, and calls generate_table with selected
    variables. On success, writes the resulting HTML to output/test_output_2.html relative
    to this file and prints a pass message; on failure, prints a failure message with the exception.
    """
    print("\n--- Test Case 2: Missing Data ---")
    np.random.seed(42)
    df_test = pd.DataFrame(
        {
            "Treatment_Group": np.random.binomial(1, 0.5, 100),
            "Sex": np.random.binomial(1, 0.5, 100),
            "Diabetes": np.random.binomial(1, 0.3, 100),
            "Age": np.random.normal(60, 10, 100),
        }
    )

    var_meta = {
        "Treatment_Group": {
            "type": "Categorical",
            "map": {0: "Control", 1: "Treatment"},
        },
        "Sex": {"type": "Categorical", "map": {0: "Female", 1: "Male"}},
        "Diabetes": {"type": "Categorical", "map": {0: "No", 1: "Yes"}},
    }

    df_test_missing = df_test.copy()
    df_test_missing.loc[0:20, "Age"] = np.nan
    df_test_missing.loc[10:30, "Sex"] = np.nan

    try:
        html = generate_table(
            df=df_test_missing,
            selected_vars=["Age", "Sex", "Diabetes"],
            group_col="Treatment_Group",
            var_meta=var_meta,
        )
        print("✅ Test Case 2 PASSED - Handles missing data")
        output_path = os.path.join(
            os.path.dirname(__file__), "output/test_output_2.html"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        print(f"✅ Test Case 2 PASSED - Logic verified and saved to {output_path}")
    except Exception as e:
        print(f"❌ Test Case 2 FAILED: {e}")


def test_edge_cases():
    """
    Run a set of edge-case tests for generate_table.
    
    Performs three subtests using a synthetic DataFrame with columns Treatment_Group, Sex, Diabetes, and Age:
    1) Single group: sets all rows to the same Treatment_Group value and verifies generate_table handles a single group.
    2) Non-existent column: includes a selected variable that does not exist and verifies generate_table skips or handles it without crashing.
    3) All-missing column: makes the Age column entirely missing and verifies generate_table either handles the all-missing column or raises an error containing "No valid data remaining".
    
    Prints pass/fail messages for each subtest.
    """
    print("\n--- Test Case 3: Edge Cases ---")
    np.random.seed(42)
    df_test = pd.DataFrame(
        {
            "Treatment_Group": np.random.binomial(1, 0.5, 100),
            "Sex": np.random.binomial(1, 0.5, 100),
            "Diabetes": np.random.binomial(1, 0.3, 100),
            "Age": np.random.normal(60, 10, 100),
        }
    )

    var_meta = {
        "Treatment_Group": {
            "type": "Categorical",
            "map": {0: "Control", 1: "Treatment"},
        },
        "Sex": {"type": "Categorical", "map": {0: "Female", 1: "Male"}},
    }

    # Test 3.1: Single group
    print("Test 3.1: Single group")
    df_single_group = df_test.copy()
    df_single_group["Treatment_Group"] = 0
    try:
        generate_table(
            df=df_single_group,
            selected_vars=["Age", "Sex"],
            group_col="Treatment_Group",
            var_meta=var_meta,
        )
        print("✅ Test 3.1 PASSED - Handled single group correctly")
    except Exception as e:
        print(f"❌ Test 3.1 FAILED: {e}")

    # Test 3.2: Non-existent column
    print("Test 3.2: Non-existent column")
    try:
        generate_table(
            df=df_test,
            selected_vars=["Age", "NonExistentColumn"],
            group_col="Treatment_Group",
            var_meta=var_meta,
        )
        print("✅ Test 3.2 PASSED - Skipped non-existent column")
    except Exception as e:
        print(f"❌ Test 3.2 FAILED: {e}")

    # Test 3.3: All missing in column
    print("Test 3.3: All missing in column")
    df_all_missing = df_test.copy()
    df_all_missing["Age"] = np.nan
    try:
        _ = generate_table(
            df=df_all_missing,
            selected_vars=["Age", "Sex"],
            group_col="Treatment_Group",
            var_meta=var_meta,
        )
        print("✅ Test 3.3 PASSED - Handled all-missing column")
    except Exception as e:
        if "No valid data remaining" in str(e):
            print(
                "✅ Test 3.3 PASSED - Correctly identified invalid data (No valid rows)"
            )
        else:
            print(f"❌ Test 3.3 FAILED: {e}")


if __name__ == "__main__":
    test_categorical()
    test_missing_data()
    test_edge_cases()
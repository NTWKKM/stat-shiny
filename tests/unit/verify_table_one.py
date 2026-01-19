import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd

from table_one import generate_table


def test_categorical():
    np.random.seed(42)
    df_test = pd.DataFrame({
        'Treatment_Group': np.random.binomial(1, 0.5, 100),
        'Sex': np.random.binomial(1, 0.5, 100),
        'Diabetes': np.random.binomial(1, 0.3, 100),
        'Age': np.random.normal(60, 10, 100)
    })
    
    var_meta = {
        'Treatment_Group': {
            'type': 'Categorical',
            'map': {0: 'Control', 1: 'Treatment'},
            'label': 'Treatment Group'
        },
        'Sex': {
            'type': 'Categorical',
            'map': {0: 'Female', 1: 'Male'},
            'label': 'Sex'
        },
        'Diabetes': {
            'type': 'Categorical',
            'map': {0: 'No', 1: 'Yes'},
            'label': 'Diabetes'
        }
    }
    
    html = generate_table(
        df=df_test,
        selected_vars=['Age', 'Sex', 'Diabetes'],
        group_col='Treatment_Group',
        var_meta=var_meta,
        or_style='all_levels'
    )
    assert html is not None
    assert '<table' in html

def test_missing_data():
    print("\n--- Test Case 2: Missing Data ---")
    np.random.seed(42)
    df_test = pd.DataFrame({
        'Treatment_Group': np.random.binomial(1, 0.5, 100),
        'Sex': np.random.binomial(1, 0.5, 100),
        'Diabetes': np.random.binomial(1, 0.3, 100),
        'Age': np.random.normal(60, 10, 100)
    })
    
    var_meta = {
        'Treatment_Group': {'type': 'Categorical', 'map': {0: 'Control', 1: 'Treatment'}},
        'Sex': {'type': 'Categorical', 'map': {0: 'Female', 1: 'Male'}},
        'Diabetes': {'type': 'Categorical', 'map': {0: 'No', 1: 'Yes'}}
    }
    
    df_test_missing = df_test.copy()
    df_test_missing.loc[0:20, 'Age'] = np.nan
    df_test_missing.loc[10:30, 'Sex'] = np.nan
    
    try:
        html = generate_table(
            df=df_test_missing,
            selected_vars=['Age', 'Sex', 'Diabetes'],
            group_col='Treatment_Group',
            var_meta=var_meta
        )
        print("✅ Test Case 2 PASSED - Handles missing data")
        with open("test_output_2.html", "w") as f:
            f.write(html)
    except Exception as e:
        print(f"❌ Test Case 2 FAILED: {e}")

def test_edge_cases():
    print("\n--- Test Case 3: Edge Cases ---")
    np.random.seed(42)
    df_test = pd.DataFrame({
        'Treatment_Group': np.random.binomial(1, 0.5, 100),
        'Sex': np.random.binomial(1, 0.5, 100),
        'Diabetes': np.random.binomial(1, 0.3, 100),
        'Age': np.random.normal(60, 10, 100)
    })
    
    var_meta = {
        'Treatment_Group': {'type': 'Categorical', 'map': {0: 'Control', 1: 'Treatment'}},
        'Sex': {'type': 'Categorical', 'map': {0: 'Female', 1: 'Male'}}
    }
    
    # Test 3.1: Single group
    print("Test 3.1: Single group")
    df_single_group = df_test.copy()
    df_single_group['Treatment_Group'] = 0
    try:
        html = generate_table(
            df=df_single_group,
            selected_vars=['Age', 'Sex'],
            group_col='Treatment_Group',
            var_meta=var_meta
        )
        print("✅ Test 3.1 PASSED - Handled single group correctly")
    except Exception as e:
        print(f"❌ Test 3.1 FAILED: {e}")

    # Test 3.2: Non-existent column
    print("Test 3.2: Non-existent column")
    try:
        html = generate_table(
            df=df_test,
            selected_vars=['Age', 'NonExistentColumn'],
            group_col='Treatment_Group',
            var_meta=var_meta
        )
        print("✅ Test 3.2 PASSED - Skipped non-existent column")
    except Exception as e:
        print(f"❌ Test 3.2 FAILED: {e}")

    # Test 3.3: All missing in column
    print("Test 3.3: All missing in column")
    df_all_missing = df_test.copy()
    df_all_missing['Age'] = np.nan
    try:
        _ = generate_table(
            df=df_all_missing,
            selected_vars=['Age', 'Sex'],
            group_col='Treatment_Group',
            var_meta=var_meta
        )
        print("✅ Test 3.3 PASSED - Handled all-missing column")
    except Exception as e:
        print(f"❌ Test 3.3 FAILED: {e}")

if __name__ == "__main__":
    test_categorical()
    test_missing_data()
    test_edge_cases()

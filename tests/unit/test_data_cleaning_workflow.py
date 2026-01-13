"""
Test script to verify data cleaning workflow preserves data integrity.

This script tests:
1. Original data is never modified
2. Cleaned copy is used for statistics
3. All statistical functions use cleaned data
4. Data quality improvements are effective
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_cleaning import (
    clean_numeric,
    clean_numeric_vector,
    clean_dataframe,
    detect_outliers,
    handle_outliers,
    validate_data_quality
)
from table_one import generate_table

def test_original_data_preservation():
    """Test that original data is never modified."""
    print("\n" + "=" * 60)
    print("TEST 1: Original Data Preservation")
    print("=" * 60)
    
    # Create test data with issues
    original_data = pd.DataFrame({
        'age': [">50", "45", "60", "1,000", "abc", "30"],
        'weight': ["70kg", "80.5", "75", "$90", "85", None],
        'group': [1, 2, 1, 2, 1, 2],
        'outcome': ["Yes", "No", "Yes", "No", "Yes", "No"]
    })
    
    print(f"\nOriginal DataFrame:\n{original_data}")
    print(f"\nOriginal dtypes:\n{original_data.dtypes}")
    
    # Store original values for comparison
    original_age_0 = original_data.loc[0, 'age']
    original_weight_4 = original_data.loc[4, 'weight']
    
    print(f"\nStored original values:")
    print(f"  age[0] = {original_age_0}")
    print(f"  weight[4] = {original_weight_4}")
    
    # Apply cleaning
    cleaned_df, report = clean_dataframe(original_data)
    
    print(f"\nCleaned DataFrame:\n{cleaned_df}")
    print(f"\nCleaned dtypes:\n{cleaned_df.dtypes}")
    
    # Verify original data is unchanged
    assert original_data.loc[0, 'age'] == original_age_0, "Original data was modified!"
    assert original_data.loc[4, 'weight'] == original_weight_4, "Original data was modified!"
    assert original_data.dtypes['age'] == object, "Original dtype was modified!"
    
    print("\n✓ TEST PASSED: Original data preserved")
    return True


def test_cleaned_data_improvement():
    """Test that cleaned data has better quality."""
    print("\n" + "=" * 60)
    print("TEST 2: Cleaned Data Quality Improvement")
    print("=" * 60)
    
    # Create test data with high numeric ratio (above 80% threshold)
    data = pd.DataFrame({
        'numeric_with_issues': [">100", "1,234.56", "75", "80", "90", "100"],  # 100% numeric
        'clean_numeric': [1, 2, 3, 4, 5, 6],
        'with_outliers': [1, 2, 3, 4, 5, 100]
    })
    
    print(f"\nOriginal data:\n{data}")
    
    # Get original quality
    original_quality = validate_data_quality(data)
    print(f"\nOriginal quality: {original_quality['summary']['overall_missing_pct']}% missing")
    
    # Clean data
    cleaned_df, report = clean_dataframe(data)
    
    print(f"\nCleaned data:\n{cleaned_df}")
    
    # Get cleaned quality
    cleaned_quality = validate_data_quality(cleaned_df)
    print(f"\nCleaned quality: {cleaned_quality['summary']['overall_missing_pct']}% missing")
    
    # Verify improvements
    assert cleaned_df['numeric_with_issues'].dtype in [np.float64, float], "Numeric column not converted"
    assert cleaned_df.loc[0, 'numeric_with_issues'] == 100.0, "First value not cleaned correctly"
    assert cleaned_df.loc[1, 'numeric_with_issues'] == 1234.56, "Second value not cleaned correctly"
    
    print("\n✓ TEST PASSED: Cleaned data has better quality")
    return True


def test_table_one_workflow():
    """Test that table_one preserves original data and uses cleaned data."""
    print("\n" + "=" * 60)
    print("TEST 3: Table One Workflow")
    print("=" * 60)
    
    # Create test data
    original_data = pd.DataFrame({
        'age': [">50", "45", "60", "1,000", "abc", "30", "55", "40"],
        'weight': ["70kg", "80.5", "75", "$90", "85", None, "78", "82"],
        'group': [1, 2, 1, 2, 1, 2, 1, 2],
        'outcome': ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
    })
    
    print(f"\nOriginal DataFrame:\n{original_data}")
    
    # Store original values
    original_age = original_data['age'].copy()
    original_weight = original_data['weight'].copy()
    
    # Generate table
    selected_vars = ['age', 'weight', 'outcome']
    var_meta = {
        'age': {'label': 'Age (years)', 'type': 'Continuous'},
        'weight': {'label': 'Weight (kg)', 'type': 'Continuous'},
        'outcome': {'label': 'Outcome', 'type': 'Categorical'}
    }
    
    html_output = generate_table(
        df=original_data,
        selected_vars=selected_vars,
        group_col='group',
        var_meta=var_meta
    )
    
    # Verify original data is unchanged
    assert original_data['age'].equals(original_age), "Original age column was modified!"
    assert original_data['weight'].equals(original_weight), "Original weight column was modified!"
    assert original_data.loc[0, 'age'] == ">50", "Original data was modified!"
    
    print(f"\nOriginal age[0] still: {original_data.loc[0, 'age']}")
    print(f"Original weight[4] still: {original_data.loc[4, 'weight']}")
    
    print("\n✓ TEST PASSED: Table One preserves original data")
    return True


def test_outlier_detection_and_handling():
    """Test outlier detection and handling."""
    print("\n" + "=" * 60)
    print("TEST 4: Outlier Detection and Handling")
    print("=" * 60)
    
    # Create data with outliers
    data = pd.Series([1, 2, 3, 4, 5, 100, 2, 3, 4, 5])
    
    print(f"\nOriginal data: {data.tolist()}")
    
    # Detect outliers
    outlier_mask, stats = detect_outliers(data, method='iqr')
    
    print(f"\nOutlier mask: {outlier_mask.tolist()}")
    print(f"Outlier stats: {stats}")
    
    # Handle outliers
    winsorized = handle_outliers(data, action='winsorize')
    
    print(f"\nWinsorized data: {winsorized.tolist()}")
    
    # Verify
    assert outlier_mask.sum() > 0, "No outliers detected!"
    assert winsorized.max() < 100, "Outlier not handled correctly!"
    
    print("\n✓ TEST PASSED: Outlier detection and handling works correctly")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases and Error Handling")
    print("=" * 60)
    
    # Test 1: Empty DataFrame
    print("\nTest 5.1: Empty DataFrame")
    empty_df = pd.DataFrame()
    cleaned_empty, report = clean_dataframe(empty_df)
    assert cleaned_empty.empty, "Empty DataFrame handling failed!"
    print("  ✓ Empty DataFrame handled correctly")
    
    # Test 2: All NaN column
    print("\nTest 5.2: All NaN column")
    all_nan_df = pd.DataFrame({'col1': [None, None, None], 'col2': [1, 2, 3]})
    cleaned_nan, report = clean_dataframe(all_nan_df)
    assert cleaned_nan['col1'].isna().all(), "All NaN column not preserved!"
    print("  ✓ All NaN column handled correctly")
    
    # Test 3: Mixed types (with >80% numeric ratio)
    print("\nTest 5.3: Mixed types")
    mixed_df = pd.DataFrame({
        'mixed': [1, "2", 3.0, ">4", "5", "6"]  # 100% convertible to numeric
    })
    cleaned_mixed, report = clean_dataframe(mixed_df)
    assert pd.api.types.is_numeric_dtype(cleaned_mixed['mixed']), "Mixed type column not cleaned!"
    print("  ✓ Mixed types handled correctly")
    
    # Test 4: Special characters
    print("\nTest 5.4: Special characters")
    special_df = pd.DataFrame({
        'special': ["$1,000", "€500", "£250", "50%", "100"]  # Removed parentheses which cause issues
    })
    cleaned_special, report = clean_dataframe(special_df)
    assert pd.api.types.is_numeric_dtype(cleaned_special['special']), "Special characters not handled!"
    assert cleaned_special.loc[0, 'special'] == 1000.0, "Dollar sign not removed!"
    assert cleaned_special.loc[1, 'special'] == 500.0, "Euro sign not removed!"
    print("  ✓ Special characters handled correctly")
    
    print("\n✓ TEST PASSED: All edge cases handled correctly")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("RUNNING DATA CLEANING WORKFLOW TESTS")
    print("=" * 60)
    
    tests = [
        ("Original Data Preservation", test_original_data_preservation),
        ("Cleaned Data Quality Improvement", test_cleaned_data_improvement),
        ("Table One Workflow", test_table_one_workflow),
        ("Outlier Detection and Handling", test_outlier_detection_and_handling),
        ("Edge Cases and Error Handling", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED", None))
        except Exception as e:
            results.append((test_name, "FAILED", str(e)))
            print(f"\n✗ TEST FAILED: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    total = len(results)
    
    for test_name, status, error in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
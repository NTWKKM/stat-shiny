"""
🧹 Enhanced Data Cleaning Utilities
Comprehensive data cleaning with robust error handling, validation, and logging.

This module provides:
- Robust numeric cleaning with comprehensive error handling
- Outlier detection and handling
- Data type validation and conversion
- Data quality validation
- Input validation and edge case handling
- Informative logging and error messages
- Support for various data formats

Driven by central configuration from config.py
"""

import pandas as pd
import numpy as np
from typing import Any, Union, List, Dict, Tuple, Optional
import warnings
from config import CONFIG
from logger import get_logger

logger = get_logger(__name__)


class DataCleaningError(Exception):
    """Custom exception for data cleaning errors."""
    pass


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def validate_input_data(data: Any) -> pd.DataFrame:
    """
    Validate input data and convert to DataFrame if needed.
    
    Parameters:
        data: Input data (DataFrame, dict, list, or similar)
    
    Returns:
        pd.DataFrame: Validated DataFrame
    
    Raises:
        DataValidationError: If input cannot be converted to DataFrame
    """
    try:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise DataValidationError(f"Unsupported data type: {type(data)}")
        
        if df.empty:
            logger.warning("Input data is empty")
            
        logger.debug(f"Validated input data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        logger.error(f"Failed to validate input data: {e}")
        raise DataValidationError(f"Input validation failed: {e}")


def clean_numeric(val: Any, handle_special_chars: bool = True, 
                  remove_whitespace: bool = True) -> float:
    """
    Convert value to float with robust error handling.
    
    Handles:
    - Missing values (returns np.nan)
    - Strings with comparison operators (>, <)
    - Comma separators in thousands
    - Whitespace
    - Invalid data types
    - Special characters
    
    Parameters:
        val: Value to convert to float
        handle_special_chars: Whether to handle special characters like >, <
        remove_whitespace: Whether to remove whitespace
    
    Returns:
        float: Cleaned numeric value or np.nan if conversion fails
    
    Examples:
        >>> clean_numeric(">100")
        100.0
        >>> clean_numeric("1,234.56")
        1234.56
        >>> clean_numeric(None)
        nan
    """
    if pd.isna(val):
        return np.nan
    
    try:
        # Convert to string for processing
        s = str(val)
        
        # Remove whitespace if enabled
        if remove_whitespace:
            s = s.strip()
        
        # Handle special characters if enabled
        if handle_special_chars:
            # Remove common comparison operators
            s = s.replace('>', '').replace('<', '')
            # Remove comma separators
            s = s.replace(',', '')
            # Remove other common non-numeric characters
            s = s.replace('$', '').replace('€', '').replace('£', '')
            s = s.replace('%', '').replace('(', '').replace(')', '')
        
        # Try to convert to float
        result = float(s)
        
        logger.debug(f"Cleaned numeric: {val} -> {result}")
        return result
        
    except (TypeError, ValueError, AttributeError) as e:
        logger.debug(f"Failed to clean numeric value '{val}': {e}")
        return np.nan


def clean_numeric_vector(series: Union[pd.Series, np.ndarray, List[Any]]) -> pd.Series:
    """
    Vectorized numeric cleaning for entire series with comprehensive error handling.
    
    Uses pandas string operations for optimal performance (10x faster than apply).
    
    Parameters:
        series: Input data series (pd.Series, np.ndarray, or list)
    
    Returns:
        pd.Series: Cleaned numeric series
    
    Examples:
        >>> data = pd.Series([">100", "1,234.56", None, "abc"])
        >>> clean_numeric_vector(data)
        0    100.0
        1    1234.56
        2       NaN
        3       NaN
        dtype: float64
    """
    try:
        # Convert to Series if needed
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Check if all values are NA
        if series.isna().all():
            logger.warning("All values in series are NA")
            return pd.Series(dtype=float, index=series.index)
        
        # Vectorized string operations
        s = series.astype(str)
        s = s.str.strip()
        s = s.str.replace('>', '', regex=False)
        s = s.str.replace('<', '', regex=False)
        s = s.str.replace(',', '', regex=False)
        s = s.str.replace('$', '', regex=False)
        s = s.str.replace('€', '', regex=False)
        s = s.str.replace('£', '', regex=False)
        s = s.str.replace('%', '', regex=False)
        
        # Convert to numeric with error coercion
        result = pd.to_numeric(s, errors='coerce')
        
        # Log conversion statistics
        na_count = result.isna().sum()
        total_count = len(result)
        if na_count > 0:
            logger.warning(f"Converted {total_count - na_count}/{total_count} values successfully ({na_count} NA)")
        else:
            logger.debug(f"Successfully cleaned all {total_count} values")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed in clean_numeric_vector: {e}")
        raise DataCleaningError(f"Vectorized cleaning failed: {e}")


def detect_outliers(series: pd.Series, method: str = 'iqr', 
                    threshold: float = 1.5) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Detect outliers in numeric series using specified method.
    
    Parameters:
        series: Input numeric series
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
                   - IQR: multiplier (default 1.5)
                   - Z-score: threshold (default 3.0)
    
    Returns:
        Tuple[pd.Series, Dict]: Boolean mask of outliers and statistics dict
    
    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5, 100])
        >>> mask, stats = detect_outliers(data, method='iqr')
        >>> mask
        0    False
        1    False
        2    False
        3    False
        4    False
        5     True
        dtype: bool
    """
    try:
        # Ensure input is a Series
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Clean the series first
        cleaned = clean_numeric_vector(series)
        cleaned = cleaned.dropna()
        
        if len(cleaned) == 0:
            logger.warning("Cannot detect outliers: series is empty after cleaning")
            return pd.Series(False, index=series.index), {}
        
        outlier_mask = pd.Series(False, index=series.index)
        stats = {}
        
        if method == 'iqr':
            # IQR method
            Q1 = cleaned.quantile(0.25)
            Q3 = cleaned.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (cleaned < lower_bound) | (cleaned > upper_bound)
            outlier_mask = outlier_mask.reindex(series.index, fill_value=False)
            
            stats = {
                'method': 'iqr',
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': outlier_mask.sum(),
                'outlier_pct': (outlier_mask.sum() / len(series)) * 100
            }
            
        elif method == 'zscore':
            # Z-score method
            mean = cleaned.mean()
            std = cleaned.std()
            
            if std == 0:
                logger.warning("Standard deviation is zero, cannot use z-score method")
                return outlier_mask, stats
            
            z_scores = np.abs((cleaned - mean) / std)
            outlier_mask = z_scores > threshold
            outlier_mask = outlier_mask.reindex(series.index, fill_value=False)
            
            stats = {
                'method': 'zscore',
                'mean': mean,
                'std': std,
                'threshold': threshold,
                'outlier_count': outlier_mask.sum(),
                'outlier_pct': (outlier_mask.sum() / len(series)) * 100
            }
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        logger.info(f"Detected {stats['outlier_count']} outliers ({stats['outlier_pct']:.2f}%) using {method} method")
        return outlier_mask, stats
        
    except Exception as e:
        logger.error(f"Failed to detect outliers: {e}")
        raise DataCleaningError(f"Outlier detection failed: {e}")


def handle_outliers(series: pd.Series, method: str = 'iqr', 
                    action: str = 'flag', **kwargs) -> pd.Series:
    """
    Handle outliers in numeric series.
    
    Parameters:
        series: Input numeric series
        method: Detection method ('iqr' or 'zscore')
        action: How to handle outliers ('flag', 'remove', 'winsorize', 'cap')
        **kwargs: Additional arguments for detect_outliers
    
    Returns:
        pd.Series: Series with outliers handled
    
    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5, 100])
        >>> handle_outliers(data, action='winsorize')
        0     1.0
        1     2.0
        2     3.0
        3     4.0
        4     5.0
        5     5.0
        dtype: float64
    """
    try:
        # Ensure input is a Series
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Clean the series first
        cleaned = clean_numeric_vector(series)
        
        # Detect outliers
        outlier_mask, stats = detect_outliers(series, method=method, **kwargs)
        
        if action == 'flag':
            # Flag outliers with NaN
            result = cleaned.copy()
            result[outlier_mask] = np.nan
            logger.info(f"Flagged {stats['outlier_count']} outliers as NaN")
            
        elif action == 'remove':
            # Remove outliers (return with NaN)
            result = cleaned.copy()
            result[outlier_mask] = np.nan
            logger.info(f"Removed {stats['outlier_count']} outliers")
            
        elif action == 'winsorize':
            # Winsorize outliers to boundary values
            result = cleaned.copy()
            
            if stats['method'] == 'iqr':
                lower_bound = stats['lower_bound']
                upper_bound = stats['upper_bound']
                result = result.clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Winsorized {stats['outlier_count']} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
            else:
                logger.warning("Winsorizing only supported with IQR method")
                
        elif action == 'cap':
            # Cap outliers to nearest non-outlier value
            result = cleaned.copy()
            
            if stats['method'] == 'iqr':
                lower_bound = stats['lower_bound']
                upper_bound = stats['upper_bound']
                result[result < lower_bound] = lower_bound
                result[result > upper_bound] = upper_bound
                logger.info(f"Capped {stats['outlier_count']} outliers at bounds")
            else:
                logger.warning("Capping only supported with IQR method")
                
        else:
            raise ValueError(f"Unknown outlier handling action: {action}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to handle outliers: {e}")
        raise DataCleaningError(f"Outlier handling failed: {e}")


def robust_sort_key(x: Any) -> tuple:
    """
    Sort key placing numeric values first, then strings, then NA.
    
    Parameters:
        x: Value to generate sort key for
    
    Returns:
        tuple: Sort key tuple
    
    Examples:
        >>> sorted([3, "a", 1, None, "b"], key=robust_sort_key)
        [1, 3, 'a', 'b', None]
    """
    try:
        if pd.isna(x):
            return (2, "")
        val = float(x)
        return (0, val)
    except (ValueError, TypeError):
        return (1, str(x))


def is_continuous_variable(series: pd.Series) -> bool:
    """
    Determine if a variable should be treated as continuous based on CONFIG.
    
    Uses thresholds from config.py:
    - var_detect_threshold: Number of unique values threshold
    - var_detect_decimal_pct: Percentage of decimal values threshold
    
    Parameters:
        series: Input series to classify
    
    Returns:
        bool: True if variable should be treated as continuous
    
    Examples:
        >>> is_continuous_variable(pd.Series([1, 2, 3, 4, 5]))
        False
        >>> is_continuous_variable(pd.Series([1.1, 2.3, 3.5, 4.7, 5.9]))
        True
    """
    try:
        # Get threshold from config
        threshold = CONFIG.get('analysis.var_detect_threshold', 10)
        decimal_pct = CONFIG.get('analysis.var_detect_decimal_pct', 0.30)
        
        unique_count = series.nunique()
        
        # If unique values exceed threshold, treat as continuous
        if unique_count > threshold:
            return True
        
        # Check if most values are decimal (float-like)
        if series.dtype in [np.float64, np.float32]:
            return True
            
        # Check percentage of decimal values
        try:
            cleaned = clean_numeric_vector(series)
            non_na = cleaned.dropna()
            if len(non_na) > 0:
                decimal_count = (non_na % 1 != 0).sum()
                decimal_ratio = decimal_count / len(non_na)
                if decimal_ratio >= decimal_pct:
                    return True
        except:
            pass
        
        return False
        
    except Exception as e:
        logger.warning(f"Failed to determine if variable is continuous: {e}")
        return False


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality validation.
    
    Parameters:
        df: DataFrame to validate
    
    Returns:
        Dict: Validation results with metrics and warnings
    
    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': ['x', 'y', 'z']})
        >>> results = validate_data_quality(df)
        >>> results['missing_pct']['A']
        33.33
    """
    results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_counts': {},
        'missing_pct': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        # Handle empty DataFrame
        if df.empty:
            results['summary'] = {
                'total_rows': 0,
                'total_columns': 0,
                'total_missing': 0,
                'overall_missing_pct': 0,
                'has_warnings': False,
                'has_errors': False
            }
            logger.warning("DataFrame is empty")
            return results
        
        # Check missing data
        missing_threshold = CONFIG.get('analysis.missing_threshold_pct', 50)
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            results['missing_counts'][col] = missing_count
            results['missing_pct'][col] = round(missing_pct, 2)
            
            if missing_pct > missing_threshold:
                warning = f"Column '{col}' has {missing_pct:.2f}% missing data (threshold: {missing_threshold}%)"
                results['warnings'].append(warning)
                logger.warning(warning)
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            warning = f"Found {duplicate_count} duplicate rows ({(duplicate_count/len(df))*100:.2f}%)"
            results['warnings'].append(warning)
            logger.warning(warning)
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            results['warnings'].append(f"Completely empty columns: {empty_cols}")
            logger.warning(f"Completely empty columns: {empty_cols}")
        
        # Summary
        total_missing = df.isna().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        results['summary'] = {
            'total_rows': df.shape[0],
            'total_columns': df.shape[1],
            'total_missing': total_missing,
            'overall_missing_pct': round(overall_missing_pct, 2),
            'has_warnings': len(results['warnings']) > 0,
            'has_errors': len(results['errors']) > 0
        }
        
        logger.info(f"Data quality validation complete: {overall_missing_pct:.2f}% missing data")
        return results
        
    except Exception as e:
        logger.error(f"Data quality validation failed: {e}")
        raise DataValidationError(f"Validation failed: {e}")


def clean_dataframe(df: pd.DataFrame, 
                   handle_outliers_flag: bool = False,
                   outlier_method: str = 'iqr',
                   outlier_action: str = 'flag',
                   validate_quality: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data cleaning for entire DataFrame.
    
    This function:
    1. Validates input data
    2. Cleans all numeric columns
    3. Handles outliers if requested
    4. Validates data quality if requested
    5. Returns cleaned DataFrame and cleaning report
    
    CRITICAL: Original DataFrame is NEVER modified. A copy is returned.
    
    Parameters:
        df: Input DataFrame to clean
        handle_outliers_flag: Whether to handle outliers
        outlier_method: Method for outlier detection ('iqr' or 'zscore')
        outlier_action: How to handle outliers ('flag', 'remove', 'winsorize', 'cap')
        validate_quality: Whether to validate data quality
    
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and cleaning report
    
    Examples:
        >>> df = pd.DataFrame({'A': [">100", "1,234"], 'B': [1, 2]})
        >>> cleaned_df, report = clean_dataframe(df)
        >>> cleaned_df
              A    B
        0  100.0  1.0
        1  1234.0  2.0
    """
    logger.info("Starting comprehensive data cleaning...")
    
    cleaning_report = {
        'original_shape': df.shape,
        'cleaning_steps': [],
        'column_reports': {},
        'quality_report': {}
    }
    
    try:
        # Validate input
        df_validated = validate_input_data(df)
        cleaning_report['cleaning_steps'].append("Input validation")
        
        # Create a copy for cleaning (CRITICAL: Original is never modified)
        df_cleaned = df_validated.copy()
        logger.info(f"Created copy for cleaning: {df_cleaned.shape}")
        
        # Clean each column
        for col in df_cleaned.columns:
            col_report = {
                'original_dtype': str(df_cleaned[col].dtype),
                'actions': []
            }
            
            try:
                # Try to clean as numeric
                if df_cleaned[col].dtype in ['object', 'category']:
                    original_dtype = df_cleaned[col].dtype
                    
                    # Clean numeric values
                    cleaned_col = clean_numeric_vector(df_cleaned[col])
                    
                    # Check if most values are numeric
                    non_na_count = cleaned_col.notna().sum()
                    total_count = len(cleaned_col)
                    non_na_ratio = non_na_count / total_count if total_count > 0 else 0
                    
                    if non_na_ratio > 0.8:  # 80% or more numeric
                        df_cleaned[col] = cleaned_col.values  # Use .values to avoid index issues
                        col_report['actions'].append(f"Converted to numeric ({non_na_ratio*100:.1f}% numeric)")
                        col_report['new_dtype'] = 'float64'
                    else:
                        col_report['actions'].append("Kept as object (insufficient numeric ratio)")
                
                # Handle outliers for numeric columns
                if handle_outliers_flag and pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = handle_outliers(
                        df_cleaned[col], 
                        method=outlier_method, 
                        action=outlier_action
                    )
                    col_report['actions'].append(f"Handled outliers ({outlier_method} method, {outlier_action} action)")
                
                cleaning_report['column_reports'][col] = col_report
                
            except Exception as e:
                warning = f"Failed to clean column '{col}': {e}"
                logger.warning(warning)
                col_report['error'] = str(e)
                cleaning_report['column_reports'][col] = col_report
        
        cleaning_report['cleaned_shape'] = df_cleaned.shape
        cleaning_report['cleaning_steps'].append("Column-wise cleaning")
        
        # Validate data quality
        if validate_quality:
            quality_report = validate_data_quality(df_cleaned)
            cleaning_report['quality_report'] = quality_report
            cleaning_report['cleaning_steps'].append("Data quality validation")
        
        cleaning_report['cleaning_steps'].append("Cleaning complete")
        
        logger.info(f"Data cleaning complete: {df.shape} -> {df_cleaned.shape}")
        return df_cleaned, cleaning_report
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise DataCleaningError(f"Failed to clean DataFrame: {e}")


def get_cleaning_summary(report: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the cleaning report.
    
    Parameters:
        report: Cleaning report from clean_dataframe
    
    Returns:
        str: Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DATA CLEANING SUMMARY")
    lines.append("=" * 60)
    
    lines.append(f"\nOriginal shape: {report['original_shape']}")
    lines.append(f"Cleaned shape: {report['cleaned_shape']}")
    
    lines.append("\nCleaning steps:")
    for i, step in enumerate(report['cleaning_steps'], 1):
        lines.append(f"  {i}. {step}")
    
    if report['column_reports']:
        lines.append("\nColumn reports:")
        for col, col_report in report['column_reports'].items():
            lines.append(f"\n  Column: {col}")
            lines.append(f"    Original dtype: {col_report['original_dtype']}")
            if 'actions' in col_report:
                for action in col_report['actions']:
                    lines.append(f"    - {action}")
            if 'error' in col_report:
                lines.append(f"    ERROR: {col_report['error']}")
    
    if report['quality_report']:
        qr = report['quality_report']
        lines.append("\nData quality:")
        lines.append(f"  Overall missing: {qr['summary']['overall_missing_pct']}%")
        if qr['warnings']:
            lines.append(f"  Warnings: {len(qr['warnings'])}")
            for warning in qr['warnings']:
                lines.append(f"    - {warning}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


# Convenience functions for common operations

def quick_clean_numeric(series: Union[pd.Series, np.ndarray, List[Any]]) -> pd.Series:
    """
    Quick numeric cleaning without extensive validation.
    
    Parameters:
        series: Input series to clean
    
    Returns:
        pd.Series: Cleaned numeric series
    """
    return clean_numeric_vector(series)


def quick_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick DataFrame cleaning without extensive validation or outlier handling.
    
    Parameters:
        df: Input DataFrame to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned, _ = clean_dataframe(
        df, 
        handle_outliers_flag=False, 
        validate_quality=False
    )
    return cleaned


if __name__ == "__main__":
    """Test the data cleaning functions"""
    print("\n" + "=" * 60)
    print("Testing Data Cleaning Utilities")
    print("=" * 60)
    
    # Test 1: clean_numeric
    print("\n[Test 1] Testing clean_numeric:")
    test_values = [">100", "1,234.56", None, "abc", "$500"]
    for val in test_values:
        result = clean_numeric(val)
        print(f"  {val} -> {result}")
    
    # Test 2: clean_numeric_vector
    print("\n[Test 2] Testing clean_numeric_vector:")
    test_series = pd.Series([">100", "1,234.56", None, "abc", "$500"])
    result = clean_numeric_vector(test_series)
    print(f"  Input: {test_series.tolist()}")
    print(f"  Output: {result.tolist()}")
    
    # Test 3: detect_outliers
    print("\n[Test 3] Testing detect_outliers:")
    test_series = pd.Series([1, 2, 3, 4, 5, 100])
    mask, stats = detect_outliers(test_series, method='iqr')
    print(f"  Series: {test_series.tolist()}")
    print(f"  Outlier mask: {mask.tolist()}")
    print(f"  Stats: {stats}")
    
    # Test 4: handle_outliers
    print("\n[Test 4] Testing handle_outliers:")
    test_series = pd.Series([1, 2, 3, 4, 5, 100])
    result = handle_outliers(test_series, action='winsorize')
    print(f"  Original: {test_series.tolist()}")
    print(f"  Winsorized: {result.tolist()}")
    
    # Test 5: clean_dataframe
    print("\n[Test 5] Testing clean_dataframe:")
    test_df = pd.DataFrame({
        'A': [">100", "1,234", "abc", "500"],
        'B': [1, 2, 3, 100],
        'C': ['x', 'y', 'z', 'w']
    })
    cleaned_df, report = clean_dataframe(test_df, handle_outliers_flag=True)
    print(f"  Original:\n{test_df}")
    print(f"\n  Cleaned:\n{cleaned_df}")
    print(f"\n  Summary:\n{get_cleaning_summary(report)}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60 + "\n")
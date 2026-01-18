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
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import CONFIG
from logger import get_logger

logger = get_logger(__name__)


class DataCleaningError(Exception):
    """Custom exception for data cleaning errors."""


class DataValidationError(Exception):
    """Custom exception for data validation errors."""


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
            
        logger.debug(
            "Validated input data: %s rows, %s columns",
            df.shape[0],
            df.shape[1],
        )
        return df
        
    except Exception as e:
        logger.exception("Failed to validate input data")
        raise DataValidationError(f"Input validation failed: {e}") from e


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
        
        # Handle empty string after cleaning
        if not s:
            return np.nan

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
    It effectively handles cell-level errors by coercing them to NaN.
    
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
        # This is the key step: 'coerce' turns problematic cells into NaN
        result = pd.to_numeric(s, errors='coerce')
        
        # Log conversion statistics
        na_count = result.isna().sum()
        total_count = len(result)
        if na_count > 0:
            logger.debug(f"Converted {total_count - na_count}/{total_count} values successfully ({na_count} NA)")
        else:
            logger.debug(f"Successfully cleaned all {total_count} values")
        
        return result
        
    except Exception as e:
        logger.exception("Failed in clean_numeric_vector")
        raise DataCleaningError(f"Vectorized cleaning failed: {e}") from e


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
        logger.exception("Failed to detect outliers")
        raise DataCleaningError(f"Outlier detection failed: {e}") from e


def handle_outliers(series: pd.Series, method: str = 'iqr', 
                    action: str = 'flag', **kwargs) -> pd.Series:
    """
    Handle outliers in numeric series.
    
    Parameters:
        series: Input numeric series
        method: Detection method ('iqr' or 'zscore')
        action: How to handle outliers ('flag', 'remove' (set to NaN), 'winsorize', 'cap')
        **kwargs: Additional arguments for detect_outliers
    
    Returns:
        pd.Series: Series with outliers handled
    """
    try:
        # Ensure input is a Series
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Clean the series first
        cleaned = clean_numeric_vector(series)
        
        # Detect outliers
        outlier_mask, stats = detect_outliers(series, method=method, **kwargs)
        if not stats:
            logger.info("No outliers detected (empty or non-numeric series)")
            return cleaned
        
        if action == 'flag':
            # Flag outliers with NaN
            result = cleaned.copy()
            result[outlier_mask] = np.nan
            logger.info(f"Flagged {stats['outlier_count']} outliers as NaN")
            
        elif action == 'remove':
            # Remove outliers (return with NaN)
            result = cleaned.copy()
            result[outlier_mask] = np.nan
            logger.info(f"Removed {stats['outlier_count']} outliers (set to NaN)")
            
        elif action == 'winsorize':
            # Winsorize outliers to boundary values
            result = cleaned.copy().astype(float)
            
            if stats['method'] == 'iqr':
                lower_bound = stats['lower_bound']
                upper_bound = stats['upper_bound']
                result = result.clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Winsorized {stats['outlier_count']} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
            else:
                logger.warning("Winsorizing only supported with IQR method")
                
        elif action == 'cap':
            # Cap outliers to nearest non-outlier value
            result = cleaned.copy().astype(float)
            
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
        logger.exception("Failed to handle outliers")
        raise DataCleaningError(f"Outlier handling failed: {e}") from e


def robust_sort_key(x: Any) -> tuple:
    """
    Sort key placing numeric values first, then strings, then NA.
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
        except (ValueError, TypeError, AttributeError):
            # Numeric conversion failed, treat as non-continuous
            logger.debug("Could not determine decimal ratio for series")
        
        return False
        
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning("Failed to determine if variable is continuous: %s", e)
        return False


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality validation.
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
        logger.exception("Data quality validation failed")
        raise DataValidationError(f"Validation failed: {e}") from e


def clean_dataframe(df: pd.DataFrame, 
                    handle_outliers_flag: bool = False,
                    outlier_method: str = 'iqr',
                    outlier_action: str = 'flag',
                    validate_quality: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive data cleaning for entire DataFrame.
    
    This function:
    1. Validates input data
    2. Cleans all numeric columns (handling individual cell errors)
    3. Handles outliers if requested
    4. Validates data quality if requested
    5. Returns cleaned DataFrame and cleaning report
    
    CRITICAL: Original DataFrame is NEVER modified. A copy is returned.
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
        
        # Create a copy for cleaning
        df_cleaned = df_validated.copy()
        logger.info(f"Created copy for cleaning: {df_cleaned.shape}")
        
        # Use lower threshold to be more aggressive in finding numeric columns
        # Default lowered to 30% to support granular cleaning of dirty data
        numeric_threshold = CONFIG.get('analysis.numeric_conversion_threshold', 0.30)

        # Clean each column
        for col in df_cleaned.columns:
            col_report = {
                'original_dtype': str(df_cleaned[col].dtype),
                'actions': []
            }
            
            try:
                # Try to clean as numeric if it's object/category
                if (
                    pd.api.types.is_object_dtype(df_cleaned[col])
                    or pd.api.types.is_string_dtype(df_cleaned[col])
                    or pd.api.types.is_categorical_dtype(df_cleaned[col])
                ):
                    original_na = df_cleaned[col].isna().sum()
                    
                    # Clean numeric values (this sets problematic cells to NaN)
                    cleaned_col = clean_numeric_vector(df_cleaned[col])
                    
                    # Check if enough values are numeric
                    non_na_count = cleaned_col.notna().sum()
                    total_count = len(cleaned_col)
                    non_na_ratio = non_na_count / total_count if total_count > 0 else 0
                    
                    # UPDATED LOGIC: Be more permissive (threshold > 0.3)
                    # If the column has at least 30% valid numbers, assume it is numeric
                    # and treat the rest as missing data (NaN) instead of rejecting the whole column.
                    if non_na_ratio > numeric_threshold:  
                        new_na = cleaned_col.isna().sum()
                        coerced_na = new_na - original_na
                        
                        df_cleaned[col] = cleaned_col.values
                        
                        action_msg = f"Converted to numeric ({non_na_ratio*100:.1f}% valid)"
                        if coerced_na > 0:
                            action_msg += f", set {coerced_na} problematic cells to NaN"
                            
                        col_report['actions'].append(action_msg)
                        col_report['new_dtype'] = 'float64'
                    else:
                        col_report['actions'].append("Kept as object (insufficient numeric signal)")
                
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
        raise DataCleaningError(f"Failed to clean DataFrame: {e}") from e


def get_cleaning_summary(report: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the cleaning report.
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


# ============================================================
# Missing Data Management Utilities
# ============================================================

def apply_missing_values_to_df(
    df: pd.DataFrame,
    var_meta: Dict[str, Any],
    missing_codes: Optional[Union[List[Any], Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Replace user-specified missing value codes with NaN.
    
    Parameters:
        df: Input DataFrame
        var_meta: Variable metadata with 'missing_values' per variable
        missing_codes: Global missing value codes (list) or per-column (dict)
    
    Returns:
        DataFrame with missing codes replaced by NaN
    """
    df_copy = df.copy()
    var_meta = var_meta or {}
    
    for col in df_copy.columns:
        # Get variable-specific missing values from metadata
        if col in var_meta and 'missing_values' in var_meta[col]:
            missing_vals = var_meta[col]['missing_values']
        elif isinstance(missing_codes, dict):
            missing_vals = missing_codes.get(col, [])
        else:
            missing_vals = missing_codes or []
        
        if not missing_vals:
            continue
            
        # Ensure missing_vals is a list for replace
        if not isinstance(missing_vals, (list, tuple, np.ndarray)):
            missing_vals = [missing_vals]
            
        # Replace all missing codes in one pass
        df_copy[col] = df_copy[col].replace(missing_vals, np.nan)
        # Also try numeric conversion for string comparisons
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            for val in missing_vals:
                try:
                    df_copy[col] = df_copy[col].replace(float(val), np.nan)
                except (ValueError, TypeError):
                    pass
    
    logger.debug(f"Applied missing value codes to {len(df_copy.columns)} columns")
    return df_copy


def detect_missing_in_variable(
    series: pd.Series,
    missing_codes: Optional[List[Any]] = None,
    already_normalized: bool = False
) -> Dict[str, Any]:
    """
    Detect and count missing values in a single variable.
    
    Parameters:
        series: Input pandas Series
        missing_codes: Optional list of values to treat as missing
        already_normalized: If True, assume coded values are already converted to NaN
    
    Returns:
        Dictionary with missing value statistics
    """
    total_count = len(series)
    
    # Count entries that are already standard NaN
    missing_nan_count = int(series.isna().sum())
    
    # Count user-specified missing codes
    missing_coded_count = 0
    if missing_codes and not already_normalized:
        for code in missing_codes:
            # Count occurrences of this code (only if not already NaN)
            count = int((series == code).sum())
            missing_coded_count += count
    
    # Total missing = NaN + coded missing
    # If already_normalized=True, coded missing will be 0 and NaN will include them
    total_missing = missing_nan_count + missing_coded_count
    
    missing_pct = (total_missing / total_count * 100) if total_count > 0 else 0.0
    
    return {
        'total_count': total_count,
        'missing_count': total_missing,
        'missing_pct': round(missing_pct, 2),
        'missing_coded_count': missing_coded_count,
        'missing_nan_count': missing_nan_count,
        'valid_count': total_count - total_missing
    }


def get_missing_summary_df(
    df: pd.DataFrame,
    var_meta: Dict[str, Any],
    missing_codes: Optional[List[Any]] = None,
    already_normalized: bool = False
) -> pd.DataFrame:
    """
    Generate summary table of missing data for all variables.
    
    Parameters:
        df: Input DataFrame
        var_meta: Variable metadata dictionary
        missing_codes: Global missing value codes (fallback)
        already_normalized: If True, assume coded values are already converted to NaN
    
    Returns:
        DataFrame with columns: Variable, Type, N_Total, N_Valid, N_Missing, Pct_Missing
    """
    summary_data = []
    
    for col in df.columns:
        original_series = df[col]
        
        # Get variable-specific missing codes
        if col in var_meta and 'missing_values' in var_meta[col]:
            var_missing_codes = var_meta[col]['missing_values']
        else:
            var_missing_codes = missing_codes or []
        
        # Detect missing values
        missing_info = detect_missing_in_variable(
            original_series, 
            var_missing_codes,
            already_normalized=already_normalized
        )
        
        # Get variable type from metadata
        var_type = var_meta.get(col, {}).get('type', 'Unknown')
        
        summary_data.append({
            'Variable': col,
            'Type': var_type,
            'N_Total': missing_info['total_count'],
            'N_Valid': missing_info['valid_count'],
            'N_Missing': missing_info['missing_count'],
            'Pct_Missing': f"{missing_info['missing_pct']:.1f}%"
        })
    
    return pd.DataFrame(summary_data)


def handle_missing_for_analysis(
    df: pd.DataFrame,
    var_meta: Dict[str, Any],
    missing_codes: Optional[List[Any]] = None,
    strategy: str = "complete-case",
    return_counts: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Apply missing data handling strategy.
    
    Parameters:
        df: Input DataFrame
        var_meta: Variable metadata
        missing_codes: Missing value codes to apply
        strategy: 'complete-case' (drop rows with any NaN) or 'drop' (keep only complete variables)
        return_counts: If True, also return before/after counts
    
    Returns:
        Cleaned DataFrame (+ counts dict if return_counts=True)
    """
    # Step 1: Apply missing codes → NaN
    df_processed = apply_missing_values_to_df(df, var_meta, missing_codes)
    
    original_rows = len(df_processed)
    
    # Step 2: Apply strategy
    if strategy == "complete-case":
        # Drop rows with any NaN
        df_clean = df_processed.dropna()
    elif strategy == "drop":
        # Drop columns with any NaN (keep only complete variables)
        df_clean = df_processed.dropna(axis=1)
    else:
        raise ValueError(f"Unknown missing data strategy: '{strategy}'. "
                         f"Supported strategies: 'complete-case', 'drop'")
    
    rows_removed = original_rows - len(df_clean)
    
    logger.info(f"Missing data handling: {original_rows} → {len(df_clean)} rows "
                f"({rows_removed} removed, strategy='{strategy}')")
    
    if return_counts:
        counts = {
            'original_rows': original_rows,
            'final_rows': len(df_clean),
            'rows_removed': rows_removed,
            'pct_removed': round((rows_removed / original_rows * 100) if original_rows > 0 else 0, 2)
        }
        return df_clean, counts
    
    return df_clean


def check_missing_data_impact(
    df_original: pd.DataFrame,
    df_clean: pd.DataFrame,
    var_meta: Dict[str, Any],
    missing_codes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compare before/after to report impact of missing data handling.
    
    Provides accurate counts even if source data contained coded missing values 
    (e.g., -99, 999) by normalizing internally if missing_codes is provided.
    
    Parameters:
        df_original: Original DataFrame (before cleaning)
        df_clean: Cleaned DataFrame (after handling)
        var_meta: Variable metadata
        missing_codes: Optional dictionary mapping variable names to missing values
    
    Returns:
        Dictionary with impact statistics
    """
    # Normalize if missing codes provided to ensure accurate counting
    if missing_codes:
        df_orig_norm = apply_missing_values_to_df(df_original, var_meta, missing_codes)
        df_clean_norm = apply_missing_values_to_df(df_clean, var_meta, missing_codes)
    else:
        df_orig_norm = df_original
        df_clean_norm = df_clean

    rows_removed = len(df_orig_norm) - len(df_clean_norm)
    pct_removed = (rows_removed / len(df_orig_norm) * 100) if len(df_orig_norm) > 0 else 0
    
    # Find which variables had missing data  
    variables_affected = []
    observations_lost = {}
    
    for col in df_orig_norm.columns:
        if col not in df_clean_norm.columns:
            continue
            
        original_missing = df_orig_norm[col].isna().sum()
        
        if original_missing > 0:
            var_label = var_meta.get(col, {}).get('label', col)
            variables_affected.append(col)
            observations_lost[col] = {
                'label': var_label,
                'count': int(original_missing),
                'pct': round((original_missing / len(df_orig_norm) * 100), 1)
            }
    
    return {
        'rows_removed': rows_removed,
        'pct_removed': round(pct_removed, 2),
        'variables_affected': variables_affected,
        'observations_lost': observations_lost
    }


# Convenience functions for common operations

def quick_clean_numeric(series: Union[pd.Series, np.ndarray, List[Any]]) -> pd.Series:
    """
    Quick numeric cleaning without extensive validation.
    """
    return clean_numeric_vector(series)


def quick_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick DataFrame cleaning without extensive validation or outlier handling.
    """
    cleaned, _ = clean_dataframe(
        df, 
        handle_outliers_flag=False, 
        validate_quality=False
    )
    return cleaned


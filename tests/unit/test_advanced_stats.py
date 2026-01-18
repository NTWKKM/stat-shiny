import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pytest
import pandas as pd
import numpy as np
from utils.advanced_stats_lib import apply_mcc, calculate_vif

# --- MCC Tests ---

def test_apply_mcc_bonferroni():
    """Test Bonferroni correction with known values."""
    p_vals = [0.01, 0.04, 0.1, 0.001]
    # Bonferroni: p * n, capped at 1.0
    # Expected: [0.04, 0.16, 0.4, 0.004] (approx)
    corrected = apply_mcc(p_vals, method='bonferroni')
    
    assert len(corrected) == 4
    assert np.isclose(corrected[0], 0.04)
    assert np.isclose(corrected[3], 0.004)
    assert all(corrected <= 1.0)

def test_apply_mcc_fdr_bh():
    """Test Benjamini-Hochberg correction."""
    # Example from a textbook or standard
    p_vals = [0.001, 0.01, 0.03, 0.05, 1.0]
    corrected = apply_mcc(p_vals, method='fdr_bh')
    
    # BH preserves order
    assert corrected[0] < corrected[1]
    assert corrected[0] >= p_vals[0] # adjusted p is generally >= raw p
    assert len(corrected) == 5

def test_apply_mcc_nans():
    """Test MCC handles NaNs gracefully."""
    p_vals = [0.01, np.nan, 0.05]
    corrected = apply_mcc(p_vals, method='bonferroni')
    
    assert len(corrected) == 3
    assert np.isnan(corrected[1])
    assert not np.isnan(corrected[0])
    # Correction factor should be based on valid p-values count? 
    # statsmodels default behavior ignores NaNs. If 2 valid, p*2.
    # 0.01 * 2 = 0.02
    assert np.isclose(corrected[0], 0.02) 

def test_apply_mcc_empty():
    """Test empty input."""
    assert apply_mcc([]).empty
    assert apply_mcc(None).empty

# --- VIF Tests ---

def test_calculate_vif_independent():
    """Test VIF with independent variables (should be low ~1)."""
    np.random.seed(42)
    df = pd.DataFrame({
        'a': np.random.normal(0, 1, 100),
        'b': np.random.normal(0, 1, 100),
        'c': np.random.normal(0, 1, 100)
    })
    
    vif_df, _ = calculate_vif(df)
    # VIFs should be close to 1 for independent comparisons
    assert not vif_df.empty
    assert all(vif_df['VIF'] < 5)

def test_calculate_vif_collinear():
    """Test VIF with highly collinear variables."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    df = pd.DataFrame({
        'x': x,
        'y': x * 2 + np.random.normal(0, 0.001, 100) # y is almost perfectly correlated with x
    })
    
    vif_df, _ = calculate_vif(df)
    # VIF should be very high
    assert vif_df[vif_df['feature'] == 'x']['VIF'].iloc[0] > 10

def test_calculate_vif_constant_handling():
    """Test that VIF adds constant internally if needed."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [2, 4, 6, 8] # Perfect correlation
    })
    vif_df, _ = calculate_vif(df)
    assert not vif_df.empty
    # 'const' should be removed from result if it was added internally
    assert 'const' not in vif_df['feature'].values

def test_calculate_vif_with_nans():
    """Test VIF handles (drops) NaNs rows."""
    df = pd.DataFrame({
        'a': [1, 2, 3, np.nan],
        'b': [5, 6, 7, 8],
        'c': [9, 10, 11, 12]
    })
    # Should drop row 4
    # Should drop row 4
    vif_df, missing_info = calculate_vif(df)
    assert not vif_df.empty
    assert len(vif_df) == 3 # 3 features: a, b, c
    
    # Check missing info
    assert missing_info['strategy'] == 'complete-case'
    assert missing_info['initial_n'] == 4
    assert missing_info['final_n'] == 3
    assert missing_info['excluded_n'] == 1

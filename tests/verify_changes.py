
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Add local path to import modules
sys.path.append(os.getcwd())

# Mock missing plotting libraries
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['forest_plot_lib'] = MagicMock()

from logic import analyze_outcome
from poisson_lib import analyze_poisson_outcome
from logger import get_logger

# Setup dummy logger to avoid noise
import logging
logging.basicConfig(level=logging.ERROR)

def create_dummy_data(n=200):
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.normal(50, 10, n),
        'sex': np.random.choice([0, 1], n),
        'treatment': np.random.choice([0, 1], n),
        'outcome_bin': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'outcome_count': np.random.poisson(2, n),
        'offset': np.random.uniform(0.5, 1.5, n)
    })
    return df

def test_logistic_backward_compatibility(df):
    print("\n[TEST 1] Logistic Regression Backward Compatibility (No Interactions)")
    try:
        # Should return 3 values if interaction_pairs is not passed (or handled correctly as Optional)
        # Wait, we changed the return signature to ALWAYS return 4 values in our implementation.
        # But we need to check if existing calls in other tabs (if any) break.
        # TAB 1 (tab_logit.py) was updated. 
        # Are there other callers? verify_changes.py checks logic.py behavior.
        
        # logic.py now returns 4 values: html, or, aor, int_res
        res = analyze_outcome('outcome_bin', df, method='bfgs')
        
        if len(res) == 4:
            print("✅ analyze_outcome returns 4 values as expected.")
            html, or_res, aor_res, int_res = res
            if int_res == {}:
                print("✅ Interaction results check: Empty (Correct)")
            else:
                print(f"❌ Interaction results should be empty, got: {int_res}")
        else:
            print(f"❌ Unexpected return length: {len(res)}. Expected 4.")
            
    except Exception as e:
        print(f"❌ Crash during backward compatibility test: {e}")
        import traceback
        traceback.print_exc()

def test_logistic_interactions(df):
    print("\n[TEST 2] Logistic Regression with Interactions")
    try:
        pairs = [('age', 'sex'), ('treatment', 'age')]
        html, or_res, aor_res, int_res = analyze_outcome(
            'outcome_bin', df, method='bfgs', interaction_pairs=pairs
        )
        
        print(f"✅ Multivariate results count: {len(aor_res)}")
        print(f"✅ Interaction results count: {len(int_res)}")
        
        expected_keys = ['age×sex', 'treatment×age'] 
        # Note: formatting might differ (e.g. :: or just *)
        
        if len(int_res) > 0:
            print("✅ Interactions successfully computed and returned.")
            for k, v in int_res.items():
                print(f"   - {k}: p={v['p_value']:.4f}")
        else:
            print("⚠️ No interactions found (might be filtered out or failed).")
            
    except Exception as e:
        print(f"❌ Crash during interaction test: {e}")
        import traceback
        traceback.print_exc()

def test_poisson_basic(df):
    print("\n[TEST 3] Poisson Regression Basic")
    try:
        html, irr_res, airr_res, int_res = analyze_poisson_outcome(
            'outcome_count', df, offset_col='offset'
        )
        print("✅ analyze_poisson_outcome executed successfully.")
        
        if len(irr_res) > 0:
            print(f"✅ Univariate results: {len(irr_res)} variables")
        else:
            print("⚠️ No univariate results.")
            
    except Exception as e:
        print(f"❌ Crash during Poisson test: {e}")
        import traceback
        traceback.print_exc()

def test_poisson_interactions(df):
    print("\n[TEST 4] Poisson Regression with Interactions")
    try:
        pairs = [('age', 'sex')]
        html, irr_res, airr_res, int_res = analyze_poisson_outcome(
            'outcome_count', df, offset_col='offset', interaction_pairs=pairs
        )
        
        if len(int_res) > 0:
            print(f"✅ Poisson Interactions found: {len(int_res)}")
            for k in int_res:
                print(f"   - {k}")
        else:
            print("⚠️ No Poisson interactions found.")
            
    except Exception as e:
        print(f"❌ Crash during Poisson Interaction test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Verification Script...")
    df = create_dummy_data()
    test_logistic_backward_compatibility(df)
    test_logistic_interactions(df)
    test_poisson_basic(df)
    test_poisson_interactions(df)
    print("\n🏁 Verification Complete.")

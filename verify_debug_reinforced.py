import pandas as pd
import numpy as np
import logging
from utils.tvc_lib import fit_tvc_cox

# Setup basic config for logger to see output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils.tvc_lib")

def test_reinforced_robustness():
    print("\n--- Testing Reinforced Robustness (Time_Months mismatch) ---")
    
    n = 20
    # Simulate data coming from transform_wide_to_long (standardized names)
    df = pd.DataFrame({
        'ID': range(n),
        'start': np.random.uniform(0, 50, n),  # Standardized name
        'stop': np.random.uniform(60, 100, n), # Standardized name
        'Status_Death': np.random.binomial(1, 0.3, n),
        'TVC_Lab': np.random.normal(0, 1, n)
    })
    
    # User tries to fit using 'Time_Months' (original UI selection)
    # This mismatches 'start'/'stop' in df
    print("Running fit_tvc_cox with start_col='Time_Months' but df has 'start'...")
    
    cph, res, clean, err, stats, missing = fit_tvc_cox(
        df, 
        start_col='Time_Months', # Expect auto-fallback to 'start'
        stop_col='Time_Months',  # Expect auto-fallback to 'stop' (assuming mismatch) 
        # Wait, usually start/stop derived from same var? Or separate? 
        # Doesn't matter, just testing mismatch logic.
        event_col='Status_Death',
        tvc_cols=['TVC_Lab']
    )
    
    if err:
        print(f"FAILED with error: {err}")
    else:
        print("SUCCESS: Model fitted despite column name mismatch!")
        print(f"Clean columns: {clean.columns.tolist()}")
        if 'start' in clean.columns:
             print("Validation: 'start' column exists.")
        else:
             print("Validation: 'start' column MISSING (Bad!)")

if __name__ == "__main__":
    test_reinforced_robustness()

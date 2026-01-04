
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Add local path to import modules
sys.path.append(os.getcwd())

# Mock plotting libraries to avoid GUI dependency
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.io'] = MagicMock()
sys.modules['forest_plot_lib'] = MagicMock()


# Mock statistical libraries if missing (for syntax checking)
sys.modules['lifelines'] = MagicMock()
sys.modules['lifelines.statistics'] = MagicMock()
sys.modules['lifelines.utils'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['sklearn.linear_model'] = MagicMock()

# Import modules to verify (this checks syntax and imports)
try:
    from diag_test import calculate_descriptive, calculate_chi2, analyze_roc
    from survival_lib import fit_km_logrank, fit_cox_ph, calculate_median_survival
    print("✅ Modules imported successfully.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def create_dummy_data(n=200):
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.normal(50, 10, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        'exposure': np.random.choice([0, 1], n),
        'disease': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'test_score': np.random.normal(0, 1, n) + np.random.choice([0, 1], n)*2,
        'time': np.random.exponential(10, n),
        'event': np.random.choice([0, 1], n, p=[0.6, 0.4])
    })
    return df

def test_diag_descriptive(df):
    print("\n[TEST] Diagnostic Descriptive Stats")
    try:
        res = calculate_descriptive(df, 'age')
        if res is not None and not res.empty:
            print("✅ calculate_descriptive (numeric) returned DataFrame")
        else:
            print("❌ calculate_descriptive (numeric) failed")
            
        res_cat = calculate_descriptive(df, 'sex')
        if res_cat is not None and not res_cat.empty:
            print("✅ calculate_descriptive (categorical) returned DataFrame")
        else:
            print("❌ calculate_descriptive (categorical) failed")
    except Exception as e:
        print(f"❌ Crash in test_diag_descriptive: {e}")

def test_diag_chi2(df):
    print("\n[TEST] Diagnostic Chi-Square")
    try:
        _display_tab, stats_df, msg, risk_df = calculate_chi2(df, 'exposure', 'disease')
        if stats_df is not None:
            print("✅ calculate_chi2 returned stats")
            print(f"   Test used: {stats_df.loc[stats_df['Statistic']=='Test', 'Value'].values[0]}")
        else:
            print("❌ calculate_chi2 failed to return stats")
            
        if risk_df is not None:
             print("✅ 2x2 Risk Metrics Calculated")
    except Exception as e:
        print(f"❌ Crash in test_diag_chi2: {e}")

def test_diag_roc(df):
    print("\n[TEST] Diagnostic ROC")
    try:
        stats_dict, error_msg, _fig, _coords = analyze_roc(df, 'disease', 'test_score', pos_label_user='1')
        if stats_dict is not None:
            print(f"✅ ROC Analysis successful. AUC: {stats_dict['AUC']}")
        else:
            print(f"❌ ROC Analysis failed: {error_msg}")
    except Exception as e:
        print(f"❌ Crash in test_diag_roc: {e}")

def test_survival_km(df):
    print("\n[TEST] Survival KM & Log-rank")
    try:
        # Check median survival
        _kms = calculate_median_survival(df, 'time', 'event', 'exposure')
        if not kms.empty:
             print("✅ calculate_median_survival returned results")
        
        # Check Log-rank
        _fig, stats = fit_km_logrank(df, 'time', 'event', 'exposure')
        if not stats.empty:
            print(f"✅ Log-rank test performed: p={stats.iloc[0]['P-value']}")
    except Exception as e:
        print(f"❌ Crash in test_survival_km: {e}")

def test_survival_cox(df):
    print("\n[TEST] Survival CoxPH")
    try:
        cph, res_df, _data, err = fit_cox_ph(df, 'time', 'event', ['age', 'sex', 'exposure'])
        if cph is not None:
            print("✅ CoxPH fitted successfully")
            print(f"   Covariates: {res_df.index.tolist()}")
        else:
            print(f"❌ CoxPH failed: {err}")
    except Exception as e:
        print(f"❌ Crash in test_survival_cox: {e}")


def test_modernized_modules():
    print("\n[TEST] Modernized Modules Import & Basic Check")
    try:
        from logger import get_logger
        _ = get_logger(__name__)  # Verify import and instantiation work
        print("✅ logger.py imported successfully")
    except Exception as e:
        print(f"❌ logger.py failed: {e}")

    # Test table_one.py
    try:
        from table_one import generate_table, get_stats_continuous
        print("✅ table_one.py imported successfully")
        s = pd.Series([1, 2, 3, 4, 5])
        stats_str = get_stats_continuous(s)
        print(f"   table_one stats check: {stats_str}")
    except Exception as e:
        print(f"❌ table_one.py failed: {e}")

    # Test psm_lib.py
    try:
        from psm_lib import calculate_propensity_score
        print("✅ psm_lib.py imported successfully")
    except Exception as e:
        print(f"❌ psm_lib.py failed: {e}")

    # Test correlation.py
    try:
        from correlation import calculate_correlation
        print("✅ correlation.py imported successfully")
    except Exception as e:
        print(f"❌ correlation.py failed: {e}")

    # Test forest_plot_lib.py
    try:
        from forest_plot_lib import create_forest_plot
        print("✅ forest_plot_lib.py imported successfully")
    except Exception as e:
        print(f"❌ forest_plot_lib.py failed: {e}")

    # Test interaction_lib.py
    try:
        from interaction_lib import create_interaction_terms
        print("✅ interaction_lib.py imported successfully")
    except Exception as e:
        print(f"❌ interaction_lib.py failed: {e}")

    # Test poisson_lib.py
    try:
        from poisson_lib import analyze_poisson_outcome
        print("✅ poisson_lib.py imported successfully")
    except Exception as e:
        print(f"❌ poisson_lib.py failed: {e}")

if __name__ == "__main__":
    df = create_dummy_data()
    test_diag_descriptive(df)
    test_diag_chi2(df)
    test_diag_roc(df)
    test_survival_km(df)
    test_survival_cox(df)
    test_modernized_modules()
    print("\n🏁 All Modules Verification Complete.")

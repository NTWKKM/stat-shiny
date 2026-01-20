
import numpy as np
import pandas as pd
import pytest

from utils.poisson_lib import analyze_poisson_outcome


@pytest.fixture
def poisson_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.normal(50, 10, n),
        "treatment": np.random.choice([0, 1], n),
        "offset_col": np.random.exponential(1, n)
    })
    # Poisson process
    mu = np.exp(0.01 * df["age"] + 0.5 * df["treatment"])
    df["visits"] = np.random.poisson(mu * df["offset_col"])
    return df

@pytest.fixture
def nb_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "x": np.random.normal(0, 1, n),
        "group": np.random.choice(["A", "B"], n)
    })
    # Negative Binomial process (Overdispersed)
    mu = np.exp(0.5 * df["x"] + np.where(df["group"] == "B", 0.5, 0))
    # Gamma mixture for overdispersion
    gamma_noise = np.random.gamma(shape=1/0.5, scale=0.5, size=n) # alpha approx 0.5
    df["events"] = np.random.poisson(mu * gamma_noise)
    return df

def test_analyze_poisson_model(poisson_data):
    """Test standard Poisson regression."""
    html_rep, irr, airr, int_res = analyze_poisson_outcome(
        outcome_name="visits",
        df=poisson_data,
        var_meta={},
        model_type="poisson"
    )
    
    # Check outputs
    assert irr is not None
    assert airr is not None
    
    # Handle variable naming (linear vs categorical)
    if "treatment" in irr:
        est = irr["treatment"]["irr"]
    elif "treatment: 1" in irr:
        est = irr["treatment: 1"]["irr"]
    else:
        pytest.fail(f"Treatment effect not found. Keys: {list(irr.keys())}")
        
    # IRR for treatment should be around exp(0.5) = 1.64
    assert 1.0 < est < 2.5

def test_analyze_nb_model(nb_data):
    """Test Negative Binomial regression."""
    html_rep, irr, airr, int_res = analyze_poisson_outcome(
        outcome_name="events",
        df=nb_data,
        var_meta={},
        model_type="negative_binomial"
    )
    
    assert irr is not None
    # Check if results exist for 'x' (linear) or 'group' (categorical)
    assert "x" in irr
    # group has levels A, B. B should be compared to A.
    assert "group: B" in irr or "group" in irr
    
def test_offset_handling(poisson_data):
    """Test regression with offset column."""
    # Run with offset
    html_rep, irr, _, _ = analyze_poisson_outcome(
        outcome_name="visits",
        df=poisson_data,
        var_meta={},
        offset_col="offset_col",
        model_type="poisson"
    )
    
    # Run without offset (should differ)
    html_rep_no, irr_no, _, _ = analyze_poisson_outcome(
        outcome_name="visits",
        df=poisson_data,
        var_meta={},
        offset_col=None,
        model_type="poisson"
    )
    
    # Estimates should differ significantly if offset matters
    assert abs(irr["age"]["irr"] - irr_no["age"]["irr"]) > 0.0001

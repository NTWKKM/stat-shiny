import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# We'll mock the entire survival_lib module context primarily to test the *patch* logic.
# However, to test the patch logic itself, we need to let it execute.
# Since we might not have firthmodels installed in this environment, we'll simulate the import.


@pytest.fixture
def mock_firth_module():
    """Mock firthmodels module to simulate its presence and behavior."""
    mock_fm = MagicMock()
    mock_class = MagicMock()

    # Simulate missing _validate_data to trigger the patch
    try:
        del mock_class._validate_data
    except AttributeError:
        pass

    mock_fm.FirthCoxPH = mock_class

    with patch.dict(sys.modules, {"firthmodels": mock_fm}):
        yield mock_fm


def test_sklearn_1_6_patch_application(mock_firth_module):
    """Test that _validate_data is patched onto FirthCoxPH if missing."""
    import importlib

    import utils.survival_lib

    # The mock_firth_module fixture puts a mock firthmodels in sys.modules.
    # It specifically deletes _validate_data from FirthCoxPH.
    # We reload utils.survival_lib to trigger the top-level try/except block.

    # Reload to run the patch logic
    importlib.reload(utils.survival_lib)

    # Now verify that the mock class has the patched method
    from firthmodels import FirthCoxPH

    assert hasattr(FirthCoxPH, "_validate_data"), (
        "Patch failed: _validate_data not added to FirthCoxPH"
    )
    assert callable(FirthCoxPH._validate_data), (
        "_validate_data should be a callable method"
    )


def test_fit_firth_cox_logic():
    """Test the data preparation and calling convention of _fit_firth_cox."""
    from utils.survival_lib import _fit_firth_cox

    # Create dummy data
    df = pd.DataFrame(
        {"time": [10, 20, 30], "event": [1, 0, 1], "var1": [1.0, 2.0, 3.0]}
    )

    # Mock FirthCoxPH class specifically within survival_lib context if possible,
    # or patch where it's used.
    with patch("utils.survival_lib.FirthCoxPH") as MockFirth:
        mock_instance = MockFirth.return_value
        mock_instance.coef_ = np.array([0.5])
        mock_instance.bse_ = np.array([0.1])
        mock_instance.pvalues_ = np.array([0.05])

        # Run function
        model, res, method = _fit_firth_cox(df, "time", "event", ["var1"])

        # Check call args
        # Should be called with X (2d array) and y (tuple of arrays)
        args, kwargs = mock_instance.fit.call_args
        X_arg = args[0]
        y_arg = args[1]

        assert isinstance(X_arg, np.ndarray)
        assert isinstance(y_arg, tuple)
        assert len(y_arg) == 2
        # event, time
        np.testing.assert_array_equal(y_arg[0], df["event"].astype(bool).values)
        np.testing.assert_array_equal(y_arg[1], df["time"].astype(float).values)

        # Check conversion to floats for X
        assert X_arg.dtype == float

        # Check results
        assert "HR" in res.columns
        assert res.iloc[0]["HR"] == np.exp(0.5)


def test_fit_cox_ph_aic_calc():
    """Test AIC extraction logic in fit_cox_ph for Firth models."""
    import utils.survival_lib

    # Mock prepare_data_for_analysis locally to avoid complex deps
    with patch("utils.survival_lib.prepare_data_for_analysis") as mock_prep:
        # Return valid data with variance
        df = pd.DataFrame({"time": [10, 20], "event": [1, 0], "v": [0, 1]})
        mock_prep.return_value = (df, {})

        # Mock _fit_firth_cox to return a mock model with loglik_
        with patch("utils.survival_lib._fit_firth_cox") as mock_fit_firth:
            mock_model = MagicMock()
            mock_model.loglik_ = -10.0  # Log likelihood
            mock_model.score.return_value = 0.7  # C-index

            mock_fit_firth.return_value = (
                mock_model,
                pd.DataFrame({"p": [0.05]}),  # res_df
                "Firth",
            )

            # Mock boolean check safely (since we modified it in code)
            with patch("utils.survival_lib.HAS_FIRTH_COX", True):
                model, res, _, _, stats, _ = utils.survival_lib.fit_cox_ph(
                    df, "time", "event", ["v"], method="firth"
                )

                # AIC = 2k - 2ln(L)
                # k = 1 (1 covariate)
                # L = -10
                # AIC = 2(1) - 2(-10) = 2 + 20 = 22

                assert stats is not None
                assert "AIC" in stats
                assert float(stats["AIC"]) == 22.0
                assert "Log-Likelihood" in stats
                assert float(stats["Log-Likelihood"]) == -10.0

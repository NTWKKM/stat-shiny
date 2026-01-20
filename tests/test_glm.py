import numpy as np
import pandas as pd

from utils import logic


def test_run_glm_gaussian_identity():
    # Simple linear regression case
    np.random.seed(42)
    X = pd.DataFrame({"x": np.linspace(0, 10, 20)})
    y = 2 * X["x"] + 1 + np.random.normal(0, 0.5, 20)

    params, ci, pval, status, metrics = logic.run_glm(
        y, X, family_name="Gaussian", link_name="identity"
    )

    assert status == "OK"
    assert params is not None
    # Intercept ~ 1, x ~ 2
    assert abs(params["x"] - 2) < 0.2
    assert abs(params["const"] - 1) < 0.5


def test_run_glm_poisson_log():
    # Poisson case
    np.random.seed(42)
    X = pd.DataFrame({"x": np.linspace(0, 1, 50)})
    # log(mu) = 0.5 * x + 1  => mu = exp(0.5*x + 1)
    mu = np.exp(0.5 * X["x"] + 1)
    y = np.random.poisson(mu)

    params, ci, pval, status, metrics = logic.run_glm(
        pd.Series(y), X, family_name="Poisson", link_name="log"
    )

    assert status == "OK"
    assert params is not None
    assert abs(params["x"] - 0.5) < 0.5


def test_run_glm_binomial_logit():
    # Logistic case
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"x": np.random.normal(0, 1, n)})
    # logit(p) = 2*x
    p = 1 / (1 + np.exp(-(2 * X["x"])))
    y = np.random.binomial(1, p)

    params, ci, pval, status, metrics = logic.run_glm(
        pd.Series(y), X, family_name="Binomial", link_name="logit"
    )

    assert status == "OK"
    assert params is not None
    # x coeff positive
    assert params["x"] > 0


def test_run_glm_invalid_inputs():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([1, 2])  # Mismatch length

    params, ci, pval, status, metrics = logic.run_glm(y, X)
    assert status != "OK"
    assert params is None

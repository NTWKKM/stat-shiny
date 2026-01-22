import numpy as np
import pandas as pd
from typing import Any
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats as sp_stats

from config import CONFIG
from logger import get_logger
from utils.data_cleaning import prepare_data_for_analysis
from utils.collinearity_lib import _run_vif

logger = get_logger(__name__)

def calculate_vif(data: pd.DataFrame, predictor_cols: list[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for predictors.
    Returns DataFrame with columns: Variable, VIF
    """
    return _run_vif(data, predictor_cols)

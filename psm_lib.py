"""
🧪 Propensity Score Matching (PSM) Library (Shiny Compatible)

Propensity score calculation and matching without Streamlit dependencies.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from logger import get_logger

logger = get_logger(__name__)


def calculate_propensity_score(df, treatment_col, covariate_cols):
    """
    Calculate propensity scores using logistic regression.
    
    Returns:
        pd.Series: Propensity scores (probability of treatment=1)
    """
    try:
        X = df[covariate_cols].fillna(df[covariate_cols].mean())
        y = df[treatment_col].astype(int)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        ps = model.predict_proba(X)[:, 1]
        logger.info(f"Propensity scores calculated: mean={ps.mean():.3f}, std={ps.std():.3f}")
        return pd.Series(ps, index=df.index)
    except Exception as e:
        logger.error(f"PSM calculation error: {e}")
        raise


def perform_matching(df, treatment_col, ps_col, caliper=0.1):
    """
    Perform 1:1 nearest neighbor matching on propensity scores.
    
    Returns:
        pd.DataFrame: Matched dataset
    """
    try:
        treated = df[df[treatment_col] == 1].copy()
        control = df[df[treatment_col] == 0].copy()
        
        matched_pairs = []
        used_controls = set()
        
        for idx, treated_row in treated.iterrows():
            ps_treated = treated_row[ps_col]
            
            distances = (control[ps_col] - ps_treated).abs()
            min_dist_idx = distances.idxmin()
            min_dist = distances[min_dist_idx]
            
            if min_dist <= caliper and min_dist_idx not in used_controls:
                matched_pairs.append(idx)
                matched_pairs.append(min_dist_idx)
                used_controls.add(min_dist_idx)
        
        matched_df = df.loc[matched_pairs].copy()
        logger.info(f"Matched {len(matched_pairs)//2} pairs (caliper={caliper})")
        return matched_df
    except Exception as e:
        logger.error(f"Matching error: {e}")
        raise

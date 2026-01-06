"""
🔗 Integration Tests for Interaction Analysis Pipeline
File: tests/integration/test_interaction_pipeline.py

Tests the Interaction Analysis workflow:
1. Creating interaction terms
2. Testing interaction improvement (Likelihood Ratio Test)
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# ✅ Correct Import based on interaction_lib.py
from interaction_lib import create_interaction_terms, test_interaction_improvement
from logic import run_binary_logit

pytestmark = pytest.mark.integration

class TestInteractionPipeline:

    @pytest.fixture
    def interaction_data(self):
        """Create dataset with known interaction effects"""
        np.random.seed(42)
        n = 200
        
        # Predictors
        age = np.random.normal(55, 10, n)
        treatment = np.random.choice([0, 1], n)
        
        # Interaction: Treatment works better if Age is lower
        # Log-odds = -1 + 0.5*Tx + 0.02*Age - 0.05*(Tx*Age)
        logit = -1 + 0.5*treatment + 0.02*age - 0.05 * (treatment * age)
        prob = 1 / (1 + np.exp(-logit))
        outcome = np.random.binomial(1, prob)
        
        df = pd.DataFrame({
            'outcome': outcome,
            'treatment': treatment,
            'age': age
        })
        return df

    def test_create_interaction_terms(self, interaction_data):
        """🔄 Test creation of interaction columns"""
        df = interaction_data
        
        # Create interaction between 'treatment' and 'age'
        pairs = [('treatment', 'age')]
        
        # Mode map (optional, inferred if None)
        df_int, meta = create_interaction_terms(df, pairs)
        
        assert not df_int.empty
        # Check if new column created (e.g. "treatment × age" or similar)
        new_cols = [c for c in df_int.columns if '×' in c or '*' in c or ':' in c]
        assert len(new_cols) > 0
        assert meta is not None

    def test_interaction_improvement_flow(self, interaction_data):
        """🔄 Test interaction improvement statistical check"""
        df = interaction_data
        
        # Check if adding interaction improves model
        result = test_interaction_improvement(
            df=df,
            outcome_col='outcome',
            base_predictors=['treatment', 'age'],
            interaction_pairs=[('treatment', 'age')],
            model_type='logistic'
        )
        
        assert result is not None
        assert 'p_value' in result
        assert 'significant' in result
        
        # Since we baked in an interaction, p-value should likely be significant (or low)
        assert isinstance(result['p_value'], float)

    def test_interaction_pipeline_integration(self, interaction_data):
        """
        🌲 End-to-End: Create Terms -> Run Model
        """
        df = interaction_data
        pairs = [('treatment', 'age')]
        
        # 1. Create terms
        df_model, _ = create_interaction_terms(df, pairs)
        
        # 2. Identify interaction column
        int_col = [c for c in df_model.columns if c not in df.columns][0]
        
        # 3. Run Logistic Regression including interaction
        y = df_model['outcome']
        X = df_model[['treatment', 'age', int_col]]
        
        params, conf, pvals, status, metrics = run_binary_logit(y, X)
        
        assert status == "OK"
        assert int_col in params.index
        # Check if model ran successfully
        assert metrics['ll'] != 0

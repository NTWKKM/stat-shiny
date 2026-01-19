"""
🔗 Integration Tests for Table One Generator Pipeline
File: tests/integration/test_table_one_pipeline.py

Tests the flow of generating baseline characteristics tables:
1. Statistical testing (T-test, Chi-square, Fisher)
2. SMD calculation
3. HTML Table Generation
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.table_one import (
    calculate_p_categorical,
    calculate_p_continuous,
    calculate_smd,
    generate_table,
)

# Mark as integration test
pytestmark = pytest.mark.integration

class TestTableOnePipeline:
    
    @pytest.fixture
    def table_one_data(self):
        """Create realistic dataset for Table 1"""
        np.random.seed(101)
        n = 200
        
        df = pd.DataFrame({
            'age': np.random.normal(60, 10, n),
            'bmi': np.random.normal(25, 4, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'smoking': np.random.choice(['Never', 'Former', 'Current'], n),
            # Group variable (e.g., Treatment vs Control)
            'group': np.random.choice(['A', 'B'], n)
        })
        
        # Add some group differences to test P-values/SMD
        # Group A is older
        df.loc[df['group'] == 'A', 'age'] += 5
        
        return df

    def test_full_table_generation_html(self, table_one_data):
        """🔄 Test HTML generation for Table 1"""
        df = table_one_data
        selected_vars = ['age', 'bmi', 'gender', 'smoking']
        
        # Generate HTML
        html_output = generate_table(
            df, 
            selected_vars, 
            group_col='group', 
            var_meta={},
            or_style='all_levels'
        )
        
        assert isinstance(html_output, str)
        assert len(html_output) > 0
        assert "<table" in html_output
        assert "Baseline Characteristics" in html_output
        
        # Check if variable labels appear
        assert "age" in html_output
        assert "gender" in html_output
        
        # Check if stats columns appear
        assert "Total (N=" in html_output
        assert "P-value" in html_output
        assert "SMD" in html_output

    def test_statistical_calculations(self, table_one_data):
        """📊 Test specific statistical functions used in Table 1"""
        df = table_one_data
        
        # 1. Continuous P-value (Age by Group)
        group_a = df[df['group'] == 'A']['age']
        group_b = df[df['group'] == 'B']['age']
        p_val, test_name = calculate_p_continuous([group_a, group_b])
        
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1
        assert test_name in ['t-test', 'Mann-Whitney U']
        
        # 2. Categorical P-value (Gender by Group)
        p_cat, test_cat = calculate_p_categorical(df, 'gender', 'group')
        assert isinstance(p_cat, float)
        assert 0 <= p_cat <= 1
        assert 'Chi-square' in test_cat or 'Fisher' in test_cat

    def test_smd_calculation_flow(self, table_one_data):
        """📏 Test Standardized Mean Difference (SMD) calculation"""
        df = table_one_data
        
        # Continuous SMD
        smd_age = calculate_smd(
            df, 'age', 'group', 'A', 'B', 
            is_cat=False
        )
        assert smd_age != "-"
        # Check if it's bolded (if > 0.1) or plain text
        # Remove HTML tags for numerical check
        smd_val = float(smd_age.replace('<b>', '').replace('</b>', ''))
        assert smd_val >= 0

    def test_table_generation_no_group(self, table_one_data):
        """🚫 Test generation without grouping variable (Descriptive only)"""
        df = table_one_data
        selected_vars = ['age', 'bmi']
        
        html_output = generate_table(
            df, 
            selected_vars, 
            group_col=None, 
            var_meta={}
        )
        
        assert "<table" in html_output
        assert "Total (N=" in html_output
        # In current implementation, the P-value column might still be in the template
        # Just check that it's empty or doesn't contain actual significance indicators
        assert 'class="p-significant"' not in html_output

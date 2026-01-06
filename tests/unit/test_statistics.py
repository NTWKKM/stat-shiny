import pytest
import pandas as pd
import numpy as np
from utils.data_cleaning import clean_names
# from logic import logistic_regression_firth # (นำเข้าฟังก์ชันที่คุณจะเทส)

def test_clean_names_basic():
    """ทดสอบว่าฟังก์ชันล้างชื่อตัวแปรทำงานถูกไหม"""
    input_cols = ["Age (years)", "Patient-ID", "BMI%"]
    expected = ["age_years", "patient_id", "bmi_percent"] 
    # ปรับ expected ตาม Logic จริงของคุณ
    assert clean_names(input_cols) == expected

def test_dataframe_handling():
    """ทดสอบกับ DataFrame จริง"""
    df = pd.DataFrame({"A B": [1], "C-D": [2]})
    df.columns = clean_names(df.columns)
    assert "a_b" in df.columns
    assert "c_d" in df.columns

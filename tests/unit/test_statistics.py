import pytest
import pandas as pd
import numpy as np
import sys
import os

# เพิ่ม path ให้ Python หาไฟล์ logic.py เจอ
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from logic import validate_logit_data, clean_numeric_value, run_binary_logit

class TestLogicFunctions:

    def test_clean_numeric_value(self):
        """ทดสอบฟังก์ชันแปลงตัวเลข"""
        assert clean_numeric_value("1,000") == 1000.0
        assert clean_numeric_value("<0.001") == 0.001
        assert clean_numeric_value("> 50") == 50.0
        assert pd.isna(clean_numeric_value("abc"))
        assert pd.isna(clean_numeric_value(None))

    def test_validate_logit_data_success(self):
        """ทดสอบข้อมูลที่ถูกต้อง"""
        df = pd.DataFrame({'age': [20, 30, 40, 50], 'sex': [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])
        valid, msg = validate_logit_data(y, df)
        assert valid is True
        assert msg == "OK"

    def test_validate_logit_data_zero_variance(self):
        """ทดสอบกรณีตัวแปรไม่มีความแปรปรวน (ค่าเหมือนกันหมด)"""
        df = pd.DataFrame({'constant': [1, 1, 1, 1], 'age': [20, 30, 40, 50]})
        y = pd.Series([0, 1, 0, 1])
        valid, msg = validate_logit_data(y, df)
        assert valid is False
        assert "zero variance" in msg

    def test_validate_logit_data_empty(self):
        """ทดสอบกรณีข้อมูลว่าง"""
        df = pd.DataFrame()
        y = pd.Series([])
        valid, msg = validate_logit_data(y, df)
        assert valid is False
        assert "Empty data" in msg

    def test_run_binary_logit_basic(self):
        """ทดสอบการรัน Logistic Regression จริงๆ ด้วยข้อมูลจำลอง"""
        # สร้างข้อมูลจำลองที่ผลลัพธ์ชัดเจน (Age สัมพันธ์กับ Outcome)
        np.random.seed(42)
        n = 100
        age = np.random.normal(50, 10, n)
        # คนแก่มีโอกาสเป็น 1 มากกว่า
        prob = 1 / (1 + np.exp(-(0.1 * (age - 50))))
        y = np.random.binomial(1, prob)
        
        df = pd.DataFrame({'age': age})
        y_series = pd.Series(y)
        
        params, conf, pvals, status, metrics = run_binary_logit(y_series, df)
        
        assert status == "OK"
        assert params is not None
        assert 'age' in params
        # ค่า Coefficient ของ age ควรเป็นบวก (ตามที่เราสร้างข้อมูล)
        assert params['age'] > 0 
        assert metrics['mcfadden'] > 0

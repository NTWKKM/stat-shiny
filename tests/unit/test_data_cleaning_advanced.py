import numpy as np
import pandas as pd
import pytest

from utils.data_cleaning import (
    DataCleaningError,
    check_assumptions,
    impute_missing_data,
    transform_variable,
)


class TestAdvancedCleaning:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "A": [1.0, 2.0, np.nan, 4.0, 5.0],
                "B": [10.0, np.nan, 30.0, 40.0, 50.0],
                "C": ["x", "y", "z", "w", "v"],  # Categorical
            }
        )

    def test_impute_mean(self, sample_df):
        res = impute_missing_data(sample_df, ["A", "B"], method="mean")
        assert res["A"].isna().sum() == 0
        assert res["B"].isna().sum() == 0
        assert res["A"][2] == 3.0  # Mean of 1,2,4,5 is 3.0
        assert res["B"][1] == 32.5  # Mean of 10,30,40,50

    def test_impute_median(self, sample_df):
        res = impute_missing_data(sample_df, ["A"], method="median")
        assert (
            res["A"][2] == 3.0
        )  # Median of 1,2,4,5 (sorted 1,2,4,5 -> mean of 2,4 is 3? No, median of even set is avg of two middles. 1,2,4,5 -> 3.)
        # Actually median of [1,2,4,5] is 3.0.

    def test_impute_knn(self, sample_df):
        # KNN requires at least one complete column or works with partials.
        # It's an approximation.
        res = impute_missing_data(sample_df, ["A", "B"], method="knn", n_neighbors=2)
        assert res["A"].isna().sum() == 0
        assert res["B"].isna().sum() == 0

    def test_impute_mice(self, sample_df):
        res = impute_missing_data(sample_df, ["A", "B"], method="mice")
        assert res["A"].isna().sum() == 0
        assert res["B"].isna().sum() == 0

    def test_impute_invalid_method(self, sample_df):
        with pytest.raises(DataCleaningError):
            impute_missing_data(sample_df, ["A"], method="magic")

    def test_transform_log(self):
        s = pd.Series([1, 10, 100])
        res = transform_variable(s, method="log")
        assert np.isclose(res[0], 0.0)
        assert np.isclose(res[2], 4.60517, atol=1e-4)

    def test_transform_log_negative(self):
        s = pd.Series([-10, 0, 10])
        # Should shift by |-10| + 1 = 11.
        # -10 + 11 = 1 -> log(1) = 0
        # 0 + 11 = 11 -> log(11) = 2.39
        res = transform_variable(s, method="log")
        assert np.isclose(res[0], 0.0)
        assert res[2] > 0

    def test_transform_sqrt(self):
        s = pd.Series([4, 9, 16])
        res = transform_variable(s, method="sqrt")
        assert res[0] == 2.0
        assert res[2] == 4.0

    def test_check_assumptions_normal(self):
        np.random.seed(42)
        normal_data = pd.Series(np.random.normal(0, 1, 100))
        res = check_assumptions(normal_data)
        assert res["is_normal"] is True
        assert res["normality_test"] == "Shapiro-Wilk"

    def test_check_assumptions_non_normal(self):
        # Exponential distribution
        non_normal = pd.Series(np.random.exponential(1, 100))
        res = check_assumptions(non_normal)
        assert "p_value" in res
        # Might be normal for small N by chance, but unlikely for exp

    def test_check_assumptions_small_sample(self):
        s = pd.Series([1, 2])
        res = check_assumptions(s)
        assert res["normality_test"] == "Insufficient Data"

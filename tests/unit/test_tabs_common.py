from tabs._common import select_variable_by_keyword


def test_select_variable_by_keyword_strings():
    cols = ["patient_id", "treatment_group", "systolic_bp", "heart_rate"]
    # Exact match
    assert select_variable_by_keyword(cols, ["systolic_bp"]) == "systolic_bp"
    # Case insensitive exact match
    assert select_variable_by_keyword(cols, ["TREATMENT_GROUP"]) == "treatment_group"
    # Token boundary / word match
    assert select_variable_by_keyword(cols, ["bp"]) == "systolic_bp"
    # Priority order across keywords
    assert select_variable_by_keyword(cols, ["heart", "bp"]) == "heart_rate"
    # Fallback to first
    assert (
        select_variable_by_keyword(cols, ["nonexistent"], default_to_first=True)
        == "patient_id"
    )
    # Fallback disabled
    assert (
        select_variable_by_keyword(cols, ["nonexistent"], default_to_first=False)
        is None
    )
    # Empty list
    assert select_variable_by_keyword([], ["age"]) is None


def test_select_variable_by_keyword_generic_types():
    # Integer column labels
    int_cols = [101, 202, 303, 404]
    assert select_variable_by_keyword(int_cols, ["202"]) == 202
    assert type(select_variable_by_keyword(int_cols, ["202"])) is int
    assert select_variable_by_keyword(int_cols, ["999"], default_to_first=True) == 101
    assert select_variable_by_keyword(int_cols, ["999"], default_to_first=False) is None

    # Custom object column labels
    class CustomColumn:
        def __init__(self, name: str, col_id: int):
            self.name = name
            self.col_id = col_id

        def __str__(self):
            return self.name

        def __repr__(self):
            return f"CustomColumn({self.name!r}, {self.col_id})"

    c1 = CustomColumn("statin_dose", 1)
    c2 = CustomColumn("aspirin_dose", 2)
    c3 = CustomColumn("age_years", 3)

    cols = [c1, c2, c3]
    assert select_variable_by_keyword(cols, ["aspirin"]) is c2
    assert select_variable_by_keyword(cols, ["years"]) is c3
    assert select_variable_by_keyword(cols, ["unknown"], default_to_first=True) is c1
    assert select_variable_by_keyword(cols, ["unknown"], default_to_first=False) is None

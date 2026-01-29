import pandas as pd
from utils.table_one_advanced import TableOneGenerator


def verify_or_styles():
    print("Verifying OR Styles with Refined Logic...")

    # 1. Setup Data - Binary Variable
    # Ref (0): 10 Pos, 10 Neg (RefGroup)
    # Event (1): 20 Pos, 10 Neg (CompGroup)
    # Variable Levels: "Neg" and "Pos".
    # Sorted: "Neg" (Ref), "Pos" (Target).

    # Logic:
    # 2x2 for "Pos" vs "Neg" (Ref).
    # Cases (CompGroup=1): Pos=20, Neg=10.
    # Controls (RefGroup=0): Pos=10, Neg=10.

    # a = Cases with Pos = 20
    # b = Controls with Pos = 10
    # c = Cases with Ref(Neg) = 10
    # d = Controls with Ref(Neg) = 10

    # OR = (a*d)/(b*c) = (20*10)/(10*10) = 200/100 = 2.0

    data = []
    # Group 0 (Reference Group)
    for _ in range(10):
        data.append({"Group": 0, "BinVar": "Pos"})
    for _ in range(10):
        data.append({"Group": 0, "BinVar": "Neg"})
    # Group 1 (Comparison Group)
    for _ in range(20):
        data.append({"Group": 1, "BinVar": "Pos"})
    for _ in range(10):
        data.append({"Group": 1, "BinVar": "Neg"})

    df = pd.DataFrame(data)
    generator = TableOneGenerator(df)

    # 2. Test "All Levels"
    print("\n[TEST] All Levels (vs Ref)")
    html_all = generator.generate(
        ["BinVar"], stratify_by="Group", or_style="all_levels"
    )
    # Should expect:
    # Neg: Ref. (or -)
    # Pos: 2.00

    assert "2.00" in html_all, f"2.00 missing in all_levels. Found: {html_all}"
    print("[PASS] 2.00 found in all_levels")

    assert "Ref." in html_all, "Ref. missing in all_levels"
    print("[PASS] Ref. found in all_levels")

    # 3. Test "Simple"
    print("\n[TEST] Simple Style")
    html_simple = generator.generate(["BinVar"], stratify_by="Group", or_style="simple")

    assert html_simple.count("2.00") == 1, (
        f"Unexpected count of 2.00: {html_simple.count('2.00')}"
    )
    print("[PASS] Only one 2.00 found")

    assert "2x2 (Pos vs Neg)" in html_simple, f"Method name incorrect: {html_simple}"
    print("[PASS] Method name correct: 2x2 (Pos vs Neg)")


if __name__ == "__main__":
    verify_or_styles()

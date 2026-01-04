import pandas as pd
import diag_test

# Sample data for Kappa calculation
df = pd.DataFrame({
    'DrA': ['Pos', 'Pos', 'Neg', 'Neg', 'Pos'],
    'DrB': ['Pos', 'Neg', 'Neg', 'Neg', 'Pos']
})

res_df, err, conf_matrix = diag_test.calculate_kappa(df, 'DrA', 'DrB')

if err:
    print(f"FAILURE: {err}")
else:
    elements = [
        {"type": "table", "header": "Statistics", "data": res_df},
        {"type": "contingency_table", "header": "Confusion Matrix (Crosstab)", "data": conf_matrix}
    ]
    report_html = diag_test.generate_report("Kappa Test", elements)
    
    # Check for "Total" and percentages in the HTML
    has_total = "Total" in report_html
    has_percentages = "(100.0%)" in report_html or "(40.0%)" in report_html
    has_rater_method = "Rater/Method 1" in report_html
    
    if has_total and has_percentages and has_rater_method:
        print("SUCCESS: Kappa Crosstab now matches Chi2 style (includes Total, Percentages, and proper Headers)")
    else:
        print("FAILURE: Formatting missing")
        if not has_total: 
            print("- Total missing")
        if not has_percentages: 
            print("- Percentages missing")
        if not has_rater_method: 
            print("- Rater/Method headers missing")
        # print(report_html)

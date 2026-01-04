import pandas as pd
import sys
import os

# Add the project directory to sys.path
sys.path.append('/Users/ntwkkm/shiny-stat/stat-shiny')

from diag_test import calculate_chi2, generate_report

def verify():
    # Create sample data
    df = pd.DataFrame({
        'Treatment': ['A']*10 + ['B']*10,
        'Outcome': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    })
    
    # Calculate chi2
    display_tab, stats_df, msg, risk_df = calculate_chi2(df, 'Treatment', 'Outcome')
    
    # Generate report
    elements = [
        {'type': 'contingency', 'header': 'Contingency Table', 'data': display_tab}
    ]
    html = generate_report("Test Report", elements)
    
    # Verify HTML content
    if 'class="contingency-table"' in html:
        print("SUCCESS: 'contingency-table' class found in HTML.")
    else:
        print("FAILURE: 'contingency-table' class NOT found in HTML.")
        
    if 'Outcome' in html and 'Treatment' in html:
        print("SUCCESS: Headers found in HTML.")
    else:
        print("FAILURE: Headers NOT found in HTML.")

    # Save HTML for manual inspection if needed
    with open('verify_output.html', 'w') as f:
        f.write(html)
    print("Verification output saved to verify_output.html")

if __name__ == "__main__":
    verify()

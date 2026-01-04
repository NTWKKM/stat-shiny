import pandas as pd

# Create dummy data
df = pd.DataFrame({
    'Category': ['1', '0', 'Total'],
    ('Diagnosis_Dr_B', '1'): ['419 (83.5%)', '137 (13.7%)', '556 (37.1%)'],
    ('Diagnosis_Dr_B', '0'): ['83 (16.5%)', '861 (86.3%)', '944 (62.9%)'],
    ('Total', ''): ['502 (100.0%)', '998 (100.0%)', '1500 (100.0%)']
})

# Construct MultiIndex columns
cols = pd.MultiIndex.from_tuples([
    ('Diagnosis_Dr_B', '1'),
    ('Diagnosis_Dr_B', '0'),
    ('Total', '')
])

data = [
    ['419 (83.5%)', '83 (16.5%)', '502 (100.0%)'],
    ['137 (13.7%)', '861 (86.3%)', '998 (100.0%)'],
    ['556 (37.1%)', '944 (62.9%)', '1500 (100.0%)']
]

df = pd.DataFrame(data, columns=cols, index=['1', '0', 'Total'])
df.index.name = 'Diagnosis_Dr_A'

html = df.to_html(classes='contingency-table')
with open('test_table.html', 'w') as f:
    f.write(html)

print("HTML structure generated in test_table.html")

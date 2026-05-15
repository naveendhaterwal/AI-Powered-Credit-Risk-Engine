import json
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/Loan_Default.csv')

report = []

report.append("## 8. Categorical Feature Audit")
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    vc = df[col].value_counts(dropna=False)
    report.append(f"### {col}")
    report.append(f"- Unique values: {len(vc)}")
    for val, count in vc.head(5).items():
        report.append(f"  - {val}: {count} ({count/len(df)*100:.2f}%)")
    report.append("")

report.append("## Target Correlation")
num_df = df.select_dtypes(include=[np.number])
if 'Status' in num_df.columns:
    corr = num_df.corr()['Status'].sort_values()
    report.append(str(corr))
    
with open('eda_summary2.md', 'w') as f:
    f.write('\n'.join(report))


import json

with open('eda_results.json', 'r') as f:
    data = json.load(f)

report = ["# Credit Risk Dataset Quality Audit & Diagnostic Report\n"]

# 1. Dataset Overview
report.append("## 1. Dataset Overview")
report.append(f"- **Rows:** {data['shape'][0]:,}")
report.append(f"- **Columns:** {data['shape'][1]}")
report.append(f"- **Target Column:** {data['target_column']}")
report.append(f"- **Memory Usage:** {data['memory_usage_mb']:.2f} MB")
report.append(f"- **Duplicate Rows:** {data['duplicates']}\n")

# 2. Missing Value Analysis
report.append("## 2. Missing Value Analysis")
for col, missing in data.get('missing', {}).items():
    report.append(f"- **{col}**: {missing['count']:,} ({missing['percentage']:.2f}%)")
report.append("\n")

# 3. Target Distribution
report.append("## 3. Target Variable Analysis")
dist = data.get('target_distribution', {})
report.append(f"Class distribution for `{data['target_column']}`:")
for val, count in dist.items():
    pct = count / data['shape'][0] * 100
    report.append(f"- {val}: {count:,} ({pct:.2f}%)")
report.append("\n")

# 4. Financial Analysis
report.append("## 4. Financial Feature Audit")
for col, stats in data.get('financial', {}).items():
    if col != 'ID' and col != 'year' and col != 'Status':
        report.append(f"### {col}")
        report.append(f"- Min: {stats['min']}")
        report.append(f"- Max: {stats['max']}")
        report.append(f"- Mean: {stats['mean']}")
        report.append(f"- 1st pctl: {stats['p1']}")
        report.append(f"- 99th pctl: {stats['p99']}")
        report.append(f"- Skew: {stats['skew']}")
        report.append(f"- Kurtosis: {stats['kurtosis']}")
        report.append(f"- Zeros: {stats['zero_count']} | Negatives: {stats['negative_count']}\n")

# Corrs
report.append("## 7. Correlation Analysis")
high_corr = data.get('high_correlations', [])
for (c1, c2, val) in high_corr:
    report.append(f"- {c1} & {c2}: {val:.2f}")

with open('eda_summary.md', 'w') as f:
    f.write('\n'.join(report))


import pandas as pd
import numpy as np
import json
import sys

def run_eda():
    df = pd.read_csv('data/raw/Loan_Default.csv')
    
    info = {}
    info['shape'] = df.shape
    info['columns'] = list(df.columns)
    info['dtypes'] = {k: str(v) for k, v in df.dtypes.items()}
    info['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
    info['duplicates'] = int(df.duplicated().sum())
    
    info['missing'] = {}
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            info['missing'][col] = {
                'count': null_count,
                'percentage': null_count / len(df) * 100
            }
            
    info['uniques'] = {col: int(df[col].nunique()) for col in df.columns}
    
    # Assume target column might be 'Status', 'Default', 'target', 'loan_status'
    target_candidates = [c for c in df.columns if c.lower() in ['status', 'default', 'target', 'loan_status']]
    target_col = target_candidates[0] if target_candidates else None
    info['target_column'] = target_col
    
    if target_col:
        info['target_distribution'] = df[target_col].value_counts().to_dict()
        
    info['describe'] = df.describe().to_dict()
    
    # Financial features percentiles
    fin_features = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c != target_col]
    info['financial'] = {}
    for col in fin_features:
        info['financial'][col] = {
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            'p1': float(df[col].quantile(0.01)) if not pd.isna(df[col].quantile(0.01)) else None,
            'p5': float(df[col].quantile(0.05)) if not pd.isna(df[col].quantile(0.05)) else None,
            'p25': float(df[col].quantile(0.25)) if not pd.isna(df[col].quantile(0.25)) else None,
            'p50': float(df[col].quantile(0.50)) if not pd.isna(df[col].quantile(0.50)) else None,
            'p75': float(df[col].quantile(0.75)) if not pd.isna(df[col].quantile(0.75)) else None,
            'p95': float(df[col].quantile(0.95)) if not pd.isna(df[col].quantile(0.95)) else None,
            'p99': float(df[col].quantile(0.99)) if not pd.isna(df[col].quantile(0.99)) else None,
            'skew': float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
            'kurtosis': float(df[col].kurtosis()) if not pd.isna(df[col].kurtosis()) else None,
            'negative_count': int((df[col] < 0).sum()) if df[col].dtype in ['float64', 'int64'] else 0,
            'zero_count': int((df[col] == 0).sum()) if df[col].dtype in ['float64', 'int64'] else 0
        }
        
    # Correlations
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.5:
                high_corr.append((corr.columns[i], corr.columns[j], float(corr.iloc[i, j])))
                
    info['high_correlations'] = high_corr
    
    # Target correlations
    if target_col and target_col in numeric_df.columns:
        target_corr = corr[target_col].to_dict()
        info['target_correlations'] = {k: v for k, v in target_corr.items() if not pd.isna(v)}
        
    with open('eda_results.json', 'w') as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    run_eda()

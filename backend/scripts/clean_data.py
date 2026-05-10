import pandas as pd
from pathlib import Path

def clean_data():
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "data" / "raw" / "Loan_Default.csv"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "Loan_Default_clean.csv"
    
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    print("Cleaning data...")
    # Drop columns not used in modeling based on EDA logic
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "Interest_rate_spread" in df.columns:
        df = df.drop(columns=["Interest_rate_spread"])
        
    # Impute missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    if "Status" in numeric_cols:
        numeric_cols.remove("Status")
        
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        
    print(f"Saving cleaned dataset to {out_path}...")
    df.to_csv(out_path, index=False)
    print("Clean dataset created successfully!")

if __name__ == "__main__":
    clean_data()

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

def train_and_export():
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "raw" / "Loan_Default.csv"
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Loading data from {data_path}...")
    if not data_path.exists():
        print("Data file not found. Please ensure Loan_Default.csv is in backend/data/raw/")
        return

    df = pd.read_csv(data_path)
    
    # Simple Preprocessing (matching EDA.ipynb logic)
    df = df.drop(columns=["ID", "Interest_rate_spread"])
    
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols.remove("Status")
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop("Status", axis=1)
    y = df["Status"]
    
    # Save feature columns
    joblib.dump(X.columns.tolist(), models_dir / "feature_columns.pkl")
    print(f"Saved {len(X.columns)} feature columns.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # We'll bake the scaler into a simple pipeline or just save models that expect scaled data.
    # For this implementation, we'll save the models directly and ensure the service handles features similarly.
    # Note: MLService currently prepares features manually.
    
    print("Applying SMOTE to balance the training dataset...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Original training shape: {X_train.shape}, {y_train.shape}")
    print(f"Resampled training shape: {X_train_resampled.shape}, {y_train_resampled.shape}")
    
    model_registry = {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    for name, model in model_registry.items():
        print(f"Training {name} on balanced dataset...")
        model.fit(X_train_resampled, y_train_resampled) # Training on resampled data
        
        # Test performance
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        filename = f"{name}_model.pkl"
        if name == "logistic": filename = "logistic_model.pkl"
        elif name == "random_forest": filename = "rf_model.pkl"
        elif name == "gradient_boosting": filename = "gb_model.pkl"
        
        joblib.dump(model, models_dir / filename)
        print(f"✓ {name} Accuracy: {acc:.4f} -> Saved to {filename}")

if __name__ == "__main__":
    train_and_export()

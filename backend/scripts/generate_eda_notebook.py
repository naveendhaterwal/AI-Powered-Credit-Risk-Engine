import os
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()

    # Introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# Credit Risk ML Pipeline: Production Banking Standard
This notebook demonstrates the end-to-end ML pipeline for credit risk assessment.
It adheres to strict banking rules, removing synthetic oversampling (SMOTE) and preventing target leakage.

**Key Features:**
1. Original authentic dataset only (`Loan_Default.csv`).
2. Robust financial constraint handling (LTV, DTI/FOIR clipping).
3. Engineered financial ratios.
4. `ColumnTransformer` for perfectly reproducible preprocessing.
5. Calibrated ensemble probabilities.
6. Hard banking deterministic rule overrides."""))

    # Imports
    nb.cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
"""))

    # Data Ingestion
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Data Ingestion\nLoad ONLY the original dataset."))
    nb.cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('../data/raw/Loan_Default.csv')
print(f"Original shape: {df.shape}")
df.head()
"""))

    # Remove Dangerous Columns
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Remove Dangerous & Leakage Columns\nDropping target leakage columns, fairness risk variables (Gender, age), and zero-variance/useless columns."))
    nb.cells.append(nbf.v4.new_code_cell("""drop_cols = [
    "Gender",
    "age",
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
    "construction_type",
    "Secured_by",
    "Security_Type",
    "open_credit"
]
# ID and year columns
drop_cols.extend(["ID", "year"])

df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])
print(f"Shape after dropping columns: {df_clean.shape}")
"""))

    # Outliers & Constraints
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Clean Financial Outliers\nImplement realistic banking constraints."))
    nb.cells.append(nbf.v4.new_code_cell("""# 1. Clip impossible ratios
if "LTV" in df_clean.columns:
    df_clean["LTV"] = df_clean["LTV"].clip(upper=150)
if "dtir1" in df_clean.columns:
    df_clean["dtir1"] = df_clean["dtir1"].clip(upper=65)

# 2. Handle income <= 0
if "income" in df_clean.columns:
    df_clean.loc[df_clean["income"] <= 0, "income"] = np.nan

# 3. Handle zero or negative property values
if "property_value" in df_clean.columns:
    df_clean.loc[df_clean["property_value"] <= 0, "property_value"] = np.nan
"""))

    # Feature Engineering
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Feature Engineering\nCreate safe financial ratios to improve underwriting realism."))
    nb.cells.append(nbf.v4.new_code_cell("""# Loan to Income
if "loan_amount" in df_clean.columns and "income" in df_clean.columns:
    df_clean["loan_to_income"] = df_clean["loan_amount"] / (df_clean["income"] + 1)

# Derived FOIR (Fixed Obligation to Income Ratio)
if "dtir1" in df_clean.columns:
    df_clean["derived_foir"] = df_clean["dtir1"] / 100.0
else:
    df_clean["derived_foir"] = 0.0

# Estimated EMI (Assuming 30yr 7% mortgage as baseline proxy)
if "loan_amount" in df_clean.columns and "term" in df_clean.columns:
    monthly_rate = 0.07 / 12
    # Simple proxy if term is missing
    tenure_months = df_clean["term"].fillna(360)
    df_clean["estimated_emi"] = df_clean["loan_amount"] * (monthly_rate * (1 + monthly_rate)**tenure_months) / ((1 + monthly_rate)**tenure_months - 1)

# Loan to Property Ratio (LTV proxy)
if "loan_amount" in df_clean.columns and "property_value" in df_clean.columns:
    df_clean["loan_to_property_ratio"] = df_clean["loan_amount"] / df_clean["property_value"]
"""))

    # Export Clean Dataset
    nb.cells.append(nbf.v4.new_markdown_cell("## Export Cleaned Training Data\nSaving the exact cleaned dataframe before passing to the sklearn pipeline."))
    nb.cells.append(nbf.v4.new_code_cell("""os.makedirs('../data/processed', exist_ok=True)
clean_csv_path = '../data/processed/clean_training_data.csv'
df_clean.to_csv(clean_csv_path, index=False)
print(f"Saved clean training dataset to {clean_csv_path}")
"""))

    # Preprocessing
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Build Proper sklearn Pipeline\nNo manual encoding. The `ColumnTransformer` will handle everything automatically."))
    nb.cells.append(nbf.v4.new_code_cell("""target = "Status"
X = df_clean.drop(columns=[target])
y = df_clean[target]

# Identify columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Define Preprocessors
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Save feature metadata for backend inference mapping
feature_metadata = {
    "numerical_columns": numerical_cols,
    "categorical_columns": categorical_cols,
    "expected_features": X.columns.tolist()
}
with open("../models/feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=2)

print("Preprocessing pipeline initialized and metadata saved.")
"""))

    # Train-test split
    nb.cells.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
"""))

    # Train Models
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. Train Models & Add Probability Calibration\nTraining `LogisticRegression`, `RandomForest`, and `GradientBoosting` inside full `Pipeline` objects. Using `class_weight='balanced'` instead of SMOTE."))
    nb.cells.append(nbf.v4.new_code_cell("""# 1. Logistic Regression
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
print("Training Logistic Regression...")
lr_model.fit(X_train, y_train)

# 2. Random Forest (Calibrated)
rf_base = RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_calibrated = CalibratedClassifierCV(estimator=rf_base, method='sigmoid', cv=3)
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_calibrated)
])
print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# 3. Gradient Boosting (Calibrated)
gb_base = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_calibrated = CalibratedClassifierCV(estimator=gb_base, method='sigmoid', cv=3)
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', gb_calibrated)
])
print("Training Gradient Boosting...")
gb_model.fit(X_train, y_train)

print("All models trained successfully!")
"""))

    # Evaluate Models
    nb.cells.append(nbf.v4.new_markdown_cell("## 7. Model Diagnostics\nEvaluate the models and generate ROC-AUC, Precision/Recall, and Calibration curves."))
    nb.cells.append(nbf.v4.new_code_cell("""def evaluate_model(model, name, X_t, y_t):
    y_pred = model.predict(X_t)
    y_prob = model.predict_proba(X_t)[:, 1]
    
    print(f"\\n--- {name} ---")
    print("ROC-AUC:", roc_auc_score(y_t, y_prob))
    print(classification_report(y_t, y_pred))
    
    return y_prob

lr_probs = evaluate_model(lr_model, "Logistic Regression", X_test, y_test)
rf_probs = evaluate_model(rf_model, "Random Forest", X_test, y_test)
gb_probs = evaluate_model(gb_model, "Gradient Boosting", X_test, y_test)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""# ROC Curve
plt.figure(figsize=(8,6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_probs)

plt.plot(fpr_lr, tpr_lr, label="LR")
plt.plot(fpr_rf, tpr_rf, label="RF (Calibrated)")
plt.plot(fpr_gb, tpr_gb, label="GB (Calibrated)")
plt.plot([0,1],[0,1], 'k--')
plt.title("ROC Curve")
plt.legend()
plt.show()

# Calibration Curve
plt.figure(figsize=(8,6))
fraction_of_positives_lr, mean_predicted_value_lr = calibration_curve(y_test, lr_probs, n_bins=10)
fraction_of_positives_rf, mean_predicted_value_rf = calibration_curve(y_test, rf_probs, n_bins=10)
fraction_of_positives_gb, mean_predicted_value_gb = calibration_curve(y_test, gb_probs, n_bins=10)

plt.plot(mean_predicted_value_lr, fraction_of_positives_lr, "s-", label="LR")
plt.plot(mean_predicted_value_rf, fraction_of_positives_rf, "s-", label="RF (Calibrated)")
plt.plot(mean_predicted_value_gb, fraction_of_positives_gb, "s-", label="GB (Calibrated)")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.title("Calibration Curves")
plt.legend()
plt.show()
"""))

    # Ensemble & Hard Rules
    nb.cells.append(nbf.v4.new_markdown_cell("## 8. Ensemble Logic & Hard Banking Rules\nDemonstrate the weighted ensemble logic and extreme borrower overrides within the notebook."))
    nb.cells.append(nbf.v4.new_code_cell("""def evaluate_borrower(borrower_dict):
    # 1. Hard Rules First
    dti = borrower_dict.get("dti", 0)
    ltv = borrower_dict.get("ltv", 0)
    foir = borrower_dict.get("foir", 0)
    credit_score = borrower_dict.get("credit_score", 0)
    income = borrower_dict.get("income", 0)
    
    reasons = []
    if credit_score < 550: reasons.append("Reject: Credit Score < 550")
    if ltv > 120: reasons.append("Reject: LTV > 120")
    if foir > 55 or dti > 55: reasons.append("Reject: DTI/FOIR > 55")
    if income <= 0: reasons.append("Manual Review: Zero/Negative Income")
        
    if reasons:
        return {"final_risk_level": "High Risk", "override_triggered": True, "reasons": reasons, "ml_score": 1.0}
    
    # 2. Build DataFrame for inference
    df_inf = pd.DataFrame([borrower_dict])
    
    # Align with expected columns
    for col in feature_metadata["expected_features"]:
        if col not in df_inf.columns:
            df_inf[col] = np.nan
            
    df_inf = df_inf[feature_metadata["expected_features"]]
    
    # 3. Predict Proba
    lr_p = lr_model.predict_proba(df_inf)[0][1]
    rf_p = rf_model.predict_proba(df_inf)[0][1]
    gb_p = gb_model.predict_proba(df_inf)[0][1]
    
    # Weighted Ensemble
    final_score = 0.25 * lr_p + 0.40 * rf_p + 0.35 * gb_p
    
    risk_level = "Low Risk"
    if final_score > 0.4: risk_level = "Medium Risk"
    if final_score > 0.7: risk_level = "High Risk"
        
    return {
        "final_risk_level": risk_level,
        "override_triggered": False,
        "reasons": [],
        "ml_score": final_score,
        "individual_scores": {"lr": lr_p, "rf": rf_p, "gb": gb_p}
    }
"""))

    nb.cells.append(nbf.v4.new_code_cell("""extreme_borrower = {
    "credit_score": 300,
    "dti": 490,
    "foir": 189,
    "loan_amount": 500000000,
    "income": 1000
}

res = evaluate_borrower(extreme_borrower)
print("Extreme Borrower Result:")
print(json.dumps(res, indent=2))
"""))

    # Save Pipelines
    nb.cells.append(nbf.v4.new_markdown_cell("## 9. Save Full Pipelines\nSave the complete `Pipeline` objects. No raw models without preprocessing."))
    nb.cells.append(nbf.v4.new_code_cell("""os.makedirs("../models", exist_ok=True)

joblib.dump(lr_model, "../models/logistic_pipeline.pkl")
joblib.dump(rf_model, "../models/rf_pipeline.pkl")
joblib.dump(gb_model, "../models/gb_pipeline.pkl")

print("All FULL pipelines exported successfully to backend/models/")
"""))

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # backend/
    
    output_path = os.path.join(project_root, 'notebooks', 'EDA.ipynb')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_notebook()
    print("EDA.ipynb generated successfully!")

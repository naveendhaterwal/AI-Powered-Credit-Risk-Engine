import requests

url = "http://127.0.0.1:8000/api/predict/risk"
payload = {
  "full_name": "Test User",
  "age": 30,
  "monthly_income": 85000,
  "employment_type": "Salaried",
  "credit_score": 300,
  "existing_loan_amount": 0,
  "existing_emi_monthly": 0,
  "loan_amount_requested": 500000000,
  "loan_purpose": "Home",
  "loan_tenure_months": 360
}

response = requests.post(url, json=payload)
print(response.status_code)
import json
print(json.dumps(response.json(), indent=2))

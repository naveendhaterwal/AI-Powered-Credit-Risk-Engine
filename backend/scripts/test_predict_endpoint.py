import requests
import json

def test_api():
    url = "http://localhost:8000/api/predict/risk"
    
    print("Testing Borrower 1 (Bad borrower - should trigger hard rules)")
    bad_payload = {
        "full_name": "Test Borrower",
        "age": 30,
        "employment_type": "Salaried",
        "monthly_income": 0.1,
        "existing_loan_amount": 50000.0,
        "credit_score": 300,
        "loan_amount_requested": 100000.0,
        "loan_tenure_months": 60,
        "loan_purpose": "Personal"
    }
    
    try:
        response = requests.post(url, json=bad_payload)
        print("Status:", response.status_code)
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error:", e)
        
    print("\n------------------\n")
    print("Testing Borrower 2 (Good borrower - should pass through ML smoothly)")
    good_payload = {
        "full_name": "Test Good Borrower",
        "age": 35,
        "employment_type": "Salaried",
        "monthly_income": 8000.0,
        "existing_loan_amount": 0.0,
        "credit_score": 750,
        "loan_amount_requested": 200000.0,
        "loan_tenure_months": 360,
        "loan_purpose": "Home"
    }
    
    try:
        response = requests.post(url, json=good_payload)
        print("Status:", response.status_code)
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_api()

from fastapi.testclient import TestClient
from main import app  

client = TestClient(app)

def test_get_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to Census Bureau Classifier API"
    
def test_predict_low_income():
    """test model prediction for <=50K case"""
    test_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predictions", json=test_data)
    assert response.status_code == 200
    assert response.json() == "<=50K" 
    
    
def test_predict_high_income():
    """test model prediction for >50K case"""
    test_data = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 193524,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"
    }
    response = client.post("/predictions", json=test_data)
    assert response.status_code == 200
    assert response.json() == ">50K"  


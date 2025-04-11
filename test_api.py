from fastapi.testclient import TestClient
from .main import app  

client = TestClient(app)

def test_get_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome!"
    
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
    assert response.json() == " <=50K" 
    
    
def test_predict_high_income():
    """test model prediction for >50K case"""
    test_data = {
    "age": 34,
    "workclass": "Private",
    "fnlgt": 37274,
    "education": "Prof-school",
    "education_num": 15,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital_gain": 14084,
    "capital_loss": 0,
    "hours_per_week": 55,
    "native_country": "United-States"
    }
    response = client.post("/predictions", json=test_data)
    assert response.status_code == 200
    assert response.json() == " >50K" 


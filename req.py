import requests

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

url = 'https://mldeploy-hmrt.onrender.com/predictions'
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

request = requests.post(url, json=test_data, headers=headers)
assert request.status_code == 200
print(request.status_code)
print(request.json())
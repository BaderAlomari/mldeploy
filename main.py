# Put the code for your API here.
from pydantic import BaseModel
from ml.data import process_data
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pathlib import Path
import joblib
from ml.model import inference
import os
class features(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    model_config = { "json_schema_extra" : {
            "examples": [
                {
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
                },
                {
                    "age": 28,
                    "workclass": "Private",
                    "fnlgt": 338409,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Prof-specialty",
                    "relationship": "Wife",
                    "race": "Black",
                    "sex": "Female",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "Cuba"
                },
                {
                    "age": 52,
                    "workclass": "Self-emp-not-inc",
                    "fnlgt": 209642,
                    "education": "HS-grad",
                    "education_num": 9,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 0,
                    "capital_loss": 0,
                    "hours_per_week": 45,
                    "native_country": "United-States"
                },
                {
                    "age": 31,
                    "workclass": "Private",
                    "fnlgt": 45781,
                    "education": "Masters",
                    "education_num": 14,
                    "marital_status": "Never-married",
                    "occupation": "Prof-specialty",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Female",
                    "capital_gain": 14084,
                    "capital_loss": 0,
                    "hours_per_week": 50,
                    "native_country": "United-States"
                },
                {
                    "age": 40,
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
            ]
        }}

app = FastAPI()

@app.get("/")
async def greetings():
    return "Welcome to Census Bureau Classifier API"

model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

categorial_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

@app.post("/predictions")
async def predictor(body: features):
    
    data = pd.DataFrame(body.__dict__,[0])
    
    data, * _ = process_data(data, categorical_features=categorial_features,
                                        training=False, encoder=encoder)
    
    pred = inference(model,data)
    
    return lb.inverse_transform(pred)[0]
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run('main:app', port=port, reload=True)
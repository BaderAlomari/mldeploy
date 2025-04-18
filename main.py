# Put the codee for your API here.
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

    class Config:
        json_schema_extra = {
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
                }
            ]
        }

app = FastAPI()

@app.get("/")
async def greetings():
    return "Welcome!"

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
    
    data = pd.DataFrame([body.dict()])
    
    data, * _ = process_data(data, categorical_features=categorial_features,
                                        training=False, encoder=encoder)
    
    pred = inference(model,data)
    
    return lb.inverse_transform(pred)[0]
if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=10000, reload=True)
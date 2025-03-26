from ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier


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

def test_categorical_features_present():
    """test all categorical features exist in the dataset."""
    data = pd.read_csv('data/census.csv')
    for feature in categorial_features:
        assert feature in data.columns, f"{feature} was not found in dataset columns."

def test_model_file():
    assert os.path.exists('model/model.pkl')

def test_model():
    """ test Random Forest model """

    model = joblib.load('model/model.pkl')
    assert isinstance(model, RandomForestClassifier)
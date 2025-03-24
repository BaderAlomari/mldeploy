import pandas as pd
from ml.data import process_data
from ml.model import inference, compute_model_metrics
import joblib
from sklearn.model_selection import train_test_split

categorial_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

data = pd.read_csv("data/census.csv")
_, test = train_test_split(data, test_size=0.20, random_state=42)

X_test, y_test, * _ = process_data(test, categorical_features=categorial_features,
                              label="salary", training=False, encoder=encoder,
                              lb=lb)

preds = inference(model, X_test)

list_res = []

for feature in categorial_features:
    for val in test[feature].unique():
        mask = test[feature]==val
        if sum(mask) == 0:
            continue
        precision, recall, fbeta = compute_model_metrics(y_test[mask],
                                                         preds[mask])
        list_res.append({"feature":feature,"val":val, "precision":precision,
                   "recall": recall, "fbeta":fbeta, "n_samples": sum(mask)})
        
slices = pd.DataFrame(list_res)
slices.to_csv("slice_output.txt", index=False, float_format="%.4f")
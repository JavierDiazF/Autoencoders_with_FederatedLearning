import os

import bentoml
from bentoml.io import JSON
import pandas as pd

BENTO_MODEL_TAG = os.environ.get("BENTO_MODEL_TAG", "rfc_class")

rfc_runner = bentoml.sklearn.get(f"{BENTO_MODEL_TAG}:latest").to_runner()

svc = bentoml.Service("random_forest_classifier", runners=[rfc_runner])

# Accepts N temperature readings at once (N=1 still works). Column-naming
# convention (v{i}) must stay identical to train.py's.
input_spec = JSON.from_sample({"values": [40.0, 39.5, 41.0]})


@svc.api(input=input_spec, output=JSON())
def predict(input_json):
    values = input_json["values"]
    columns = [f"v{i}" for i in range(len(values))]
    df = pd.DataFrame([values], columns=columns)
    return rfc_runner.predict.run(df)

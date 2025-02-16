from fastapi import FastAPI
import mlflow.sklearn
import numpy as np

app = FastAPI()

# Load trained model
model = mlflow.sklearn.load_model("model")

@app.post("/predict/")
def predict(data: dict):
    input_data = np.array([list(data.values())]).reshape(1, -1)
    pred = model.predict(input_data)
    return {"Survived": int(pred[0])}

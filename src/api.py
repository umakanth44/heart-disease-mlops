from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("models/heart_model.pkl")

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    # Convert input dict to DataFrame
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}

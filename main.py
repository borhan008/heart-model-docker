from fastapi import FastAPI
import joblib
import numpy as np


from schema import HeartDiseasesInput

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
)


model = joblib.load("model.joblib")

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"  
]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "model_type":type(model).__name__,
        "model_features": FEATURES,
        "tot_features": len(FEATURES),
    }


@app.post("/predict")
def predict(input_data: HeartDiseasesInput):
    input_array = np.array([[
        input_data.age,
        input_data.sex,
        input_data.cp,
        input_data.trestbps,
        input_data.chol,
        input_data.fbs,
        input_data.restecg,
        input_data.thalach,
        input_data.exang,
        input_data.oldpeak,
        input_data.slope,
        input_data.ca,
        input_data.thal
    ]])
    prediction = model.predict(input_array)
    return {"prediction": prediction[0]}
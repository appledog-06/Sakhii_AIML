from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('LR1.pkl')  # Replace with the path to your saved model
scaler = joblib.load('scaler.pkl')  # Ensure the scaler is saved during training and available here

app = FastAPI()

# Define the input data structure
class PCOSInput(BaseModel):
    Follicle_No_R: int
    Follicle_No_L: int
    Skin_darkening: int  # Y/N: 1/0
    hair_growth: int  # Y/N: 1/0
    Weight_gain: int  # Y/N: 1/0
    Cycle_length: int
    AMH: float
    Fast_food: int  # Y/N: 1/0
    Cycle_R_I: int  # R/I: 1/0
    FSH_LH: float
    PRL: float
    Pimples: int  # Y/N: 1/0
    Age: int
    BMI: float

# Define the prediction response structure
class PCOSPrediction(BaseModel):
    prediction: int
    probability: float

# Define the prediction endpoint
@app.post("/predict/", response_model=PCOSPrediction)
def predict(data: PCOSInput):
    # Convert input data to array
    input_data = np.array([[data.Follicle_No_R, data.Follicle_No_L, data.Skin_darkening, data.hair_growth,
                            data.Weight_gain, data.Cycle_length, data.AMH, data.Fast_food, data.Cycle_R_I,
                            data.FSH_LH, data.PRL, data.Pimples, data.Age, data.BMI]])
    
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]  # Probability for class 1 (PCOS)

    return PCOSPrediction(prediction=int(prediction[0]), probability=probability)

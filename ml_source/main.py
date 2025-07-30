from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("C:\\Users\\Manasa\\Documents\\covid19\\ml_source\\covid_diag.pkl")

# Define FastAPI app
app = FastAPI()
# create class
# Define input data format
class inp_data(BaseModel):
    Age:int
    Gender: int
    Fever: int
    Cough: int
    Fatigue: int
    Breathlessness: int
    Comorbidity: int
    Stage: int
    Type:int
    Tumor_Size:float

@app.get("/")
def root():
    return {"message": "COVID Survival Rate Predictor is running!"}

@app.post("/predict")
def predict(data: inp_data):
    # Convert input to DataFrame
    # input to dict form like keys ans values.
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]
    return {"Predicted Survival Rate": prediction}

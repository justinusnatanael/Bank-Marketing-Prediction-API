from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load the saved model and scaler
model = joblib.load('log_reg.pkl')
scaler = joblib.load('scaler.pkl')

# Define a function to preprocess the input data
def preprocess_input(data):
    # Convert categorical columns to dummy variables
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    data = pd.get_dummies(data, columns=categorical_cols)

    # Ensure all expected columns are present
    expected_cols = [
        'age', 'duration', 'campaign', 'pdays', 'previous', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
        'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician',
        'job_unemployed', 'job_unknown', 'marital_divorced', 'marital_married', 'marital_single', 'marital_unknown',
        'education_basic.4y', 'education_basic.6y', 'education_basic.9y', 'education_high.school', 'education_illiterate',
        'education_professional.course', 'education_university.degree', 'education_unknown', 'default_no', 'default_unknown',
        'default_yes', 'housing_no', 'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes', 
        'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 
        'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep', 'day_of_week_fri', 'day_of_week_mon', 
        'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed', 'poutcome_failure', 'poutcome_nonexistent', 'poutcome_success'
    ]

    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_cols]

    # Scale the features
    data_scaled = scaler.transform(data)

    return data_scaled

class ClientData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bank Machine Learning Model API"}

@app.post("/predict")
def predict(client_data: ClientData):
    data = pd.DataFrame([client_data.dict()])
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)
    prediction_probability = model.predict_proba(processed_data)
    return {
        'prediction': prediction[0],
        'probability': prediction_probability[0].tolist()
    }

# Example usage for testing 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Bank Customer Churn Prediction API
Author: Shivam Pandey
Description: End-to-end ML deployment using FastAPI
"""



from fastapi import FastAPI 
import numpy as np 
import pandas as pd
import joblib
from pydantic import BaseModel, Field, validator
from typing import Literal
from enum import Enum


#load the model
model = joblib.load('churn_pipeline.pkl')

app = FastAPI(title = 'Bank Customer Churn Prediction')

#Input Data

class ChurnInput(BaseModel):

    CreditScore: int = Field(..., ge=300, le=900, example=650)
    Geography: Literal["France", "Spain", "Germany"] = Field(..., example="France")
    Gender: Literal["Male", "Female"] = Field(..., example="Male")
    Age: int = Field(..., ge=18, le=100, example=35)
    Tenure: int = Field(..., ge=0, le=10, example=5)
    Balance: float = Field(..., ge=0, example=60000)
    NumOfProducts: int = Field(..., ge=1, le=4, example=2)

    HasCrCard: Literal["yes", "no"] = Field(..., example="yes")
    IsActiveMember: Literal["yes", "no"] = Field(..., example="yes")

    EstimatedSalary: float = Field(..., ge=0, example=70000)


# FEATURE ENGINEERING

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[18, 30, 45, 60, 100],
        labels=['Young', 'Adult', 'MidAge', 'Senior']
    )

    df['ZeroBalance'] = (df['Balance'] == 0).astype(int)

    df['EngagementScore'] = df['NumOfProducts'] + df['IsActiveMember']

    return df

# ROUTES

@app.get('/')
def home():
    return {'messege' : "Bank Customer Churn Prediction API is running 🚀"}


@app.post('/predict')
def prediction(data:ChurnInput):
    
    df = pd.DataFrame([data.dict()])

    df["HasCrCard"] = df["HasCrCard"].map({"yes": 1, "no": 0})
    df["IsActiveMember"] = df["IsActiveMember"].map({"yes": 1, "no": 0})


    # Feature engineering
    df = feature_engineering(df)

    prob = model.predict_proba(df)[0][1]
    prediction = 1 if prob >= 0.35 else 0


    result = "Customer will churn" if prediction == 1 else "Customer will not churn"

    if prob >= 0.75:
        risk = "High Risk"
    elif prob >= 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
    "prediction": result,
    "churn_probability": round(float(prob), 4),
    "risk_category": risk
    }


import pytest
import os
import joblib
import numpy as np
from backend.predictor import ChurnPredictor

@pytest.fixture
def predictor_instance():
    return ChurnPredictor()

@pytest.fixture
def valid_customer_data():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 191.00
    }

def test_model_and_pipeline_loaded(predictor_instance):
    assert predictor_instance.model is not None, "Model failed to load."
    assert predictor_instance.pipeline is not None, "Preprocessing pipeline failed to load."
    assert len(predictor_instance.feature_names) > 0, "Feature names metadata is empty."

def test_prediction_output_schema(predictor_instance, valid_customer_data):
    result = predictor_instance.predict(valid_customer_data)
    
    assert "prediction" in result
    assert result["prediction"] in ["Churn", "No Churn"]
    
    assert "probability" in result
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    
    assert "risk_level" in result
    assert result["risk_level"] in ["Low", "Medium", "High"]
    
    assert "top_factors" in result
    assert isinstance(result["top_factors"], list)
    assert len(result["top_factors"]) > 0
    
    assert "recommendations" in result
    assert isinstance(result["recommendations"], list)

def test_high_risk_vs_loyal_prediction(predictor_instance):
    high_risk_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 98.00,
        "TotalCharges": 98.00
    }
    
    loyal_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 70,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.00,
        "TotalCharges": 4200.00
    }
    
    res_high = predictor_instance.predict(high_risk_data)
    res_loyal = predictor_instance.predict(loyal_data)
    
    assert res_high["probability"] > res_loyal["probability"]
    assert res_high["risk_level"] in ["High", "Medium"]
    assert res_loyal["risk_level"] in ["Low", "Medium"]

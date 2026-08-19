import pytest
import pandas as pd
import numpy as np
from ml.src.feature_engineering import FeatureEngineer
from ml.src.preprocessing import clean_raw_data, build_preprocessing_pipeline, get_transformed_feature_names

@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame([{
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
        "Churn": "No"
    }])

def test_clean_raw_data(sample_raw_dataframe):
    # Test blank space string conversion in TotalCharges
    df_blank = sample_raw_dataframe.copy()
    df_blank.loc[0, 'TotalCharges'] = " "
    cleaned = clean_raw_data(df_blank)
    
    assert isinstance(cleaned['TotalCharges'].iloc[0], (int, float, np.number))
    assert not np.isnan(cleaned['TotalCharges'].iloc[0])

def test_feature_engineering_columns(sample_raw_dataframe):
    fe = FeatureEngineer()
    transformed = fe.transform(sample_raw_dataframe)
    
    expected_cols = [
        'tenure_group', 'num_services', 'charge_per_service',
        'has_streaming', 'has_security', 'contract_risk', 'charges_to_tenure_ratio'
    ]
    for col in expected_cols:
        assert col in transformed.columns, f"Engineered column {col} missing in output."
        
    assert transformed['tenure_group'].iloc[0] == '0-12m'
    assert transformed['contract_risk'].iloc[0] == 'High'
    assert transformed['has_security'].iloc[0] == 'Yes' # Due to OnlineBackup
    assert 'customerID' not in transformed.columns

def test_preprocessing_pipeline_transform(sample_raw_dataframe):
    pipeline = build_preprocessing_pipeline()
    X = sample_raw_dataframe.drop(columns=['Churn', 'customerID'])
    
    X_proc = pipeline.fit_transform(X)
    assert isinstance(X_proc, np.ndarray)
    assert X_proc.shape[0] == 1
    assert X_proc.shape[1] > 0
    assert not np.isnan(X_proc).any()
    
    feature_names = get_transformed_feature_names(pipeline)
    assert len(feature_names) == X_proc.shape[1]

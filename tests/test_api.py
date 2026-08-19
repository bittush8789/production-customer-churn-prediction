import pytest
import json
from backend.app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def valid_payload():
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 4,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 382.0
    }

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"

def test_stats_endpoint(client):
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = response.get_json()
    assert "dataset_stats" in data
    assert "total_customers" in data["dataset_stats"]
    assert "models_comparison" in data

def test_sample_profiles_endpoint(client):
    response = client.get('/api/sample?type=high_risk')
    assert response.status_code == 200
    data = response.get_json()
    assert data["profile_type"] == "high_risk"
    assert "MonthlyCharges" in data["data"]

def test_predict_success(client, valid_payload):
    response = client.post(
        '/predict',
        data=json.dumps(valid_payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert "top_factors" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]

def test_predict_missing_field(client, valid_payload):
    del valid_payload["Contract"]
    response = client.post(
        '/predict',
        data=json.dumps(valid_payload),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Missing required fields" in data["message"]

def test_predict_invalid_data_type(client, valid_payload):
    valid_payload["tenure"] = "invalid_string"
    response = client.post(
        '/predict',
        data=json.dumps(valid_payload),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_predict_empty_payload(client):
    response = client.post(
        '/predict',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400

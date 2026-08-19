import os
from flask import Blueprint, request, jsonify, send_from_directory
from backend.predictor import ChurnPredictor

api_bp = Blueprint('api_bp', __name__)
predictor = ChurnPredictor()

SAMPLE_PROFILES = {
    "high_risk": {
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
    },
    "loyal_customer": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 68,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.20,
        "TotalCharges": 4365.60
    },
    "moderate_risk": {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 14,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 79.85,
        "TotalCharges": 1117.90
    }
}

@api_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Customer Churn Prediction API",
        "version": "1.0.0"
    }), 200

@api_bp.route('/predict', methods=['POST'])
def predict_churn():
    if not request.is_json:
        return jsonify({
            "error": "Bad Request",
            "message": "Request payload must be a valid JSON object."
        }), 400
        
    data = request.get_json()
    if not data:
        return jsonify({
            "error": "Bad Request",
            "message": "Empty request body provided."
        }), 400
        
    is_valid, err_msg = predictor.validate_input(data)
    if not is_valid:
        return jsonify({
            "error": "Validation Error",
            "message": err_msg
        }), 400
        
    try:
        result = predictor.predict(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": "Prediction Error",
            "message": str(e)
        }), 500

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    meta = predictor.meta
    dataset_stats = meta.get("dataset_stats", {
        "total_customers": 7043,
        "churned_customers": 1869,
        "retained_customers": 5174,
        "churn_rate": 26.54,
        "high_risk_estimate": 1972
    })
    
    models_comparison = meta.get("all_models_comparison", [])
    best_model_name = meta.get("best_model_name", "XGBoost")
    
    return jsonify({
        "dataset_stats": dataset_stats,
        "best_model": best_model_name,
        "models_comparison": models_comparison
    }), 200

@api_bp.route('/sample', methods=['GET'])
def get_sample_profiles():
    profile_type = request.args.get('type', 'high_risk')
    profile = SAMPLE_PROFILES.get(profile_type, SAMPLE_PROFILES['high_risk'])
    return jsonify({
        "profile_type": profile_type,
        "data": profile
    }), 200

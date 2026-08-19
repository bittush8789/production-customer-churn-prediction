import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")

class ChurnPredictor:
    """
    Production-ready Customer Churn Predictor encapsulating preprocessing,
    model inference, and dynamic SHAP feature attributions.
    """
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "churn_model.pkl")
        self.pipeline_path = os.path.join(models_dir, "preprocessing.pkl")
        self.meta_path = os.path.join(models_dir, "model_meta.json")
        
        self.model = None
        self.pipeline = None
        self.meta = {}
        self.feature_names = []
        self.explainer = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.pipeline_path):
            print(f"Warning: Artifacts not found in {self.models_dir}. Please run training first.")
            return
            
        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)
        
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.feature_names = self.meta.get("feature_names", [])
        
        # Initialize SHAP explainer
        try:
            if hasattr(self.model, "feature_importances_") or "XGB" in type(self.model).__name__ or "Forest" in type(self.model).__name__:
                self.explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, "coef_"):
                self.explainer = shap.LinearExplainer(self.model, masker=shap.maskers.Independent(np.zeros((1, len(self.feature_names)))))
        except Exception as e:
            print(f"Notice: Initializing fallback SHAP explainer ({e})")
            self.explainer = None

    def validate_input(self, data: dict) -> tuple[bool, str]:
        """
        Validates customer input fields and data types.
        """
        required_fields = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        missing = [f for f in required_fields if f not in data or data[f] is None or data[f] == ""]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
            
        try:
            tenure = float(data.get('tenure', 0))
            if tenure < 0:
                return False, "Tenure must be a non-negative number."
        except ValueError:
            return False, "Tenure must be a numeric value."
            
        try:
            monthly = float(data.get('MonthlyCharges', 0))
            if monthly < 0:
                return False, "MonthlyCharges must be a non-negative number."
        except ValueError:
            return False, "MonthlyCharges must be a numeric value."
            
        try:
            total = float(data.get('TotalCharges', 0))
            if total < 0:
                return False, "TotalCharges must be a non-negative number."
        except ValueError:
            return False, "TotalCharges must be a numeric value."
            
        return True, ""

    def predict(self, customer_data: dict) -> dict:
        """
        Runs preprocessing, inference, risk scoring, and SHAP explainability.
        """
        if self.model is None or self.pipeline is None:
            self._load_artifacts()
            if self.model is None:
                raise RuntimeError("Model artifacts not found. Please train model first.")
                
        # Convert dictionary to single-row DataFrame
        df = pd.DataFrame([customer_data])
        
        # Ensure correct datatypes
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
        
        # Preprocess features
        X_proc = self.pipeline.transform(df)
        
        # Inference
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_proc)[0, 1])
        else:
            decision = float(self.model.decision_function(X_proc)[0])
            proba = float(1.0 / (1.0 + np.exp(-decision)))
            
        proba = round(proba, 4)
        
        # Classify risk level
        if proba >= 0.65:
            risk_level = "High"
            prediction = "Churn"
        elif proba >= 0.35:
            risk_level = "Medium"
            prediction = "Churn" if proba >= 0.50 else "No Churn"
        else:
            risk_level = "Low"
            prediction = "No Churn"
            
        # Calculate SHAP explainability
        top_factors, factor_details = self._explain_prediction(X_proc, customer_data)
        
        # Generate actionable retention strategies
        recommendations = self._generate_retention_recommendations(customer_data, proba, top_factors)
        
        return {
            "prediction": prediction,
            "probability": proba,
            "probability_pct": round(proba * 100, 1),
            "risk_level": risk_level,
            "top_factors": top_factors,
            "factor_details": factor_details,
            "recommendations": recommendations,
            "model_used": self.meta.get("best_model_name", "Trained ML Classifier")
        }

    def _explain_prediction(self, X_proc: np.ndarray, raw_data: dict) -> tuple[list[str], list[dict]]:
        """
        Extracts SHAP feature attributions and formats human-readable explanations.
        """
        shap_values = None
        if self.explainer is not None:
            try:
                raw_shap = self.explainer.shap_values(X_proc)
                if isinstance(raw_shap, list):
                    shap_values = raw_shap[1][0]
                elif len(raw_shap.shape) == 2:
                    shap_values = raw_shap[0]
                elif len(raw_shap.shape) == 3:
                    shap_values = raw_shap[0, :, 1]
            except Exception as e:
                print(f"SHAP explanation computation error: {e}")
                shap_values = None
                
        # If SHAP is unavailable or linear fallback
        if shap_values is None:
            if hasattr(self.model, "feature_importances_"):
                shap_values = self.model.feature_importances_ * X_proc[0]
            elif hasattr(self.model, "coef_"):
                shap_values = self.model.coef_[0] * X_proc[0]
            else:
                shap_values = np.zeros(X_proc.shape[1])
                
        # Pair feature names with SHAP values
        contributions = []
        for i, val in enumerate(shap_values):
            feat = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            contributions.append({
                "feature": feat,
                "shap_value": float(val),
                "abs_val": abs(float(val))
            })
            
        contributions.sort(key=lambda x: x["abs_val"], reverse=True)
        
        readable_factors = []
        factor_details = []
        
        for item in contributions:
            feat = item["feature"]
            val = item["shap_value"]
            impact = "Increases Risk" if val > 0 else "Reduces Risk"
            
            description = self._map_feature_to_description(feat, val, raw_data)
            if description and description not in readable_factors:
                readable_factors.append(description)
                factor_details.append({
                    "factor": description,
                    "impact": impact,
                    "weight": round(val, 3),
                    "is_risk_driver": val > 0
                })
                if len(readable_factors) >= 5:
                    break
                    
        # Ensure at least 3 factors
        if len(readable_factors) < 3:
            fallback_factors = self._rule_based_factors(raw_data)
            for f in fallback_factors:
                if f not in readable_factors:
                    readable_factors.append(f)
                    factor_details.append({
                        "factor": f,
                        "impact": "Increases Risk",
                        "weight": 0.25,
                        "is_risk_driver": True
                    })
                if len(readable_factors) >= 5:
                    break
                    
        return readable_factors[:5], factor_details[:5]

    def _map_feature_to_description(self, feat_name: str, shap_val: float, raw: dict) -> str:
        """
        Translates raw pipeline feature names into clear business explanations.
        """
        contract = str(raw.get("Contract", ""))
        tenure = float(raw.get("tenure", 0))
        monthly = float(raw.get("MonthlyCharges", 0))
        internet = str(raw.get("InternetService", ""))
        payment = str(raw.get("PaymentMethod", ""))
        tech_support = str(raw.get("TechSupport", ""))
        online_sec = str(raw.get("OnlineSecurity", ""))
        
        if "Contract" in feat_name or "contract_risk" in feat_name:
            if "Month-to-month" in contract or "High" in feat_name:
                return "Month-to-month contract (High churn volatility)"
            elif "Two year" in contract:
                return "Two-year long term contract (Strong retention factor)"
            elif "One year" in contract:
                return "One-year contract commitment (Moderate retention)"
                
        if "tenure" in feat_name:
            if tenure <= 12:
                return f"Low customer tenure ({int(tenure)} months - onboarding vulnerability)"
            elif tenure >= 48:
                return f"High tenure loyalty ({int(tenure)} months of relationship)"
            else:
                return f"Mid-tier tenure duration ({int(tenure)} months)"
                
        if "MonthlyCharges" in feat_name or "charge_per_service" in feat_name or "charges_to_tenure_ratio" in feat_name:
            if monthly > 80:
                return f"High monthly charges (${monthly:.2f}/mo billing pressure)"
            elif monthly < 35:
                return f"Low monthly cost (${monthly:.2f}/mo affordable tier)"
                
        if "InternetService" in feat_name:
            if "Fiber optic" in internet:
                return "Fiber optic broadband subscription (Higher churn sensitivity)"
            elif "DSL" in internet:
                return "DSL internet connection (Stable low-cost tier)"
            elif "No" in internet:
                return "No internet service (Low complexity user)"
                
        if "PaymentMethod" in feat_name:
            if "Electronic check" in payment:
                return "Payment via Electronic Check (Manual payment friction)"
            elif "automatic" in payment:
                return f"Automated billing setup ({payment})"
                
        if "TechSupport" in feat_name or "has_security" in feat_name:
            if tech_support == "No":
                return "Absence of Tech Support assistance"
            elif tech_support == "Yes":
                return "Active Tech Support coverage (Enhances satisfaction)"
                
        if "OnlineSecurity" in feat_name:
            if online_sec == "No":
                return "Lack of Online Security add-on"
            elif online_sec == "Yes":
                return "Online Security protection enabled"
                
        if "PaperlessBilling" in feat_name:
            if raw.get("PaperlessBilling") == "Yes":
                return "Paperless billing active"
                
        if "SeniorCitizen" in feat_name:
            if raw.get("SeniorCitizen") in [1, "1", "Yes"]:
                return "Senior Citizen customer profile"
                
        if "num_services" in feat_name:
            return "Multi-product adoption (Higher switching friction)"
            
        return ""

    def _rule_based_factors(self, raw: dict) -> list[str]:
        """
        Rule-based backup factors in case SHAP signals are subtle.
        """
        factors = []
        if raw.get("Contract") == "Month-to-month":
            factors.append("Month-to-month contract")
        if float(raw.get("tenure", 0)) <= 12:
            factors.append(f"Short tenure ({int(raw.get('tenure', 0))} months)")
        if float(raw.get("MonthlyCharges", 0)) >= 75:
            factors.append(f"High monthly charges (${raw.get('MonthlyCharges')})")
        if raw.get("TechSupport") == "No":
            factors.append("No technical support subscription")
        if raw.get("InternetService") == "Fiber optic":
            factors.append("Fiber optic internet plan")
        return factors

    def _generate_retention_recommendations(self, raw: dict, proba: float, top_factors: list[str]) -> list[dict]:
        """
        Generates contextual retention recommendations based on customer attributes and risk level.
        """
        recs = []
        contract = raw.get("Contract", "")
        tenure = float(raw.get("tenure", 0))
        monthly = float(raw.get("MonthlyCharges", 0))
        tech_support = raw.get("TechSupport", "")
        payment = raw.get("PaymentMethod", "")
        
        if contract == "Month-to-month":
            recs.append({
                "action": "Offer Annual Contract Incentive",
                "detail": "Provide a 15% discount on a 1-year or 2-year commitment to eliminate month-to-month churn risk.",
                "priority": "High"
            })
            
        if tech_support == "No" and raw.get("InternetService") != "No":
            recs.append({
                "action": "Complimentary Tech Support Trial",
                "detail": "Enroll customer in a 3-month free Tech Support and Security bundle to improve product experience.",
                "priority": "High" if proba >= 0.65 else "Medium"
            })
            
        if "Electronic check" in payment:
            recs.append({
                "action": "Migrate to Auto-Pay Discount",
                "detail": "Offer a $5 monthly bill credit for switching to automated Credit Card or Bank Transfer payments.",
                "priority": "Medium"
            })
            
        if monthly > 80 and tenure < 24:
            recs.append({
                "action": "Loyalty Plan Optimization",
                "detail": "Review high monthly spending and suggest a customized bundle to deliver better perceived value.",
                "priority": "High"
            })
            
        if not recs:
            recs.append({
                "action": "Standard Customer Check-In",
                "detail": "Maintain regular customer touchpoints and periodically share satisfaction surveys.",
                "priority": "Low"
            })
            
        return recs[:3]

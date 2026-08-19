import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer for Telco Customer Churn feature engineering.
    
    Engineered features:
    - tenure_group: Binned tenure buckets (0-12m, 13-24m, 25-48m, 49-60m, >60m)
    - num_services: Count of active services subscribed (0 to 9)
    - charge_per_service: MonthlyCharges / (num_services + 1)
    - has_streaming: Flag indicating presence of StreamingTV or StreamingMovies
    - has_security: Flag indicating presence of OnlineSecurity or TechSupport or DeviceProtection
    - contract_risk: High/Medium/Low risk classification based on contract commitment
    - charges_to_tenure_ratio: MonthlyCharges / (tenure + 1)
    """
    def __init__(self):
        self.service_cols = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies'
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Create copy to prevent mutating the original DataFrame
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X).copy()
            
        # Clean TotalCharges if present as string
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
            
        # Ensure tenure and MonthlyCharges are numeric
        if 'tenure' in df.columns:
            df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
        if 'MonthlyCharges' in df.columns:
            df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0)
            
        # 1. Tenure Group
        if 'tenure' in df.columns:
            bins = [-1, 12, 24, 48, 60, np.inf]
            labels = ['0-12m', '13-24m', '25-48m', '49-60m', '>60m']
            df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels).astype(str)
        else:
            df['tenure_group'] = '0-12m'
            
        # 2. Number of Subscribed Services
        service_count = pd.Series(0, index=df.index)
        for col in self.service_cols:
            if col in df.columns:
                service_count += (df[col] == 'Yes').astype(int)
        if 'InternetService' in df.columns:
            service_count += (df['InternetService'].isin(['DSL', 'Fiber optic'])).astype(int)
        df['num_services'] = service_count
        
        # 3. Charge per service ratio
        if 'MonthlyCharges' in df.columns:
            df['charge_per_service'] = df['MonthlyCharges'] / (df['num_services'] + 1)
        else:
            df['charge_per_service'] = 0.0
            
        # 4. Has Streaming
        streaming_tv = (df['StreamingTV'] == 'Yes') if 'StreamingTV' in df.columns else False
        streaming_movies = (df['StreamingMovies'] == 'Yes') if 'StreamingMovies' in df.columns else False
        df['has_streaming'] = np.where(streaming_tv | streaming_movies, 'Yes', 'No')
        
        # 5. Has Security / Protection / Support
        sec = (df['OnlineSecurity'] == 'Yes') if 'OnlineSecurity' in df.columns else False
        tech = (df['TechSupport'] == 'Yes') if 'TechSupport' in df.columns else False
        dev = (df['DeviceProtection'] == 'Yes') if 'DeviceProtection' in df.columns else False
        backup = (df['OnlineBackup'] == 'Yes') if 'OnlineBackup' in df.columns else False
        df['has_security'] = np.where(sec | tech | dev | backup, 'Yes', 'No')
        
        # 6. Contract Risk
        if 'Contract' in df.columns:
            df['contract_risk'] = df['Contract'].map({
                'Month-to-month': 'High',
                'One year': 'Medium',
                'Two year': 'Low'
            }).fillna('Medium')
        else:
            df['contract_risk'] = 'Medium'
            
        # 7. Charges to Tenure ratio
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['charges_to_tenure_ratio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        else:
            df['charges_to_tenure_ratio'] = 0.0
            
        # Remove customerID if present
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
            
        return df

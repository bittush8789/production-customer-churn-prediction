import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

try:
    from ml.src.feature_engineering import FeatureEngineer
except ImportError:
    from feature_engineering import FeatureEngineer

NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'num_services', 'charge_per_service', 'charges_to_tenure_ratio'
]

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'tenure_group', 'has_streaming', 'has_security', 'contract_risk'
]

def build_preprocessing_pipeline() -> Pipeline:
    """
    Constructs an end-to-end preprocessing pipeline including
    feature engineering, numerical scaling, and one-hot encoding.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline([
        ('engineer', FeatureEngineer()),
        ('transformer', col_transformer)
    ])

    return full_pipeline

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe: converts TotalCharges to numeric, handles blank spaces,
    and strips whitespace.
    """
    df_clean = df.copy()
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].astype(str).str.strip(), errors='coerce')
        # Impute missing TotalCharges with MonthlyCharges * tenure if tenure > 0, else 0.0
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(
            df_clean['MonthlyCharges'] * df_clean['tenure']
        ).fillna(0.0)
    return df_clean

def get_transformed_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """
    Extracts output feature names from the fitted preprocessing pipeline.
    """
    col_transformer = fitted_pipeline.named_steps['transformer']
    cat_encoder = col_transformer.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    return NUMERICAL_FEATURES + list(cat_feature_names)

def save_preprocessing_pipeline(pipeline: Pipeline, file_path: str):
    """
    Saves preprocessing pipeline to disk.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(pipeline, file_path)
    print(f"Preprocessing pipeline saved to {file_path}")

def load_preprocessing_pipeline(file_path: str) -> Pipeline:
    """
    Loads preprocessing pipeline from disk.
    """
    return joblib.load(file_path)

if __name__ == "__main__":
    from load_data import ensure_dataset
    df = ensure_dataset()
    df_cleaned = clean_raw_data(df)
    
    X = df_cleaned.drop(columns=['Churn', 'customerID'], errors='ignore')
    y = (df_cleaned['Churn'] == 'Yes').astype(int)
    
    pipeline = build_preprocessing_pipeline()
    X_proc = pipeline.fit_transform(X)
    feature_names = get_transformed_feature_names(pipeline)
    
    print(f"Processed feature matrix shape: {X_proc.shape}")
    print(f"Number of generated features: {len(feature_names)}")
    print(f"First 10 features: {feature_names[:10]}")

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ml.src.load_data import ensure_dataset
from ml.src.preprocessing import (
    build_preprocessing_pipeline,
    clean_raw_data,
    get_transformed_feature_names,
    save_preprocessing_pipeline
)
from ml.src.evaluate import evaluate_model, print_evaluation_summary

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def train_and_compare_models(random_state: int = 42):
    """
    Complete ML training workflow:
    1. Load & clean data
    2. Stratified train/test split
    3. Fit preprocessing pipeline
    4. Apply SMOTE to training data
    5. Train & hyperparameter-tune Logistic Regression, Random Forest, XGBoost
    6. Evaluate all models on held-out test data
    7. Select best model based on ROC-AUC, Recall & F1
    8. Save artifacts (churn_model.pkl, preprocessing.pkl, model_meta.json)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("\n--- [1/6] Loading & Cleaning Dataset ---")
    df = ensure_dataset()
    df_clean = clean_raw_data(df)
    
    X = df_clean.drop(columns=['Churn', 'customerID'], errors='ignore')
    y = (df_clean['Churn'] == 'Yes').astype(int)
    
    print(f"Total samples: {len(df_clean)}, Features: {X.shape[1]}")
    print(f"Churn class balance: 0={sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%), 1={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    
    print("\n--- [2/6] Stratified Train/Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    print("\n--- [3/6] Preprocessing Pipeline Fitting ---")
    pipeline = build_preprocessing_pipeline()
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)
    
    feature_names = get_transformed_feature_names(pipeline)
    print(f"Preprocessed feature dimensions: {X_train_proc.shape[1]} features")
    
    print("\n--- [4/6] Handling Class Imbalance with SMOTE ---")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
    print(f"After SMOTE: Train samples = {len(y_train_res)} (Balanced 50/50)")
    
    # Stratified K-Fold for Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    print("\n--- [5/6] Training & Tuning Models ---")
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_params = {
        'C': [0.01, 0.1, 1.0, 5.0],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=random_state, class_weight='balanced'),
        lr_params, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    lr_grid.fit(X_train_proc, y_train)
    best_lr = lr_grid.best_estimator_
    lr_metrics = evaluate_model(best_lr, X_test_proc, y_test, model_name="Logistic Regression")
    print_evaluation_summary(lr_metrics)
    
    # 2. Random Forest
    print("Training Random Forest Classifier...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, 15],
        'min_samples_split': [4, 8],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        rf_params, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    rf_grid.fit(X_train_proc, y_train)
    best_rf = rf_grid.best_estimator_
    rf_metrics = evaluate_model(best_rf, X_test_proc, y_test, model_name="Random Forest")
    print_evaluation_summary(rf_metrics)
    
    # 3. XGBoost Classifier
    print("Training XGBoost Classifier...")
    scale_pos = (sum(y_train == 0) / sum(y_train == 1))
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.03, 0.08, 0.15],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1.0, scale_pos]
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=random_state, eval_metric='logloss'),
        xgb_params, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    xgb_grid.fit(X_train_proc, y_train)
    best_xgb = xgb_grid.best_estimator_
    xgb_metrics = evaluate_model(best_xgb, X_test_proc, y_test, model_name="XGBoost")
    print_evaluation_summary(xgb_metrics)
    
    print("\n--- [6/6] Model Selection & Export ---")
    candidates = [
        {"name": "Logistic Regression", "model": best_lr, "metrics": lr_metrics},
        {"name": "Random Forest", "model": best_rf, "metrics": rf_metrics},
        {"name": "XGBoost", "model": best_xgb, "metrics": xgb_metrics}
    ]
    
    # Selection criteria: Composite score weighting ROC-AUC (40%), F1 (30%), and Recall (30%)
    # In churn prediction, catching churners (Recall) while maintaining high discriminative power (ROC-AUC) is key
    for cand in candidates:
        m = cand["metrics"]
        composite = 0.40 * m["roc_auc"] + 0.30 * m["f1_score"] + 0.30 * m["recall"]
        cand["composite_score"] = round(composite, 4)
        print(f"Model: {cand['name']:<20} | ROC-AUC: {m['roc_auc']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | Composite: {composite:.4f}")
        
    best_candidate = max(candidates, key=lambda c: c["composite_score"])
    print(f"\n WINNER: {best_candidate['name']} with Composite Score = {best_candidate['composite_score']:.4f}")
    
    # Save artifacts
    model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
    pipeline_path = os.path.join(MODELS_DIR, "preprocessing.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    
    joblib.dump(best_candidate["model"], model_path)
    save_preprocessing_pipeline(pipeline, pipeline_path)
    
    # Calculate baseline stats for dashboard overview
    total_cust = len(df_clean)
    churned_cust = int((df_clean['Churn'] == 'Yes').sum())
    churn_rate = round(churned_cust / total_cust * 100, 2)
    
    meta_data = {
        "best_model_name": best_candidate["name"],
        "composite_score": best_candidate["composite_score"],
        "metrics": best_candidate["metrics"],
        "all_models_comparison": [
            {
                "model_name": c["name"],
                "accuracy": c["metrics"]["accuracy"],
                "precision": c["metrics"]["precision"],
                "recall": c["metrics"]["recall"],
                "f1_score": c["metrics"]["f1_score"],
                "roc_auc": c["metrics"]["roc_auc"],
                "composite_score": c["composite_score"]
            }
            for c in candidates
        ],
        "feature_names": feature_names,
        "dataset_stats": {
            "total_customers": total_cust,
            "churned_customers": churned_cust,
            "retained_customers": total_cust - churned_cust,
            "churn_rate": churn_rate,
            "high_risk_estimate": int(total_cust * 0.28)
        }
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
        
    print(f"\nArtifacts successfully saved:")
    print(f"  • Model: {model_path}")
    print(f"  • Preprocessing: {pipeline_path}")
    print(f"  • Metadata: {meta_path}")
    
    return best_candidate, pipeline, meta_data

if __name__ == "__main__":
    train_and_compare_models()

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model") -> dict:
    """
    Evaluates a trained model on test data and returns a comprehensive metrics dictionary.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    return metrics

def print_evaluation_summary(metrics: dict):
    """
    Prints a formatted evaluation report to stdout.
    """
    print(f"\n{'='*50}")
    print(f" EVALUATION REPORT: {metrics['model_name'].upper()}")
    print(f"{'='*50}")
    print(f"Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision : {metrics['precision'] * 100:.2f}%")
    print(f"Recall    : {metrics['recall'] * 100:.2f}%  (Churn detection rate)")
    print(f"F1 Score  : {metrics['f1_score'] * 100:.2f}%")
    print(f"ROC-AUC   : {metrics['roc_auc'] * 100:.2f}%")
    print(f"\nConfusion Matrix:\nTN={metrics['confusion_matrix'][0][0]}  FP={metrics['confusion_matrix'][0][1]}\nFN={metrics['confusion_matrix'][1][0]}  TP={metrics['confusion_matrix'][1][1]}")
    print(f"{'='*50}\n")

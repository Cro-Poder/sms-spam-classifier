"""
Version 2.1 Training Pipeline: Utility-Fraud Augmented Feature Extraction & Calibrated Linear SVM.
Trains on historical + V2.1 modern utility-augmented training folds.
Saves artifacts/final_model_v2_1.joblib.
Preserves Version 1 and Version 2 artifacts untouched.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from src.data_preprocessing import load_raw_dataset
from src.data_preprocessing_v2 import V2WordPreprocessor, char_clean_v2
from src.evaluation import plot_confusion_matrix

sns.set_theme(style="whitegrid", font_scale=1.1)


def build_v2_1_pipeline(
    word_max_features: int = 3800,
    char_max_features: int = 4800,
    C: float = 1.0,
    cv_calib: int = 5,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs the Version 2.1 production pipeline:
    1. Word-level TF-IDF (1-2 n-grams with deobfuscation and stemming)
    2. Character-level TF-IDF (3-5 char_wb n-grams)
    3. FeatureUnion combining word and sub-word representations
    4. Calibrated Linear SVM with 5-fold cross-validation Platt scaling
    """
    combined_features = FeatureUnion([
        ('word_tfidf', Pipeline([
            ('prep', V2WordPreprocessor(stem=True, remove_stopwords=True)),
            ('vec', TfidfVectorizer(
                max_features=word_max_features,
                ngram_range=(1, 2),
                sublinear_tf=True
            ))
        ])),
        ('char_tfidf', Pipeline([
            ('vec', TfidfVectorizer(
                preprocessor=char_clean_v2,
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=char_max_features,
                sublinear_tf=True
            ))
        ]))
    ])
    
    base_svc = LinearSVC(C=C, random_state=random_state, max_iter=2500)
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_svc,
        method='sigmoid',
        cv=cv_calib
    )
    
    return Pipeline([
        ('features', combined_features),
        ('classifier', calibrated_clf)
    ])


def run_training_v2_1(
    original_data_path: str = "data/spam.csv",
    modern_train_path: str = "data/modern_sms_v2_1_augmented_train.csv",
    base_modern_eval_path: str = "data/modern_sms_heldout_eval.csv",
    extended_modern_eval_path: str = "data/modern_sms_v2_1_heldout_eval.csv",
    artifacts_dir: str = "artifacts",
    figures_dir: str = "figures",
    random_state: int = 42
):
    """
    Executes Version 2.1 training, evaluation, and serialization workflow.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 80)
    print("VERSION 2.1 TRAINING & CALIBRATION PIPELINE")
    print("=" * 80)
    
    # 1. Load and Split Original Dataset (strictly identical split)
    df_orig = load_raw_dataset(original_data_path)
    X_orig = df_orig['text']
    y_orig = df_orig['label']
    
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig,
        test_size=0.20,
        random_state=random_state,
        stratify=y_orig
    )
    print(f"Historical Train: {len(X_train_orig)} | Historical Held-Out Test: {len(X_test_orig)}")
    
    # 2. Augment Training Set with V2.1 Modern Data (Preserving strict isolation)
    if os.path.exists(modern_train_path):
        df_modern_train = pd.read_csv(modern_train_path)
        print(f"Loaded {len(df_modern_train)} modern V2.1 training templates ({sum(df_modern_train['label'] == 0)} Ham, {sum(df_modern_train['label'] == 1)} Spam).")
        X_train_combined = pd.concat([X_train_orig, df_modern_train['text']], ignore_index=True)
        y_train_combined = pd.concat([y_train_orig, df_modern_train['label']], ignore_index=True)
    else:
        print("Warning: Modern V2.1 training path not found. Using historical data only.")
        X_train_combined = X_train_orig
        y_train_combined = y_train_orig
        
    print(f"Total V2.1 Training Corpus: {len(X_train_combined)} messages.")
    
    # 3. Fit Version 2.1 Champion Pipeline
    pipeline_v2_1 = build_v2_1_pipeline(random_state=random_state)
    print("Fitting V2.1 FeatureUnion and Calibrated Linear SVM...")
    pipeline_v2_1.fit(X_train_combined, y_train_combined)
    print("Fitting complete.")
    
    # 4. Evaluate on Historical Held-Out Test Set (1,034 samples)
    y_pred_hist = pipeline_v2_1.predict(X_test_orig)
    y_probs_hist = pipeline_v2_1.predict_proba(X_test_orig)[:, 1]
    
    cm_hist = confusion_matrix(y_test_orig, y_pred_hist)
    tn_h, fp_h, fn_h, tp_h = cm_hist.ravel()
    
    hist_metrics = {
        "accuracy": float(accuracy_score(y_test_orig, y_pred_hist)),
        "precision": float(precision_score(y_test_orig, y_pred_hist)),
        "recall": float(recall_score(y_test_orig, y_pred_hist)),
        "f1_score": float(f1_score(y_test_orig, y_pred_hist)),
        "roc_auc": float(roc_auc_score(y_test_orig, y_probs_hist)),
        "pr_auc": float(average_precision_score(y_test_orig, y_probs_hist)),
        "false_positives": int(fp_h),
        "false_negatives": int(fn_h),
        "total_test_samples": int(len(y_test_orig))
    }
    
    print("\n--- HISTORICAL TEST SET EVALUATION (V2.1) ---")
    print(f"Accuracy:        {hist_metrics['accuracy']:.4f}")
    print(f"Precision:       {hist_metrics['precision']:.4f}")
    print(f"Recall:          {hist_metrics['recall']:.4f}")
    print(f"F1-Score:        {hist_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:         {hist_metrics['roc_auc']:.4f}")
    print(f"PR-AUC:          {hist_metrics['pr_auc']:.4f}")
    print(f"Confusion Matrix: TN={tn_h}, FP={fp_h}, FN={fn_h}, TP={tp_h}")
    
    # 5. Evaluate on Base Modern Held-Out Eval Set (60 samples)
    base_modern_metrics = {}
    if os.path.exists(base_modern_eval_path):
        df_base_eval = pd.read_csv(base_modern_eval_path)
        y_pred_base = pipeline_v2_1.predict(df_base_eval['text'])
        y_probs_base = pipeline_v2_1.predict_proba(df_base_eval['text'])[:, 1]
        cm_b = confusion_matrix(df_base_eval['label'], y_pred_base)
        tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
        
        base_modern_metrics = {
            "accuracy": float(accuracy_score(df_base_eval['label'], y_pred_base)),
            "precision": float(precision_score(df_base_eval['label'], y_pred_base)),
            "recall": float(recall_score(df_base_eval['label'], y_pred_base)),
            "f1_score": float(f1_score(df_base_eval['label'], y_pred_base)),
            "roc_auc": float(roc_auc_score(df_base_eval['label'], y_probs_base)),
            "pr_auc": float(average_precision_score(df_base_eval['label'], y_probs_base)),
            "false_positives": int(fp_b),
            "false_negatives": int(fn_b),
            "total_samples": int(len(df_base_eval))
        }
        print("\n--- BASE MODERN HELDOUT SET EVALUATION (60 samples) ---")
        print(f"Accuracy:        {base_modern_metrics['accuracy']:.4f}")
        print(f"Precision:       {base_modern_metrics['precision']:.4f}")
        print(f"Recall:          {base_modern_metrics['recall']:.4f}")
        print(f"F1-Score:        {base_modern_metrics['f1_score']:.4f}")
        print(f"PR-AUC:          {base_modern_metrics['pr_auc']:.4f}")
        print(f"Confusion Matrix: TN={tn_b}, FP={fp_b}, FN={fn_b}, TP={tp_b}")
        
    # 6. Evaluate on Extended V2.1 Modern Held-Out Eval Set (80 samples: 40 Ham, 40 Spam)
    ext_modern_metrics = {}
    if os.path.exists(extended_modern_eval_path):
        df_ext_eval = pd.read_csv(extended_modern_eval_path)
        y_pred_ext = pipeline_v2_1.predict(df_ext_eval['text'])
        y_probs_ext = pipeline_v2_1.predict_proba(df_ext_eval['text'])[:, 1]
        cm_e = confusion_matrix(df_ext_eval['label'], y_pred_ext)
        tn_e, fp_e, fn_e, tp_e = cm_e.ravel()
        
        ext_modern_metrics = {
            "accuracy": float(accuracy_score(df_ext_eval['label'], y_pred_ext)),
            "precision": float(precision_score(df_ext_eval['label'], y_pred_ext)),
            "recall": float(recall_score(df_ext_eval['label'], y_pred_ext)),
            "f1_score": float(f1_score(df_ext_eval['label'], y_pred_ext)),
            "roc_auc": float(roc_auc_score(df_ext_eval['label'], y_probs_ext)),
            "pr_auc": float(average_precision_score(df_ext_eval['label'], y_probs_ext)),
            "false_positives": int(fp_e),
            "false_negatives": int(fn_e),
            "total_samples": int(len(df_ext_eval))
        }
        print("\n--- EXTENDED V2.1 MODERN HELDOUT EVALUATION (80 samples, incl. utility fraud) ---")
        print(f"Accuracy:        {ext_modern_metrics['accuracy']:.4f}")
        print(f"Precision:       {ext_modern_metrics['precision']:.4f}")
        print(f"Recall:          {ext_modern_metrics['recall']:.4f}")
        print(f"F1-Score:        {ext_modern_metrics['f1_score']:.4f}")
        print(f"PR-AUC:          {ext_modern_metrics['pr_auc']:.4f}")
        print(f"Confusion Matrix: TN={tn_e}, FP={fp_e}, FN={fn_e}, TP={tp_e}")

    # 7. Save Model and Metadata Artifacts (Preserving V1 and V2)
    model_artifact_path = os.path.join(artifacts_dir, "final_model_v2_1.joblib")
    joblib.dump(pipeline_v2_1, model_artifact_path)
    print(f"\nSaved V2.1 model artifact to: {model_artifact_path}")
    
    metadata = {
        "model_version": "2.1.0",
        "model_name": "Calibrated Linear Support Vector Machine (Utility-Augmented Multi-Feature)",
        "classifier_type": "CalibratedClassifierCV(LinearSVC)",
        "feature_extraction": "FeatureUnion(Word TF-IDF (1,2) + Char_wb TF-IDF (3,5))",
        "calibration_method": "Platt Sigmoid Scaling (5-fold CV)",
        "training_samples_total": len(X_train_combined),
        "historical_train_samples": len(X_train_orig),
        "modern_augmented_train_samples": len(df_modern_train) if os.path.exists(modern_train_path) else 0,
        "historical_test_metrics": hist_metrics,
        "base_modern_eval_metrics": base_modern_metrics,
        "extended_modern_eval_metrics": ext_modern_metrics
    }
    
    metadata_path = os.path.join(artifacts_dir, "model_metadata_v2_1.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved V2.1 metadata to: {metadata_path}")
    
    # 8. Save Confusion Matrix Figure
    cm_fig_path = os.path.join(figures_dir, "champion_v2_1_confusion_matrix.png")
    plot_confusion_matrix(
        cm_hist,
        labels=["HAM", "SPAM"],
        title="Version 2.1 Calibrated SVM Confusion Matrix (Historical Test Set)",
        save_path=cm_fig_path
    )
    print(f"Saved V2.1 confusion matrix to: {cm_fig_path}")


if __name__ == "__main__":
    run_training_v2_1()


"""
Version 2 Training Pipeline: Multi-Feature Extraction & Calibrated Linear SVM.
Trains on historical + modern training folds and saves artifacts/final_model_v2.joblib.
Preserves Version 1 artifacts untouched.
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


def build_v2_pipeline(
    word_max_features: int = 3500,
    char_max_features: int = 4500,
    C: float = 1.0,
    cv_calib: int = 5,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs the Version 2 production pipeline:
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


def run_training_v2(
    original_data_path: str = "data/spam.csv",
    modern_train_path: str = "data/modern_sms_augmented_train.csv",
    modern_eval_path: str = "data/modern_sms_heldout_eval.csv",
    artifacts_dir: str = "artifacts",
    figures_dir: str = "figures",
    random_state: int = 42
):
    """
    Executes Version 2 training, evaluation, and serialization workflow.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 80)
    print("VERSION 2 TRAINING & CALIBRATION PIPELINE")
    print("=" * 80)
    
    # 1. Load and Split Original Dataset (strictly identical split)
    df_orig = load_raw_dataset(original_data_path)
    X_orig = df_orig['text']
    y_orig = df_orig['label']
    
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig, test_size=0.20, random_state=random_state, stratify=y_orig
    )
    
    # 2. Augment Training Fold with Modern Training Samples
    if os.path.exists(modern_train_path):
        df_mod_train = pd.read_csv(modern_train_path)
        X_train_aug = pd.concat([X_train_orig, df_mod_train['text']], ignore_index=True)
        y_train_aug = pd.concat([y_train_orig, df_mod_train['label']], ignore_index=True)
        print(f"Historical Train samples: {len(X_train_orig)}")
        print(f"Modern Train samples added: {len(df_mod_train)} (Ham: {(df_mod_train['label']==0).sum()}, Spam: {(df_mod_train['label']==1).sum()})")
        print(f"Total V2 Train samples: {len(X_train_aug)}")
    else:
        X_train_aug, y_train_aug = X_train_orig, y_train_orig
        print("Modern train path not found; training on original split only.")
        
    # 3. Build and Fit Pipeline
    pipeline_v2 = build_v2_pipeline(random_state=random_state)
    print("\nFitting Version 2 Pipeline (FeatureUnion + Calibrated LinearSVC)...")
    pipeline_v2.fit(X_train_aug, y_train_aug)
    print("Fitting complete.")
    
    # 4. Evaluate on Historical Test Set (1,034 samples)
    preds_orig = pipeline_v2.predict(X_test_orig)
    probs_orig = pipeline_v2.predict_proba(X_test_orig)[:, 1]
    
    acc_orig = float(accuracy_score(y_test_orig, preds_orig))
    prec_orig = float(precision_score(y_test_orig, preds_orig, zero_division=0))
    rec_orig = float(recall_score(y_test_orig, preds_orig, zero_division=0))
    f1_orig = float(f1_score(y_test_orig, preds_orig, zero_division=0))
    auc_orig = float(roc_auc_score(y_test_orig, probs_orig))
    pr_auc_orig = float(average_precision_score(y_test_orig, probs_orig))
    
    cm_orig = confusion_matrix(y_test_orig, preds_orig)
    tn_orig, fp_orig, fn_orig, tp_orig = cm_orig.ravel()
    
    print("\n--- Version 2 Results on Historical Test Set (1,034 samples) ---")
    print(f"Accuracy:  {acc_orig * 100:.2f}%")
    print(f"Precision: {prec_orig * 100:.2f}%")
    print(f"Recall:    {rec_orig * 100:.2f}%")
    print(f"F1-Score:  {f1_orig * 100:.2f}%")
    print(f"ROC-AUC:   {auc_orig:.4f}")
    print(f"PR-AUC:    {pr_auc_orig:.4f}")
    print(f"Confusion Matrix: TN={tn_orig}, FP={fp_orig}, FN={fn_orig}, TP={tp_orig}")
    
    # 5. Evaluate on Modern Heldout Test Set (60 samples)
    acc_mod, prec_mod, rec_mod, f1_mod, auc_mod, pr_auc_mod = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    tn_mod, fp_mod, fn_mod, tp_mod = 0, 0, 0, 0
    
    if os.path.exists(modern_eval_path):
        df_mod_eval = pd.read_csv(modern_eval_path)
        preds_mod = pipeline_v2.predict(df_mod_eval['text'])
        probs_mod = pipeline_v2.predict_proba(df_mod_eval['text'])[:, 1]
        
        acc_mod = float(accuracy_score(df_mod_eval['label'], preds_mod))
        prec_mod = float(precision_score(df_mod_eval['label'], preds_mod, zero_division=0))
        rec_mod = float(recall_score(df_mod_eval['label'], preds_mod, zero_division=0))
        f1_mod = float(f1_score(df_mod_eval['label'], preds_mod, zero_division=0))
        auc_mod = float(roc_auc_score(df_mod_eval['label'], probs_mod))
        pr_auc_mod = float(average_precision_score(df_mod_eval['label'], probs_mod))
        
        cm_mod = confusion_matrix(df_mod_eval['label'], preds_mod)
        tn_mod, fp_mod, fn_mod, tp_mod = cm_mod.ravel()
        
        print("\n--- Version 2 Results on Modern Held-Out Evaluation Set (60 samples) ---")
        print(f"Accuracy:  {acc_mod * 100:.2f}%")
        print(f"Precision: {prec_mod * 100:.2f}%")
        print(f"Recall:    {rec_mod * 100:.2f}%")
        print(f"F1-Score:  {f1_mod * 100:.2f}%")
        print(f"Confusion Matrix: TN={tn_mod}, FP={fp_mod}, FN={fn_mod}, TP={tp_mod}")
        
    # 6. Save Version 2 Pipeline Artifact
    v2_model_path = os.path.join(artifacts_dir, "final_model_v2.joblib")
    joblib.dump(pipeline_v2, v2_model_path)
    print(f"\nSerialized Version 2 Pipeline saved to: {v2_model_path}")
    
    # 7. Plot Version 2 Confusion Matrices
    fig_cm_orig = plot_confusion_matrix(
        cm_orig,
        title="Version 2: Historical Test Confusion Matrix",
        save_path=os.path.join(figures_dir, "champion_v2_confusion_matrix.png")
    )
    
    # 8. Save Metadata
    metadata_v2 = {
        "version": "2.0.0",
        "model_name": "Calibrated Linear SVM (Combined Word + Char TF-IDF)",
        "model_path": v2_model_path,
        "historical_test_metrics": {
            "accuracy": acc_orig,
            "precision": prec_orig,
            "recall": rec_orig,
            "f1_score": f1_orig,
            "roc_auc": auc_orig,
            "pr_auc": pr_auc_orig,
            "true_negatives": int(tn_orig),
            "false_positives": int(fp_orig),
            "false_negatives": int(fn_orig),
            "true_positives": int(tp_orig),
            "confusion_matrix": cm_orig.tolist()
        },
        "modern_eval_metrics": {
            "accuracy": acc_mod,
            "precision": prec_mod,
            "recall": rec_mod,
            "f1_score": f1_mod,
            "roc_auc": auc_mod,
            "pr_auc": pr_auc_mod,
            "true_negatives": int(tn_mod),
            "false_positives": int(fp_mod),
            "false_negatives": int(fn_mod),
            "true_positives": int(tp_mod)
        },
        "train_samples": len(X_train_aug),
        "test_samples": len(X_test_orig),
        "modern_eval_samples": len(df_mod_eval) if os.path.exists(modern_eval_path) else 0,
        "random_state": random_state
    }
    
    with open(os.path.join(artifacts_dir, "model_metadata_v2.json"), "w") as f:
        json.dump(metadata_v2, f, indent=4)
    print(f"Version 2 metadata saved to: {os.path.join(artifacts_dir, 'model_metadata_v2.json')}")
    print("=" * 80)
    return metadata_v2


if __name__ == "__main__":
    run_training_v2()

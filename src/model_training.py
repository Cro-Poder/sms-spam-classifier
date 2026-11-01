"""
Model Training, Cross-Validation, Hyperparameter Tuning, and Pipeline Serialization.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve

from src.data_preprocessing import load_raw_dataset, TextPreprocessor
from src.evaluation import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves
)


def build_pipeline(classifier, max_features=3000, ngram_range=(1, 1)):
    """
    Constructs an end-to-end Pipeline ensuring strict prevention of data leakage.
    Input raw strings -> TextPreprocessor -> TfidfVectorizer -> Classifier.
    """
    return Pipeline([
        ('preprocessor', TextPreprocessor(stem=True, remove_stopwords=True)),
        ('vectorizer', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ('classifier', classifier)
    ])


def run_training_and_experiments(
    data_path: str = "data/spam.csv",
    artifacts_dir: str = "artifacts",
    figures_dir: str = "figures",
    random_state: int = 42
):
    """
    Executes the full machine learning training workflow:
    1. Loads and deduplicates dataset.
    2. Performs stratified train-test split (80/20).
    3. Runs 5-Fold Stratified Cross-Validation on the training set across 6 models.
    4. Tunes hyperparameters on promising models using GridSearchCV.
    5. Fits pipelines on training data and evaluates on held-out test data.
    6. Selects the champion model based on Precision, Recall, and F1-score.
    7. Serializes the genuinely fitted pipeline and saves metadata and visualizations.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 70)
    print("STEP 1: LOADING DATASET AND STRATIFIED SPLIT")
    print("=" * 70)
    df = load_raw_dataset(data_path)
    print(f"Total dataset records (after deduplication): {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts(normalize=True).rename({0: 'HAM (0)', 1: 'SPAM (1)'}) * 100}")
    
    X = df['text']
    y = df['label']
    
    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    print(f"\nTrain set: {len(X_train)} samples (Ham: {np.sum(y_train == 0)}, Spam: {np.sum(y_train == 1)})")
    print(f"Test set:  {len(X_test)} samples (Ham: {np.sum(y_test == 0)}, Spam: {np.sum(y_test == 1)})")
    
    # Candidate Classifiers
    candidate_classifiers = {
        "MultinomialNB": MultinomialNB(alpha=0.2),
        "BernoulliNB": BernoulliNB(alpha=0.2),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=random_state),
        "SVC": SVC(kernel='sigmoid', gamma=1.0, probability=True, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    }
    
    print("\n" + "=" * 70)
    print("STEP 2: 5-FOLD STRATIFIED CROSS-VALIDATION (TRAINING SET ONLY)")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    cv_records = []
    fitted_pipelines = {}
    
    for name, clf in candidate_classifiers.items():
        pipe = build_pipeline(clf)
        print(f"Running 5-fold CV for: {name} ...", end=" ", flush=True)
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        
        cv_records.append({
            "Model": name,
            "CV_Accuracy_Mean": float(np.mean(cv_res['test_accuracy'])),
            "CV_Accuracy_Std": float(np.std(cv_res['test_accuracy'])),
            "CV_Precision_Mean": float(np.mean(cv_res['test_precision'])),
            "CV_Precision_Std": float(np.std(cv_res['test_precision'])),
            "CV_Recall_Mean": float(np.mean(cv_res['test_recall'])),
            "CV_Recall_Std": float(np.std(cv_res['test_recall'])),
            "CV_F1_Mean": float(np.mean(cv_res['test_f1'])),
            "CV_F1_Std": float(np.std(cv_res['test_f1'])),
            "CV_ROC_AUC_Mean": float(np.mean(cv_res['test_roc_auc'])),
            "CV_ROC_AUC_Std": float(np.std(cv_res['test_roc_auc']))
        })
        print("Done.")
        
    cv_summary_df = pd.DataFrame(cv_records)
    print("\n--- 5-Fold Stratified Cross-Validation Summary ---")
    print(cv_summary_df[["Model", "CV_Accuracy_Mean", "CV_Precision_Mean", "CV_Recall_Mean", "CV_F1_Mean", "CV_ROC_AUC_Mean"]].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("STEP 3: HYPERPARAMETER TUNING VIA GRIDSEARCHCV")
    print("=" * 70)
    
    # Tune MultinomialNB and LogisticRegression on training set
    mnb_pipeline = build_pipeline(MultinomialNB())
    mnb_param_grid = {
        'vectorizer__max_features': [2500, 3000, 4000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.1, 0.2, 0.5, 1.0]
    }
    print("Tuning MultinomialNB ...")
    mnb_grid = GridSearchCV(
        mnb_pipeline,
        mnb_param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=1
    )
    mnb_grid.fit(X_train, y_train)
    print(f"Best MNB Params: {mnb_grid.best_params_} | Best CV F1: {mnb_grid.best_score_:.4f}")
    
    # Tune SVC
    svc_pipeline = build_pipeline(SVC(probability=True, random_state=random_state))
    svc_param_grid = {
        'vectorizer__max_features': [3000],
        'classifier__C': [0.5, 1.0, 2.0],
        'classifier__kernel': ['sigmoid', 'linear']
    }
    print("Tuning SVC ...")
    svc_grid = GridSearchCV(
        svc_pipeline,
        svc_param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=1
    )
    svc_grid.fit(X_train, y_train)
    print(f"Best SVC Params: {svc_grid.best_params_} | Best CV F1: {svc_grid.best_score_:.4f}")
    
    # Replace pipelines with tuned configurations
    candidate_pipelines = {
        "MultinomialNB (Tuned)": mnb_grid.best_estimator_,
        "SVC (Tuned)": svc_grid.best_estimator_,
        "BernoulliNB": build_pipeline(candidate_classifiers["BernoulliNB"]),
        "LogisticRegression": build_pipeline(candidate_classifiers["LogisticRegression"]),
        "RandomForest": build_pipeline(candidate_classifiers["RandomForest"]),
        "ExtraTrees": build_pipeline(candidate_classifiers["ExtraTrees"])
    }
    
    print("\n" + "=" * 70)
    print("STEP 4: FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)
    
    test_results = []
    roc_dict = {}
    
    for name, pipe in candidate_pipelines.items():
        # Fit on entire training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Predict on test set
        y_pred = pipe.predict(X_test)
        
        y_proba = None
        if hasattr(pipe.named_steps['classifier'], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_dict[name] = (fpr, tpr, roc_records_auc := evaluate_predictions(y_test, y_pred, y_proba)["roc_auc"])
            
        metrics = evaluate_predictions(y_test, y_pred, y_proba)
        test_results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1_Score": metrics["f1_score"],
            "Specificity": metrics["specificity"],
            "ROC_AUC": metrics["roc_auc"],
            "FP": metrics["false_positives"],
            "FN": metrics["false_negatives"],
            "TP": metrics["true_positives"],
            "TN": metrics["true_negatives"],
            "Confusion_Matrix": metrics["confusion_matrix"]
        })
        
    test_results_df = pd.DataFrame(test_results)
    print("\n--- Test Set Evaluation Results ---")
    print(test_results_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "FP", "FN"]].to_string(index=False))
    
    # Save comparison dataframe to CSV
    test_results_df.to_csv(os.path.join(artifacts_dir, "model_comparison.csv"), index=False)
    cv_summary_df.to_csv(os.path.join(artifacts_dir, "cv_summary.csv"), index=False)
    
    # Generate and save comparison plots
    plot_model_comparison(test_results_df, save_path=os.path.join(figures_dir, "model_comparison.png"))
    plot_roc_curves(roc_dict, save_path=os.path.join(figures_dir, "roc_curves.png"))
    
    print("\n" + "=" * 70)
    print("STEP 5: MODEL SELECTION & ARTIFACT SERIALIZATION")
    print("=" * 70)
    
    # Model Selection Logic:
    # False positives in spam filtering mean important legitimate communications are lost.
    # Therefore, 100% precision (FP = 0) with maximal F1 score is the optimal criterion.
    zero_fp_models = test_results_df[test_results_df["FP"] == 0]
    if not zero_fp_models.empty:
        best_model_name = zero_fp_models.sort_values(by="F1_Score", ascending=False).iloc[0]["Model"]
    else:
        # Fallback to highest precision, then highest F1
        best_model_name = test_results_df.sort_values(by=["Precision", "F1_Score"], ascending=[False, False]).iloc[0]["Model"]
        
    print(f"Selected Champion Model: {best_model_name}")
    champion_pipeline = fitted_pipelines[best_model_name]
    champion_metrics = test_results_df[test_results_df["Model"] == best_model_name].iloc[0].to_dict()
    
    # Plot champion confusion matrix
    cm_arr = np.array(champion_metrics["Confusion_Matrix"])
    plot_confusion_matrix(
        cm_arr,
        title=f"Confusion Matrix: {best_model_name}",
        save_path=os.path.join(figures_dir, "champion_confusion_matrix.png")
    )
    
    # Save champion pipeline
    model_path = os.path.join(artifacts_dir, "final_model.joblib")
    joblib.dump(champion_pipeline, model_path)
    print(f"Serialized full fitted pipeline saved to: {model_path}")
    
    # Verify the saved model works immediately
    loaded_model = joblib.load(model_path)
    test_sample = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
    verification_pred = loaded_model.predict(test_sample)
    verification_proba = loaded_model.predict_proba(test_sample)[:, 1] if hasattr(loaded_model, "predict_proba") else None
    print(f"Verification test prediction for sample spam: {verification_pred[0]} (Proba: {verification_proba})")
    
    # Metadata export
    metadata = {
        "model_name": best_model_name,
        "model_path": model_path,
        "test_metrics": {
            "accuracy": champion_metrics["Accuracy"],
            "precision": champion_metrics["Precision"],
            "recall": champion_metrics["Recall"],
            "f1_score": champion_metrics["F1_Score"],
            "specificity": champion_metrics["Specificity"],
            "roc_auc": champion_metrics["ROC_AUC"],
            "false_positives": champion_metrics["FP"],
            "false_negatives": champion_metrics["FN"],
            "true_positives": champion_metrics["TP"],
            "true_negatives": champion_metrics["TN"],
            "confusion_matrix": champion_metrics["Confusion_Matrix"]
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": random_state,
        "pipeline_steps": list(champion_pipeline.named_steps.keys()),
        "hyperparameters": str(champion_pipeline.get_params())
    }
    
    with open(os.path.join(artifacts_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model metadata saved to: {os.path.join(artifacts_dir, 'model_metadata.json')}")
    print("=" * 70)
    print("TRAINING & EVALUATION COMPLETE!")
    print("=" * 70)
    return metadata, test_results_df


if __name__ == "__main__":
    run_training_and_experiments()

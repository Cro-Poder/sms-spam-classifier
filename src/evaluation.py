"""
Model Evaluation and Visualization Module for SMS Spam Classification.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def evaluate_predictions(y_true, y_pred, y_proba=None) -> dict:
    """
    Computes standard evaluation metrics for binary classification.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_proba: Predicted probabilities for positive class (spam, 1).
        
    Returns:
        dict: Summary of metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    auc = None
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            auc = None
            
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "roc_auc": auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist()
    }


def plot_confusion_matrix(cm, labels=["HAM", "SPAM"], title="Confusion Matrix", save_path=None):
    """
    Plots a visually clear Confusion Matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=10)
    ax.set_title(title, fontweight="bold", pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, save_path=None):
    """
    Plots a multi-bar chart comparing accuracy, precision, recall, and f1 across models.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    melted = comparison_df.melt(
        id_vars=["Model"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Score"
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("Set2", len(available_metrics))
    
    sns.barplot(
        data=melted,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette,
        ax=ax
    )
    
    ax.set_title("Model Comparison on Held-Out Test Set", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Model Architecture", fontweight="bold")
    ax.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
    ax.set_ylim(0.70, 1.02)
    ax.legend(title="Metric", loc="lower right")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig


def plot_roc_curves(roc_dict: dict, save_path=None):
    """
    Plots superimposed ROC Curves for models that output probability estimates.
    roc_dict structure: {model_name: (fpr, tpr, auc_score)}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, (fpr, tpr, auc_val) in roc_dict.items():
        label = f"{name} (AUC = {auc_val:.4f})" if auc_val is not None else name
        ax.plot(fpr, tpr, lw=2, label=label)
        
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", pad=15)
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig


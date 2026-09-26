"""
NEXORA Model Evaluation Pipeline
Evaluates the trained wake-word classifier on held-out test data.
Computes:
- Overall Accuracy
- NEXORA Precision, Recall, F1
- Confusion Matrix
- False NEXORA Detections (False Positives)
- Missed NEXORA Detections (False Negatives)
- Generates logs/evaluation_report.json and logs/confusion_matrix.png
"""

import os
import sys
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from software.app import config


def evaluate_model(
    model_path: Path = config.MODEL_PATH,
    data_path: Path = config.DATASET_DIR / "dataset_features.npz",
    report_output_path: Path = config.EVALUATION_REPORT_PATH,
    cm_image_path: Path = config.CONFUSION_MATRIX_IMAGE,
) -> Dict[str, Any]:
    """
    Evaluates trained model on test data and produces formal report and chart.
    """
    print("==================================================")
    print(" NEXORA MODEL EVALUATION & BENCHMARKING")
    print(f" Model Path: {model_path}")
    print(f" Data Path : {data_path}")
    print("==================================================")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found. Train the model first.")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file {data_path} not found. Prepare the dataset first.")

    pipeline = joblib.load(str(model_path))
    data = np.load(str(data_path))

    X_test, y_test = data["X_test"], data["y_test"]
    X_val, y_val = data["X_val"], data["y_val"]

    # If test set is empty, fall back to val or full dataset for initial evaluation
    eval_set_name = "test"
    X_eval, y_eval = X_test, y_test

    if len(X_eval) == 0:
        if len(X_val) > 0:
            print("[INFO] Test set is empty, using validation set for evaluation.")
            X_eval, y_eval = X_val, y_val
            eval_set_name = "validation"
        else:
            print("[INFO] Test and val sets are empty, evaluating on training set (warning: small dataset).")
            X_eval, y_eval = data["X_train"], data["y_train"]
            eval_set_name = "train (diagnostic)"

    sample_count = len(X_eval)
    print(f"Evaluating on {sample_count} samples from {eval_set_name} set.")

    if sample_count == 0:
        print("[ERROR] No samples available for evaluation.")
        report = {"status": "NO_EVALUATION_SAMPLES", "sample_count": 0}
        with open(report_output_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    y_pred = pipeline.predict(X_eval)
    y_prob = pipeline.predict_proba(X_eval)

    classes = config.CLASSES
    n_classes = len(classes)
    acc = float(accuracy_score(y_eval, y_pred))

    # Confusion matrix with fixed labels [0, 1, 2]
    cm = confusion_matrix(y_eval, y_pred, labels=[0, 1, 2])

    # Per-class metrics
    p, r, f1, s = precision_recall_fscore_support(
        y_eval, y_pred, labels=[0, 1, 2], zero_division=0
    )

    nexora_idx = config.CLASS_TO_IDX["NEXORA"]
    nexora_precision = float(p[nexora_idx])
    nexora_recall = float(r[nexora_idx])
    nexora_f1 = float(f1[nexora_idx])

    # False NEXORA Detections (False Positives: actual was NOT Nexora, but predicted Nexora)
    false_nexora_detections = int(cm[0, nexora_idx] + cm[1, nexora_idx])

    # Missed NEXORA Detections (False Negatives: actual was Nexora, but predicted Silence or Unknown)
    missed_nexora_detections = int(cm[nexora_idx, 0] + cm[nexora_idx, 1])

    is_small_dataset = sample_count < 30

    print(f"\n--- Evaluation Results ({eval_set_name.upper()} SET) ---")
    print(f"Overall Accuracy          : {acc * 100:.2f}%")
    print(f"NEXORA Precision          : {nexora_precision * 100:.2f}%")
    print(f"NEXORA Recall (TPR)       : {nexora_recall * 100:.2f}%")
    print(f"NEXORA F1-Score           : {nexora_f1:.4f}")
    print(f"False NEXORA Detections   : {false_nexora_detections}")
    print(f"Missed NEXORA Detections  : {missed_nexora_detections}")

    print("\nConfusion Matrix:")
    print(f"                Pred: SILENCE  UNKNOWN  NEXORA")
    for i, cls_name in enumerate(classes):
        print(f"Actual: {cls_name:8s}       {cm[i, 0]:5d}    {cm[i, 1]:5d}   {cm[i, 2]:5d}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=classes,
        yticklabels=classes,
        title=f"NEXORA KWS Confusion Matrix ({eval_set_name.title()} Set)",
        ylabel="Ground Truth Label",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    cm_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(cm_image_path), dpi=200)
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_image_path}")

    report = {
        "evaluation_dataset": eval_set_name,
        "sample_count": sample_count,
        "is_dataset_too_small": is_small_dataset,
        "overall_accuracy": acc,
        "nexora_metrics": {
            "precision": nexora_precision,
            "recall": nexora_recall,
            "f1_score": nexora_f1,
            "false_nexora_detections": false_nexora_detections,
            "missed_nexora_detections": missed_nexora_detections,
        },
        "per_class_metrics": {
            classes[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1_score": float(f1[i]),
                "support": int(s[i])
            }
            for i in range(n_classes)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_image": str(cm_image_path),
        "notes": (
            "Dataset sample size is under 30 samples. Recommended to record more variations."
            if is_small_dataset else "Evaluation completed with sufficient test samples."
        )
    }

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation report written to: {report_output_path}")
    print("==================================================")
    return report


if __name__ == "__main__":
    evaluate_model()

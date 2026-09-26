"""
NEXORA Wake Word Model Training Pipeline
Trains a lightweight Neural Network (MLP) on 16 kHz MFCC features.
Target classes: SILENCE, UNKNOWN, NEXORA.
Architecture designed for direct INT8 / TinyML conversion on ESP32-S3.

Saves:
- models/nexora_model.joblib (Trained model + scaler pipeline)
- models/model_metadata.json (Detailed model metrics and architecture specs)
"""

import os
import sys
import json
import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from software.app import config


def train_model(
    data_path: Path = config.DATASET_DIR / "dataset_features.npz",
    model_output_path: Path = config.MODEL_PATH,
    metadata_output_path: Path = config.MODEL_METADATA_PATH,
    hidden_layer_sizes: tuple = (64, 32),
    max_iter: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains a lightweight neural network for wake-word classification.
    """
    print("==================================================")
    print(" NEXORA MODEL TRAINING (TinyML Compatible MLP)")
    print(f" Dataset source: {data_path}")
    print("==================================================")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset feature file {data_path} not found. Please run 'prepare_dataset.py' first."
        )

    data = np.load(str(data_path))
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples      : {len(X_test)}")
    print(f"Feature dimension : {X_train.shape[1] if len(X_train) > 0 else 'N/A'}")

    if len(X_train) == 0:
        raise ValueError("Training dataset contains 0 samples. Collect samples first.")

    unique_classes = np.unique(y_train)
    print(f"Classes present in training set: {[config.IDX_TO_CLASS.get(c, c) for c in unique_classes]}")

    if len(unique_classes) < 2:
        print("[WARNING] Fewer than 2 distinct classes in dataset. At least 2 classes required to train a valid classifier.")
        print("Please collect samples for NEXORA, UNKNOWN, and SILENCE.")

    # Create Lightweight Pipeline: StandardScaler -> MLPClassifier
    # Architecture: Input(1274) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(3, Softmax)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=min(32, max(4, len(X_train))),
        learning_rate_init=0.001,
        early_stopping=False,
        random_state=random_state,
        verbose=False
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", mlp)
    ])

    print("\nStarting training...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_duration_sec = time.time() - start_time
    print(f"Training completed in {training_duration_sec:.2f} seconds.")

    # Validation accuracy
    val_acc = 0.0
    if len(X_val) > 0:
        y_val_pred = pipeline.predict(X_val)
        val_acc = float(accuracy_score(y_val, y_val_pred))
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    else:
        print("Validation Accuracy: N/A (Insufficient validation samples)")

    # Test accuracy
    test_acc = 0.0
    if len(X_test) > 0:
        y_test_pred = pipeline.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_test_pred))
        print(f"Test Accuracy      : {test_acc * 100:.2f}%")
    else:
        print("Test Accuracy      : N/A (Insufficient test samples)")

    # Save Pipeline Model
    joblib.dump(pipeline, str(model_output_path))
    model_size_bytes = os.path.getsize(str(model_output_path))
    print(f"Saved model to: {model_output_path} ({model_size_bytes / 1024:.2f} KB)")

    # Compute parameter counts
    coefs = mlp.coefs_
    intercepts = mlp.intercepts_
    total_params = sum(w.size for w in coefs) + sum(b.size for b in intercepts)

    metadata = {
        "model_type": "Multi-Layer Perceptron (MLP) Classifier",
        "framework": "scikit-learn (TinyML weights extractable to C array)",
        "input_shape": [config.EXPECTED_NUM_FRAMES, config.N_MFCC],
        "input_dim_flattened": config.FLATTENED_FEATURE_DIM,
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "total_parameters": int(total_params),
        "estimated_int8_weights_kb": float(total_params / 1024.0),
        "output_classes": config.CLASSES,
        "classes_in_training": [config.IDX_TO_CLASS.get(int(c), str(c)) for c in unique_classes],
        "model_size_bytes": int(model_size_bytes),
        "training_sample_count": int(len(X_train)),
        "validation_sample_count": int(len(X_val)),
        "test_sample_count": int(len(X_test)),
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "training_duration_sec": training_duration_sec,
        "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_target_notes": "Compatible with ESP32-S3 SRAM (512KB) and ESP-NN / TinyML acceleration"
    }

    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_output_path}")
    print("==================================================")
    return metadata


if __name__ == "__main__":
    train_model()

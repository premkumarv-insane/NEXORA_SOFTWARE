"""
NEXORA Keyword Spotting (KWS) Inference Module
Performs feature extraction and small custom classifier inference.
Outputs probabilities for: [SILENCE, UNKNOWN, NEXORA]
Tracks software execution latency for feature extraction and inference.
"""

import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from software.app import config
    from software.app.features import FeatureExtractor
except ImportError:
    import config
    from features import FeatureExtractor


class NexoraKWS:
    """
    Local Wake Word Classifier for 'NEXORA'.
    Pure local execution — NO external APIs, NO cloud services.
    """

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        metadata_path: Path = config.MODEL_METADATA_PATH,
        confidence_threshold: float = config.KWS_CONFIDENCE_THRESHOLD
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = FeatureExtractor()

        self.model = None
        self.metadata = {}
        self.classes = config.CLASSES
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        """Loads trained model weights and metadata."""
        if not self.model_path.exists():
            print(f"[KWS WARNING] Model file {self.model_path} not found. Train model first.")
            return

        try:
            self.model = joblib.load(str(self.model_path))
            if self.metadata_path.exists():
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            self.is_loaded = True
            print(f"[KWS] Model loaded successfully: {self.metadata.get('model_type', 'MLP Classifier')}")
        except Exception as e:
            print(f"[KWS ERROR] Failed to load model from {self.model_path}: {e}")
            self.is_loaded = False

    def predict(self, audio_window: np.ndarray) -> Dict[str, Any]:
        """
        Runs feature extraction and classification on an audio window (16000 samples).
        
        Returns:
            Dict containing:
            - probabilities: dict of {class_name: prob}
            - predicted_class: str
            - nexora_score: float (prob of NEXORA)
            - confidence: float
            - feature_latency_ms: float
            - inference_latency_ms: float
            - total_latency_ms: float
        """
        if not self.is_loaded or self.model is None:
            # Fallback if model not trained yet
            return {
                "probabilities": {"SILENCE": 1.0, "UNKNOWN": 0.0, "NEXORA": 0.0},
                "predicted_class": "SILENCE",
                "nexora_score": 0.0,
                "confidence": 1.0,
                "feature_latency_ms": 0.0,
                "inference_latency_ms": 0.0,
                "total_latency_ms": 0.0,
                "model_loaded": False
            }

        # 1. Feature Extraction & Latency Timing
        t0 = time.perf_counter()
        features = self.feature_extractor.extract_features(audio_window, flatten=True)
        t1 = time.perf_counter()
        feature_latency_ms = (t1 - t0) * 1000.0

        # Reshape for single sample prediction
        X = features.reshape(1, -1)

        # 2. Classifier Inference & Timing
        t2 = time.perf_counter()
        probs = self.model.predict_proba(X)[0]
        t3 = time.perf_counter()
        inference_latency_ms = (t3 - t2) * 1000.0
        total_latency_ms = (t3 - t0) * 1000.0

        # Map to class names
        prob_dict = {}
        for idx, cls_name in enumerate(self.classes):
            if idx < len(probs):
                prob_dict[cls_name] = float(probs[idx])
            else:
                prob_dict[cls_name] = 0.0

        nexora_score = prob_dict.get("NEXORA", 0.0)
        best_class = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_class]

        return {
            "probabilities": prob_dict,
            "predicted_class": best_class,
            "nexora_score": nexora_score,
            "confidence": confidence,
            "feature_latency_ms": feature_latency_ms,
            "inference_latency_ms": inference_latency_ms,
            "total_latency_ms": total_latency_ms,
            "model_loaded": True
        }

from pathlib import Path
from typing import Any

from app.core.config import MODEL_FILE
import joblib
import numpy as np


class DeepfakeClassifier:
    """
    Deepfake voice classifier.

    Pipeline:
        Embedding
            ↓
        StandardScaler
            ↓
        SVM Classifier
            ↓
        Prediction + Confidence
    """

    # Below this confidence, prediction will be marked as UNCERTAIN
    UNCERTAIN_THRESHOLD = 60.0

    def __init__(self) -> None:

        model_path = (
            Path(__file__).resolve().parents[2]
            / "training"
            / "models"
            / MODEL_FILE
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        saved = joblib.load(model_path)
        print("=" * 60)
        print("MODEL PATH:", model_path)
        print("MODEL FILE:", MODEL_FILE)
        print("MODEL KEYS:", saved.keys())
        print("=" * 60)


        self.model = saved["model"]
        self.scaler = saved["scaler"]
        
        self.model_name = saved["model_name"]
        self.accuracy = saved["accuracy"]
        self.version = saved["version"]
        self.feature_type = saved["feature_type"]
        self.embedding_size = saved["embedding_size"]

    def _validate_embedding(
        self,
        embedding: np.ndarray
    ) -> None:

        if embedding.ndim != 1:
            raise ValueError(
                "Embedding must be one-dimensional."
            )

        if embedding.shape[0] != self.embedding_size:
            raise ValueError(
                f"Expected embedding size "
                f"{self.embedding_size}, "
                f"got {embedding.shape[0]}."
            )

        if np.isnan(embedding).any():
            raise ValueError(
                "Embedding contains NaN values."
            )

        if np.isinf(embedding).any():
            raise ValueError(
                "Embedding contains infinite values."
            )

    @staticmethod
    def _confidence_label(
        probability: float
    ) -> str:

        if probability >= 95:
            return "Very High"

        if probability >= 90:
            return "High"

        if probability >= 80:
            return "Medium"

        if probability >= 70:
            return "Low"

        return "Very Low"

    def predict(
        self,
        embedding: np.ndarray
    ) -> dict[str, Any]:

        self._validate_embedding(embedding)

        embedding = self.scaler.transform(
            [embedding]
        )

        prediction = int(
            self.model.predict(embedding)[0]
        )

        probabilities = self.model.predict_proba(
            embedding
        )[0]

        probability = float(
            probabilities[prediction]
        )

        probability_percent = round(
            probability * 100,
            2
        )

        if probability_percent < self.UNCERTAIN_THRESHOLD:

            label = "UNCERTAIN"

            message = (
                "Model confidence is too low. "
                "Please upload a clearer recording."
            )

        else:

            label = (
                "REAL"
                if prediction == 0
                else "FAKE"
            )

            message = "Prediction completed successfully."

        return {

            "prediction": label,

            "probability": probability_percent,

            "confidence": self._confidence_label(
                probability_percent
            ),

            "message": message,

            "model": self.model_name,

            "model_version": self.version,

            "accuracy": round(
                self.accuracy * 100,
                2
            ),

            "feature_type": self.feature_type,

            "embedding_size": self.embedding_size

        }


classifier = DeepfakeClassifier()
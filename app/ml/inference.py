import time
from pathlib import Path
from typing import Any

from app.ml.classifier import classifier
from app.ml.feature_extractor import feature_extractor


class InferenceEngine:
    """
    Runs the complete ML inference pipeline.

    Pipeline:
        Audio File
            ↓
        Audio Preprocessing
            ↓
        Wav2Vec2 Feature Extraction
            ↓
        SVM Classification
            ↓
        Prediction Response
    """

    def predict(
        self,
        file_path: str |Path
    ) -> dict[str, Any]:

        total_start = time.perf_counter()

        # ---------------------------------
        # Feature Extraction
        # ---------------------------------

        (
            embedding,
            audio_statistics,
            performance
        ) = feature_extractor.extract_embedding(
            file_path
        )

        # ---------------------------------
        # Classification
        # ---------------------------------

        classify_start = time.perf_counter()

        result = classifier.predict(
            embedding
        )

        performance["classification_ms"] = round(
            (time.perf_counter() - classify_start) * 1000,
            2
        )

        # ---------------------------------
        # Total Time
        # ---------------------------------

        total_time = (
            time.perf_counter() - total_start
        ) * 1000

        result["processing_time_ms"] = round(
            total_time,
            2
        )

        performance["total_ms"] = round(
            total_time,
            2
        )

        result["performance"] = performance

        result["audio_statistics"] = audio_statistics

        return result


inference_engine = InferenceEngine()
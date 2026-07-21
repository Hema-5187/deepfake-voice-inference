import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from app.ml.model_loader import model_loader
from app.ml.preprocessing import audio_preprocessor


class FeatureExtractor:
    """
    Extract Wav2Vec2 embeddings and collect performance metrics.
    """

    def __init__(self):
        
        model_loader.load()
        self.processor = model_loader.processor
        self.model = model_loader.model
        self.device = model_loader.device

    def _prepare_inputs(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> Dict[str, torch.Tensor]:

        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )

        return {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

    def _extract_hidden_states(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state

    @staticmethod
    def _mean_pool(
        hidden_states: torch.Tensor
    ) -> np.ndarray:

        embedding = (
            hidden_states
            .mean(dim=1)
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return embedding

    @staticmethod
    def _validate_embedding(
        embedding: np.ndarray
    ) -> None:

        if embedding.ndim != 1:
            raise ValueError("Embedding must be one-dimensional.")

        if embedding.shape[0] != 768:
            raise ValueError(
                f"Expected 768 dimensions, got {embedding.shape[0]}"
            )

        if np.isnan(embedding).any():
            raise ValueError("Embedding contains NaN values.")

        if np.isinf(embedding).any():
            raise ValueError("Embedding contains Inf values.")

    def extract_embedding(
        self,
        file_path: str | Path
    ) -> tuple[np.ndarray, dict, dict]:

        # -----------------------------
        # Audio Preprocessing
        # -----------------------------

        preprocess_start = time.perf_counter()

        waveform, sample_rate, stats = (
            audio_preprocessor.preprocess(file_path)
        )

        preprocess_time = (
            time.perf_counter() - preprocess_start
        ) * 1000

        # -----------------------------
        # Wav2Vec2
        # -----------------------------

        embedding_start = time.perf_counter()

        inputs = self._prepare_inputs(
            waveform,
            sample_rate
        )

        hidden_states = self._extract_hidden_states(
            inputs
        )

        embedding = self._mean_pool(
            hidden_states
        )

        embedding_time = (
            time.perf_counter() - embedding_start
        ) * 1000

        self._validate_embedding(
            embedding
        )

        performance = {

            "preprocessing_ms": round(
                preprocess_time,
                2
            ),

            "embedding_ms": round(
                embedding_time,
                2
            )

        }

        return embedding, stats, performance


feature_extractor = FeatureExtractor()
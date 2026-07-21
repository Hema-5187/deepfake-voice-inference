import time
from pathlib import Path
from typing import Dict, Tuple

import librosa
import noisereduce as nr
import numpy as np


class AudioPreprocessor:
    """
    Handles audio loading, preprocessing and validation.

    Pipeline:
        Load Audio
            ↓
        Validate
            ↓
        Trim Silence
            ↓
        Noise Reduction (Optional)
            ↓
        Normalize
            ↓
        Statistics
    """

    TARGET_SAMPLE_RATE = 16000

    MIN_DURATION_SECONDS = 2.0
    MAX_DURATION_SECONDS = 30.0

    SILENCE_TOP_DB = 20

    RMS_THRESHOLD = 0.005

    CLIPPING_THRESHOLD = 0.98

    # ==========================
    # Performance Setting
    # ==========================
    # False is recommended for production.
    ENABLE_NOISE_REDUCTION = False

    def __init__(self):
        self.performance = {}

    def load_audio(
        self,
        file_path: str | Path
    ) -> Tuple[np.ndarray, int]:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {file_path}"
            )

        waveform, sample_rate = librosa.load(
            file_path,
            sr=self.TARGET_SAMPLE_RATE,
            mono=True
        )

        if waveform.size == 0:
            raise ValueError("Audio file is empty.")

        return waveform.astype(np.float32), sample_rate

    def validate_duration(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> float:

        duration = len(waveform) / sample_rate

        if duration < self.MIN_DURATION_SECONDS:
            raise ValueError(
                f"Audio is too short ({duration:.2f}s). "
                f"Minimum allowed duration is "
                f"{self.MIN_DURATION_SECONDS:.1f}s."
            )

        if duration > self.MAX_DURATION_SECONDS:
            raise ValueError(
                f"Audio is too long ({duration:.2f}s). "
                f"Maximum allowed duration is "
                f"{self.MAX_DURATION_SECONDS:.1f}s."
            )

        return duration

    def trim_silence(
        self,
        waveform: np.ndarray
    ) -> np.ndarray:

        waveform, _ = librosa.effects.trim(
            waveform,
            top_db=self.SILENCE_TOP_DB
        )

        return waveform

    def reduce_noise(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:

        return nr.reduce_noise(
            y=waveform,
            sr=sample_rate
        )

    def normalize(
        self,
        waveform: np.ndarray
    ) -> np.ndarray:

        peak = np.max(np.abs(waveform))

        if peak > 0:
            waveform = waveform / peak

        return waveform.astype(np.float32)

    def validate_waveform(
        self,
        waveform: np.ndarray
    ) -> None:

        if np.isnan(waveform).any():
            raise ValueError("Audio contains NaN values.")

        if np.isinf(waveform).any():
            raise ValueError("Audio contains infinite values.")

        rms = float(
            librosa.feature.rms(y=waveform).mean()
        )

        if rms < self.RMS_THRESHOLD:
            raise ValueError(
                "Recording is too quiet. Please record again."
            )

    def get_statistics(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> Dict:

        duration = round(
            len(waveform) / sample_rate,
            2
        )

        rms = float(
            librosa.feature.rms(y=waveform).mean()
        )

        peak = float(
            np.max(np.abs(waveform))
        )

        return {

            "duration_seconds": duration,

            "sample_rate": sample_rate,

            "num_samples": len(waveform),

            "rms_energy": round(rms, 6),

            "peak_amplitude": round(peak, 6)

        }

    def preprocess(
        self,
        file_path: str | Path
    ) -> Tuple[np.ndarray, int, Dict]:

        self.performance = {}

        # ----------------------------
        # Load Audio
        # ----------------------------

        start = time.perf_counter()

        waveform, sample_rate = self.load_audio(
            file_path
        )

        self.performance["load_audio_ms"] = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        # ----------------------------
        # Duration Validation
        # ----------------------------

        start = time.perf_counter()

        self.validate_duration(
            waveform,
            sample_rate
        )

        self.performance["duration_validation_ms"] = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        # ----------------------------
        # Trim Silence
        # ----------------------------

        start = time.perf_counter()

        waveform = self.trim_silence(
            waveform
        )

        self.performance["trim_silence_ms"] = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        # ----------------------------
        # Noise Reduction (Optional)
        # ----------------------------

        if self.ENABLE_NOISE_REDUCTION:

            start = time.perf_counter()

            waveform = self.reduce_noise(
                waveform,
                sample_rate
            )

            self.performance["noise_reduction_ms"] = round(
                (time.perf_counter() - start) * 1000,
                2
            )

        else:

            self.performance["noise_reduction_ms"] = 0.0

        # ----------------------------
        # Normalize
        # ----------------------------

        start = time.perf_counter()

        waveform = self.normalize(
            waveform
        )

        self.performance["normalization_ms"] = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        # ----------------------------
        # Validate Waveform
        # ----------------------------

        start = time.perf_counter()

        self.validate_waveform(
            waveform
        )

        self.performance["waveform_validation_ms"] = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        statistics = self.get_statistics(
            waveform,
            sample_rate
        )

        return waveform, sample_rate, statistics


audio_preprocessor = AudioPreprocessor()
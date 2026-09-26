"""
NEXORA Local Automatic Speech Recognition (ASR) Interface
Converts captured command audio to textual transcript.
Uses Vosk open-source offline speech recognition toolkit.
100% Local execution — NO cloud speech APIs used.

Interface:
audio (16 kHz PCM / WAV file) -> transcript (str)
"""

import os
import sys
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Union, Optional

try:
    from software.app import config
except ImportError:
    import config


class LocalASREngine:
    """
    Open-source local speech-to-text engine using Vosk.
    Runs 100% on laptop CPU with zero internet connection required.
    """

    def __init__(self, model_name: str = "vosk-model-small-en-us-0.15"):
        self.model_name = model_name
        self.vosk_available = False
        self.recognizer = None
        self.model = None
        self.engine_type = "UNINITIALIZED"

        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes Vosk offline model."""
        try:
            import vosk
            vosk.SetLogLevel(-1)  # Suppress verbose Kaldi logging
            
            # Check ~/.cache/vosk or default paths
            cache_model_dir = Path.home() / ".cache" / "vosk" / self.model_name
            if cache_model_dir.exists():
                self.model = vosk.Model(str(cache_model_dir))
            else:
                # Try language loader
                self.model = vosk.Model(lang=config.ASR_LANG)

            self.vosk_available = True
            self.engine_type = "REAL_LOCAL_VOSK_ASR"
            print(f"[ASR] Local Vosk model loaded successfully ({self.engine_type}).")
        except Exception as e:
            self.vosk_available = False
            self.engine_type = "MOCK_FALLBACK_TEST_ONLY"
            print(f"[ASR WARNING] Local Vosk initialization notice: {e}")
            print("[ASR] Falling back to clearly marked Mock/Test ASR interface.")

    def transcribe(self, audio_input: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Transcribes 16 kHz audio buffer or WAV file into text.
        
        Args:
            audio_input: 1D numpy array (float32 [-1.0, 1.0]) or path to WAV file.
            
        Returns:
            Dict containing:
            - transcript: str
            - confidence: float
            - latency_ms: float
            - engine: str ("REAL_LOCAL_VOSK_ASR" or "MOCK_FALLBACK_TEST_ONLY")
            - is_mock: bool
        """
        t_start = time.perf_counter()

        # Load audio if path was provided
        if isinstance(audio_input, (str, Path)):
            audio_data, sr = sf.read(str(audio_input), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
        else:
            audio_data = audio_input

        # Convert float32 [-1, 1] to int16 PCM bytes
        audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        raw_bytes = audio_int16.tobytes()

        transcript = ""
        is_mock = not self.vosk_available

        if self.vosk_available and self.model is not None:
            import vosk
            rec = vosk.KaldiRecognizer(self.model, config.SAMPLE_RATE)
            rec.SetWords(True)

            # Process audio chunks
            chunk_size = 4000
            for i in range(0, len(raw_bytes), chunk_size):
                chunk = raw_bytes[i:i + chunk_size]
                rec.AcceptWaveform(chunk)

            final_res = json.loads(rec.FinalResult())
            transcript = final_res.get("text", "").strip()

            # If empty, check partial or silence
            if not transcript:
                transcript = "(no audible speech detected)"
        else:
            # Explicit Mock/Test fallback interface (Do NOT pretend this is real ASR)
            transcript = "[MOCK_TEST_TRANSCRIPT] turn on the laboratory lights"

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return {
            "transcript": transcript,
            "latency_ms": latency_ms,
            "engine": self.engine_type,
            "is_mock": is_mock,
            "audio_duration_sec": len(audio_data) / float(config.SAMPLE_RATE)
        }

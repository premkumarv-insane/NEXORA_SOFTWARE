"""
NEXORA Command Capture Module
Captures command audio immediately following confirmed wake word detection.
Saves audio to disk as 16 kHz mono WAV and prepares it for local ASR.

Guarantees:
- Dormant until NEXORA CONFIRMED.
- Zero audio is routed to ASR prior to wake-word confirmation.
"""

import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional

try:
    from software.app import config
except ImportError:
    import config


class CommandCapture:
    """
    Records command audio following confirmed wake word trigger.
    Features:
    - Fixed or VAD-based capture duration
    - 16 kHz Mono PCM audio recording
    - Automatic persistence to recordings/ directory
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        default_duration: float = config.COMMAND_DURATION_SEC,
        recordings_dir: Path = config.RECORDINGS_DIR,
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.default_duration = default_duration
        self.recordings_dir = recordings_dir
        self.device_index = device_index
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    def capture_command(
        self,
        duration: Optional[float] = None,
        save_file: bool = True
    ) -> Tuple[np.ndarray, Optional[Path], float]:
        """
        Records command audio from the microphone.
        
        Args:
            duration: Recording duration in seconds (defaults to config.COMMAND_DURATION_SEC)
            save_file: If True, writes WAV file to recordings directory
            
        Returns:
            Tuple of:
            - audio_array (np.ndarray float32)
            - filepath (Path or None)
            - capture_duration_sec (float)
        """
        rec_duration = duration if duration is not None else self.default_duration
        total_samples = int(rec_duration * self.sample_rate)

        t_start = time.perf_counter()

        # Capture audio block
        audio_data = sd.rec(
            total_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device_index
        )
        sd.wait()

        capture_duration = time.perf_counter() - t_start
        audio = audio_data.flatten()

        saved_path = None
        if save_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"command_{timestamp}.wav"
            saved_path = self.recordings_dir / filename
            audio_clipped = np.clip(audio, -1.0, 1.0)
            sf.write(str(saved_path), audio_clipped, self.sample_rate, subtype="PCM_16")

        return audio, saved_path, capture_duration

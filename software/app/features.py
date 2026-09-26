"""
NEXORA Feature Extraction Pipeline
Lightweight, reproducible acoustic feature extractor designed for 16 kHz audio.
Implements:
16 kHz PCM -> Framing -> Windowing -> FFT -> Mel Filterbank -> Log-Mel -> MFCC
Architecture and mathematics directly correspond to TinyML / ESP32-S3 implementations.

Configuration Documentation:
- Sample Rate: 16000 Hz
- Frame Length: 25.0 ms (400 samples)
- Frame Step: 10.0 ms (160 samples)
- FFT Length: 512 points
- Mel Filters: 40 triangular filterbanks (0 Hz to 8000 Hz)
- MFCC Coefficients: 13 (C0 - C12)
- Target Window Duration: 1.0 s (16000 samples)
- Output Feature Tensor Shape: (98 frames, 13 coefficients) = 1274 scalar values
"""

import numpy as np
from scipy.fftpack import dct
import soundfile as sf
from typing import Tuple, Optional, Union
from pathlib import Path

try:
    from software.app import config
except ImportError:
    import config


class FeatureExtractor:
    """
    Lightweight audio feature extractor for Wake Word Detection.
    Computes Log-Mel Spectrogram and MFCC features using pure numpy/scipy operations.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        frame_length_ms: float = config.FRAME_LENGTH_MS,
        frame_step_ms: float = config.FRAME_STEP_MS,
        n_fft: int = config.N_FFT,
        n_mels: int = config.N_MELS,
        n_mfcc: int = config.N_MFCC,
        preemphasis: float = config.PREEMPHASIS_COEFF,
        feature_type: str = config.FEATURE_TYPE,
    ):
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_length_ms / 1000.0))
        self.frame_step = int(sample_rate * (frame_step_ms / 1000.0))
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.preemphasis = preemphasis
        self.feature_type = feature_type

        # Precompute Hann window
        self.window = np.hanning(self.frame_length).astype(np.float32)

        # Precompute Mel Filterbank
        self.mel_filters = self._build_mel_filterbank()

    def _hz_to_mel(self, hz: np.ndarray) -> np.ndarray:
        """Convert frequency in Hz to Mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        """Convert Mel scale to frequency in Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> np.ndarray:
        """
        Constructs triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
        """
        f_min = 0.0
        f_max = self.sample_rate / 2.0  # Nyquist frequency: 8000 Hz

        mel_min = self._hz_to_mel(np.array(f_min))
        mel_max = self._hz_to_mel(np.array(f_max))

        # Linearly spaced points in Mel scale
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # FFT bin indices
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(np.int32)

        num_fft_bins = self.n_fft // 2 + 1
        filterbank = np.zeros((self.n_mels, num_fft_bins), dtype=np.float32)

        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1]
            f_m = bins[m]
            f_m_plus = bins[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return filterbank

    def pad_or_trim(self, audio: np.ndarray, target_length: int = config.SAMPLES_PER_WINDOW) -> np.ndarray:
        """
        Pads audio with zero reflection or crops to the target length (16000 samples / 1.0s).
        """
        if len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant", constant_values=0)
        elif len(audio) > target_length:
            audio = audio[:target_length]
        return audio.astype(np.float32)

    def extract_features(
        self,
        audio: np.ndarray,
        flatten: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract MFCC or Log-Mel features from raw 16 kHz audio buffer.
        
        Args:
            audio: 1D numpy array of audio samples (float32 normalized [-1.0, 1.0])
            flatten: If True, returns 1D vector (1274,), else 2D tensor (98, 13)
            normalize: If True, applies Cepstral Mean Subtraction (CMS)
            
        Returns:
            Extracted feature array
        """
        # Ensure correct window length
        audio = self.pad_or_trim(audio)

        # 1. Pre-emphasis filter: y[t] = x[t] - alpha * x[t-1]
        emphasized_audio = np.append(audio[0], audio[1:] - self.preemphasis * audio[:-1])

        # 2. Framing
        num_samples = len(emphasized_audio)
        num_frames = 1 + int(np.floor((num_samples - self.frame_length) / self.frame_step))

        if num_frames <= 0:
            num_frames = 1

        indices = (
            np.tile(np.arange(0, self.frame_length), (num_frames, 1)) +
            np.tile(np.arange(0, num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        )
        frames = emphasized_audio[indices]

        # 3. Windowing (Hann window)
        frames *= self.window

        # 4. Magnitude FFT & Power Spectrum
        mag_fft = np.abs(np.fft.rfft(frames, n=self.n_fft))
        power_spectrum = (mag_fft ** 2) / float(self.n_fft)

        # 5. Mel Filterbank Energy
        mel_energies = np.dot(power_spectrum, self.mel_filters.T)
        mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)

        # 6. Log-Mel Energies
        log_mel = np.log(mel_energies)

        if self.feature_type == "logmel":
            features = log_mel
        else:
            # 7. Discrete Cosine Transform (DCT-II) -> MFCC
            # Take first n_mfcc coefficients
            mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, : self.n_mfcc]
            features = mfcc

        # 8. Normalization (Cepstral Mean Subtraction)
        if normalize:
            features = features - (np.mean(features, axis=0, keepdims=True) + 1e-8)

        # Truncate or pad to exactly EXPECTED_NUM_FRAMES (98 frames)
        if features.shape[0] < config.EXPECTED_NUM_FRAMES:
            pad_width = config.EXPECTED_NUM_FRAMES - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode="edge")
        elif features.shape[0] > config.EXPECTED_NUM_FRAMES:
            features = features[:config.EXPECTED_NUM_FRAMES, :]

        if flatten:
            return features.flatten().astype(np.float32)
        return features.astype(np.float32)

    def extract_from_file(self, wav_path: Union[str, Path], flatten: bool = True) -> Optional[np.ndarray]:
        """
        Loads a WAV file, converts to 16 kHz mono, and extracts features.
        """
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  # Convert stereo to mono

            if sr != self.sample_rate:
                from scipy import signal
                num_resampled = int(len(audio) * float(self.sample_rate) / float(sr))
                audio = signal.resample(audio, num_resampled)

            return self.extract_features(audio, flatten=flatten)
        except Exception as e:
            print(f"[ERROR] Failed to extract features from {wav_path}: {e}")
            return None

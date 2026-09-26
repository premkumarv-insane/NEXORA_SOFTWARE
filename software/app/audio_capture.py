"""
NEXORA Real-Time Audio Capture
Continuously reads 16 kHz Mono PCM audio from the laptop microphone.
Maintains a thread-safe ring buffer and produces sliding audio windows
for real-time Wake Word feature extraction and inference.
"""

import time
import queue
import threading
import numpy as np
import sounddevice as sd
from typing import Optional, Generator

try:
    from software.app import config
except ImportError:
    import config


class AudioCaptureStream:
    """
    Manages continuous 16 kHz Mono audio capture from laptop microphone.
    Provides overlapping sliding audio windows of duration WINDOW_DURATION_SEC
    at intervals of STEP_DURATION_SEC.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
        window_duration: float = config.WINDOW_DURATION_SEC,
        step_duration: float = config.STEP_DURATION_SEC,
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.device_index = device_index

        self.window_samples = int(self.sample_rate * self.window_duration)
        self.step_samples = int(self.sample_rate * self.step_duration)

        # Ring buffer holds recent samples
        self.buffer_lock = threading.Lock()
        self.ring_buffer = np.zeros(self.window_samples, dtype=np.float32)

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._queue = queue.Queue(maxsize=50)

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback invoked by PortAudio / sounddevice with incoming audio."""
        if status:
            # Drop/overflow warning
            pass

        # Mono float32 samples
        audio_chunk = indata[:, 0].astype(np.float32)

        with self.buffer_lock:
            # Shift buffer left and append new audio chunk
            chunk_len = len(audio_chunk)
            if chunk_len >= self.window_samples:
                self.ring_buffer[:] = audio_chunk[-self.window_samples:]
            else:
                self.ring_buffer = np.roll(self.ring_buffer, -chunk_len)
                self.ring_buffer[-chunk_len:] = audio_chunk

            current_window = self.ring_buffer.copy()

        # Place copy in consumer queue if there's room
        try:
            self._queue.put_nowait(current_window)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(current_window)
            except (queue.Empty, queue.Full):
                pass

    def start(self):
        """Starts the audio capture stream."""
        if self._running:
            return

        self._running = True
        with self.buffer_lock:
            self.ring_buffer.fill(0.0)

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Blocksize set to step_samples to receive audio in steps (e.g. 200 ms chunks)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.step_samples,
            device=self.device_index,
            callback=self._audio_callback
        )
        self._stream.start()

    def stop(self):
        """Stops the audio capture stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def get_latest_window(self) -> np.ndarray:
        """Returns the most recent 1.0s audio window (thread-safe)."""
        with self.buffer_lock:
            return self.ring_buffer.copy()

    def stream_windows(self) -> Generator[np.ndarray, None, None]:
        """
        Generator yielding 1.0 second audio windows every step interval.
        """
        while self._running:
            try:
                window = self._queue.get(timeout=0.5)
                yield window
            except queue.Empty:
                continue

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

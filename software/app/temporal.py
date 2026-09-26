"""
NEXORA Temporal Verification Module
Prevents single-frame false triggers by requiring:
- Consecutive positive frames above configurable threshold
- Sliding confidence score smoothing
- Cooldown period after confirmed activation
- Clear state transitions: IDLE -> CANDIDATE -> CONFIRMED / REJECTED -> COOLDOWN
"""

import time
from collections import deque
from typing import Dict, Any, Tuple

try:
    from software.app import config
except ImportError:
    import config


class TemporalVerifier:
    """
    Temporal verifier for NEXORA wake-word triggers.
    Ensures that acoustic activations persist across time before triggering downstream ASR.
    """

    STATE_IDLE = "LISTENING"
    STATE_CANDIDATE = "NEXORA CANDIDATE"
    STATE_CONFIRMED = "NEXORA CONFIRMED"
    STATE_REJECTED = "NEXORA REJECTED"
    STATE_COOLDOWN = "COOLDOWN"

    def __init__(
        self,
        threshold: float = config.KWS_CONFIDENCE_THRESHOLD,
        consecutive_frames_required: int = config.TEMPORAL_CONSECUTIVE_FRAMES,
        window_size: int = config.TEMPORAL_WINDOW_SIZE,
        cooldown_sec: float = config.TEMPORAL_COOLDOWN_SEC
    ):
        self.threshold = threshold
        self.consecutive_frames_required = consecutive_frames_required
        self.window_size = window_size
        self.cooldown_sec = cooldown_sec

        self.confidence_history = deque(maxlen=window_size)
        self.consecutive_count = 0
        self.last_confirmed_time = 0.0
        self.current_state = self.STATE_IDLE
        self.total_candidates = 0
        self.total_confirmations = 0
        self.total_rejections = 0

    def update(self, nexora_score: float) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Processes a new frame's NEXORA score.
        
        Args:
            nexora_score: Probability [0.0, 1.0] of NEXORA from KWS classifier.
            
        Returns:
            Tuple of:
            - is_confirmed: True ONLY on the exact frame NEXORA is confirmed
            - state_string: Current state label
            - metadata: Dict with consecutive counts, smoothed score, timestamps
        """
        now = time.time()
        self.confidence_history.append(nexora_score)
        smoothed_score = sum(self.confidence_history) / len(self.confidence_history)

        # Check if in cooldown period
        if (now - self.last_confirmed_time) < self.cooldown_sec:
            self.current_state = self.STATE_COOLDOWN
            remaining = self.cooldown_sec - (now - self.last_confirmed_time)
            return False, self.STATE_COOLDOWN, {
                "cooldown_remaining_sec": remaining,
                "consecutive_count": 0,
                "smoothed_score": smoothed_score
            }

        is_above_threshold = (nexora_score >= self.threshold)
        is_confirmed = False

        if is_above_threshold:
            self.consecutive_count += 1
            if self.consecutive_count >= self.consecutive_frames_required:
                self.current_state = self.STATE_CONFIRMED
                self.last_confirmed_time = now
                self.consecutive_count = 0
                self.total_confirmations += 1
                is_confirmed = True
            else:
                self.current_state = self.STATE_CANDIDATE
                self.total_candidates += 1
        else:
            if self.current_state == self.STATE_CANDIDATE:
                self.current_state = self.STATE_REJECTED
                self.total_rejections += 1
            else:
                self.current_state = self.STATE_IDLE
            self.consecutive_count = 0

        info = {
            "score": nexora_score,
            "smoothed_score": smoothed_score,
            "consecutive_count": self.consecutive_count,
            "threshold": self.threshold,
            "required_consecutive": self.consecutive_frames_required,
            "total_confirmations": self.total_confirmations,
            "total_rejections": self.total_rejections
        }

        return is_confirmed, self.current_state, info

    def reset(self):
        """Resets the verifier state."""
        self.confidence_history.clear()
        self.consecutive_count = 0
        self.last_confirmed_time = 0.0
        self.current_state = self.STATE_IDLE

"""
NEXORA Context-Aware Adaptive KWS Controller
Deterministic edge controller dynamically modulating KWS inference frequency
based on real-time acoustic context estimation.

Pipeline Architecture:
Audio Window (16 kHz PCM)
    ↓
AcousticContextEstimator (RMS energy, noise floor, speech likelihood)
    ↓ Context: [QUIET | ACTIVE | CANDIDATE]
AdaptivePolicy (Deterministic decision & periodic safety check)
    ↓
if should_evaluate:
    NexoraKWS.predict()
    TemporalVerifier.update()
else:
    Preserve TemporalVerifier state safely (zero false confirmations)

Key Guarantees:
1. Lightweight & Deterministic: O(N) operations (~0.01 ms execution overhead).
2. Safety Constraint (MAX_KWS_GAP_MS): Periodic evaluation prevents missed wake words during silence.
3. Candidate Priority: CANDIDATE state forces KWS evaluation on 100% of available windows.
4. Verifier Isolation: AdaptivePolicy NEVER directly confirms a wake word.
   TemporalVerifier remains solely responsible for state confirmation.
"""

import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

try:
    from software.app import config
except ImportError:
    import config


@dataclass
class AcousticContext:
    """Represents the acoustic environmental properties of an audio window."""
    state: str                 # "QUIET", "ACTIVE", or "CANDIDATE"
    activity_level: float      # RMS energy of window
    speech_likelihood: float   # Normalized [0.0, 1.0] proxy based on energy above noise floor
    noise_floor: float         # Current estimated ambient noise floor
    is_speech: bool            # True if energy exceeds activity threshold
    peak_amplitude: float      # Max absolute amplitude in window
    snr_db: float              # Approximate SNR in dB relative to noise floor

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveDecision:
    """Represents the controller decision for the current audio window."""
    should_evaluate: bool      # True: run feature extraction & KWS inference; False: skip
    context_state: str         # Context state from estimator
    reason: str                # Human-readable explanation of decision
    gap_ms: float              # Milliseconds elapsed since last KWS evaluation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AcousticContextEstimator:
    """
    Lightweight, deterministic acoustic context estimator.
    Computes RMS signal energy, running noise floor, and speech likelihood
    in O(N) time with minimal CPU footprint.
    """

    STATE_QUIET = "QUIET"
    STATE_ACTIVE = "ACTIVE"
    STATE_CANDIDATE = "CANDIDATE"

    def __init__(
        self,
        activity_threshold: float = config.ADAPTIVE_ACTIVITY_THRESHOLD,
        candidate_threshold: float = config.ADAPTIVE_CANDIDATE_THRESHOLD,
        initial_noise_floor: float = 0.0005
    ):
        self.activity_threshold = activity_threshold
        self.candidate_threshold = candidate_threshold
        self.noise_floor = initial_noise_floor

    def estimate_context(
        self,
        audio_window: np.ndarray,
        last_kws_score: float = 0.0,
        current_temporal_state: Optional[str] = None
    ) -> AcousticContext:
        """
        Estimates the acoustic context for an audio window.

        Args:
            audio_window: 1D float32 audio samples.
            last_kws_score: Previous frame's NEXORA score from KWS classifier.
            current_temporal_state: Current state of TemporalVerifier (e.g. "NEXORA CANDIDATE").

        Returns:
            AcousticContext dataclass.
        """
        if len(audio_window) == 0:
            rms = 0.0
            peak = 0.0
        else:
            rms = float(np.sqrt(np.mean(audio_window ** 2)))
            peak = float(np.max(np.abs(audio_window)))

        is_speech = rms >= self.activity_threshold

        # Update running noise floor during quiet periods
        if not is_speech:
            self.noise_floor = float(np.clip(
                0.95 * self.noise_floor + 0.05 * rms,
                1e-6,
                0.05
            ))

        # Approximate SNR (dB)
        noise_power = max(1e-12, self.noise_floor ** 2)
        signal_power = max(1e-12, rms ** 2)
        snr_db = float(10.0 * np.log10(signal_power / noise_power))

        # Normalized speech likelihood [0.0, 1.0]
        dynamic_range = max(1e-5, self.activity_threshold * 2.0)
        speech_likelihood = float(np.clip(
            (rms - self.noise_floor) / dynamic_range,
            0.0,
            1.0
        ))

        # Context State Determination:
        # 1. CANDIDATE: Prior frame was candidate or temporal verifier is in candidate state
        if (
            current_temporal_state == "NEXORA CANDIDATE"
            or last_kws_score >= self.candidate_threshold
        ):
            state = self.STATE_CANDIDATE
        # 2. ACTIVE: Acoustic energy indicates speech or intentional sound
        elif is_speech:
            state = self.STATE_ACTIVE
        # 3. QUIET: Low energy ambient silence
        else:
            state = self.STATE_QUIET

        return AcousticContext(
            state=state,
            activity_level=rms,
            speech_likelihood=speech_likelihood,
            noise_floor=self.noise_floor,
            is_speech=is_speech,
            peak_amplitude=peak,
            snr_db=snr_db
        )


class AdaptivePolicy:
    """
    Deterministic decision policy for adaptive KWS inference.
    Controls evaluation frequency while enforcing periodic safety checks.
    """

    def __init__(self, max_kws_gap_ms: float = config.MAX_KWS_GAP_MS):
        self.max_kws_gap_ms = max_kws_gap_ms
        self.last_eval_time: Optional[float] = None
        self.consecutive_skips = 0

        # Runtime telemetry counters
        self.total_windows = 0
        self.kws_evaluations = 0
        self.kws_skipped = 0

    def should_evaluate(
        self,
        context: AcousticContext,
        current_time: Optional[float] = None
    ) -> AdaptiveDecision:
        """
        Determines whether KWS feature extraction & inference must be performed.

        Policy:
        - CANDIDATE: Always evaluate (maximum verification effort).
        - ACTIVE: Always evaluate (normal speech monitoring).
        - QUIET: Skip evaluation unless gap >= MAX_KWS_GAP_MS.

        Args:
            context: AcousticContext from estimator.
            current_time: Optional wall-clock timestamp (seconds). Uses time.time() if None.

        Returns:
            AdaptiveDecision dataclass.
        """
        now = current_time if current_time is not None else time.time()

        if self.last_eval_time is None:
            # First window bootstrap
            gap_ms = float(self.max_kws_gap_ms)
            eval_required = True
            reason = "Initial bootstrap evaluation"
        else:
            gap_ms = (now - self.last_eval_time) * 1000.0

            if context.state == AcousticContextEstimator.STATE_CANDIDATE:
                eval_required = True
                reason = "CANDIDATE context forces continuous KWS evaluation"
            elif context.state == AcousticContextEstimator.STATE_ACTIVE:
                eval_required = True
                reason = "ACTIVE speech context requires normal KWS evaluation"
            else:
                # QUIET state: check safety gap
                if gap_ms >= self.max_kws_gap_ms:
                    eval_required = True
                    reason = f"Periodic safety check (gap {gap_ms:.1f}ms >= {self.max_kws_gap_ms:.1f}ms)"
                else:
                    eval_required = False
                    reason = f"QUIET context: skipped evaluation (gap {gap_ms:.1f}ms < {self.max_kws_gap_ms:.1f}ms)"

        # Update telemetry counters
        self.total_windows += 1
        if eval_required:
            self.kws_evaluations += 1
            self.last_eval_time = now
            self.consecutive_skips = 0
        else:
            self.kws_skipped += 1
            self.consecutive_skips += 1

        return AdaptiveDecision(
            should_evaluate=eval_required,
            context_state=context.state,
            reason=reason,
            gap_ms=gap_ms
        )

    def reset(self):
        """Resets internal state, counters, and timers."""
        self.last_eval_time = None
        self.consecutive_skips = 0
        self.total_windows = 0
        self.kws_evaluations = 0
        self.kws_skipped = 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns cumulative performance statistics."""
        skip_rate = (
            (self.kws_skipped / self.total_windows * 100.0)
            if self.total_windows > 0 else 0.0
        )
        return {
            "total_windows": self.total_windows,
            "kws_evaluations": self.kws_evaluations,
            "kws_skipped": self.kws_skipped,
            "kws_skip_rate_pct": skip_rate,
            "consecutive_skips": self.consecutive_skips,
            "max_kws_gap_ms": self.max_kws_gap_ms
        }

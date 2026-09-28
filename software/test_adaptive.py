"""
Unit Tests for NEXORA Adaptive Context-Aware KWS Computation
Verifies:
1. Quiet audio -> QUIET state
2. Active audio -> ACTIVE state
3. Candidate context -> CANDIDATE state
4. MAX_KWS_GAP_MS prevents indefinite skipping
5. Candidate state forces KWS evaluation
6. Adaptive policy never directly confirms a wake word
7. Existing TemporalVerifier remains responsible for confirmation
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from software.app import config
from software.app.adaptive import (
    AcousticContextEstimator,
    AdaptivePolicy,
    AcousticContext,
    AdaptiveDecision
)
from software.app.temporal import TemporalVerifier


def test_quiet_audio_quiet_state():
    """1. Quiet audio -> QUIET state."""
    estimator = AcousticContextEstimator(activity_threshold=0.005)
    # Zero energy audio
    silence = np.zeros(16000, dtype=np.float32)
    ctx = estimator.estimate_context(silence)
    assert ctx.state == AcousticContextEstimator.STATE_QUIET, f"Expected QUIET, got {ctx.state}"
    assert ctx.is_speech is False
    assert ctx.activity_level == 0.0

    # Low noise room ambient audio (RMS = 0.001 < 0.005)
    low_noise = np.random.normal(0, 0.001, 16000).astype(np.float32)
    ctx_noise = estimator.estimate_context(low_noise)
    assert ctx_noise.state == AcousticContextEstimator.STATE_QUIET, f"Expected QUIET, got {ctx_noise.state}"
    print("[PASS] Test 1: Quiet audio -> QUIET state")


def test_active_audio_active_state():
    """2. Active audio -> ACTIVE state."""
    estimator = AcousticContextEstimator(activity_threshold=0.005)
    # Active speech level audio (RMS = 0.040 > 0.005)
    speech_signal = (np.sin(np.linspace(0, 100 * np.pi, 16000)) * 0.05).astype(np.float32)
    ctx = estimator.estimate_context(speech_signal)
    assert ctx.state == AcousticContextEstimator.STATE_ACTIVE, f"Expected ACTIVE, got {ctx.state}"
    assert ctx.is_speech is True
    assert ctx.activity_level > 0.005
    assert ctx.speech_likelihood > 0.0
    print("[PASS] Test 2: Active audio -> ACTIVE state")


def test_candidate_context_candidate_state():
    """3. Candidate context -> CANDIDATE state."""
    estimator = AcousticContextEstimator(candidate_threshold=0.60)
    silence = np.zeros(16000, dtype=np.float32)

    # Triggered by prior KWS score >= candidate_threshold
    ctx_score = estimator.estimate_context(silence, last_kws_score=0.85)
    assert ctx_score.state == AcousticContextEstimator.STATE_CANDIDATE, f"Expected CANDIDATE, got {ctx_score.state}"

    # Triggered by temporal verifier state
    ctx_temp = estimator.estimate_context(silence, current_temporal_state="NEXORA CANDIDATE")
    assert ctx_temp.state == AcousticContextEstimator.STATE_CANDIDATE, f"Expected CANDIDATE, got {ctx_temp.state}"
    print("[PASS] Test 3: Candidate context -> CANDIDATE state")


def test_max_gap_prevents_indefinite_skipping():
    """4. MAX_KWS_GAP_MS prevents indefinite skipping."""
    max_gap_ms = 800.0
    policy = AdaptivePolicy(max_kws_gap_ms=max_gap_ms)
    quiet_ctx = AcousticContext(
        state="QUIET",
        activity_level=0.0001,
        speech_likelihood=0.0,
        noise_floor=0.0005,
        is_speech=False,
        peak_amplitude=0.0002,
        snr_db=-10.0
    )

    # Frame 0 at t = 0.0s: Initial bootstrap evaluation
    d0 = policy.should_evaluate(quiet_ctx, current_time=0.0)
    assert d0.should_evaluate is True, "First frame should evaluate for bootstrap"

    # Frame 1 at t = 0.2s (gap = 200ms < 800ms): Skip
    d1 = policy.should_evaluate(quiet_ctx, current_time=0.20)
    assert d1.should_evaluate is False, "Frame at 200ms should be skipped"

    # Frame 2 at t = 0.4s (gap = 400ms < 800ms): Skip
    d2 = policy.should_evaluate(quiet_ctx, current_time=0.40)
    assert d2.should_evaluate is False, "Frame at 400ms should be skipped"

    # Frame 3 at t = 0.6s (gap = 600ms < 800ms): Skip
    d3 = policy.should_evaluate(quiet_ctx, current_time=0.60)
    assert d3.should_evaluate is False, "Frame at 600ms should be skipped"

    # Frame 4 at t = 0.8s (gap = 800ms >= 800ms): Must evaluate due to safety constraint!
    d4 = policy.should_evaluate(quiet_ctx, current_time=0.80)
    assert d4.should_evaluate is True, "Frame at 800ms MUST evaluate (MAX_KWS_GAP_MS exceeded)"
    assert "Periodic safety check" in d4.reason

    print("[PASS] Test 4: MAX_KWS_GAP_MS prevents indefinite skipping")


def test_candidate_state_forces_evaluation():
    """5. Candidate state forces KWS evaluation."""
    policy = AdaptivePolicy(max_kws_gap_ms=800.0)
    cand_ctx = AcousticContext(
        state="CANDIDATE",
        activity_level=0.03,
        speech_likelihood=0.9,
        noise_floor=0.0005,
        is_speech=True,
        peak_amplitude=0.1,
        snr_db=30.0
    )

    # Initialize at t=0
    policy.should_evaluate(cand_ctx, current_time=0.0)

    # Immediately on next frame at t=0.05 (gap only 50ms): CANDIDATE still forces evaluation
    d_cand = policy.should_evaluate(cand_ctx, current_time=0.05)
    assert d_cand.should_evaluate is True, "CANDIDATE context must force continuous evaluation"
    assert "CANDIDATE" in d_cand.reason
    print("[PASS] Test 5: Candidate state forces KWS evaluation")


def test_adaptive_policy_never_directly_confirms():
    """6. Adaptive policy never directly confirms a wake word."""
    policy = AdaptivePolicy()
    cand_ctx = AcousticContext(
        state="CANDIDATE",
        activity_level=0.1,
        speech_likelihood=1.0,
        noise_floor=0.0005,
        is_speech=True,
        peak_amplitude=0.5,
        snr_db=50.0
    )
    decision = policy.should_evaluate(cand_ctx, current_time=1.0)
    assert isinstance(decision, AdaptiveDecision)
    assert hasattr(decision, "should_evaluate")
    # Decision must NOT have confirmation attribute or issue confirmation
    assert not hasattr(decision, "is_confirmed"), "AdaptiveDecision must not contain is_confirmed"
    assert not hasattr(policy, "confirm"), "AdaptivePolicy must not have confirm method"
    assert not hasattr(policy, "is_confirmed"), "AdaptivePolicy must not track confirmation directly"
    print("[PASS] Test 6: Adaptive policy never directly confirms a wake word")


def test_temporal_verifier_remains_responsible():
    """7. Existing TemporalVerifier remains responsible for confirmation."""
    temporal = TemporalVerifier(threshold=0.75, consecutive_frames_required=2)
    estimator = AcousticContextEstimator()
    policy = AdaptivePolicy()

    # Simulated candidate frame with high score (0.90)
    high_score = 0.90
    sim_audio = np.random.normal(0, 0.04, 16000).astype(np.float32)

    # Frame 1:
    ctx1 = estimator.estimate_context(sim_audio, last_kws_score=0.0)
    dec1 = policy.should_evaluate(ctx1, current_time=0.0)
    assert dec1.should_evaluate is True

    # KWS -> TemporalVerifier
    is_conf1, state1, _ = temporal.update(high_score)
    assert not is_conf1, "Frame 1 must NOT confirm when consecutive_frames_required=2"
    assert state1 == TemporalVerifier.STATE_CANDIDATE, "Frame 1 must enter STATE_CANDIDATE"

    # Frame 2:
    ctx2 = estimator.estimate_context(sim_audio, last_kws_score=high_score, current_temporal_state=state1)
    assert ctx2.state == AcousticContextEstimator.STATE_CANDIDATE
    dec2 = policy.should_evaluate(ctx2, current_time=0.2)
    assert dec2.should_evaluate is True

    # KWS -> TemporalVerifier
    is_conf2, state2, _ = temporal.update(high_score)
    assert is_conf2 is True, "Frame 2 must be confirmed by TemporalVerifier"
    assert state2 == TemporalVerifier.STATE_CONFIRMED, "State must be CONFIRMED"
    print("[PASS] Test 7: Existing TemporalVerifier remains responsible for confirmation")


def run_all_tests():
    print("==================================================")
    print(" NEXORA ADAPTIVE CONTROLLER UNIT TESTS")
    print("==================================================")
    test_quiet_audio_quiet_state()
    test_active_audio_active_state()
    test_candidate_context_candidate_state()
    test_max_gap_prevents_indefinite_skipping()
    test_candidate_state_forces_evaluation()
    test_adaptive_policy_never_directly_confirms()
    test_temporal_verifier_remains_responsible()
    print("==================================================")
    print(" ALL 7 ADAPTIVE CONTROLLER UNIT TESTS PASSED")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()

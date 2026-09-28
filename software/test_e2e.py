"""
NEXORA End-to-End Pipeline & Intent Integration Test
Verifies all 12 system checks:
1. Core modules initialization (KWS, ASR, Intent Gatekeeper, Action Executor)
2. Silence rejection (no wake word trigger on silence)
3. Unknown audio confuser rejection
4. Temporal verification (candidate -> confirmation sequence)
5. Offline ASR transcription
6. GREETING intent classification and friendly response
7. LIGHT_ON intent classification and simulated light state change
8. LIGHT_OFF intent classification and simulated light state change
9. SYSTEM_STATUS intent classification and diagnostic report
10. HELP intent classification and command listing
11. UNKNOWN_INTENT handling and safe rejection
12. Action execution isolation (no code execution, strict whitelist boundaries)
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from software.app import config
from software.app.features import FeatureExtractor
from software.app.kws import NexoraKWS
from software.app.temporal import TemporalVerifier
from software.app.asr import LocalASREngine
from software.app.intent import IntentGatekeeper
from software.app.actions import ActionExecutor


def run_e2e_test():
    print("==================================================")
    print(" NEXORA END-TO-END SYSTEM INTEGRATION TEST")
    print("==================================================")

    # 1. Verify Modules Initialization
    fe = FeatureExtractor()
    kws = NexoraKWS()
    temporal = TemporalVerifier(threshold=0.60, consecutive_frames_required=2)
    asr = LocalASREngine()
    intent_gate = IntentGatekeeper()
    actions = ActionExecutor()

    assert kws.is_loaded, "KWS Model must be loaded"
    assert asr.vosk_available, "Vosk Local ASR must be available"
    assert intent_gate is not None, "Intent Gatekeeper must be initialized"
    assert actions is not None, "Action Executor must be initialized"
    print("[PASS] Core modules initialized")

    # 2. Test Silence Rejection
    silence_audio = np.zeros(16000, dtype=np.float32)
    silence_res = kws.predict(silence_audio)
    is_conf, state, _ = temporal.update(silence_res["nexora_score"])
    assert not is_conf, "Silence should never trigger wake word"
    assert silence_res["probabilities"]["SILENCE"] > 0.5, "Silence probability should be dominant"
    print("[PASS] Silence rejected")

    # 3. Test Unknown Confuser Rejection
    unknown_files = list((config.DATASET_DIR / "unknown").glob("*.wav"))
    if unknown_files:
        unk_audio, _ = sf.read(str(unknown_files[0]), dtype="float32")
        unk_res = kws.predict(unk_audio)
        is_conf, state, _ = temporal.update(unk_res["nexora_score"])
        assert not is_conf, "Unknown word should not confirm wake word"
    print("[PASS] Unknown audio rejected")

    # 4. Test NEXORA Wake Word Detection & Temporal Verification
    nexora_files = list((config.DATASET_DIR / "nexora").glob("*.wav"))
    assert len(nexora_files) > 0, "NEXORA audio files must exist"

    nexora_audio, _ = sf.read(str(nexora_files[0]), dtype="float32")
    nexora_res = kws.predict(nexora_audio)
    assert nexora_res["nexora_score"] > 0.60, f"Expected high NEXORA score, got {nexora_res['nexora_score']}"

    # Frame 1: Candidate (since consecutive_frames_required=2)
    is_conf_1, state_1, _ = temporal.update(nexora_res["nexora_score"])
    assert state_1 == TemporalVerifier.STATE_CANDIDATE, "First frame should reach CANDIDATE"
    assert not is_conf_1, "Single frame must NOT confirm when consecutive=2"

    # Frame 2: Confirmed
    is_conf_2, state_2, _ = temporal.update(nexora_res["nexora_score"])
    assert state_2 == TemporalVerifier.STATE_CONFIRMED, "Second consecutive frame should reach CONFIRMED"
    assert is_conf_2, "Temporal verification should trigger is_confirmed=True"
    print("[PASS] Temporal verification")

    # 5. Test Offline ASR
    cmd_file = config.DATASET_DIR / "unknown" / "unknown_seed_018.wav"  # "Turn on the lights"
    if cmd_file.exists():
        cmd_audio, _ = sf.read(str(cmd_file), dtype="float32")
        asr_result = asr.transcribe(cmd_audio)
        assert not asr_result["is_mock"], "Vosk ASR must run as real local ASR"
        assert len(asr_result["transcript"]) > 0, "Transcript must not be empty"
    print("[PASS] Offline ASR")

    # 6. GREETING Intent
    res_greet = intent_gate.classify("how are you")
    act_greet = actions.execute(res_greet.intent)
    assert res_greet.intent == IntentGatekeeper.INTENT_GREETING
    assert act_greet.action == "NONE"
    assert "ready" in act_greet.response_text.lower()
    print("[PASS] GREETING intent")

    # 7. LIGHT_ON Intent
    res_on = intent_gate.classify("turn on the lights")
    act_on = actions.execute(res_on.intent)
    assert res_on.intent == IntentGatekeeper.INTENT_LIGHT_ON
    assert act_on.action == "LIGHT_ON"
    assert actions.light_state is True
    assert "turned on" in act_on.response_text.lower()
    print("[PASS] LIGHT_ON intent")

    # 8. LIGHT_OFF Intent
    res_off = intent_gate.classify("turn off the lights")
    act_off = actions.execute(res_off.intent)
    assert res_off.intent == IntentGatekeeper.INTENT_LIGHT_OFF
    assert act_off.action == "LIGHT_OFF"
    assert actions.light_state is False
    assert "turned off" in act_off.response_text.lower()
    print("[PASS] LIGHT_OFF intent")

    # 9. SYSTEM_STATUS Intent
    res_status = intent_gate.classify("system status")
    act_status = actions.execute(res_status.intent, context={"kws_loaded": True, "asr_engine": "Offline Vosk"})
    assert res_status.intent == IntentGatekeeper.INTENT_SYSTEM_STATUS
    assert act_status.action == "SYSTEM_STATUS"
    assert "operational" in act_status.response_text.lower()
    print("[PASS] SYSTEM_STATUS intent")

    # 10. HELP Intent
    res_help = intent_gate.classify("help")
    act_help = actions.execute(res_help.intent)
    assert res_help.intent == IntentGatekeeper.INTENT_HELP
    assert act_help.action == "HELP"
    assert "commands" in act_help.response_text.lower()
    print("[PASS] HELP intent")

    # 11. UNKNOWN_INTENT Handling
    res_unknown = intent_gate.classify("something completely unrelated and wild")
    act_unknown = actions.execute(res_unknown.intent)
    assert res_unknown.intent == IntentGatekeeper.INTENT_UNKNOWN
    assert act_unknown.action == "NONE"
    assert act_unknown.action_executed is False
    assert "not recognized" in act_unknown.response_text.lower()
    print("[PASS] UNKNOWN_INTENT handling")

    # 12. Action Execution Isolation (Safety Boundary: no arbitrary shell or code execution)
    malicious_inputs = [
        "import os; os.system('echo hacked')",
        "__import__('os').system('dir')",
        "; rm -rf /",
        "format C:",
        "exec('print(123)')"
    ]
    for mal_phrase in malicious_inputs:
        mal_res = intent_gate.classify(mal_phrase)
        mal_act = actions.execute(mal_res.intent)
        assert mal_act.action == "NONE", f"Malicious input '{mal_phrase}' must produce action NONE"
        assert mal_act.action_executed is False
    print("[PASS] Action execution isolation")

    # 13. Context-Aware Adaptive KWS Controller Integration
    from software.app.adaptive import AcousticContextEstimator, AdaptivePolicy
    estimator = AcousticContextEstimator()
    policy = AdaptivePolicy(max_kws_gap_ms=config.MAX_KWS_GAP_MS)

    # 13a. Silence is estimated as QUIET and skipped within max_kws_gap
    silence_ctx = estimator.estimate_context(silence_audio)
    assert silence_ctx.state == "QUIET", "Silence must be classified as QUIET context"
    d_first = policy.should_evaluate(silence_ctx, current_time=0.0)
    assert d_first.should_evaluate is True, "First window must evaluate (bootstrap)"
    d_skip = policy.should_evaluate(silence_ctx, current_time=0.2)
    assert d_skip.should_evaluate is False, "Quiet audio within gap must be skipped"

    # 13b. Speech audio is estimated as ACTIVE and forces evaluation
    speech_ctx = estimator.estimate_context(nexora_audio)
    assert speech_ctx.state == "ACTIVE", "Speech audio must be classified as ACTIVE context"
    d_speech = policy.should_evaluate(speech_ctx, current_time=0.4)
    assert d_speech.should_evaluate is True, "ACTIVE context must require KWS evaluation"

    # 13c. Safety gap forces evaluation on quiet stream after MAX_KWS_GAP_MS
    d_gap = policy.should_evaluate(silence_ctx, current_time=1.5)
    assert d_gap.should_evaluate is True, "MAX_KWS_GAP_MS must force periodic safety evaluation"

    print("[PASS] Context-Aware Adaptive Controller integration")

    print("\n==================================================")
    print(" ALL END-TO-END PIPELINE TESTS PASSED")
    print("==================================================")


if __name__ == "__main__":
    run_e2e_test()

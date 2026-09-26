"""
NEXORA - Low-Latency Edge Voice Activator Software Prototype
SIH26172 Working Software Prototype

Pipeline:
Laptop Mic (16 kHz Mono PCM)
  ↓ Audio framing (1.0s window, 200ms step)
  ↓ MFCC features (40 Mel, 13 MFCCs)
  ↓ Small custom classifier (SILENCE / UNKNOWN / NEXORA)
  ↓ Temporal verification (sliding window + consecutive frames)
  ↓ NEXORA CONFIRMED
  ↓ Command audio capture (DORMANT before confirmation)
  ↓ Open-source Local ASR (Vosk)
  ↓ Transcript
  ↓ Intent Gatekeeper (Deterministic Edge Classifier)
  ↓ Action Executor (Safe software-only simulation)
  ↓ Local Response
  ↓ Return to LISTENING
"""

import os
import sys
import time
import argparse
import tracemalloc
import numpy as np
from pathlib import Path

# Allow importing project packages
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from software.app import config
from software.app.audio_capture import AudioCaptureStream
from software.app.kws import NexoraKWS
from software.app.temporal import TemporalVerifier
from software.app.command_capture import CommandCapture
from software.app.asr import LocalASREngine
from software.app.intent import IntentGatekeeper
from software.app.actions import ActionExecutor


# System States
STATE_LISTENING = "LISTENING"
STATE_CANDIDATE = "CANDIDATE"
STATE_CONFIRMED = "CONFIRMED"
STATE_CAPTURING = "CAPTURING"
STATE_TRANSCRIBING = "TRANSCRIBING"
STATE_EXECUTING = "EXECUTING"
STATE_RESPONDING = "RESPONDING"


def get_memory_usage_mb() -> float:
    """Returns current process RAM usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024.0 * 1024.0)
    except ImportError:
        current, peak = tracemalloc.get_traced_memory()
        return peak / (1024.0 * 1024.0)


def print_banner(kws: NexoraKWS, asr: LocalASREngine, threshold: float, consecutive_frames: int):
    """Prints the NEXORA demo terminal header."""
    print("=" * 60)
    print("   NEXORA — EDGE VOICE ACTIVATOR (SOFTWARE MVP)             ")
    print("   SIH26172 Edge Acoustic Processing Prototype             ")
    print("=" * 60)
    print(f" Microphone   : ACTIVE (16000 Hz, Mono PCM)")
    print(f" KWS Engine   : LOCAL (Trained Custom Classifier)")
    print(f" Model Type   : {kws.metadata.get('model_type', 'MLP Neural Classifier')}")
    print(f" Wake word    : NEXORA (Threshold: {threshold:.2f})")
    print(f" Temporal req : {consecutive_frames} frame(s)")
    print(f" ASR Engine   : {asr.engine_type} (100% Offline)")
    print(f" Intent Gate  : LOCAL RULE-BASED (Offline)")
    print(f" Streaming    : DORMANT (Activates ONLY after NEXORA Confirmation)")
    print("=" * 60)
    print(f" [{STATE_LISTENING}] Listening for wake word: 'NEXORA'...")
    print(" (Press Ctrl+C to terminate session)\n")


def run_intent_test() -> int:
    """
    Executes a software-only test of the intent gatekeeper and action executor.
    Does not require microphone hardware.
    """
    print("=" * 60)
    print(" NEXORA INTENT GATEKEEPER TEST MODE (Software-Only)")
    print("=" * 60)

    gatekeeper = IntentGatekeeper()
    executor = ActionExecutor()

    test_phrases = [
        ("how are you", IntentGatekeeper.INTENT_GREETING, "NONE"),
        ("turn on the lights", IntentGatekeeper.INTENT_LIGHT_ON, "LIGHT_ON"),
        ("turn off the lights", IntentGatekeeper.INTENT_LIGHT_OFF, "LIGHT_OFF"),
        ("system status", IntentGatekeeper.INTENT_SYSTEM_STATUS, "SYSTEM_STATUS"),
        ("help", IntentGatekeeper.INTENT_HELP, "HELP"),
        ("something completely unrelated", IntentGatekeeper.INTENT_UNKNOWN, "NONE"),
    ]

    all_passed = True
    for phrase, expected_intent, expected_action in test_phrases:
        res = gatekeeper.classify(phrase)
        act = executor.execute(res.intent)
        status = "PASS" if (res.intent == expected_intent and act.action == expected_action) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"Transcript : \"{phrase}\"")
        print(f"[INTENT]   : {res.intent} (Confidence: {res.confidence_label})")
        print(f"[ACTION]   : {act.action} - {act.status_message}")
        print(f"[RESPONSE] : {act.response_text}")
        print(f"Result     : [{status}]\n")

    print("-" * 60)
    if all_passed:
        print("[SUCCESS] All 6 test phrases classified and executed successfully.")
        return 0
    else:
        print("[FAIL] One or more test phrases failed verification.")
        return 1


def run_prototype(
    threshold: float = config.KWS_CONFIDENCE_THRESHOLD,
    consecutive_frames: int = config.TEMPORAL_CONSECUTIVE_FRAMES,
    command_duration: float = config.COMMAND_DURATION_SEC,
    device_index: int = None,
    max_loops: int = None,
):
    """
    Runs the real-time NEXORA wake word detection, command capture,
    intent classification, and action response loop.
    """
    tracemalloc.start()

    # 1. Initialize Modules
    print("\n[INIT] Initializing acoustic and recognition modules...")
    kws = NexoraKWS(confidence_threshold=threshold)
    temporal = TemporalVerifier(
        threshold=threshold,
        consecutive_frames_required=consecutive_frames,
        cooldown_sec=config.TEMPORAL_COOLDOWN_SEC
    )
    command_recorder = CommandCapture(
        default_duration=command_duration,
        device_index=device_index
    )
    asr = LocalASREngine()
    intent_gatekeeper = IntentGatekeeper()
    action_executor = ActionExecutor()

    print_banner(kws, asr, threshold, consecutive_frames)

    if not kws.is_loaded:
        print("[NOTICE] Running with uninitialized KWS weights.")
        print("         To achieve accurate detection, record samples and run 'train.py'.")

    # 2. Start Continuous Audio Stream
    audio_stream = AudioCaptureStream(
        sample_rate=config.SAMPLE_RATE,
        window_duration=config.WINDOW_DURATION_SEC,
        step_duration=config.STEP_DURATION_SEC,
        device_index=device_index
    )

    last_log_time = 0.0
    loop_count = 0
    current_system_state = STATE_LISTENING

    try:
        with audio_stream:
            for window in audio_stream.stream_windows():
                loop_count += 1
                if max_loops and loop_count > max_loops:
                    break

                # State: LISTENING
                current_system_state = STATE_LISTENING

                # Step 1: KWS Inference
                res = kws.predict(window)
                probs = res["probabilities"]
                nexora_score = res["nexora_score"]

                # Step 2: Temporal Verification
                is_confirmed, temp_state, temp_info = temporal.update(nexora_score)

                if temp_state == TemporalVerifier.STATE_CANDIDATE:
                    current_system_state = STATE_CANDIDATE

                # Step 3: Throttled Terminal Telemetry
                now = time.time()
                if (now - last_log_time) >= 0.40 or temp_state in [
                    TemporalVerifier.STATE_CANDIDATE,
                    TemporalVerifier.STATE_CONFIRMED,
                    TemporalVerifier.STATE_REJECTED
                ]:
                    last_log_time = now
                    score_str = (
                        f"NEXORA: {probs.get('NEXORA', 0.0):.2f} | "
                        f"UNKNOWN: {probs.get('UNKNOWN', 0.0):.2f} | "
                        f"SILENCE: {probs.get('SILENCE', 0.0):.2f}"
                    )
                    state_tag = f"[{temp_state:16s}]"
                    print(f"\r{state_tag} {score_str} (Consecutive: {temp_info.get('consecutive_count', 0)})", end="")
                    if temp_state != TemporalVerifier.STATE_IDLE and temp_state != TemporalVerifier.STATE_COOLDOWN:
                        print()  # Advance line on candidate/confirmed/rejected

                # Step 4: Handle Wake Word Confirmation & Downstream Pipeline
                if is_confirmed:
                    current_system_state = STATE_CONFIRMED

                    print("\n" + "=" * 60)
                    print(" >>> [WAKE WORD DETECTED] NEXORA CONFIRMED <<<")
                    print(f" Confidence: {nexora_score:.3f} (Threshold: {threshold:.2f})")
                    print(f" Temporal frames verified: {consecutive_frames}")
                    print("=" * 60)

                    # State: CAPTURING
                    current_system_state = STATE_CAPTURING
                    print("\n[CAPTURE]")
                    print(f"Recording command speech ({command_duration:.1f}s window)...")

                    cmd_audio, cmd_file, cap_dur = command_recorder.capture_command(
                        duration=command_duration
                    )
                    print(f"Command captured: {len(cmd_audio)} samples ({cap_dur:.2f}s)")
                    if cmd_file:
                        print(f"Archived to: {cmd_file.name}")

                    # State: TRANSCRIBING
                    current_system_state = STATE_TRANSCRIBING
                    print("\n[ASR]")
                    asr_res = asr.transcribe(cmd_audio)
                    transcript = asr_res["transcript"]
                    print(f"Transcript: \"{transcript}\"")

                    # State: EXECUTING (Intent Gatekeeper + Action Executor)
                    current_system_state = STATE_EXECUTING
                    t_intent_0 = time.perf_counter()
                    intent_res = intent_gatekeeper.classify(transcript)
                    t_intent_ms = (time.perf_counter() - t_intent_0) * 1000.0

                    t_act_0 = time.perf_counter()
                    action_res = action_executor.execute(
                        intent_res.intent,
                        context={
                            "kws_loaded": kws.is_loaded,
                            "asr_engine": asr_res["engine"],
                            "transcript": transcript
                        }
                    )
                    t_act_ms = (time.perf_counter() - t_act_0) * 1000.0
                    total_post_asr_ms = t_intent_ms + t_act_ms

                    print("\n[INTENT]")
                    print(f"Detected: {intent_res.intent}")
                    print(f"Confidence: {intent_res.confidence_label}")

                    print("\n[ACTION]")
                    print(f"{action_res.action}")
                    print(f"{action_res.status_message}")

                    # State: RESPONDING
                    current_system_state = STATE_RESPONDING
                    print("\n[RESPONSE]")
                    print(f"{action_res.response_text}")

                    # Step 5: Software Performance Telemetry
                    ram_mb = get_memory_usage_mb()
                    total_kws_latency = res["total_latency_ms"]
                    model_size_kb = (
                        kws.metadata.get("model_size_bytes", 0) / 1024.0
                        if kws.metadata else 0.0
                    )

                    print("\n" + "-" * 60)
                    print(" [SOFTWARE PERFORMANCE MEASUREMENTS]")
                    print(" (Note: Measurements taken on Host PC - NOT ESP32 target)")
                    print(f"  - Feature extraction latency : {res['feature_latency_ms']:.2f} ms")
                    print(f"  - Model inference latency    : {res['inference_latency_ms']:.2f} ms")
                    print(f"  - Total KWS detection latency: {total_kws_latency:.2f} ms")
                    print(f"  - Command capture duration   : {cap_dur:.2f} s")
                    print(f"  - Local ASR latency          : {asr_res['latency_ms']:.2f} ms")
                    print(f"  - Intent classification lat. : {t_intent_ms:.3f} ms")
                    print(f"  - Action execution latency   : {t_act_ms:.3f} ms")
                    print(f"  - Total post-ASR latency     : {total_post_asr_ms:.3f} ms")
                    print(f"  - KWS Model file size        : {model_size_kb:.2f} KB")
                    print(f"  - Process RAM usage (RSS)    : {ram_mb:.2f} MB")
                    print("-" * 60)
                    print(" [STATUS] READY FOR NEXT COMMAND. Listening for 'NEXORA'...\n")

                    # State: Return to LISTENING
                    current_system_state = STATE_LISTENING

    except KeyboardInterrupt:
        print("\n\n[EXIT] Terminating NEXORA session cleanly.")
    finally:
        audio_stream.stop()
        tracemalloc.stop()
        print("[EXIT] Audio stream closed. System shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="NEXORA Real-Time Wake Word, Intent & Action Activator")
    parser.add_argument("--threshold", type=float, default=config.KWS_CONFIDENCE_THRESHOLD, help="KWS confidence threshold [0.0 - 1.0]")
    parser.add_argument("--consecutive", type=int, default=config.TEMPORAL_CONSECUTIVE_FRAMES, help="Consecutive frames required for confirmation")
    parser.add_argument("--command-duration", type=float, default=config.COMMAND_DURATION_SEC, help="Command capture duration in seconds")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index")
    parser.add_argument("--test-run", action="store_true", help="Runs for 5 seconds to verify pipeline without infinite loop")
    parser.add_argument("--intent-test", action="store_true", help="Runs software-only intent gatekeeper test and exits")

    args = parser.parse_args()

    if args.intent_test:
        sys.exit(run_intent_test())

    max_loops = 25 if args.test_run else None  # 25 loops * 200ms = 5 seconds

    run_prototype(
        threshold=args.threshold,
        consecutive_frames=args.consecutive,
        command_duration=args.command_duration,
        device_index=args.device,
        max_loops=max_loops
    )


if __name__ == "__main__":
    main()

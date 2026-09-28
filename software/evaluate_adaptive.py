"""
NEXORA Adaptive KWS Robustness & Baseline Evaluation Harness
Evaluates identical audio streams across two modes:
1. BASELINE (Continuous fixed-rate KWS evaluation)
2. ADAPTIVE (Context-aware adaptive KWS computation)

Test Categories:
A. WAKE WORD    - Known NEXORA target recordings
B. NON-WAKE     - Ordinary speech commands and conversational phrases
C. SILENCE      - Low-energy ambient recordings
D. NOISY WAKE   - NEXORA wake recordings mixed with background noise
E. UNKNOWN      - Confuser words and out-of-vocabulary phrases

Key Principles:
- Identical Input: Exactly the same audio stream is passed to Baseline and Adaptive.
- No Invented Metrics: Latencies and energy savings are strictly bounded by empirical measurements.
- Temporal Integrity: Evaluates standard TemporalVerifier configurations (--consecutive 1 and --consecutive 2).
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from software.app import config
from software.app.kws import NexoraKWS
from software.app.temporal import TemporalVerifier
from software.app.adaptive import AcousticContextEstimator, AdaptivePolicy


@dataclass
class StreamRunResult:
    windows: int
    evaluations: int
    skipped: int
    confirmations: int
    rejected_candidates: int
    kws_compute_ms: float
    total_compute_ms: float
    confirmation_timestamps: List[float]


def evaluate_stream(
    stream: np.ndarray,
    mode: str,
    kws: NexoraKWS,
    consecutive_frames: int = 1,
    threshold: float = config.KWS_CONFIDENCE_THRESHOLD,
    max_kws_gap_ms: float = config.MAX_KWS_GAP_MS
) -> StreamRunResult:
    """
    Evaluates an audio stream using either BASELINE or ADAPTIVE mode.
    Sliding window parameters: 16,000 samples window, 3,200 samples step (200 ms).
    """
    window_size = config.SAMPLES_PER_WINDOW  # 16000
    step_size = config.SAMPLES_PER_STEP      # 3200

    temporal = TemporalVerifier(
        threshold=threshold,
        consecutive_frames_required=consecutive_frames,
        cooldown_sec=config.TEMPORAL_COOLDOWN_SEC
    )

    estimator = AcousticContextEstimator() if mode == "ADAPTIVE" else None
    policy = AdaptivePolicy(max_kws_gap_ms=max_kws_gap_ms) if mode == "ADAPTIVE" else None

    total_windows = 0
    evaluations = 0
    skipped = 0
    confirmations = 0
    rejected = 0
    kws_compute_ms = 0.0
    total_compute_ms = 0.0
    conf_timestamps: List[float] = []
    last_score = 0.0

    num_samples = len(stream)
    if num_samples < window_size:
        # Pad to at least one window
        stream = np.pad(stream, (0, window_size - num_samples), mode="constant")
        num_samples = len(stream)

    for step_idx, start in enumerate(range(0, num_samples - window_size + 1, step_size)):
        total_windows += 1
        window = stream[start : start + window_size]
        t_sec = step_idx * config.STEP_DURATION_SEC

        t_step_start = time.perf_counter()

        if mode == "ADAPTIVE":
            ctx = estimator.estimate_context(
                window,
                last_kws_score=last_score,
                current_temporal_state=temporal.current_state
            )
            decision = policy.should_evaluate(ctx, current_time=t_sec)
            should_eval = decision.should_evaluate
        else:
            should_eval = True

        if should_eval:
            evaluations += 1
            t_kws_0 = time.perf_counter()
            res = kws.predict(window)
            t_kws_1 = time.perf_counter()
            kws_compute_ms += (t_kws_1 - t_kws_0) * 1000.0

            score = res["nexora_score"]
            last_score = score
            is_conf, state, _ = temporal.update(score)

            if is_conf:
                confirmations += 1
                conf_timestamps.append(t_sec)
            if state == TemporalVerifier.STATE_REJECTED:
                rejected += 1
        else:
            skipped += 1
            # Preserve temporal state safely (zero confirmation)
            is_conf = False

        total_compute_ms += (time.perf_counter() - t_step_start) * 1000.0

    return StreamRunResult(
        windows=total_windows,
        evaluations=evaluations,
        skipped=skipped,
        confirmations=confirmations,
        rejected_candidates=rejected,
        kws_compute_ms=kws_compute_ms,
        total_compute_ms=total_compute_ms,
        confirmation_timestamps=conf_timestamps
    )


def load_dataset_samples() -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """
    Loads audio samples from repository datasets into standard evaluation categories.
    Returns:
        Dict mapping category name to list of (sample_id, audio_array) tuples.
    """
    categories: Dict[str, List[Tuple[str, np.ndarray]]] = {
        "Wake": [],
        "Non-Wake": [],
        "Silence": [],
        "Unknown": [],
        "Noisy Wake": []
    }

    nex_dir = config.DATASET_DIR / "nexora"
    sil_dir = config.DATASET_DIR / "silence"
    unk_dir = config.DATASET_DIR / "unknown"

    # 1. Category A: WAKE (All 20 target wake words)
    nex_files = sorted(list(nex_dir.glob("*.wav")))
    wake_samples = []
    for f in nex_files:
        audio, _ = sf.read(str(f), dtype="float32")
        wake_samples.append((f.stem, audio))
    categories["Wake"] = wake_samples

    # 2. Category C: SILENCE (All 15 silence recordings)
    sil_files = sorted(list(sil_dir.glob("*.wav")))
    sil_samples = []
    for f in sil_files:
        audio, _ = sf.read(str(f), dtype="float32")
        sil_samples.append((f.stem, audio))
    categories["Silence"] = sil_samples

    # 3. Category E: UNKNOWN (First 15 confuser words: Nebula, Nectar, Nokia, Alexa, etc.)
    unk_files = sorted(list(unk_dir.glob("*.wav")))
    unk_confusers = []
    for f in unk_files[:15]:
        audio, _ = sf.read(str(f), dtype="float32")
        unk_confusers.append((f.stem, audio))
    categories["Unknown"] = unk_confusers

    # 4. Category B: NON-WAKE SPEECH (Speech command phrases: "turn on lights", "system online", etc.)
    non_wake_speech = []
    for f in unk_files[15:25]:
        audio, _ = sf.read(str(f), dtype="float32")
        non_wake_speech.append((f.stem, audio))
    categories["Non-Wake"] = non_wake_speech

    # 5. Category D: NOISY WAKE (Wake words mixed with real ambient background recordings)
    noisy_wake_samples = []
    for i, (wake_id, wake_audio) in enumerate(wake_samples):
        noise_audio = sil_samples[i % len(sil_samples)][1]
        reps = int(np.ceil(len(wake_audio) / len(noise_audio))) + 1
        tiled_noise = np.tile(noise_audio, reps)[:len(wake_audio)]
        # Mix wake audio with ambient background noise
        noisy_audio = wake_audio + tiled_noise
        noisy_wake_samples.append((f"{wake_id}_noisy", noisy_audio))
    categories["Noisy Wake"] = noisy_wake_samples

    return categories


def run_evaluation(
    consecutive_frames: int = 1,
    threshold: float = config.KWS_CONFIDENCE_THRESHOLD,
    max_kws_gap_ms: float = config.MAX_KWS_GAP_MS
) -> Dict[str, Any]:
    """
    Executes the full evaluation across all categories with identical audio streams.
    """
    kws = NexoraKWS(confidence_threshold=threshold)
    dataset = load_dataset_samples()

    # Pre-load ambient silence padding for realistic stream framing (0.8s lead, 0.8s trail)
    sil_audio = dataset["Silence"][0][1] if dataset["Silence"] else np.zeros(12800, dtype=np.float32)
    lead_silence = sil_audio[:int(0.8 * config.SAMPLE_RATE)]
    trail_silence = sil_audio[:int(0.8 * config.SAMPLE_RATE)]

    category_results: Dict[str, Dict[str, Any]] = {}

    total_windows = 0
    total_base_evals = 0
    total_adapt_evals = 0
    total_adapt_skips = 0
    total_base_compute_ms = 0.0
    total_adapt_compute_ms = 0.0

    base_true_detections = 0
    adapt_true_detections = 0
    base_false_activations = 0
    adapt_false_activations = 0

    for cat_name, samples in dataset.items():
        cat_samples_count = len(samples)
        cat_base_confirms = 0
        cat_adapt_confirms = 0
        cat_base_evals = 0
        cat_adapt_evals = 0
        cat_adapt_skips = 0
        cat_windows = 0

        is_wake_target = cat_name in ["Wake", "Noisy Wake"]

        for sample_id, audio in samples:
            # Construct identical stream: [0.8s ambient silence + sample + 0.8s ambient silence]
            stream = np.concatenate([lead_silence, audio, trail_silence])

            # 1. Run BASELINE on stream
            base_res = evaluate_stream(
                stream,
                mode="BASELINE",
                kws=kws,
                consecutive_frames=consecutive_frames,
                threshold=threshold,
                max_kws_gap_ms=max_kws_gap_ms
            )

            # 2. Run ADAPTIVE on the exact same stream
            adapt_res = evaluate_stream(
                stream,
                mode="ADAPTIVE",
                kws=kws,
                consecutive_frames=consecutive_frames,
                threshold=threshold,
                max_kws_gap_ms=max_kws_gap_ms
            )

            # Accumulate category metrics
            cat_windows += base_res.windows
            cat_base_evals += base_res.evaluations
            cat_adapt_evals += adapt_res.evaluations
            cat_adapt_skips += adapt_res.skipped
            cat_base_confirms += (1 if base_res.confirmations > 0 else 0)
            cat_adapt_confirms += (1 if adapt_res.confirmations > 0 else 0)

            total_base_compute_ms += base_res.kws_compute_ms
            total_adapt_compute_ms += adapt_res.kws_compute_ms

        total_windows += cat_windows
        total_base_evals += cat_base_evals
        total_adapt_evals += cat_adapt_evals
        total_adapt_skips += cat_adapt_skips

        if is_wake_target:
            base_true_detections += cat_base_confirms
            adapt_true_detections += cat_adapt_confirms
        else:
            base_false_activations += cat_base_confirms
            adapt_false_activations += cat_adapt_confirms

        category_results[cat_name] = {
            "samples": cat_samples_count,
            "baseline_confirmations": cat_base_confirms,
            "adaptive_confirmations": cat_adapt_confirms,
            "windows": cat_windows,
            "baseline_evaluations": cat_base_evals,
            "adaptive_evaluations": cat_adapt_evals,
            "adaptive_skips": cat_adapt_skips,
            "skip_rate_pct": (cat_adapt_skips / cat_windows * 100.0) if cat_windows > 0 else 0.0
        }

    # Global Calculations
    skip_rate = (total_adapt_skips / total_windows * 100.0) if total_windows > 0 else 0.0
    compute_reduction = (
        (1.0 - (total_adapt_compute_ms / max(1e-5, total_base_compute_ms))) * 100.0
    )

    if base_true_detections > 0:
        retention_pct = (adapt_true_detections / base_true_detections) * 100.0
        retention_str = f"{retention_pct:.1f}% ({adapt_true_detections}/{base_true_detections})"
    else:
        retention_pct = 0.0
        retention_str = f"N/A (0 baseline detections)"

    # Print Report adhering strictly to specification
    print("=" * 56)
    print("NEXORA ADAPTIVE KWS ROBUSTNESS EVALUATION")
    print("=" * 56)
    print("\nConfiguration:")
    print(f"Threshold: {threshold:.2f}")
    print(f"Consecutive frames: {consecutive_frames}")
    print(f"MAX_KWS_GAP_MS: {max_kws_gap_ms:.1f} ms")
    print("\n" + "-" * 56)
    print("CATEGORY RESULTS")
    print("-" * 56)
    print(f"{'Category':<15} {'Samples':>8} {'Baseline':>10} {'Adaptive':>10}")
    for cat in ["Wake", "Non-Wake", "Silence", "Unknown", "Noisy Wake"]:
        r = category_results[cat]
        print(f"{cat:<15} {r['samples']:>8} {r['baseline_confirmations']:>10} {r['adaptive_confirmations']:>10}")

    print("\n" + "-" * 56)
    print("COMPUTE")
    print("-" * 56)
    print(f"Baseline KWS evaluations: {total_base_evals}")
    print(f"Adaptive KWS evaluations: {total_adapt_evals}")
    print(f"KWS skip rate: {skip_rate:.1f}%")
    print(f"Host-PC KWS compute reduction: {compute_reduction:.1f}%")

    print("\n" + "-" * 56)
    print("DETECTION")
    print("-" * 56)
    print(f"Baseline true detections: {base_true_detections}")
    print(f"Adaptive true detections: {adapt_true_detections}")
    print(f"Baseline false activations: {base_false_activations}")
    print(f"Adaptive false activations: {adapt_false_activations}")
    print(f"\nAdaptive detection retention: {retention_str}")
    print("Detection latency: Detection latency not measurable from current recording metadata.")

    print("\n" + "-" * 56)
    print("IMPORTANT LIMITATION")
    print("-" * 56)
    print("Host-PC compute measurements are NOT electrical ESP32")
    print("energy measurements.")
    print()
    print("MAX_KWS_GAP_MS provides a bounded evaluation fallback;")
    print("it does NOT mathematically guarantee wake-word detection.")
    print("=" * 56 + "\n")

    report_payload = {
        "configuration": {
            "threshold": threshold,
            "consecutive_frames": consecutive_frames,
            "max_kws_gap_ms": max_kws_gap_ms
        },
        "category_results": category_results,
        "compute": {
            "baseline_evaluations": total_base_evals,
            "adaptive_evaluations": total_adapt_evals,
            "adaptive_skips": total_adapt_skips,
            "kws_skip_rate_pct": float(skip_rate),
            "baseline_kws_compute_ms": float(total_base_compute_ms),
            "adaptive_kws_compute_ms": float(total_adapt_compute_ms),
            "host_pc_compute_reduction_pct": float(compute_reduction)
        },
        "detection": {
            "baseline_true_detections": base_true_detections,
            "adaptive_true_detections": adapt_true_detections,
            "baseline_false_activations": base_false_activations,
            "adaptive_false_activations": adapt_false_activations,
            "adaptive_detection_retention_pct": float(retention_pct),
            "detection_latency": "Detection latency not measurable from current recording metadata."
        }
    }

    out_file = config.LOGS_DIR / f"adaptive_evaluation_consecutive_{consecutive_frames}.json"
    with open(out_file, "w") as f:
        json.dump(report_payload, f, indent=2)

    return report_payload


def main():
    parser = argparse.ArgumentParser(description="NEXORA Adaptive KWS Robustness Evaluation Harness")
    parser.add_argument("--consecutive", type=int, default=1, help="Consecutive frames required for confirmation (default: 1)")
    parser.add_argument("--threshold", type=float, default=config.KWS_CONFIDENCE_THRESHOLD, help="Confidence threshold [0.0 - 1.0]")
    parser.add_argument("--max-gap-ms", type=float, default=config.MAX_KWS_GAP_MS, help="Max evaluation gap in ms")
    parser.add_argument("--run-both-consecutive", action="store_true", help="Evaluates consecutive=1 and consecutive=2 consecutively")

    args = parser.parse_args()

    if args.run_both_consecutive:
        print("\n>>> EVALUATING CONFIGURATION A: --consecutive 1 <<<\n")
        run_evaluation(consecutive_frames=1, threshold=args.threshold, max_kws_gap_ms=args.max_gap_ms)
        print("\n>>> EVALUATING CONFIGURATION B: --consecutive 2 <<<\n")
        run_evaluation(consecutive_frames=2, threshold=args.threshold, max_kws_gap_ms=args.max_gap_ms)
    else:
        run_evaluation(consecutive_frames=args.consecutive, threshold=args.threshold, max_kws_gap_ms=args.max_gap_ms)


if __name__ == "__main__":
    main()
